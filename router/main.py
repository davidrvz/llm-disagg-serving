"""
Router / gateway — the single entry point clients talk to.

Flow per request:
  1. POST /generate  →  pick prefill worker via scheduler
  2. Forward prompt  →  prefill worker returns KV cache + input_ids
  3. Pick decode worker, forward KV cache + input_ids
  4a. stream=False  →  wait for full text, return JSON
  4b. stream=True   →  proxy the decode worker's SSE stream to the client

TODO: Add request-level tracing (opentelemetry).
TODO: Surface per-request latency breakdown in response headers.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from router.models import (
    DecodeRequest,
    DecodeResponse,
    GenerateRequest,
    GenerateResponse,
    PrefillRequest,
    PrefillResponse,
)
from router.scheduler import RoundRobinScheduler
from workers.config import MODEL_NAME

_scheduler = RoundRobinScheduler(
    prefill_urls=os.getenv("PREFILL_URLS", "http://localhost:8001").split(","),
    decode_urls=os.getenv("DECODE_URLS", "http://localhost:8002").split(","),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Single shared client for all outbound calls — avoids per-request TCP overhead.
    async with httpx.AsyncClient() as client:
        app.state.http = client
        yield


app = FastAPI(title="LLM Disagg Router", lifespan=lifespan)


def _http(request: Request) -> httpx.AsyncClient:
    return request.app.state.http


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
async def config() -> dict:
    """Return the active model and worker pool — used by the UI."""
    return {
        "model_name": MODEL_NAME,
        "prefill_workers": _scheduler.prefill_urls,
        "decode_workers": _scheduler.decode_urls,
    }


async def _prefill(
    client: httpx.AsyncClient,
    prefill_url: str,
    payload: PrefillRequest,
) -> PrefillResponse:
    resp = await client.post(
        f"{prefill_url}/prefill",
        json=payload.model_dump(),
        timeout=120.0,
    )
    if resp.status_code != 200:
        raise HTTPException(502, f"Prefill worker error: {resp.text}")
    return PrefillResponse(**resp.json())


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse | StreamingResponse:
    request_id = str(uuid.uuid4())
    prefill_url = _scheduler.next_prefill()
    decode_url = _scheduler.next_decode()
    client = _http(request)

    prefill_result = await _prefill(
        client,
        prefill_url,
        PrefillRequest(
            request_id=request_id,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        ),
    )

    decode_payload = DecodeRequest(
        request_id=request_id,
        kv_cache_b64=prefill_result.kv_cache_b64,
        input_ids=prefill_result.input_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )

    if req.stream:
        return _stream_response(client, decode_url, decode_payload, prefill_result, prefill_url, decode_url)

    # ── Non-streaming path ────────────────────────────────────────────────────
    resp = await client.post(
        f"{decode_url}/decode",
        json=decode_payload.model_dump(),
        timeout=300.0,
    )
    if resp.status_code != 200:
        raise HTTPException(502, f"Decode worker error: {resp.text}")
    decode_result = DecodeResponse(**resp.json())

    return GenerateResponse(
        text=decode_result.text,
        prompt_tokens=prefill_result.prompt_tokens,
        generated_tokens=decode_result.generated_tokens,
        prefill_worker=prefill_url,
        decode_worker=decode_url,
    )


def _stream_response(
    client: httpx.AsyncClient,
    decode_url: str,
    decode_payload: DecodeRequest,
    prefill_result: PrefillResponse,
    prefill_worker: str,
    decode_worker: str,
) -> StreamingResponse:
    """Proxy the decode worker's SSE stream straight to the client.

    The decode worker emits:
        data: <token text>\n\n
        data: [DONE]\n\n

    We prepend a metadata event so the client knows prompt_tokens and which
    workers handled the request, then forward every subsequent event as-is.
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        meta = (
            f"data: [META] prompt_tokens={prefill_result.prompt_tokens}"
            f" prefill_worker={prefill_worker}"
            f" decode_worker={decode_worker}\n\n"
        )
        yield meta

        async with client.stream(
            "POST",
            f"{decode_url}/decode/stream",
            json=decode_payload.model_dump(),
            timeout=None,
        ) as resp:
            if resp.status_code != 200:
                yield f"data: [ERROR] decode worker returned {resp.status_code}\n\n"
                return
            async for chunk in resp.aiter_text():
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")
