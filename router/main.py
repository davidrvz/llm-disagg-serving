"""
Router / gateway — the single entry point clients talk to.

Flow per request:
  1. POST /generate  →  pick prefill worker via scheduler
  2. Forward prompt  →  prefill worker returns KV cache + input_ids
  3. Pick decode worker, forward KV cache + input_ids
  4. Return final text to client

TODO: Add streaming support (SSE) for step 3.
TODO: Add request-level tracing (opentelemetry).
"""

import os
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from router.models import (
    DecodeRequest,
    DecodeResponse,
    GenerateRequest,
    GenerateResponse,
    PrefillRequest,
    PrefillResponse,
)
from router.scheduler import RoundRobinScheduler

app = FastAPI(title="LLM Disagg Router")

_scheduler = RoundRobinScheduler(
    prefill_urls=os.getenv("PREFILL_URLS", "http://prefill:8001").split(","),
    decode_urls=os.getenv("DECODE_URLS", "http://decode:8002").split(","),
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """End-to-end generate endpoint.

    Orchestrates a prefill worker followed by a decode worker and returns
    the completed text.  Streaming (req.stream=True) is not yet implemented.

    TODO: Implement SSE streaming path.
    TODO: Surface per-request latency breakdown in response headers.
    """
    request_id = str(uuid.uuid4())
    prefill_url = _scheduler.next_prefill()
    decode_url = _scheduler.next_decode()

    # ── Step 1: Prefill ───────────────────────────────────────────────────────
    prefill_payload = PrefillRequest(
        request_id=request_id,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{prefill_url}/prefill", json=prefill_payload.model_dump())

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Prefill worker error: {resp.text}")

    prefill_result = PrefillResponse(**resp.json())

    # ── Step 2: Decode ────────────────────────────────────────────────────────
    decode_payload = DecodeRequest(
        request_id=request_id,
        kv_cache_b64=prefill_result.kv_cache_b64,
        input_ids=prefill_result.input_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{decode_url}/decode", json=decode_payload.model_dump())

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Decode worker error: {resp.text}")

    decode_result = DecodeResponse(**resp.json())

    return GenerateResponse(
        text=decode_result.text,
        prompt_tokens=prefill_result.prompt_tokens,
        generated_tokens=decode_result.generated_tokens,
        prefill_worker=prefill_url,
        decode_worker=decode_url,
    )
