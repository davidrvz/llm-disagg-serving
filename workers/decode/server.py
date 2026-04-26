"""FastAPI server that wraps DecodeWorker and exposes it over HTTP."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from router.models import DecodeRequest, DecodeResponse
from workers.config import MODEL_NAME
from workers.decode.worker import DecodeWorker

logging.basicConfig(level=logging.INFO)

worker = DecodeWorker()
# Single-threaded executor keeps all GPU ops on one thread.
_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    worker.load(MODEL_NAME)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Decode Worker", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest) -> DecodeResponse:
    """Run the full decode loop and return completed text.

    Runs in a thread pool so the event loop stays free during inference.

    TODO: Add request queuing / backpressure.
    """
    loop = asyncio.get_running_loop()
    text, generated_tokens = await loop.run_in_executor(
        _executor,
        lambda: worker.decode(
            req.kv_cache_b64, req.input_ids, req.max_new_tokens, req.temperature
        ),
    )
    return DecodeResponse(
        request_id=req.request_id,
        text=text,
        generated_tokens=generated_tokens,
    )


@app.post("/decode/stream")
async def decode_stream(req: DecodeRequest) -> StreamingResponse:
    """Stream tokens one at a time as SSE events.

    Emits:
        data: <token text>\\n\\n   — for each token
        data: [DONE]\\n\\n         — when generation is complete

    The sync generator runs inside a single-threaded executor so GPU ops
    never block the event loop and always run on the same thread.
    """
    _done = object()

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        gen = worker.decode_stream(
            req.kv_cache_b64, req.input_ids, req.max_new_tokens, req.temperature
        )
        while True:
            token = await loop.run_in_executor(_executor, next, gen, _done)
            if token is _done:
                break
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("workers.decode.server:app", host="0.0.0.0", port=8002, reload=False)
