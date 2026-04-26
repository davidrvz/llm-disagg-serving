"""FastAPI server that wraps PrefillWorker and exposes it over HTTP."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from router.models import PrefillRequest, PrefillResponse
from workers.config import MODEL_NAME
from workers.prefill.worker import PrefillWorker

logging.basicConfig(level=logging.INFO)

worker = PrefillWorker()
# Single-threaded executor keeps all GPU ops on one thread.
_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    worker.load(MODEL_NAME)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Prefill Worker", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/prefill", response_model=PrefillResponse)
async def prefill(req: PrefillRequest) -> PrefillResponse:
    """Tokenize and prefill the prompt, returning a serialized KV cache.

    The forward pass is blocking (GPU/CPU bound), so it runs in a thread pool
    to avoid stalling the event loop for other requests.

    TODO: Add request queuing / backpressure when the GPU is saturated.
    """
    loop = asyncio.get_running_loop()
    kv_cache_b64, input_ids, prompt_tokens, _ = await loop.run_in_executor(
        _executor, worker.prefill, req.prompt
    )
    return PrefillResponse(
        request_id=req.request_id,
        kv_cache_b64=kv_cache_b64,
        input_ids=input_ids,
        prompt_tokens=prompt_tokens,
    )


if __name__ == "__main__":
    uvicorn.run("workers.prefill.server:app", host="0.0.0.0", port=8001, reload=False)
