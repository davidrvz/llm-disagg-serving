"""FastAPI server that wraps PrefillWorker and exposes it over HTTP."""

import logging

import uvicorn
from fastapi import FastAPI

from router.models import PrefillRequest, PrefillResponse
from workers.prefill.worker import PrefillWorker

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Prefill Worker")
worker = PrefillWorker()


@app.on_event("startup")
async def startup() -> None:
    worker.load()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/prefill", response_model=PrefillResponse)
async def prefill(req: PrefillRequest) -> PrefillResponse:
    """Tokenize and prefill the prompt, returning a serialized KV cache.

    TODO: Run worker.prefill() in a thread pool (run_in_executor) so the
          event loop is not blocked during the forward pass.
    TODO: Add request queuing / backpressure when the GPU is saturated.
    """
    kv_cache_b64, input_ids, prompt_tokens = worker.prefill(req.prompt)
    return PrefillResponse(
        request_id=req.request_id,
        kv_cache_b64=kv_cache_b64,
        input_ids=input_ids,
        prompt_tokens=prompt_tokens,
    )


if __name__ == "__main__":
    uvicorn.run("workers.prefill.server:app", host="0.0.0.0", port=8001, reload=False)
