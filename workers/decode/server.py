"""FastAPI server that wraps DecodeWorker and exposes it over HTTP."""

import logging

import uvicorn
from fastapi import FastAPI

from router.models import DecodeRequest, DecodeResponse
from workers.decode.worker import DecodeWorker

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Decode Worker")
worker = DecodeWorker()


@app.on_event("startup")
async def startup() -> None:
    worker.load()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest) -> DecodeResponse:
    """Run the autoregressive decode loop and return the completed text.

    TODO: Run worker.decode() in a thread pool executor.
    TODO: Replace with a streaming SSE endpoint for token-by-token output.
    """
    text, generated_tokens = worker.decode(
        kv_cache_b64=req.kv_cache_b64,
        input_ids=req.input_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return DecodeResponse(
        request_id=req.request_id,
        text=text,
        generated_tokens=generated_tokens,
    )


if __name__ == "__main__":
    uvicorn.run("workers.decode.server:app", host="0.0.0.0", port=8002, reload=False)
