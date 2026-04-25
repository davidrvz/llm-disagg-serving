from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    prefill_worker: str
    decode_worker: str


# ── Internal worker messages ───────────────────────────────────────────────────

class PrefillRequest(BaseModel):
    request_id: str
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 1.0


class PrefillResponse(BaseModel):
    request_id: str
    # KV cache serialized by kv_transfer.serializer (base64-encoded bytes)
    kv_cache_b64: str
    input_ids: list[int]
    prompt_tokens: int


class DecodeRequest(BaseModel):
    request_id: str
    kv_cache_b64: str
    input_ids: list[int]
    max_new_tokens: int = 128
    temperature: float = 1.0


class DecodeResponse(BaseModel):
    request_id: str
    text: str
    generated_tokens: int
