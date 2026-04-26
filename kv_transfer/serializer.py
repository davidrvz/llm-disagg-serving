"""
Serialize and deserialize KV cache tensors for transfer between workers.

Current strategy: pickle → base64 string (simple, zero extra deps).
This is intentionally naive — it's the right starting point for profiling
before optimizing the transfer path.

TODO: Benchmark pickle vs. numpy tobytes vs. safetensors.
TODO: Compress with lz4 or zstd when transfer happens over the network.
TODO: Replace entirely with zero-copy shared memory path once the
      kv_transfer.transport layer is wired up.
"""

from __future__ import annotations

import base64
import io
import pickle
from typing import Any

import torch
from transformers.cache_utils import DynamicCache


def serialize_kv_cache(past_key_values: Any) -> str:
    """Serialize a HuggingFace past_key_values cache to a base64 string.

    Accepts both the legacy tuple-of-2-tuples format and the newer
    DynamicCache object (transformers ≥ 4.47, which iterates as 3-tuples).
    Only key and value tensors are preserved; any extra fields are dropped.

    Returns
    -------
    str
        Base64-encoded pickle blob safe to embed in JSON.

    TODO: Profile and replace pickle with a faster binary format.
    """
    buf = io.BytesIO()
    # Index by position ([0], [1]) rather than unpacking so this works with
    # both 2-tuple and 3-tuple (key, value, None) iteration formats.
    cpu_kv = tuple(
        (item[0].detach().cpu(), item[1].detach().cpu()) for item in past_key_values
    )
    pickle.dump(cpu_kv, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def deserialize_kv_cache(
    kv_cache_b64: str,
    device: str = "cpu",
) -> DynamicCache:
    """Deserialize a base64 string back into a DynamicCache.

    Parameters
    ----------
    kv_cache_b64:
        Value produced by :func:`serialize_kv_cache`.
    device:
        Target device for the tensors ("cpu", "cuda", "mps").

    Returns
    -------
    DynamicCache
        Ready to pass as past_key_values to any HuggingFace CausalLM.

    TODO: Skip the device copy when device=="cpu" (tensors are already there).
    """
    raw = base64.b64decode(kv_cache_b64.encode("ascii"))
    cpu_kv: list[tuple[torch.Tensor, torch.Tensor]] = pickle.loads(raw)  # noqa: S301
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(cpu_kv):
        cache.update(k.to(device), v.to(device), layer_idx=layer_idx)
    return cache
