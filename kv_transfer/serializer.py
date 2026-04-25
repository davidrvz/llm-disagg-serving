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


def serialize_kv_cache(past_key_values: tuple[tuple[torch.Tensor, ...], ...]) -> str:
    """Serialize a HuggingFace past_key_values tuple to a base64 string.

    Parameters
    ----------
    past_key_values:
        A tuple of (key, value) tensor pairs, one per transformer layer,
        as returned by any HuggingFace CausalLM with use_cache=True.

    Returns
    -------
    str
        Base64-encoded pickle blob safe to embed in JSON.

    TODO: Move tensors to CPU before serializing if they live on GPU/MPS.
    TODO: Profile and replace pickle with a faster binary format.
    """
    buf = io.BytesIO()
    # Detach and move to CPU so the bytes are portable across workers
    cpu_kv = tuple(
        (k.detach().cpu(), v.detach().cpu()) for k, v in past_key_values
    )
    pickle.dump(cpu_kv, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def deserialize_kv_cache(
    kv_cache_b64: str,
    device: str = "cpu",
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Deserialize a base64 string back into a past_key_values tuple.

    Parameters
    ----------
    kv_cache_b64:
        Value produced by :func:`serialize_kv_cache`.
    device:
        Target device for the tensors ("cpu", "cuda", "mps").

    Returns
    -------
    tuple[tuple[torch.Tensor, torch.Tensor], ...]
        Ready to pass as past_key_values to a HuggingFace model.

    TODO: Skip the copy when device=="cpu" (tensors are already on CPU).
    """
    raw = base64.b64decode(kv_cache_b64.encode("ascii"))
    cpu_kv: Any = pickle.loads(raw)  # noqa: S301
    return tuple(
        (k.to(device), v.to(device)) for k, v in cpu_kv
    )
