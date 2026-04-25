"""
Abstract transport layer for moving KV cache tensors between workers.

The hierarchy of transfer strategies, roughly ordered by latency:

  1. SharedMemoryTransport  — same machine, zero copy (current impl)
  2. RDMATransport          — across nodes, GPU-direct RDMA (TODO)
  3. HTTPTransport          — fallback, what the router uses today via base64

Start with SharedMemoryTransport during development; swap in RDMA when you
scale to multi-node.
"""

from __future__ import annotations

import abc
from multiprocessing import shared_memory
from typing import Any

import torch


class KVTransport(abc.ABC):
    """Abstract base: put/get a KV cache identified by a request_id."""

    @abc.abstractmethod
    def put(self, request_id: str, past_key_values: Any) -> None:
        """Store *past_key_values* under *request_id*."""

    @abc.abstractmethod
    def get(self, request_id: str) -> Any:
        """Retrieve and remove *past_key_values* for *request_id*."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources."""


class SharedMemoryTransport(KVTransport):
    """In-process / same-machine transport via multiprocessing.shared_memory.

    The prefill worker calls put(); the decode worker calls get().
    Both must live in the same OS process group (e.g. spawned by the same
    docker-compose service or the same Python process).

    TODO: Implement actual shared_memory allocation sized to the KV tensor.
    TODO: Add a lock / semaphore so get() blocks until put() completes.
    TODO: Benchmark against pickle+socket to confirm the zero-copy benefit.
    TODO: Implement cleanup so stale blocks don't leak on worker crash.
    """

    def __init__(self) -> None:
        # request_id → (SharedMemory block, metadata dict)
        self._store: dict[str, tuple[shared_memory.SharedMemory, dict]] = {}

    def put(self, request_id: str, past_key_values: Any) -> None:
        """Serialize KV tensors into a shared memory block.

        TODO: Replace stub with real numpy-view shared memory write.
        """
        raise NotImplementedError(
            "SharedMemoryTransport.put() is not yet implemented. "
            "Use kv_transfer.serializer (HTTP path) for now."
        )

    def get(self, request_id: str) -> Any:
        """Read KV tensors back from shared memory.

        TODO: Replace stub with real numpy-view shared memory read.
        """
        raise NotImplementedError(
            "SharedMemoryTransport.get() is not yet implemented. "
            "Use kv_transfer.serializer (HTTP path) for now."
        )

    def close(self) -> None:
        for shm, _ in self._store.values():
            shm.close()
            shm.unlink()
        self._store.clear()


class RDMATransport(KVTransport):
    """Multi-node GPU-direct RDMA transport (future).

    TODO: Implement using NCCL send/recv or UCX for cross-node KV migration.
          This unlocks true disaggregation across separate physical machines.
    """

    def put(self, request_id: str, past_key_values: Any) -> None:
        raise NotImplementedError("RDMATransport is not yet implemented.")

    def get(self, request_id: str) -> Any:
        raise NotImplementedError("RDMATransport is not yet implemented.")

    def close(self) -> None:
        pass
