"""
Round-robin scheduler for prefill and decode worker pools.

TODO: Replace with work-stealing or load-aware scheduling once you have
      real latency metrics from the latency_breakdown notebook.
"""

import itertools
from typing import Iterator


class RoundRobinScheduler:
    def __init__(self, prefill_urls: list[str], decode_urls: list[str]) -> None:
        self._prefill_cycle: Iterator[str] = itertools.cycle(prefill_urls)
        self._decode_cycle: Iterator[str] = itertools.cycle(decode_urls)
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls

    def next_prefill(self) -> str:
        """Return the URL of the next prefill worker."""
        return next(self._prefill_cycle)

    def next_decode(self) -> str:
        """Return the URL of the next decode worker."""
        return next(self._decode_cycle)

    def add_prefill(self, url: str) -> None:
        """Hot-add a prefill worker.

        TODO: Rebuild the cycle atomically so in-flight requests are unaffected.
        """
        self.prefill_urls.append(url)
        self._prefill_cycle = itertools.cycle(self.prefill_urls)

    def add_decode(self, url: str) -> None:
        """Hot-add a decode worker.

        TODO: Same as add_prefill — needs atomic cycle rebuild.
        """
        self.decode_urls.append(url)
        self._decode_cycle = itertools.cycle(self.decode_urls)
