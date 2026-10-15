"""Bounded, run-scoped evidence the PEP sends with policy checks.

Evidence (last reads, retrieval counts, compiled prompts) belongs to one run.
Entries are keyed by trace id, dropped when the local root span ends, and
bounded by expiry and size so traces that never end here cannot grow memory.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

MAX_TRACES = 1024
TTL_S = 3600.0
MAX_READS_PER_SCOPE = 64

_lock = threading.Lock()


class EvidenceStore:
    """LRU map with a per-entry TTL. Each key carries the trace id it belongs to."""

    def __init__(self, max_entries: int = MAX_TRACES, ttl_s: float = TTL_S) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._items: "OrderedDict[Any, tuple[float, str, Any]]" = OrderedDict()

    def get(self, key: Any) -> Any:
        with _lock:
            item = self._items.get(key)
            if item is None:
                return None
            ts, _trace_id, value = item
            if time.monotonic() - ts > self.ttl_s:
                del self._items[key]
                return None
            return value

    def update(self, key: Any, trace_id: str, fn: Callable[[Any], Any]) -> None:
        """Replace the value at key with fn(current or None), refreshing its expiry."""
        with _lock:
            item = self._items.pop(key, None)
            current = None
            if item is not None and time.monotonic() - item[0] <= self.ttl_s:
                current = item[2]
            self._items[key] = (time.monotonic(), trace_id, fn(current))
            self._evict()

    def forget_trace(self, trace_id: str) -> None:
        with _lock:
            for key in [k for k, item in self._items.items() if item[1] == trace_id]:
                del self._items[key]

    def clear(self) -> None:
        with _lock:
            self._items.clear()

    def __len__(self) -> int:
        with _lock:
            return len(self._items)

    def _evict(self) -> None:
        now = time.monotonic()
        while self._items:
            ts = next(iter(self._items.values()))[0]
            if len(self._items) > self.max_entries or now - ts > self.ttl_s:
                self._items.popitem(last=False)
            else:
                break


# (tenant_id, agent_id, trace_id) -> {(tool, resource_id): {"tool", "read_at", "field", "resource_id"?}}
last_reads = EvidenceStore()
# trace_id -> {"present": bool, "chunk_count": ...}
retrieval_by_trace = EvidenceStore()
# trace_id -> {"name", "label"?, "version_id"?}
prompt_by_trace = EvidenceStore()

_STORES = (last_reads, retrieval_by_trace, prompt_by_trace)


def trace_key(trace_id: Any) -> Optional[str]:
    if trace_id is None:
        return None
    if isinstance(trace_id, int):
        return format(trace_id, "032x") if trace_id else None
    text = str(trace_id).strip().lower()
    return text if text and text.strip("0") else None


def forget_trace(trace_id: Any) -> None:
    """Drop all evidence recorded for a finished run."""
    key = trace_key(trace_id)
    if key:
        for store in _STORES:
            store.forget_trace(key)
