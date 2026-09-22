"""Small thread-safe in-memory TTL cache for searches and pages."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, max_size: int = 128, ttl_seconds: int = 1800) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            created_at, value = item
            if monotonic() - created_at > self.ttl_seconds:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return deepcopy(value)

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._items[key] = (monotonic(), deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


search_cache: TTLCache[list[dict]] = TTLCache(max_size=100, ttl_seconds=900)
page_cache: TTLCache[dict] = TTLCache(max_size=100, ttl_seconds=3600)