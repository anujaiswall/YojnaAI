"""A small in-memory cache so repeat questions skip the LLM and vector search."""
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, max_items: int = 500, ttl_seconds: int = 3600):
        self.max_items = max_items
        self.ttl = ttl_seconds
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    @staticmethod
    def key(question: str) -> str:
        # "What is PM-KISAN?" and "what is pm-kisan" share one entry.
        return " ".join(question.lower().split()).rstrip("?.! ")

    def get(self, question: str) -> Any | None:
        k = self.key(question)
        item = self._data.get(k)
        if item is None:
            return None
        saved_at, value = item
        if time.time() - saved_at > self.ttl:
            del self._data[k]
            return None
        self._data.move_to_end(k)  # mark as recently used
        return value

    def set(self, question: str, value: Any) -> None:
        k = self.key(question)
        self._data[k] = (time.time(), value)
        self._data.move_to_end(k)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)  # drop the least recently used

    def __len__(self) -> int:
        return len(self._data)
