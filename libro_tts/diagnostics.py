"""Bounded in-memory diagnostics for noisy model generators."""
from __future__ import annotations

import io


class TailCapture(io.TextIOBase):
    def __init__(self, limit: int = 8192):
        super().__init__()
        if limit < 1:
            raise ValueError("Diagnostic capture limit must be positive.")
        self.limit = limit
        self._tail = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError("Cannot write to closed diagnostic capture.")
        self._tail = (self._tail + text[-self.limit:])[-self.limit:]
        return len(text)

    def getvalue(self) -> str:
        return self._tail
