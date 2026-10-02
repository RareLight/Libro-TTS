from __future__ import annotations

import sys


class ChunkProgressBar:
    def __init__(self, desc: str, enabled: bool = True, total: int | None = None):
        self._bar = None
        if not enabled or not sys.stdout.isatty():
            return

        try:
            from tqdm.auto import tqdm
        except Exception:
            return

        self._bar = tqdm(
            total=total,
            desc=desc,
            unit="chunk",
            dynamic_ncols=True,
            leave=False,
        )

    def update(self, step: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(step)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
