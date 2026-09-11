"""Best-effort same-host account lock for the single S2 worker."""

from __future__ import annotations

import os
from pathlib import Path


class AccountProcessLock:
    """Use exclusive creation so a second worker cannot start the same account."""

    def __init__(self, lock_path: str | Path) -> None:
        self.lock_path = Path(lock_path)
        self._handle = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self.lock_path.open("x", encoding="utf-8")
            self._handle.write(str(os.getpid()))
            self._handle.flush()
        except FileExistsError as exc:
            raise RuntimeError(f"account lock already held: {self.lock_path}") from exc

    def release(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "AccountProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()
