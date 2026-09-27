"""Stop a question while it is still running."""

from __future__ import annotations

import threading
from collections.abc import Callable


class QuestionCancelled(Exception):
    """The user stopped a question before an answer was kept."""


class RunControl:
    """Flag a run as cancelled and close anything still talking to the model."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._closers: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def attach_closer(self, closer: Callable[[], None]) -> None:
        """Close this resource when the run is cancelled."""
        with self._lock:
            self._closers.append(closer)
            already = self._cancelled.is_set()
        if already:
            _close(closer)

    def cancel(self) -> None:
        """Stop the run and close attached model clients."""
        with self._lock:
            self._cancelled.set()
            closers = list(self._closers)
        for closer in closers:
            _close(closer)

    def raise_if_cancelled(self) -> None:
        """Raise when the user has already asked to stop."""
        if self._cancelled.is_set():
            raise QuestionCancelled()


def _close(closer: Callable[[], None]) -> None:
    try:
        closer()
    except Exception:
        return
