"""Terminal progress for long ontology commands."""

from __future__ import annotations

import sys
import threading
from typing import Protocol, TextIO


def computing_status(done: int, total: int) -> str:
    """Percent line for embedding or loading concepts."""
    percent = 0 if total <= 0 else round(100 * done / total)
    return f"Computing {percent}% · {done} / {total} concepts"


class Progress(Protocol):
    """Report a named stage and an in-place counter."""

    def stage(self, message: str) -> None:
        """Print one status line and leave the cursor on the next line."""

    def tick(self, message: str) -> None:
        """Replace the current status line with an updated counter."""


class StderrProgress:
    """Write stage lines to stderr so the answer stays on stdout."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr
        self._ticking = False

    def stage(self, message: str) -> None:
        if self._ticking:
            print(file=self._stream, flush=True)
            self._ticking = False
        print(message, file=self._stream, flush=True)

    def tick(self, message: str) -> None:
        print(f"\r{message}", end="", file=self._stream, flush=True)
        self._ticking = True


class WorkingStatus:
    """Show a short waiting line while the model looks for an answer."""

    _FRAMES = ("beep boop", "now cooking", "still simmering")

    def __init__(self, enabled: bool = True, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self._stream = stream or sys.stderr
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _render(frame: str) -> str:
        """Keep the phrase against the label and erase any leftover tail."""
        return f"\rWorking · {frame}\033[K"

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        def _run() -> None:
            index = 0
            while not self._stop.is_set():
                frame = self._FRAMES[index % len(self._FRAMES)]
                print(self._render(frame), end="", file=self._stream, flush=True)
                index += 1
                if self._stop.wait(0.7):
                    break

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            if self.enabled:
                print(file=self._stream, flush=True)
