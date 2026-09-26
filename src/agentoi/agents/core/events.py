from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

Event = dict[str, Any]
Observer = Any


def _deliver(observer: Observer, event: Event) -> None:
    handler = getattr(observer, "on_event", None)
    if callable(handler):
        handler(event)
    elif callable(observer):
        observer(event)


def emit_event(
    event_type: str,
    *,
    observers: Iterable[Observer] = (),
    oot_observer: Any = None,
    logger: logging.Logger | None = None,
    **data: Any,
) -> Event:
    """Create and synchronously deliver an agent event.

    Observer failures are intentionally isolated from the agent run, matching
    the historical event contract.
    """

    event: Event = {"event": event_type, "ts": time.time(), **data}
    recipients = list(observers)
    if oot_observer is not None:
        recipients.append(oot_observer)

    delivered: set[int] = set()
    for observer in recipients:
        # Treat a bound on_event method and its owning observer as one recipient.
        identity = id(getattr(observer, "__self__", observer))
        if identity in delivered:
            continue
        delivered.add(identity)
        try:
            _deliver(observer, event)
        except Exception as exc:
            if logger is not None:
                logger.debug("Observer error: %s", exc)
    return event
