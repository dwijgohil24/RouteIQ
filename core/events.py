"""
Observer Pattern — in-process event bus.

Views previously wrote directly into st.session_state and other views read from
it — tight coupling with no clear contract. With EventBus, publishers emit named
events; subscribers register handlers. Neither side knows about the other.

Usage:
    bus = get_event_bus()
    bus.subscribe(ITINERARY_GENERATED, my_handler)
    bus.publish(ITINERARY_GENERATED, {"itinerary": itin, "constraints": c})
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


# ── Well-known event names (single source of truth) ──────────────────────────

ITINERARY_GENERATED  = "itinerary.generated"
DATA_LOADED          = "data.loaded"
CONSTRAINTS_UPDATED  = "constraints.updated"


# ── EventBus ──────────────────────────────────────────────────────────────────

class EventBus:
    """Synchronous publish/subscribe bus. Handlers are called in subscription order."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        if handler not in self._handlers[event]:
            self._handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        self._handlers[event] = [
            h for h in self._handlers[event] if h is not handler
        ]

    def publish(self, event: str, payload: Any = None) -> None:
        for handler in list(self._handlers[event]):
            handler(payload)

    def clear(self, event: str | None = None) -> None:
        if event:
            self._handlers[event] = []
        else:
            self._handlers.clear()


# ── Module-level singleton ────────────────────────────────────────────────────

_bus = EventBus()


def get_event_bus() -> EventBus:
    return _bus
