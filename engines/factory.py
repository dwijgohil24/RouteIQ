"""
Factory Pattern — engine construction.

app.py calls EngineFactory, never engine constructors directly.
Swapping an engine implementation (e.g. a different LLM provider) requires
changing only this file, not the entry point.
"""
from __future__ import annotations

from .ai  import AIEngine
from .ml  import MLEngine
from .routing import RoutingStrategy, OSRMStrategy, HaversineStrategy


class EngineFactory:
    """Creates and wires engine instances. All constructor details live here."""

    # ── individual creators ───────────────────────────────────────────────────

    @staticmethod
    def create_routing_strategy() -> RoutingStrategy:
        return OSRMStrategy(fallback=HaversineStrategy())

    @staticmethod
    def create_ml_engine() -> MLEngine:
        return MLEngine()

    @staticmethod
    def create_ai_engine() -> AIEngine:
        return AIEngine()

    # ── bulk creator — returns a plain dict so callers stay import-free ───────

    @classmethod
    def create_all(cls) -> dict:
        return {
            "ai": cls.create_ai_engine(),
            "ml": cls.create_ml_engine(),
        }
