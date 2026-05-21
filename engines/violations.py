"""
Chain of Responsibility Pattern — itinerary violation checking.

Each checker handles exactly one concern (time windows / driver hours / capacity).
Adding a new rule means adding a new class and appending it to build_chain() —
existing checkers are never modified (Open/Closed Principle).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta


# ── Abstract Handler ──────────────────────────────────────────────────────────

class ViolationChecker(ABC):
    def __init__(self) -> None:
        self._next: ViolationChecker | None = None

    def set_next(self, checker: ViolationChecker) -> ViolationChecker:
        self._next = checker
        return checker          # enables fluent chaining: a.set_next(b).set_next(c)

    def check(self, stops: list, constraints: dict, base_dt: datetime,
              violations: list) -> None:
        self._check(stops, constraints, base_dt, violations)
        if self._next:
            self._next.check(stops, constraints, base_dt, violations)

    @abstractmethod
    def _check(self, stops: list, constraints: dict, base_dt: datetime,
               violations: list) -> None: ...

    # ── shared helper ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_dt(hhmm: str, base: datetime) -> datetime:
        try:
            h, m = map(int, str(hhmm).strip().split(":"))
            return base.replace(hour=h, minute=m, second=0, microsecond=0)
        except Exception:
            return base


# ── Concrete Handler 1 — time-window compliance ───────────────────────────────

class TimeWindowChecker(ViolationChecker):
    def _check(self, stops, constraints, base_dt, violations):
        for s in stops:
            tw_start_raw = s.get("time_window_start", "")
            tw_end_raw   = s.get("time_window_end",   "")
            if not tw_start_raw or not tw_end_raw:
                continue

            arr      = self._to_dt(s.get("arrival_time", "00:00"), base_dt)
            tw_start = self._to_dt(tw_start_raw, base_dt)
            tw_end   = self._to_dt(tw_end_raw,   base_dt)

            if arr < tw_start:
                wait = int((tw_start - arr).seconds / 60)
                violations.append({
                    "sequence":      s["sequence"],
                    "location_name": s.get("location_name", ""),
                    "type":          "time_window",
                    "severity":      "Warning",
                    "detail": (
                        f"Arrives at {s.get('arrival_time')} but window opens at "
                        f"{tw_start_raw}. Driver waits {wait} min."
                    ),
                })
            elif arr > tw_end:
                late = int((arr - tw_end).seconds / 60)
                violations.append({
                    "sequence":      s["sequence"],
                    "location_name": s.get("location_name", ""),
                    "type":          "time_window",
                    "severity":      "Critical",
                    "detail": (
                        f"Arrives at {s.get('arrival_time')} — window closed at "
                        f"{tw_end_raw}. Late by {late} min. SLA breach."
                    ),
                })


# ── Concrete Handler 2 — driver working-hours limit ──────────────────────────

class DriverHoursChecker(ViolationChecker):
    def _check(self, stops, constraints, base_dt, violations):
        max_hours = constraints.get("max_hours", 10)
        day_end   = self._to_dt(
            constraints.get("start_time", "08:00"), base_dt
        ) + timedelta(hours=max_hours)

        for s in stops:
            dep = self._to_dt(s.get("departure_time", "00:00"), base_dt)
            if dep > day_end:
                over = int((dep - day_end).seconds / 60)
                violations.append({
                    "sequence":      s["sequence"],
                    "location_name": s.get("location_name", ""),
                    "type":          "driver_hours",
                    "severity":      "Critical",
                    "detail": (
                        f"Departure at {s.get('departure_time')} exceeds "
                        f"{max_hours}h limit by {over} min."
                    ),
                })


# ── Concrete Handler 3 — vehicle capacity ────────────────────────────────────

class CapacityChecker(ViolationChecker):
    def _check(self, stops, constraints, base_dt, violations):
        capacity_kg   = constraints.get("vehicle_capacity_kg")
        total_load_kg = sum(s.get("load_kg", 0) for s in stops)
        if capacity_kg and total_load_kg > capacity_kg:
            violations.append({
                "sequence":      0,
                "location_name": "All stops",
                "type":          "capacity",
                "severity":      "Critical",
                "detail": (
                    f"Total load {total_load_kg} kg exceeds vehicle capacity "
                    f"{capacity_kg} kg by {total_load_kg - capacity_kg} kg."
                ),
            })


# ── Assembly — build the full chain ──────────────────────────────────────────

def build_chain() -> ViolationChecker:
    """Return the head of the assembled checker chain."""
    head = TimeWindowChecker()
    head.set_next(DriverHoursChecker()).set_next(CapacityChecker())
    return head
