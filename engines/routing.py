"""
Strategy Pattern — routing algorithms.

Callers depend only on RoutingStrategy, never on httpx or OSRM directly.
Swap OSRMStrategy for any other provider (Google Maps, HERE, mock) by changing
the module-level default; no call-site changes needed.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import httpx


# ── Abstract Strategy ─────────────────────────────────────────────────────────

class RoutingStrategy(ABC):
    @abstractmethod
    def route(self, coords: list[tuple[float, float]]) -> dict:
        """Return legs + totals for an ordered list of (lat, lon) pairs."""


# ── Concrete Strategy A — straight-line Haversine ────────────────────────────

class HaversineStrategy(RoutingStrategy):
    AVG_SPEED_KMPH = 35.0

    @staticmethod
    def haversine_km(c1: tuple, c2: tuple) -> float:
        lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
        lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(a))

    def route(self, coords: list) -> dict:
        legs = []
        for i in range(len(coords) - 1):
            d = self.haversine_km(coords[i], coords[i + 1])
            legs.append({
                "distance_km":  round(d, 2),
                "duration_min": round(d / self.AVG_SPEED_KMPH * 60, 1),
            })
        return {
            "legs":               legs,
            "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
            "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
            "source":             "haversine_fallback",
            "route_geometry":     None,
        }


# ── Concrete Strategy B — OSRM real-road routing (with polyline geometry) ────

class OSRMStrategy(RoutingStrategy):
    BASE_URL = "http://router.project-osrm.org/route/v1/driving/"

    def __init__(self, fallback: RoutingStrategy | None = None):
        self._fallback = fallback or HaversineStrategy()

    def route(self, coords: list) -> dict:
        waypoints = ";".join(f"{lon},{lat}" for lat, lon in coords)
        url = (
            self.BASE_URL + waypoints
            + "?overview=simplified&geometries=geojson&steps=false&annotations=false"
        )
        try:
            resp = httpx.get(url, timeout=8.0)
            data = resp.json()
            if data.get("code") != "Ok":
                raise ValueError(f"OSRM non-OK code: {data.get('code')}")
            osrm_route = data["routes"][0]
            legs = [
                {
                    "distance_km":  round(leg["distance"] / 1000, 2),
                    "duration_min": round(leg["duration"] / 60,   1),
                }
                for leg in osrm_route["legs"]
            ]
            # GeoJSON coords are [lon, lat] — flip to [lat, lon] for Folium
            geom = osrm_route.get("geometry", {}).get("coordinates", [])
            geometry = [[c[1], c[0]] for c in geom] if geom else None
            return {
                "legs":               legs,
                "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
                "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
                "source":             "osrm",
                "route_geometry":     geometry,
            }
        except Exception:
            return self._fallback.route(coords)


# ── Module-level default (swap here to change globally) ──────────────────────

_default: RoutingStrategy = OSRMStrategy(fallback=HaversineStrategy())


def get_default_strategy() -> RoutingStrategy:
    return _default


def set_default_strategy(strategy: RoutingStrategy) -> None:
    global _default
    _default = strategy
