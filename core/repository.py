"""
Repository Pattern — data access.

DataLoader previously mixed file-existence checks, CSV parsing, date coercion,
and generation logic in one class. Each Repository now owns exactly one data
concern. The coupled generation step (routes depend on stops) is isolated in
_DataSeeder so repositories themselves stay independent.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import pandas as pd


# ── Abstract Repository ───────────────────────────────────────────────────────

class DataRepository(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame: ...

    @abstractmethod
    def exists(self) -> bool: ...


# ── Internal seeder (coupled generation lives here, not in repositories) ──────

class _DataSeeder:
    """Generates all three CSVs together when any one is missing."""

    def __init__(self, stops_path: str, routes_path: str, kb_path: str) -> None:
        self._stops  = stops_path
        self._routes = routes_path
        self._kb     = kb_path

    def seed_if_needed(self) -> None:
        if (os.path.exists(self._stops)
                and os.path.exists(self._routes)
                and os.path.exists(self._kb)):
            return
        from generate_data import generate_stops, generate_routes, generate_kb
        stops  = generate_stops(60)
        routes = generate_routes(stops)
        kb     = generate_kb()
        stops.to_csv(self._stops,  index=False)
        routes.to_csv(self._routes, index=False)
        kb.to_csv(self._kb,         index=False)


# ── Concrete Repositories ─────────────────────────────────────────────────────

class StopsRepository(DataRepository):
    def __init__(self, path: str = "stops.csv",
                 seeder: _DataSeeder | None = None) -> None:
        self._path   = path
        self._seeder = seeder

    def exists(self) -> bool:
        return os.path.exists(self._path)

    def load(self) -> pd.DataFrame:
        if not self.exists() and self._seeder:
            self._seeder.seed_if_needed()
        return pd.read_csv(
            self._path, parse_dates=["time_window_start", "time_window_end"]
        )


class RoutesRepository(DataRepository):
    def __init__(self, path: str = "routes.csv",
                 seeder: _DataSeeder | None = None) -> None:
        self._path   = path
        self._seeder = seeder

    def exists(self) -> bool:
        return os.path.exists(self._path)

    def load(self) -> pd.DataFrame:
        if not self.exists() and self._seeder:
            self._seeder.seed_if_needed()
        return pd.read_csv(self._path)


class KBRepository(DataRepository):
    def __init__(self, path: str = "logistics_kb.csv",
                 seeder: _DataSeeder | None = None) -> None:
        self._path   = path
        self._seeder = seeder

    def exists(self) -> bool:
        return os.path.exists(self._path)

    def load(self) -> pd.DataFrame:
        if not self.exists() and self._seeder:
            self._seeder.seed_if_needed()
        return pd.read_csv(self._path)


# ── Factory helper — builds a wired set of repositories ──────────────────────

def make_repositories(
    stops_path:  str = "stops.csv",
    routes_path: str = "routes.csv",
    kb_path:     str = "logistics_kb.csv",
) -> tuple[StopsRepository, RoutesRepository, KBRepository]:
    seeder = _DataSeeder(stops_path, routes_path, kb_path)
    return (
        StopsRepository(stops_path,   seeder),
        RoutesRepository(routes_path, seeder),
        KBRepository(kb_path,         seeder),
    )
