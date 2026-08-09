"""
DataLoader — thin facade over the Repository layer.

Callers (app.py) use DataLoader as before; internally it delegates to
StopsRepository, RoutesRepository, and KBRepository (Repository Pattern).
The coupled generation step lives in core.repository._DataSeeder.
"""
import pandas as pd

from core.repository import make_repositories


class DataLoader:
    def __init__(self):
        self._stops_repo, self._routes_repo, self._kb_repo = make_repositories()

    def load_stops(self) -> pd.DataFrame:
        return self._stops_repo.load()

    def load_routes(self) -> pd.DataFrame:
        return self._routes_repo.load()

    def load_kb(self) -> pd.DataFrame:
        return self._kb_repo.load()

    def get_summary_stats(self, stops_df: pd.DataFrame, routes_df: pd.DataFrame) -> dict:
        if stops_df.empty:
            return {}
        return {
            "total_stops":         len(stops_df),
            "total_routes":        stops_df["route_id"].nunique() if "route_id" in stops_df.columns else 0,
            "total_distance_km":   round(routes_df["distance_km"].sum(), 1) if not routes_df.empty else 0,
            "avg_stops_per_route": round(stops_df.groupby("route_id").size().mean(), 1) if "route_id" in stops_df.columns else 0,
            "on_time_pct":         round(100 * (stops_df["status"] == "On Time").mean(), 1) if "status" in stops_df.columns else 0,
            "high_priority":       int((stops_df["priority"] == "High").sum()) if "priority" in stops_df.columns else 0,
        }
