import os
import pandas as pd


class DataLoader:
    STOPS_FILE  = "stops.csv"
    ROUTES_FILE = "routes.csv"
    KB_FILE     = "logistics_kb.csv"

    def load_stops(self):
        if not os.path.exists(self.STOPS_FILE):
            self._generate_and_save()
        return pd.read_csv(self.STOPS_FILE, parse_dates=["time_window_start", "time_window_end"])

    def load_routes(self):
        if not os.path.exists(self.ROUTES_FILE):
            self._generate_and_save()
        return pd.read_csv(self.ROUTES_FILE)

    def load_kb(self):
        if not os.path.exists(self.KB_FILE):
            self._generate_and_save()
        return pd.read_csv(self.KB_FILE)

    def _generate_and_save(self):
        from generate_data import generate_stops, generate_routes, generate_kb
        stops  = generate_stops(60)
        routes = generate_routes(stops)
        kb     = generate_kb()
        stops.to_csv(self.STOPS_FILE,  index=False)
        routes.to_csv(self.ROUTES_FILE, index=False)
        kb.to_csv(self.KB_FILE,         index=False)

    def get_summary_stats(self, stops_df, routes_df):
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
