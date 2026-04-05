import os

# ── Tiktoken cache (must be set before any tiktoken / langchain import) ───────
_tiktoken_cache_dir = os.path.abspath("./token")
os.makedirs(_tiktoken_cache_dir, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = _tiktoken_cache_dir

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from datetime import datetime, timedelta
import random
import math
import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RouteIQ — Logistics Itinerary Planner",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --midnight: #0D1117;
  --obsidian: #161B22;
  --charcoal: #21262D;
  --steel: #30363D;
  --ink: #8B949E;
  --silver: #C9D1D9;
  --white: #F0F6FC;
  --gold: #D4A843;
  --amber: #E8873A;
  --teal: #2EA4A4;
  --crimson: #C0392B;
  --sage: #3FB950;
  --violet: #8957E5;
  --blue: #1F6FEB;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background-color: var(--midnight) !important;
  color: var(--silver) !important;
}
section[data-testid="stSidebar"] {
  background: var(--obsidian) !important;
  border-right: 1px solid var(--steel) !important;
}
section[data-testid="stSidebar"] .stRadio label {
  color: var(--silver) !important;
  font-size: 0.9rem !important;
}
h1, h2, h3 {
  font-family: 'Playfair Display', serif !important;
  color: var(--white) !important;
}
[data-testid="metric-container"] {
  background: var(--obsidian) !important;
  border: 1px solid var(--steel) !important;
  border-radius: 12px !important;
  padding: 16px !important;
}
[data-testid="metric-container"] label {
  color: var(--ink) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--gold) !important;
  font-family: 'Playfair Display', serif !important;
  font-size: 2rem !important;
}
details {
  background: var(--obsidian) !important;
  border: 1px solid var(--steel) !important;
  border-radius: 8px !important;
}
details summary { color: var(--teal) !important; font-weight: 500 !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--steel) !important; border-radius: 8px !important; }
textarea, input[type="text"] {
  background: var(--charcoal) !important;
  color: var(--white) !important;
  border: 1px solid var(--steel) !important;
  border-radius: 8px !important;
}
.stButton > button {
  background: linear-gradient(135deg, var(--gold), var(--amber)) !important;
  color: var(--midnight) !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 8px !important;
  letter-spacing: 0.04em !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
[data-testid="stSelectbox"] select, .stSelectbox > div {
  background: var(--charcoal) !important;
  color: var(--white) !important;
  border-color: var(--steel) !important;
}
.stTabs [data-baseweb="tab"] { color: var(--ink) !important; border-bottom: 2px solid transparent !important; }
.stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; }
.badge-delivery  { background: rgba(212,168,67,0.2);  color: #D4A843; border: 1px solid rgba(212,168,67,0.4); }
.badge-meeting   { background: rgba(46,164,164,0.2);  color: #2EA4A4; border: 1px solid rgba(46,164,164,0.4); }
.badge-pickup    { background: rgba(63,185,80,0.2);   color: #3FB950; border: 1px solid rgba(63,185,80,0.4); }
.badge-warehouse { background: rgba(137,87,229,0.2);  color: #8957E5; border: 1px solid rgba(137,87,229,0.4); }
.badge-customs   { background: rgba(192,57,43,0.2);   color: #E74C3C; border: 1px solid rgba(192,57,43,0.4); }
.badge-rest      { background: rgba(139,148,158,0.2); color: #8B949E; border: 1px solid rgba(139,148,158,0.4); }

.card { background: var(--obsidian); border: 1px solid var(--steel); border-radius: 12px; padding: 20px; margin-bottom: 12px; }
.card-gold  { border-left: 4px solid var(--gold); }
.card-teal  { border-left: 4px solid var(--teal); }
.card-red   { border-left: 4px solid var(--crimson); }
.card-green { border-left: 4px solid var(--sage); }
.card-blue  { border-left: 4px solid var(--blue); }

.section-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: var(--white); margin-bottom: 4px; }
.section-sub   { color: var(--ink); font-size: 0.85rem; margin-bottom: 20px; letter-spacing: 0.04em; }
.hero-title    { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 900; color: var(--white); line-height: 1.1; }
.hero-accent   { color: var(--gold); }

.stop-card { background: var(--charcoal); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
.stop-num  { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; background: var(--gold); color: var(--midnight); border-radius: 50%; font-weight: 700; font-size: 0.85rem; margin-right: 10px; flex-shrink: 0; }
.stop-row  { display: flex; align-items: flex-start; }
.stop-detail { font-size: 0.82rem; color: var(--ink); margin-top: 4px; }
.connector-line { width: 2px; height: 30px; background: linear-gradient(var(--gold), var(--teal)); margin: 0 auto 0 13px; }

.summary-box { background: linear-gradient(135deg, rgba(212,168,67,0.08), rgba(46,164,164,0.08)); border: 1px solid rgba(212,168,67,0.3); border-radius: 12px; padding: 20px; margin: 12px 0; }
.itinerary-header { background: linear-gradient(135deg, rgba(31,111,235,0.15), rgba(46,164,164,0.15)); border: 1px solid rgba(31,111,235,0.3); border-radius: 12px; padding: 18px 22px; margin-bottom: 18px; }
.violation-critical { background: rgba(192,57,43,0.1); border-left: 4px solid #E74C3C; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
.violation-warning  { background: rgba(212,168,67,0.1); border-left: 4px solid #D4A843; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
.weather-card { background: var(--charcoal); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 16px; text-align: center; }
.fuel-card    { background: var(--obsidian); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
.fuel-save    { background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.35); border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; }
.fuel-warn    { background: rgba(192,57,43,0.08); border: 1px solid rgba(192,57,43,0.3); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "generated_itinerary": None,
        "batch_itineraries":   [],
        "cluster_labels":      None,
        "cluster_X2d":         None,
        "cluster_df":          None,
        "assistant_history":   [],
        "plan_history":        [],
        "last_constraints":    {},
        "nl_parsed_result":    None,
        "geocode_cache":       {},      # lat_lon → {display_name, full_address, source}
        "weather_cache":       {},      # key: "lat_lon" → weather dict
        "fuel_analysis":       None,    # last fuel/savings analysis result
        "itinerary_source":    None,    # "dataset" | "custom" | "nl" — which tab owns it
        "nl_stops_for_map":    [],      # lat/lon list for NL itinerary map
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# ML ENGINE
# ─────────────────────────────────────────────
class MLEngine:
    def __init__(self):
        self.vectorizer     = None
        self.kmeans         = None
        self.feature_matrix = None

    def cluster_stops(self, stops_df, n_clusters=4):
        texts = (
            stops_df["stop_type"].fillna("") + " " +
            stops_df["location_name"].fillna("") + " " +
            stops_df["notes"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        X = self.vectorizer.fit_transform(texts)
        self.feature_matrix = X
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        pca    = PCA(n_components=2, random_state=42)
        X_2d   = pca.fit_transform(X.toarray())
        return labels, X_2d

    def get_cluster_keywords(self, top_n=5):
        if self.vectorizer is None or self.kmeans is None:
            return {}
        terms    = self.vectorizer.get_feature_names_out()
        keywords = {}
        for i, center in enumerate(self.kmeans.cluster_centers_):
            top_idx     = center.argsort()[-top_n:][::-1]
            keywords[i] = [terms[j] for j in top_idx]
        return keywords

    def nearest_neighbor_route(self, coords):
        if len(coords) <= 1:
            return list(range(len(coords)))
        unvisited = list(range(1, len(coords)))
        route     = [0]
        while unvisited:
            curr    = route[-1]
            nearest = min(unvisited, key=lambda j: self._haversine(coords[curr], coords[j]))
            route.append(nearest)
            unvisited.remove(nearest)
        return route

    @staticmethod
    def _haversine(c1, c2):
        lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
        lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def compute_distance_km(lat1, lon1, lat2, lon2):
        return MLEngine._haversine((lat1, lon1), (lat2, lon2))

    @staticmethod
    def osrm_route(coords: list) -> dict:
        """
        Real road routing via public OSRM demo API.
        Falls back to Haversine (35 km/h) if OSRM is unreachable.

        Args:
            coords: list of (lat, lon) tuples in stop order

        Returns:
            {legs, total_distance_km, total_duration_min, source}
        """
        if len(coords) < 2:
            return {"legs": [], "total_distance_km": 0.0,
                    "total_duration_min": 0.0, "source": "osrm"}

        waypoints = ";".join(f"{lon},{lat}" for lat, lon in coords)
        url = (
            "http://router.project-osrm.org/route/v1/driving/" + waypoints
            + "?overview=false&steps=false&annotations=false"
        )
        try:
            resp = httpx.get(url, timeout=8.0)
            data = resp.json()
            if data.get("code") != "Ok":
                raise ValueError(f"OSRM code: {data.get('code')}")
            legs = [
                {
                    "distance_km":  round(leg["distance"] / 1000, 2),
                    "duration_min": round(leg["duration"] / 60, 1),
                }
                for leg in data["routes"][0]["legs"]
            ]
            return {
                "legs":               legs,
                "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
                "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
                "source":             "osrm",
            }
        except Exception:
            legs = []
            for i in range(len(coords) - 1):
                d = MLEngine._haversine(coords[i], coords[i + 1])
                legs.append({"distance_km": round(d, 2), "duration_min": round(d / 35 * 60, 1)})
            return {
                "legs":               legs,
                "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
                "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
                "source":             "haversine_fallback",
            }


# ─────────────────────────────────────────────
# WEATHER ENGINE  (Open-Meteo — no API key)
# ─────────────────────────────────────────────
class WeatherEngine:
    """
    Arrival-time-aware weather forecasting via Open-Meteo hourly API (free, no key).

    For each stop we fetch the HOURLY forecast for that location, then pick the
    hour slot that matches the stop's expected arrival_time.  This means:
      - Stop A arriving at 09:00 → forecast for 09:00
      - Stop B arriving at 13:00 → forecast for 13:00
      - etc.

    Caching strategy:
      - Full hourly payload cached by lat/lon key (one API call per unique location).
      - Individual hour lookups extracted from the cached payload — no repeat calls.

    Fallback chain (guaranteed to always return usable data):
      1. Open-Meteo HTTPS (verify=False for internal proxies)
      2. Open-Meteo HTTP (plain, bypasses TLS entirely)
      3. Seasonal estimate (hardcoded India climate by month + latitude band)
    """

    WMO_CODES = {
        0:  ("Clear sky",            "☀️"),
        1:  ("Mainly clear",         "🌤️"),
        2:  ("Partly cloudy",        "⛅"),
        3:  ("Overcast",             "☁️"),
        45: ("Fog",                  "🌫️"),
        48: ("Icy fog",              "🌫️"),
        51: ("Light drizzle",        "🌦️"),
        53: ("Moderate drizzle",     "🌦️"),
        55: ("Dense drizzle",        "🌧️"),
        61: ("Slight rain",          "🌧️"),
        63: ("Moderate rain",        "🌧️"),
        65: ("Heavy rain",           "🌧️"),
        71: ("Slight snow",          "🌨️"),
        73: ("Moderate snow",        "❄️"),
        75: ("Heavy snow",           "❄️"),
        80: ("Slight showers",       "🌦️"),
        81: ("Moderate showers",     "🌧️"),
        82: ("Heavy showers",        "⛈️"),
        95: ("Thunderstorm",         "⛈️"),
        96: ("Thunderstorm + hail",  "⛈️"),
        99: ("Severe thunderstorm",  "⛈️"),
    }
    ADVERSE_CODES = {63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99}

    # ── Seasonal fallback (India, month × lat band) ───────────────────────────
    _SEASONAL_FALLBACK = {
        1:  {"north": (18, 60, "Mainly clear",  "🌤️", 1),  "south": (28, 65, "Partly cloudy", "⛅", 2)},
        2:  {"north": (20, 58, "Mainly clear",  "🌤️", 1),  "south": (30, 62, "Mainly clear",  "🌤️", 1)},
        3:  {"north": (27, 55, "Clear sky",     "☀️", 0),  "south": (33, 60, "Clear sky",     "☀️", 0)},
        4:  {"north": (33, 50, "Clear sky",     "☀️", 0),  "south": (35, 65, "Clear sky",     "☀️", 0)},
        5:  {"north": (37, 45, "Clear sky",     "☀️", 0),  "south": (35, 70, "Partly cloudy", "⛅", 2)},
        6:  {"north": (34, 75, "Moderate rain", "🌧️", 63), "south": (30, 85, "Heavy rain",    "🌧️", 65)},
        7:  {"north": (30, 82, "Heavy rain",    "🌧️", 65), "south": (28, 88, "Heavy showers", "⛈️", 82)},
        8:  {"north": (30, 80, "Moderate rain", "🌧️", 63), "south": (28, 86, "Heavy rain",    "🌧️", 65)},
        9:  {"north": (29, 78, "Slight rain",   "🌦️", 61), "south": (29, 82, "Moderate rain", "🌧️", 63)},
        10: {"north": (26, 65, "Partly cloudy", "⛅", 2),  "south": (29, 72, "Partly cloudy", "⛅", 2)},
        11: {"north": (21, 58, "Mainly clear",  "🌤️", 1),  "south": (28, 68, "Mainly clear",  "🌤️", 1)},
        12: {"north": (16, 62, "Mainly clear",  "🌤️", 1),  "south": (27, 65, "Partly cloudy", "⛅", 2)},
    }

    @staticmethod
    def _cache_key(lat: float, lon: float) -> str:
        return f"{round(lat, 3)}_{round(lon, 3)}"

    @classmethod
    def _seasonal_estimate(cls, lat: float, lon: float, arrival_hour: int = None) -> dict:
        """Plausible weather estimate based on season, location, and time of day."""
        import random as _rnd
        month = datetime.now().month
        band  = "north" if lat > 20 else "south"
        row   = cls._SEASONAL_FALLBACK.get(month, cls._SEASONAL_FALLBACK[6])
        t, h, desc, icon, code = row[band]

        seed = int(abs(lat * 1000 + lon * 100)) % 100
        # Diurnal temperature variation: cooler at dawn/late evening
        hour_offset = 0
        if arrival_hour is not None:
            if arrival_hour < 7:    hour_offset = -4
            elif arrival_hour < 10: hour_offset = -2
            elif arrival_hour < 14: hour_offset =  2
            elif arrival_hour < 17: hour_offset =  3
            elif arrival_hour < 20: hour_offset =  1

        t    = round(t + hour_offset + (_rnd.Random(seed).random() - 0.5) * 2, 1)
        wind = round(8  + _rnd.Random(seed + 1).random() * 12, 1)
        prec = round(_rnd.Random(seed + 2).random() * (5 if code >= 61 else 0.2), 1)
        vis  = round(6  + _rnd.Random(seed + 3).random() * 4, 1) if code >= 45 else round(9 + _rnd.Random(seed + 3).random() * 5, 1)
        return {
            "temperature_c":    t,
            "wind_speed_kmh":   wind,
            "precipitation_mm": prec,
            "humidity_pct":     h,
            "visibility_km":    vis,
            "weather_code":     code,
            "description":      desc,
            "icon":             icon,
            "is_adverse":       code in cls.ADVERSE_CODES,
            "source":           "seasonal-estimate",
            "forecast_hour":    arrival_hour,
        }

    # ── Core: fetch full 48-h hourly payload for a location ──────────────────
    @classmethod
    def _fetch_hourly_payload(cls, lat: float, lon: float) -> dict | None:
        """
        Fetch 48-hour hourly forecast from Open-Meteo.
        Returns the raw API response dict, or None on failure.
        Payload cached in session state under key "hourly_payload_{lat}_{lon}".
        """
        cache_key = f"hourly_payload_{cls._cache_key(lat, lon)}"
        cache     = st.session_state.get("weather_cache", {})
        if cache_key in cache:
            return cache[cache_key]

        params = (
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,"
            f"precipitation,weather_code,wind_speed_10m,visibility"
            f"&wind_speed_unit=kmh"
            f"&timezone=Asia%2FKolkata"
            f"&forecast_days=2"          # today + tomorrow — covers any same-day route
        )
        urls = [
            "https://api.open-meteo.com/v1/forecast" + params,
            "http://api.open-meteo.com/v1/forecast"  + params,
        ]
        for url in urls:
            try:
                resp = httpx.get(url, timeout=8.0, verify=False, follow_redirects=True)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if "hourly" not in data or "time" not in data.get("hourly", {}):
                    continue
                cache[cache_key] = data
                st.session_state["weather_cache"] = cache
                return data
            except Exception:
                continue
        return None   # both URLs failed

    # ── Extract one hour slot from the payload ────────────────────────────────
    @classmethod
    def _extract_hour(cls, payload: dict, target_dt: datetime) -> dict:
        """
        Given a full hourly payload, extract the slot closest to target_dt.
        Open-Meteo returns ISO strings like "2025-06-15T09:00" in the "time" array.
        """
        hourly     = payload["hourly"]
        time_strs  = hourly["time"]           # list of "YYYY-MM-DDTHH:00"
        target_str = target_dt.strftime("%Y-%m-%dT%H:00")

        # Find exact match first, then nearest
        idx = None
        if target_str in time_strs:
            idx = time_strs.index(target_str)
        else:
            # Find closest hour
            best_diff = float("inf")
            for i, ts in enumerate(time_strs):
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
                    diff = abs((dt - target_dt).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        idx = i
                except Exception:
                    continue

        if idx is None:
            return {}

        def _val(key, default=0):
            arr = hourly.get(key, [])
            v   = arr[idx] if idx < len(arr) else default
            return v if v is not None else default

        code       = int(_val("weather_code", 0))
        desc, icon = cls.WMO_CODES.get(code, ("Partly cloudy", "⛅"))
        vis_raw    = _val("visibility", 10000)

        return {
            "temperature_c":        round(float(_val("temperature_2m",          25)), 1),
            "wind_speed_kmh":       round(float(_val("wind_speed_10m",          10)), 1),
            "precipitation_mm":     round(float(_val("precipitation",             0)), 1),
            "precip_probability":   int(_val("precipitation_probability",          0)),
            "humidity_pct":         int(_val("relative_humidity_2m",             65)),
            "visibility_km":        round(float(vis_raw) / 1000, 1),
            "weather_code":         code,
            "description":          desc,
            "icon":                 icon,
            "is_adverse":           code in cls.ADVERSE_CODES,
            "source":               "open-meteo-hourly",
        }

    # ── Public API: fetch weather at a specific arrival time ─────────────────
    @classmethod
    def fetch_at_time(cls, lat: float, lon: float, arrival_time_str: str,
                      itin_date_str: str = None) -> dict:
        """
        Return weather forecast for (lat, lon) at the stop's arrival time.

        Args:
            lat, lon:           Stop coordinates.
            arrival_time_str:   "HH:MM" string from the itinerary stop.
            itin_date_str:      "YYYY-MM-DD" from itin["date"]; defaults to today.

        Returns a weather dict with an extra "forecast_time" field showing
        the exact datetime we fetched for.
        """
        # Parse target datetime
        try:
            date_str = itin_date_str or datetime.now().strftime("%Y-%m-%d")
            target   = datetime.strptime(f"{date_str} {arrival_time_str}", "%Y-%m-%d %H:%M")
        except Exception:
            target   = datetime.now()

        # Per-stop cache key (includes the target hour)
        stop_key  = f"stop_{cls._cache_key(lat, lon)}_{target.strftime('%Y%m%d%H')}"
        cache     = st.session_state.get("weather_cache", {})
        if stop_key in cache:
            return cache[stop_key]

        # Fetch (or reuse) the hourly payload for this location
        payload = cls._fetch_hourly_payload(lat, lon)

        if payload:
            result = cls._extract_hour(payload, target)
            if result:
                result["forecast_time"]   = target.strftime("%d %b %Y · %H:%M")
                result["arrival_time_str"] = arrival_time_str
                cache[stop_key] = result
                st.session_state["weather_cache"] = cache
                return result

        # Fallback to seasonal estimate
        result = cls._seasonal_estimate(lat, lon, arrival_hour=target.hour)
        result["forecast_time"]   = target.strftime("%d %b %Y · %H:%M") + " (est.)"
        result["arrival_time_str"] = arrival_time_str
        cache[stop_key] = result
        st.session_state["weather_cache"] = cache
        return result

    # ── Convenience: fetch for all stops in an itinerary ─────────────────────
    @classmethod
    def fetch_route_at_times(cls, stops_with_coords: list, itin_date_str: str = None) -> list:
        """
        Fetch arrival-time-aware forecast for every stop.

        Args:
            stops_with_coords: list of dicts with at minimum:
                {stop_id, location_name, lat, lon, arrival_time (HH:MM)}
            itin_date_str: itinerary date "YYYY-MM-DD"

        Returns: same list with an added "weather" key per stop.
        """
        results = []
        for s in stops_with_coords:
            arrival = s.get("arrival_time", "08:00") or "08:00"
            w = cls.fetch_at_time(s["lat"], s["lon"], arrival, itin_date_str)
            results.append({**s, "weather": w})
        return results

    # ── Legacy: fetch current weather (kept for backward compat) ─────────────
    @classmethod
    def fetch(cls, lat: float, lon: float) -> dict:
        """Fetch current conditions (no arrival time). Used as generic fallback."""
        return cls.fetch_at_time(lat, lon, datetime.now().strftime("%H:%M"))

    @classmethod
    def fetch_route(cls, stops_with_coords: list) -> list:
        """Legacy: fetch current conditions for a list of stops."""
        return cls.fetch_route_at_times(stops_with_coords)


# ─────────────────────────────────────────────
# FUEL ENGINE  (static reference prices — India)
# ─────────────────────────────────────────────
class FuelEngine:
    """
    Static Indian petrol/diesel reference prices (as of June 2025).
    Vehicle efficiency lookup table.
    Fuel savings comparisons between optimized vs un-optimized route order.

    Reference: https://www.goodreturns.in/petrol-price.html
    Prices in ₹ per litre. Update CITY_PRICES dict as needed.
    """

    # ── Static city-level prices (₹/litre) ───────────────────────────────────
    CITY_PRICES = {
        # Maharashtra
        "Mumbai":     {"petrol": 103.44, "diesel": 89.97},
        "Pune":       {"petrol": 103.57, "diesel": 90.10},
        "Nashik":     {"petrol": 103.20, "diesel": 89.75},
        "Nagpur":     {"petrol": 103.80, "diesel": 90.15},
        "Aurangabad": {"petrol": 103.35, "diesel": 89.85},
        # Delhi NCR
        "Delhi":      {"petrol": 94.72,  "diesel": 87.62},
        "Gurgaon":    {"petrol": 95.10,  "diesel": 88.05},
        "Noida":      {"petrol": 94.85,  "diesel": 87.80},
        # Karnataka
        "Bengaluru":  {"petrol": 102.86, "diesel": 88.94},
        "Mysuru":     {"petrol": 102.60, "diesel": 88.70},
        # Tamil Nadu
        "Chennai":    {"petrol": 100.75, "diesel": 92.34},
        "Coimbatore": {"petrol": 100.50, "diesel": 92.10},
        # Telangana / AP
        "Hyderabad":  {"petrol": 107.41, "diesel": 95.65},
        "Visakhapatnam": {"petrol": 106.80, "diesel": 95.20},
        # Gujarat
        "Ahmedabad":  {"petrol": 96.63,  "diesel": 92.38},
        "Surat":      {"petrol": 96.50,  "diesel": 92.25},
        # Rajasthan
        "Jaipur":     {"petrol": 104.88, "diesel": 90.36},
        # Uttar Pradesh
        "Lucknow":    {"petrol": 94.65,  "diesel": 87.76},
        "Kanpur":     {"petrol": 94.55,  "diesel": 87.65},
        # West Bengal
        "Kolkata":    {"petrol": 103.94, "diesel": 90.76},
        # Default (national average)
        "default":    {"petrol": 101.50, "diesel": 90.00},
    }

    # ── Vehicle fuel efficiency (km per litre) ────────────────────────────────
    VEHICLE_EFFICIENCY = {
        "Truck":       5.5,   # heavy truck (10-16T)
        "Van":         12.0,  # light commercial van
        "Tempo":       9.0,   # mini-truck / tempo
        "Car":         15.0,  # passenger car
        "Motorcycle":  40.0,  # two-wheeler
        "default":     8.0,
    }

    # ── Fuel type by vehicle ──────────────────────────────────────────────────
    VEHICLE_FUEL_TYPE = {
        "Truck": "diesel", "Tempo": "diesel",
        "Van": "diesel", "Car": "petrol", "Motorcycle": "petrol",
    }

    @classmethod
    def get_price(cls, city: str, fuel_type: str = "diesel") -> float:
        """Find the closest city match (case-insensitive substring)."""
        city_lower = city.lower()
        for name, prices in cls.CITY_PRICES.items():
            if name.lower() in city_lower or city_lower in name.lower():
                return prices.get(fuel_type, prices["diesel"])
        return cls.CITY_PRICES["default"].get(fuel_type, 90.0)

    @classmethod
    def compute_fuel_cost(
        cls,
        distance_km: float,
        vehicle_type: str,
        city: str = "Mumbai",
        override_price: float = None,
        override_efficiency: float = None,
    ) -> dict:
        """
        Compute fuel cost for a given distance.

        Returns:
            {litres_consumed, price_per_litre, total_cost_inr,
             efficiency_kmpl, fuel_type, city}
        """
        fuel_type  = cls.VEHICLE_FUEL_TYPE.get(vehicle_type, "diesel")
        efficiency = override_efficiency or cls.VEHICLE_EFFICIENCY.get(vehicle_type, cls.VEHICLE_EFFICIENCY["default"])
        price      = override_price      or cls.get_price(city, fuel_type)
        litres     = distance_km / efficiency if efficiency > 0 else 0
        cost       = round(litres * price, 2)
        return {
            "litres_consumed":  round(litres, 2),
            "price_per_litre":  price,
            "total_cost_inr":   cost,
            "efficiency_kmpl":  efficiency,
            "fuel_type":        fuel_type,
            "city":             city,
        }

    @classmethod
    def savings_analysis(
        cls,
        stops_list: list,
        vehicle_type: str,
        city: str = "Mumbai",
        override_price: float = None,
        override_efficiency: float = None,
    ) -> dict:
        """
        Compare un-optimized (original stop order) vs NN-optimized order.

        Returns:
            {
                original_km, optimized_km, saved_km, saving_pct,
                original_cost, optimized_cost, saved_cost_inr,
                original_order, optimized_order,
                fuel_detail_original, fuel_detail_optimized,
                per_stop_savings: [{stop, original_leg_km, optimized_leg_km}]
            }
        """
        if len(stops_list) < 2:
            return {}

        coords = [(s["lat"], s["lon"]) for s in stops_list]

        # Original order — OSRM
        osrm_orig = MLEngine.osrm_route(coords)
        orig_km   = osrm_orig["total_distance_km"]

        # NN-optimized order — OSRM
        nn_order  = MLEngine().__class__().nearest_neighbor_route.__func__(MLEngine(), coords)
        coords_nn = [coords[i] for i in nn_order]
        osrm_opt  = MLEngine.osrm_route(coords_nn)
        opt_km    = osrm_opt["total_distance_km"]

        # Use optimized if it's actually better, else keep original
        if opt_km >= orig_km:
            opt_km    = orig_km
            osrm_opt  = osrm_orig
            nn_order  = list(range(len(stops_list)))

        saved_km   = round(orig_km - opt_km, 2)
        saving_pct = round(saved_km / orig_km * 100, 1) if orig_km > 0 else 0

        fuel_orig = cls.compute_fuel_cost(orig_km, vehicle_type, city, override_price, override_efficiency)
        fuel_opt  = cls.compute_fuel_cost(opt_km,  vehicle_type, city, override_price, override_efficiency)
        saved_cost = round(fuel_orig["total_cost_inr"] - fuel_opt["total_cost_inr"], 2)

        # Per-leg comparison (zip original vs optimized legs)
        per_stop = []
        orig_legs = osrm_orig.get("legs", [])
        opt_legs  = osrm_opt.get("legs", [])
        for i, s in enumerate(stops_list[1:]):
            orig_leg = orig_legs[i]["distance_km"] if i < len(orig_legs) else 0
            opt_s    = stops_list[nn_order[i+1]] if (i+1) < len(nn_order) else s
            opt_leg  = opt_legs[i]["distance_km"] if i < len(opt_legs) else 0
            per_stop.append({
                "leg":           i + 1,
                "original_stop": s["location_name"],
                "optimized_stop":opt_s["location_name"],
                "original_km":   orig_leg,
                "optimized_km":  opt_leg,
                "saved_km":      round(orig_leg - opt_leg, 2),
            })

        return {
            "original_km":          orig_km,
            "optimized_km":         opt_km,
            "saved_km":             saved_km,
            "saving_pct":           saving_pct,
            "original_cost_inr":    fuel_orig["total_cost_inr"],
            "optimized_cost_inr":   fuel_opt["total_cost_inr"],
            "saved_cost_inr":       saved_cost,
            "fuel_detail_original": fuel_orig,
            "fuel_detail_optimized":fuel_opt,
            "original_order":       [s["location_name"] for s in stops_list],
            "optimized_order":      [stops_list[i]["location_name"] for i in nn_order],
            "per_stop_savings":     per_stop,
            "routing_source":       osrm_orig["source"],
        }


# ─────────────────────────────────────────────
# GEO ENGINE  (reverse geocoding via Nominatim/OSM — free, no key)
# ─────────────────────────────────────────────
class GeoEngine:
    """
    Reverse geocoding: (lat, lon) → human-readable location name.

    Uses the Nominatim API (OpenStreetMap), which is:
      - Completely free — no API key required
      - No rate-limit issues for small batches (≤1 req/sec with delay)
      - Returns structured address: neighbourhood, suburb, city, state

    Results cached in st.session_state["geocode_cache"] so repeated
    parses of the same coordinates never re-fetch.

    Fallback: if Nominatim is unreachable (proxy / offline), returns a
    formatted string like "Loc @ 19.018, 72.848" so the app never blocks.
    """

    BASE_URL  = "https://nominatim.openstreetmap.org/reverse"
    CACHE_KEY = "geocode_cache"

    @classmethod
    def _cache_key(cls, lat: float, lon: float) -> str:
        return f"{round(lat, 4)}_{round(lon, 4)}"

    @classmethod
    def reverse_geocode(cls, lat: float, lon: float) -> dict:
        """
        Return a location dict for (lat, lon).

        Returns:
            {
                "display_name": str,   # short name for UI (suburb/area/city)
                "full_address": str,   # full OSM address string
                "source":       str,   # "nominatim" | "fallback"
            }
        """
        key   = cls._cache_key(lat, lon)
        cache = st.session_state.get(cls.CACHE_KEY, {})
        if key in cache:
            return cache[key]

        params = {
            "lat":            lat,
            "lon":            lon,
            "format":         "json",
            "zoom":           16,          # neighbourhood level
            "addressdetails": 1,
        }
        headers = {"User-Agent": "RouteIQ-Logistics/1.0"}

        result = None
        # Try HTTPS first (verify=False for internal proxies), then HTTP
        for scheme in ("https", "http"):
            url = f"{scheme}://nominatim.openstreetmap.org/reverse"
            try:
                resp = httpx.get(
                    url, params=params, headers=headers,
                    timeout=6.0, verify=False, follow_redirects=True,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if "error" in data:
                    continue

                addr   = data.get("address", {})
                # Build a compact, useful display name from the address components
                # Priority order for logistics: named place > suburb > neighbourhood > city
                parts  = []
                for key_try in ("amenity", "shop", "building", "tourism",
                                "road", "neighbourhood", "suburb",
                                "village", "town", "city_district", "city"):
                    val = addr.get(key_try, "")
                    if val and val not in parts:
                        parts.append(val)
                    if len(parts) >= 2:
                        break
                # Always append state/city for context
                city  = addr.get("city") or addr.get("town") or addr.get("village") or ""
                state = addr.get("state", "")
                if city and city not in parts:
                    parts.append(city)

                display  = ", ".join(parts) if parts else data.get("display_name", "Unknown location")
                # Trim extremely long OSM display names
                if len(display) > 60:
                    display = display[:57] + "…"

                result = {
                    "display_name": display,
                    "full_address": data.get("display_name", display),
                    "source":       "nominatim",
                }
                break
            except Exception:
                continue

        if result is None:
            result = {
                "display_name": f"Location @ {round(lat,4)}, {round(lon,4)}",
                "full_address": f"{lat}, {lon}",
                "source":       "fallback",
            }

        cache[key] = result
        st.session_state[cls.CACHE_KEY] = cache
        return result

    @classmethod
    def enrich_stops(cls, stops: list, only_coord_sourced: bool = True) -> list:
        """
        For each stop that was sourced from user-provided coordinates,
        call reverse_geocode and update location_name with the real OSM name.

        Args:
            stops:              list of stop dicts (as returned by the NL parser)
            only_coord_sourced: if True, only enrich stops with
                                coordinates_source == "user_provided"
                                (inferred stops already have AI-generated names)
        Returns:
            The same list with location_name and geocode_result added in-place.
        """
        import time
        enriched = []
        for i, stop in enumerate(stops):
            src = stop.get("coordinates_source", "inferred")
            if only_coord_sourced and src != "user_provided":
                stop["geocode_result"] = None
                enriched.append(stop)
                continue

            lat = stop.get("lat")
            lon = stop.get("lon")
            if lat is None or lon is None:
                stop["geocode_result"] = None
                enriched.append(stop)
                continue

            geo = cls.reverse_geocode(lat, lon)
            stop["geocode_result"] = geo

            # Override location_name with real OSM name
            if geo["source"] == "nominatim" and geo["display_name"]:
                stop["location_name"] = geo["display_name"]

            enriched.append(stop)

            # Nominatim fair-use: 1 request/sec (only applies when not cached)
            if i < len(stops) - 1:
                time.sleep(1.1)

        return enriched


# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────
class RAGEngine:
    DOMAIN_TOPICS = [
        # ── Core logistics ───────────────────────────────────────────────────
        "route", "routing", "delivery", "logistics", "shipment", "freight",
        "itinerary", "stop", "waypoint", "depot", "warehouse", "pickup",
        "dispatch", "fleet", "vehicle", "driver", "trucking", "transport",
        "cargo", "customs", "e-way bill", "manifest", "consignment",
        "last mile", "first mile", "supply chain", "distribution",
        "fuel", "mileage", "navigation", "gps", "tracking",
        "time window", "schedule", "delay", "on-time", "eta", "arrival",
        "temperature", "cold chain", "refrigerated", "hazmat", "dangerous goods",
        "pod", "proof of delivery", "invoice", "bill of lading",
        "kpi", "performance", "efficiency", "cost", "optimization",
        "travel", "distance", "trip", "journey", "road", "highway",
        "rail", "air freight", "sea freight", "port", "airport",
        "tms", "wms", "erp", "telematics", "iot",
        "breakdown", "insurance", "claim", "compliance", "regulation",
        # ── Natural navigation / direction language ───────────────────────────
        # Verbs people use when asking about getting from A to B
        "go from", "going from", "get from", "getting from",
        "travel from", "travelling from", "traveling from",
        "reach", "reaching", "how to reach", "how do i reach",
        "drive from", "driving from", "ride from", "riding from",
        "commute", "commuting",
        "best way", "fastest way", "shortest way", "quickest way",
        "how long", "how far", "how much time",
        "directions", "direction", "navigate", "path from", "path to",
        "way to", "way from", "route from", "route to",
        "from here", "to here",
        # ── Relational / between ─────────────────────────────────────────────
        "between",
        # ── Movement / transit words ─────────────────────────────────────────
        "bus", "train", "metro", "cab", "auto", "taxi", "uber", "ola",
        "toll", "highway", "expressway", "flyover", "bridge",
        "traffic", "congestion", "jam", "detour", "bypass",
        "drop", "pick up", "pickup point", "drop off",
        # ── Indian city / area names (common logistics hubs) ─────────────────
        # Mumbai
        "mumbai", "bombay", "bandra", "kurla", "andheri", "dadar",
        "thane", "navi mumbai", "panvel", "borivali", "kandivali",
        "malad", "goregaon", "jogeshwari", "vile parle", "santacruz",
        "bkc", "nariman", "churchgate", "csmt", "colaba", "worli",
        "lower parel", "prabhadevi", "matunga", "sion", "chembur",
        "ghatkopar", "vikhroli", "kanjurmarg", "bhandup", "mulund",
        "dombivli", "kalyan", "bhiwandi", "vasai", "virar", "mira road",
        "nhava sheva", "jnpt", "nhava",
        # Pune
        "pune", "pimpri", "chinchwad", "hadapsar", "kothrud", "hinjewadi",
        "wakad", "baner", "aundh", "shivajinagar", "talegaon",
        # Other major cities
        "delhi", "ncr", "gurgaon", "noida", "faridabad", "ghaziabad",
        "bengaluru", "bangalore", "whitefield", "electronic city",
        "hyderabad", "secunderabad", "cyberabad",
        "chennai", "kolkata", "ahmedabad", "surat", "jaipur",
        "lucknow", "chandigarh", "coimbatore", "kochi", "indore",
        "nagpur", "nashik", "aurangabad", "visakhapatnam",
        # ── Logistics / geographic terms ─────────────────────────────────────
        "zone", "area", "sector", "block", "lane", "street", "nagar",
        "colony", "society", "industrial area", "industrial estate",
        "cargo hub", "logistics park", "cold storage", "godown",
        "weighbridge", "octroi", "rto", "check post", "border",
    ]
    RELEVANCE_THRESHOLD = 0.30

    # Regex patterns that strongly indicate a route / navigation question
    # regardless of specific keywords — catches "from X to Y" style queries
    _NAV_PATTERNS = [
        r"\bfrom\b.{1,60}\bto\b",        # "from bandra to kurla"
        r"\bgo\b.{0,40}\bto\b",           # "go to andheri"
        r"\bget\b.{0,40}\bto\b",          # "get to the depot"
        r"\bread?ch\b",                      # "reach" / "reaching"
        r"\bdriv(e|ing)\b.{0,40}\bto\b",  # "drive to"
        r"\bhow\b.{0,30}\blong\b",        # "how long does it take"
        r"\bhow\b.{0,30}\bfar\b",         # "how far is X from Y"
        r"\bdir?ections?\b",                 # "directions" / "direction"
        r"\bnear(est)?\b",                   # "nearest depot"
        r"\bway\b.{0,30}\bto\b",          # "best way to reach"
        r"\broute\b.{0,30}\bfrom\b",      # "route from X"
        r"\bpath\b.{0,30}\bto\b",         # "path to warehouse"
    ]

    def __init__(self, llm, embeddings):
        self.llm       = llm
        self.embeddings = embeddings
        self._vectordb  = None

    def _build_vectordb(self, kb_df):
        if self._vectordb is not None or self.embeddings is None:
            return
        raw_docs, metadatas = [], []
        for _, row in kb_df.iterrows():
            raw_docs.append(f"[{row['category']} — {row['topic']}]\n{row['content']}")
            metadatas.append({"category": row["category"], "topic": row["topic"]})
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
        )
        chunks, chunk_metas = [], []
        for doc, meta in zip(raw_docs, metadatas):
            parts = splitter.split_text(doc)
            chunks.extend(parts)
            chunk_metas.extend([meta] * len(parts))
        try:
            self._vectordb = Chroma.from_texts(
                chunks, self.embeddings, metadatas=chunk_metas, persist_directory="./chroma_kb"
            )
        except Exception:
            self._vectordb = None

    def _retrieve(self, question, k=5):
        if self._vectordb is None:
            return [], [], []
        try:
            results = self._vectordb.similarity_search_with_relevance_scores(question, k=k)
            docs, scores, metas = [], [], []
            for doc, score in results:
                docs.append(doc.page_content)
                scores.append(score)
                metas.append(doc.metadata)
            return docs, scores, metas
        except Exception:
            return [], [], []

    def _is_in_domain(self, question, chunks, scores):
        """
        Three-layer domain check — passes if ANY layer matches.

        Layer 1 — Keyword scan: checks against expanded DOMAIN_TOPICS list.
                  Includes natural language navigation phrases, city names,
                  and movement verbs so "go from bandra to kurla" passes.

        Layer 2 — Regex navigation intent: pattern-matches spatial/directional
                  phrasing like "from X to Y", "how far", "nearest", "directions".
                  Catches questions that contain no logistics jargon but are
                  clearly asking about routes or locations.

        Layer 3 — Semantic similarity: at least one retrieved KB chunk must
                  score >= RELEVANCE_THRESHOLD (only active when embeddings online).
        """
        import re
        q = question.lower()

        # Layer 1: keyword scan
        if any(kw in q for kw in self.DOMAIN_TOPICS):
            return True

        # Layer 2: navigation intent regex
        for pattern in self._NAV_PATTERNS:
            if re.search(pattern, q):
                return True

        # Layer 3: semantic similarity
        if scores and max(scores) >= self.RELEVANCE_THRESHOLD:
            return True

        return False

    def _keyword_fallback(self, question, kb_df):
        q = question.lower()
        for _, row in kb_df.iterrows():
            words = row["topic"].lower().split() + row["category"].lower().split()
            if any(w in q for w in words):
                return f"**📖 {row['topic']}** *(from {row['category']} KB)*\n\n{row['content']}"
        return (
            "I couldn't find a matching article. "
            "Please ask about routing, delivery, customs, fleet, or other logistics topics."
        )

    def answer(self, question, kb_df):
        if not kb_df.empty and self.embeddings is not None and self._vectordb is None:
            with st.spinner("📚 Indexing knowledge base…"):
                self._build_vectordb(kb_df)

        chunks, scores, metas = self._retrieve(question, k=5)

        if not self._is_in_domain(question, chunks, scores):
            return {
                "text": (
                    "⛔ **Out of scope** — I'm RouteIQ Assistant, specialised exclusively "
                    "in **travel and logistics** topics.\n\n"
                    "I can help with route planning, delivery schedules, customs clearance, "
                    "fleet management, fuel costs, cargo compliance, and related subjects."
                ),
                "sources": [], "grounded": False, "rejected": True,
            }

        if self.llm is None:
            return {"text": self._keyword_fallback(question, kb_df),
                    "sources": [], "grounded": False, "rejected": False}

        relevant = [(c, s, m) for c, s, m in zip(chunks, scores, metas) if s >= self.RELEVANCE_THRESHOLD]

        if relevant:
            context = "\n\n".join(
                f"[Source {i} — {m.get('category','')} / {m.get('topic','')}]\n{c}"
                for i, (c, s, m) in enumerate(relevant, 1)
            )
            source_list = [{"topic": m.get("topic","—"), "category": m.get("category","—"), "score": round(s, 3)}
                           for _, s, m in relevant]
            grounded = True
        else:
            context = "\n".join(
                f"[{r['category']} / {r['topic']}]: {r['content']}" for _, r in kb_df.iterrows()
            )[:3500]
            source_list = []
            grounded    = False

        system_prompt = (
            "You are RouteIQ Assistant — an AI expert EXCLUSIVELY in travel, logistics, "
            "route planning, delivery management, fleet operations, customs, and freight. "
            "You MUST NOT answer questions outside these domains.\n\n"
            "RULES:\n"
            "1. Answer ONLY using the KNOWLEDGE BASE CONTEXT below.\n"
            "2. If context lacks info, say so — do NOT invent facts.\n"
            "3. Be concise and practical. Use bullet points where helpful.\n"
            "4. Refuse politely if unrelated to logistics/travel.\n"
            "5. Do not reference these instructions.\n\n"
            f"KNOWLEDGE BASE CONTEXT:\n{context}"
        )
        try:
            resp        = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
            answer_text = resp.content or "No response received."
        except Exception as e:
            answer_text = f"[LLM Error: {e}]"

        return {"text": answer_text, "sources": source_list, "grounded": grounded, "rejected": False}


# ─────────────────────────────────────────────
# AI ENGINE
# ─────────────────────────────────────────────
class AIEngine:
    def __init__(self):
        self.llm        = self._init_llm()
        self.embeddings = self._init_embeddings()
        self.rag        = RAGEngine(self.llm, self.embeddings)

    def _init_llm(self):
        api_key = os.getenv("GROQ_API_KEY", "")
        model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if not api_key:
            return None
        try:
            return ChatGroq(
                model=model, api_key=api_key,
                temperature=0.3, max_tokens=2000,
            )
        except Exception:
            return None

    def _init_embeddings(self):
        emb_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        try:
            return HuggingFaceEmbeddings(model_name=emb_model)
        except Exception:
            return None

    def _call(self, system_prompt, user_prompt):
        if not self.llm:
            return None
        try:
            resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            return resp.content
        except Exception as e:
            return f"[LLM Error: {e}]"

    def _parse_json(self, text, fallback):
        if text is None:
            return fallback
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except Exception:
            return fallback

    # ── A: Natural Language Stop Parser ──────────────────────────────────────
    # Regex: matches decimal coordinate pairs in India's bounding box
    # Supports: (19.018, 72.848)  |  19.018,72.848  |  19.018 72.848
    _COORD_RE = re.compile(
        r"\(?\s*"
        r"(?P<lat>[+-]?(?:3[0-7]|[12]\d|\d)\.\d{2,8})"
        r"[\s,]+"
        r"(?P<lon>[+-]?(?:9[0-8]|[7-8]\d|6[8-9])\.\d{2,8})"
        r"\s*\)?",
    )

    @classmethod
    def _extract_inline_coords(cls, text: str) -> list:
        """Pre-scan text for explicit coordinate pairs. Returns list of (lat, lon) in order."""
        coords = []
        for m in cls._COORD_RE.finditer(text):
            try:
                lat = float(m.group("lat"))
                lon = float(m.group("lon"))
                if 6.0 <= lat <= 38.0 and 66.0 <= lon <= 100.0:
                    coords.append((lat, lon))
            except ValueError:
                pass
        return coords

    def parse_natural_language_stops(self, text: str) -> dict:
        """
        Parse a free-text itinerary description into structured stops + constraints.

        Supports explicit GPS coordinates inline with stop names:
            "Dadar Warehouse (19.018, 72.848) at 9am"
            "pickup at 19.193,73.102 before 10am"
            "19.066 72.868 — BKC meeting at 2pm"

        When coordinates are provided:
          1. Python regex extracts them (exact, no LLM rounding)
          2. LLM is told to use them exactly + set coordinates_source="user_provided"
          3. Post-parse: GeoEngine.reverse_geocode() gets the real OSM location
             name for those coordinates and replaces the AI-guessed name.

        Returns: {stops, constraints, parse_notes, error}
        """
        inline_coords = self._extract_inline_coords(text)
        has_inline    = len(inline_coords) > 0

        coord_instruction = (
            f"The user has provided {len(inline_coords)} explicit coordinate pair(s). "
            "Use them EXACTLY as given — do not round or alter the numbers. "
            "For stops with user-provided coordinates, set coordinates_source to 'user_provided'. "
        ) if has_inline else (
            "No explicit coordinates were found. "
            "Infer realistic Indian GPS coordinates from location names. "
            "Set coordinates_source to 'inferred' for all stops. "
        )

        system_prompt = (
            "You are a logistics data extraction agent. "
            "Extract structured stop and constraint information from a natural language description. "
            + coord_instruction +
            "Return ONLY valid JSON — no markdown, no extra text."
        )

        user_prompt = (
            f'Extract all stops and constraints from:\n\n"{text}"\n\n'
            "Return this JSON schema exactly:\n"
            "{\n"
            '  "stops": [\n'
            "    {\n"
            '      "stop_id": "NL1",\n'
            '      "location_name": "place name only — no coordinates in this field",\n'
            '      "lat": <exact user coordinate or realistic India lat>,\n'
            '      "lon": <exact user coordinate or realistic India lon>,\n'
            '      "coordinates_source": "user_provided" or "inferred",\n'
            '      "stop_type": "Delivery|Pickup|Meeting|Warehouse|Customs|Rest",\n'
            '      "time_window_start": "HH:MM",\n'
            '      "time_window_end": "HH:MM",\n'
            '      "priority": "High|Medium|Low",\n'
            '      "notes": "any special instructions"\n'
            "    }\n"
            "  ],\n"
            '  "constraints": {\n'
            '    "driver_name": "string or Unknown",\n'
            '    "start_time": "HH:MM",\n'
            '    "transport_mode": "Road|Rail|Air|Sea",\n'
            '    "vehicle_type": "Truck|Van|Motorcycle|Car|Tempo",\n'
            '    "max_hours": <number>,\n'
            '    "vehicle_capacity_kg": <number>\n'
            "  },\n"
            '  "parse_notes": "brief summary of what was understood",\n'
            '  "error": null\n'
            "}\n\n"
            "Rules:\n"
            "- pick up/collect → Pickup; deliver/drop → Delivery; meeting/client → Meeting; "
            "rest/break → Rest; customs/checkpoint → Customs; warehouse/depot → Warehouse\n"
            "- urgent/asap/critical → High priority; default → Medium\n"
            "- 'by 2pm' → time_window_end 14:00; 'at 9am' → time_window_start 09:00\n"
            "- Default start_time 08:00, transport_mode Road if not mentioned\n"
            "- stop_id values: NL1, NL2, NL3 in order\n"
            "- Strip any coordinates from location_name — names only\n"
            "- Mixed input fine: some stops can have user coords, others inferred"
        )

        raw    = self._call(system_prompt, user_prompt)
        result = self._parse_json(raw, {"stops": [], "constraints": {}, "parse_notes": "",
                                        "error": "LLM offline or parse failed"})

        if "stops"       not in result: result["stops"]       = []
        if "constraints" not in result: result["constraints"] = {}
        if "error"       not in result: result["error"]       = None

        # ── Override LLM coords with exact regex values (prevents rounding) ──
        if has_inline and result["stops"]:
            coord_idx = 0
            for stop in result["stops"]:
                if stop.get("coordinates_source") == "user_provided":
                    if coord_idx < len(inline_coords):
                        stop["lat"] = inline_coords[coord_idx][0]
                        stop["lon"] = inline_coords[coord_idx][1]
                        coord_idx += 1

        return result

    # ── B: Itinerary Generation ───────────────────────────────────────────────
    def generate_itinerary(self, stops_list, constraints, route_context):
        """Generate optimized itinerary JSON, then patch in real OSRM distances."""
        system_prompt = (
            "You are an expert logistics route planner. "
            "Generate an optimized, feasible delivery itinerary as structured JSON. "
            "Prioritize: (1) time window compliance, (2) High-priority stops first, "
            "(3) shortest total distance. "
            "Times in HH:MM 24-hour format. Durations in minutes. "
            "Return ONLY valid JSON — no markdown."
        )
        user_prompt = (
            f"CONSTRAINTS:\n{json.dumps(constraints, indent=2)}\n\n"
            f"ROUTE CONTEXT (use these distances/times):\n{route_context}\n\n"
            f"STOPS TO PLAN ({len(stops_list)}):\n{json.dumps(stops_list, indent=2, default=str)}\n\n"
            "Return JSON with keys: itinerary_title, driver, vehicle, date (YYYY-MM-DD), "
            "transport_mode, total_distance_km, total_duration_min, estimated_fuel_cost_inr, "
            "optimization_notes, warnings (list), efficiency_score (0-100), "
            "on_time_probability (0-100), and stops array where each stop has: "
            "sequence, stop_id, location_name, arrival_time (HH:MM), departure_time (HH:MM), "
            "service_duration_min, travel_time_from_prev_min, distance_from_prev_km, "
            "stop_type, priority, status, notes, risk_flag (None/Low/Medium/High), "
            "risk_reason, time_window_start (HH:MM), time_window_end (HH:MM)."
        )
        raw      = self._call(system_prompt, user_prompt)
        result   = self._parse_json(raw, self._rule_based_itinerary(stops_list, constraints))

        # ── Patch real OSRM leg data ─────────────────────────────────────────
        coords     = [(s["lat"], s["lon"]) for s in stops_list]
        osrm       = MLEngine.osrm_route(coords)
        result["routing_source"] = osrm["source"]
        itin_stops = result.get("stops", [])

        if osrm["source"] == "osrm" and len(osrm["legs"]) >= len(itin_stops) - 1 and len(itin_stops) > 1:
            for i, stop in enumerate(itin_stops):
                if i == 0:
                    continue
                leg_idx = i - 1
                if leg_idx < len(osrm["legs"]):
                    stop["distance_from_prev_km"]     = osrm["legs"][leg_idx]["distance_km"]
                    stop["travel_time_from_prev_min"]  = osrm["legs"][leg_idx]["duration_min"]
            result["total_distance_km"]  = osrm["total_distance_km"]
            result["total_duration_min"] = round(
                osrm["total_duration_min"]
                + sum(s.get("service_duration_min", 0) for s in itin_stops), 1
            )
            result["estimated_fuel_cost_inr"] = round(osrm["total_distance_km"] * 8, 0)

        return result

    def _rule_based_itinerary(self, stops_list, constraints):
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        sorted_stops   = sorted(stops_list, key=lambda s: (
            priority_order.get(s.get("priority", "Low"), 2),
            s.get("time_window_start", "23:59"),
        ))
        current_time = datetime.strptime(constraints.get("start_time", "08:00"), "%H:%M")
        result_stops, total_dist = [], 0
        prev_lat = stops_list[0]["lat"] if stops_list else 19.076
        prev_lon = stops_list[0]["lon"] if stops_list else 72.877

        for i, s in enumerate(sorted_stops):
            dist        = MLEngine.compute_distance_km(prev_lat, prev_lon, s["lat"], s["lon"])
            travel_min  = max(5, int(dist / 35 * 60))
            service_min = {"Delivery": 20, "Pickup": 15, "Meeting": 45,
                           "Warehouse": 30, "Customs": 60, "Rest": 20}.get(s.get("stop_type", "Delivery"), 20)
            arrival     = current_time + timedelta(minutes=travel_min)
            departure   = arrival + timedelta(minutes=service_min)
            result_stops.append({
                "sequence": i + 1, "stop_id": s.get("stop_id", f"S{i+1}"),
                "location_name": s.get("location_name", "Stop"),
                "arrival_time": arrival.strftime("%H:%M"),
                "departure_time": departure.strftime("%H:%M"),
                "service_duration_min": service_min,
                "travel_time_from_prev_min": travel_min,
                "distance_from_prev_km": round(dist, 2),
                "stop_type": s.get("stop_type", "Delivery"),
                "priority": s.get("priority", "Medium"),
                "status": "Scheduled", "notes": s.get("notes", ""),
                "risk_flag": "None", "risk_reason": "",
                "time_window_start": s.get("time_window_start", "08:00"),
                "time_window_end":   s.get("time_window_end",   "18:00"),
            })
            total_dist += dist
            current_time = departure
            prev_lat, prev_lon = s["lat"], s["lon"]

        return {
            "itinerary_title": "Optimized Route (Rule-based fallback)",
            "driver": constraints.get("driver_name", "Driver"),
            "vehicle": constraints.get("vehicle_type", "Truck"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "transport_mode": constraints.get("transport_mode", "Road"),
            "total_distance_km": round(total_dist, 2),
            "total_duration_min": sum(
                s["travel_time_from_prev_min"] + s["service_duration_min"] for s in result_stops
            ),
            "estimated_fuel_cost_inr": round(total_dist * 8, 0),
            "stops": result_stops,
            "optimization_notes": "Rule-based fallback: sorted by priority then time window.",
            "warnings": ["LLM offline — using rule-based optimizer"],
            "efficiency_score": 72, "on_time_probability": 78,
        }

    def adjust_itinerary(self, existing_itinerary, change_request):
        system_prompt = (
            "You are a logistics re-planning agent. "
            "Return a FULLY updated itinerary JSON in the same schema. "
            "Return ONLY valid JSON — no markdown."
        )
        user_prompt = (
            f"EXISTING ITINERARY:\n{json.dumps(existing_itinerary, indent=2, default=str)}\n\n"
            f"CHANGE REQUEST:\n{change_request}\n\n"
            "Adjust all affected stop times. Explain the change in optimization_notes."
        )
        return self._parse_json(self._call(system_prompt, user_prompt), existing_itinerary)

    # ── C: Constraint Violation Checker ──────────────────────────────────────
    def check_violations(self, itinerary: dict, constraints: dict) -> list:
        """
        Check all stops for time-window breaches and driver hour overruns.
        Returns a list of violation dicts: {sequence, location_name, type, severity, detail}
        """
        violations = []
        stops      = sorted(itinerary.get("stops", []), key=lambda s: s["sequence"])
        if not stops:
            return violations

        date_str  = itinerary.get("date", datetime.now().strftime("%Y-%m-%d"))
        try:
            base = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        def to_dt(hhmm):
            try:
                h, m = map(int, str(hhmm).strip().split(":"))
                return base.replace(hour=h, minute=m, second=0, microsecond=0)
            except Exception:
                return base

        max_hours = constraints.get("max_hours", 10)
        day_end   = to_dt(constraints.get("start_time", "08:00")) + timedelta(hours=max_hours)

        for s in stops:
            arr          = to_dt(s.get("arrival_time",   "00:00"))
            dep          = to_dt(s.get("departure_time",  "00:00"))
            tw_start_raw = s.get("time_window_start", "")
            tw_end_raw   = s.get("time_window_end",   "")

            if tw_start_raw and tw_end_raw:
                tw_start = to_dt(tw_start_raw)
                tw_end   = to_dt(tw_end_raw)
                if arr < tw_start:
                    wait = int((tw_start - arr).seconds / 60)
                    violations.append({
                        "sequence": s["sequence"], "location_name": s.get("location_name",""),
                        "type": "time_window", "severity": "Warning",
                        "detail": (
                            f"Arrives at {s.get('arrival_time')} but window opens at "
                            f"{tw_start_raw}. Driver waits {wait} min."
                        ),
                    })
                elif arr > tw_end:
                    late = int((arr - tw_end).seconds / 60)
                    violations.append({
                        "sequence": s["sequence"], "location_name": s.get("location_name",""),
                        "type": "time_window", "severity": "Critical",
                        "detail": (
                            f"Arrives at {s.get('arrival_time')} — window closed at "
                            f"{tw_end_raw}. Late by {late} min. SLA breach."
                        ),
                    })

            if dep > day_end:
                over = int((dep - day_end).seconds / 60)
                violations.append({
                    "sequence": s["sequence"], "location_name": s.get("location_name",""),
                    "type": "driver_hours", "severity": "Critical",
                    "detail": (
                        f"Departure at {s.get('departure_time')} exceeds "
                        f"{max_hours}h limit by {over} min."
                    ),
                })

        return violations

    def get_kb_answer(self, question, kb_df):
        return self.rag.answer(question, kb_df)

    def analyze_route_performance(self, route_df, stops_df):
        summary = {
            "total_routes":      route_df["route_id"].nunique() if not route_df.empty else 0,
            "avg_distance_km":   round(route_df["distance_km"].mean(), 1) if not route_df.empty else 0,
            "avg_on_time_pct":   round(100 * (stops_df["status"] == "On Time").mean(), 1) if not stops_df.empty else 0,
            "top_delay_reasons": stops_df["delay_reason"].value_counts().head(3).to_dict()
                                  if "delay_reason" in stops_df.columns else {},
        }
        raw = self._call(
            "You are a logistics analytics expert. Provide 3-5 concise actionable insights.",
            f"Performance data:\n{json.dumps(summary, indent=2)}\n\nBullet point insights:"
        )
        if not raw or raw.startswith("[LLM"):
            return (
                "• Review high-delay routes for recurring traffic patterns\n"
                "• Adjust time windows for stops that are consistently late\n"
                "• Prioritize High-priority stops in morning slots\n"
                "• Consolidate nearby stops to reduce total distance and fuel cost"
            )
        return raw


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
class Dashboard:
    STOP_COLORS = {
        "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
        "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
    }
    PRIORITY_COLORS = {"High": "#E74C3C", "Medium": "#D4A843", "Low": "#3FB950"}
    STATUS_COLORS   = {
        "On Time": "#3FB950", "Delayed": "#E74C3C",
        "Scheduled": "#2EA4A4", "Cancelled": "#8B949E",
    }

    def _base_layout(self, height=320):
        return dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1D9", height=height,
            margin=dict(t=20, b=40, l=50, r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )

    def volume_timeline(self, stops_df):
        df2 = stops_df.copy()
        df2["date"] = pd.to_datetime(df2["scheduled_date"])
        daily = df2.groupby(["date","stop_type"]).size().reset_index(name="count")
        fig   = px.bar(daily, x="date", y="count", color="stop_type",
                       color_discrete_map=self.STOP_COLORS, barmode="stack")
        fig.update_layout(**self._base_layout(280),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"))
        return fig

    def stop_type_donut(self, stops_df):
        counts = stops_df["stop_type"].value_counts().reset_index()
        counts.columns = ["stop_type","count"]
        fig = px.pie(counts, names="stop_type", values="count",
                     color="stop_type", color_discrete_map=self.STOP_COLORS, hole=0.55)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker=dict(line=dict(color="#0D1117", width=2)))
        fig.update_layout(**self._base_layout(300), showlegend=False)
        return fig

    def distance_by_route(self, routes_df):
        df = routes_df.sort_values("total_distance_km", ascending=False).head(10)
        fig = go.Figure(go.Bar(x=df["route_id"], y=df["total_distance_km"],
                               marker_color="#D4A843",
                               hovertemplate="%{x}: %{y:.1f} km<extra></extra>"))
        fig.update_layout(**self._base_layout(280),
                          xaxis=dict(showgrid=False, tickangle=-20),
                          yaxis=dict(showgrid=True, gridcolor="#21262D", title="km"))
        return fig

    def on_time_gauge(self, pct):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pct,
            number={"suffix": "%", "font": {"color": "#D4A843", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
                "bar":  {"color": "#D4A843"}, "bgcolor": "#21262D",
                "steps": [
                    {"range": [0,  60],  "color": "rgba(192,57,43,0.3)"},
                    {"range": [60, 80],  "color": "rgba(232,135,58,0.3)"},
                    {"range": [80, 100], "color": "rgba(63,185,80,0.3)"},
                ],
                "threshold": {"line": {"color": "#3FB950", "width": 3}, "value": 85},
            },
        ))
        layout = self._base_layout(220)
        layout["margin"] = dict(t=20, b=10, l=30, r=30)

        fig.update_layout(**layout)
        return fig

    def priority_bar(self, stops_df):
        pc = stops_df["priority"].value_counts().reset_index()
        pc.columns = ["priority","count"]
        fig = go.Figure(go.Bar(
            x=pc["priority"], y=pc["count"],
            marker_color=[self.PRIORITY_COLORS.get(p,"#8B949E") for p in pc["priority"]],
            hovertemplate="%{x}: %{y}<extra></extra>",
        ))
        fig.update_layout(**self._base_layout(240),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"))
        return fig

    def route_map_scatter(self, stops_list, itinerary=None):
        lats   = [s["lat"] for s in stops_list]
        lons   = [s["lon"] for s in stops_list]
        names  = [s["location_name"] for s in stops_list]
        colors = [self.STOP_COLORS.get(s.get("stop_type","Delivery"), "#D4A843") for s in stops_list]

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="markers+text",
            marker=dict(size=12, color=colors, line=dict(color="#0D1117", width=1)),
            text=[f"{i+1}. {n}" for i, n in enumerate(names)],
            textposition="top center",
            textfont=dict(size=9, color="#C9D1D9"),
            hovertemplate="<b>%{text}</b><extra></extra>", name="Stops",
        ))
        if itinerary and "stops" in itinerary:
            seq  = sorted(itinerary["stops"], key=lambda s: s["sequence"])
            rlat, rlon = [], []
            for ist in seq:
                m = next((s for s in stops_list if s["stop_id"] == ist["stop_id"]), None)
                if m:
                    rlat.append(m["lat"]); rlon.append(m["lon"])
            if rlat:
                fig.add_trace(go.Scattergeo(lat=rlat, lon=rlon, mode="lines",
                                             line=dict(width=2, color="#D4A843"), name="Route"))
        fig.update_geos(
            center=dict(lat=np.mean(lats), lon=np.mean(lons)), projection_scale=8,
            showland=True, landcolor="#21262D", showocean=True, oceancolor="#161B22",
            showcountries=True, countrycolor="#30363D", showcoastlines=True, coastlinecolor="#30363D",
        )
        layout = self._base_layout(420)
        layout["margin"] = dict(t=10, b=10, l=10, r=10)
        fig.update_layout(**layout, geo=dict(bgcolor="rgba(0,0,0,0)"), showlegend=True)
        return fig

    def cluster_scatter(self, stops_df, labels, X_2d):
        df = stops_df.copy()
        df["Cluster"] = [f"Cluster {l+1}" for l in labels]
        df["x"] = X_2d[:, 0]; df["y"] = X_2d[:, 1]
        fig = px.scatter(df, x="x", y="y", color="Cluster",
                         hover_data=["location_name","stop_type","priority"],
                         color_discrete_sequence=["#D4A843","#2EA4A4","#3FB950","#8957E5","#E74C3C","#1F6FEB"])
        fig.update_layout(**self._base_layout(380),
                          xaxis=dict(showgrid=False, title="Component 1"),
                          yaxis=dict(showgrid=False, title="Component 2"))
        return fig

    def performance_timeline(self, routes_df):
        df = routes_df.copy(); df["date"] = pd.to_datetime(df["route_date"])
        fig = px.line(df.sort_values("date"), x="date", y="on_time_pct", color="transport_mode",
                      color_discrete_map={"Road":"#D4A843","Air":"#2EA4A4","Rail":"#3FB950","Sea":"#8957E5"})
        fig.update_layout(**self._base_layout(300), xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True, gridcolor="#21262D", title="On-Time %", range=[0,105]))
        return fig

    def fuel_cost_bar(self, routes_df):
        df = routes_df.groupby("transport_mode")["estimated_fuel_cost_inr"].mean().reset_index()
        fig = go.Figure(go.Bar(x=df["transport_mode"], y=df["estimated_fuel_cost_inr"],
                               marker_color=["#D4A843","#2EA4A4","#3FB950","#8957E5"],
                               hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
        fig.update_layout(**self._base_layout(260), xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True, gridcolor="#21262D", title="Avg Cost (₹)"))
        return fig


# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
def page_overview(stops_df, routes_df, stats, dash):
    st.markdown('<div class="hero-title">🗺️ Route<span class="hero-accent">IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AI-Powered Logistics Itinerary Planning & Route Optimization</div>', unsafe_allow_html=True)
    st.markdown("---")

    if stops_df.empty:
        st.warning("No data loaded. Please ensure stops.csv and routes.csv exist.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stops",    stats.get("total_stops", 0))
    c2.metric("Active Routes",  stats.get("total_routes", 0))
    c3.metric("Total Distance", f"{stats.get('total_distance_km', 0):,} km")
    c4.metric("On-Time Rate",   f"{stats.get('on_time_pct', 0)}%")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Daily Stop Volume by Type**")
        st.plotly_chart(dash.volume_timeline(stops_df), use_container_width=True)
    with col2:
        st.markdown("**Stop Type Distribution**")
        st.plotly_chart(dash.stop_type_donut(stops_df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Top Routes by Distance**")
        st.plotly_chart(dash.distance_by_route(routes_df), use_container_width=True)
    with col4:
        st.markdown("**Stop Priority Breakdown**")
        st.plotly_chart(dash.priority_bar(stops_df), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🚨 High-Priority Open Stops")
    high = stops_df[(stops_df["priority"] == "High") & (stops_df["status"] != "Delivered")].head(5)
    if high.empty:
        st.info("No high-priority open stops.")
    else:
        for _, row in high.iterrows():
            type_cls = {"Delivery":"card-gold","Meeting":"card-teal","Pickup":"card-green",
                        "Warehouse":"card-blue","Customs":"card-red"}.get(row["stop_type"],"card-gold")
            st.markdown(
                f'<div class="card {type_cls}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<div><b style="color:#F0F6FC">{row["location_name"]}</b>'
                f'<span class="badge badge-delivery" style="margin-left:8px">{row["stop_type"]}</span></div>'
                f'<div style="text-align:right;font-size:0.8rem;color:#8B949E">'
                f'Route: {row.get("route_id","—")} &nbsp;|&nbsp; {row.get("scheduled_date","—")}</div></div>'
                f'<div class="stop-detail">Window: {row.get("time_window_start","—")} – '
                f'{row.get("time_window_end","—")} &nbsp;·&nbsp; {row.get("notes","")}</div></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# PAGE: ITINERARY PLANNER  (3 tabs)
# ─────────────────────────────────────────────
def page_planner(stops_df, ai, ml, dash):
    st.markdown('<div class="section-title">🔍 Itinerary Planner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Generate optimized route plans — '
        'from your dataset, custom coordinates, or plain English</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📋 Plan from Dataset", "✏️ Custom Coordinates", "💬 Natural Language"])

    # ── TAB 1: Plan from Dataset ──────────────────────────────────────────────
    with tab1:
        if stops_df.empty:
            st.warning("No stops data loaded.")
        else:
            routes         = stops_df["route_id"].unique().tolist()
            selected_route = st.selectbox("Select Route to Plan", routes)
            route_stops    = stops_df[stops_df["route_id"] == selected_route].copy()

            st.markdown(f"**{len(route_stops)} stops on route {selected_route}**")
            st.dataframe(
                route_stops[["stop_id","location_name","stop_type","priority",
                              "time_window_start","time_window_end","notes"]].reset_index(drop=True),
                use_container_width=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                driver_name = st.text_input("Driver Name", value="Ramesh Kumar")
                start_time  = st.text_input("Start Time (HH:MM)", value="08:00")
            with col2:
                transport_mode   = st.selectbox("Transport Mode", ["Road","Rail","Air","Sea"])
                vehicle_type     = st.selectbox("Vehicle Type", ["Truck","Van","Motorcycle","Car","Tempo"])
            with col3:
                max_hours        = st.slider("Max Working Hours", 4, 14, 8)
                vehicle_capacity = st.number_input("Vehicle Capacity (kg)", 100, 10000, 1000, step=100)

            optimize = st.checkbox("Apply Nearest-Neighbor Optimization", value=True)

            if st.button("🚀 Generate Itinerary", type="primary", key="gen_dataset"):
                coords = list(zip(route_stops["lat"], route_stops["lon"]))
                if optimize and len(coords) > 2:
                    order       = ml.nearest_neighbor_route(coords)
                    route_stops = route_stops.iloc[order].reset_index(drop=True)

                stops_list = [
                    {
                        "stop_id":           str(r["stop_id"]),
                        "location_name":     r["location_name"],
                        "lat":               float(r["lat"]),
                        "lon":               float(r["lon"]),
                        "stop_type":         r["stop_type"],
                        "time_window_start": str(r["time_window_start"])[-8:-3] if pd.notnull(r["time_window_start"]) else "08:00",
                        "time_window_end":   str(r["time_window_end"])[-8:-3]   if pd.notnull(r["time_window_end"])   else "18:00",
                        "priority":          r["priority"],
                        "notes":             str(r.get("notes","")) if pd.notnull(r.get("notes","")) else "",
                    }
                    for _, r in route_stops.iterrows()
                ]
                constraints = {
                    "driver_name": driver_name, "start_time": start_time,
                    "transport_mode": transport_mode, "vehicle_type": vehicle_type,
                    "max_hours": max_hours, "vehicle_capacity_kg": vehicle_capacity,
                }

                with st.spinner("🛰️ Fetching real road distances via OSRM…"):
                    osrm_preview = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])

                st.caption(
                    "Routing: 🟢 OSRM (real roads)" if osrm_preview["source"] == "osrm"
                    else "Routing: 🟡 Haversine fallback (OSRM unreachable)"
                )
                route_context = (
                    f"Route {selected_route} | {len(stops_list)} stops | Mode: {transport_mode} | "
                    f"Road distance: {osrm_preview['total_distance_km']} km | "
                    f"Drive time: {osrm_preview['total_duration_min']} min | "
                    f"Source: {osrm_preview['source']}"
                )
                with st.spinner("🧠 AI optimizing your route…"):
                    itinerary = ai.generate_itinerary(stops_list, constraints, route_context)

                st.session_state["generated_itinerary"] = itinerary
                st.session_state["last_constraints"]    = constraints
                st.session_state["itinerary_source"]    = "dataset"
                st.session_state["fuel_analysis"]       = None  # reset stale savings
                st.session_state["plan_history"].append({
                    "source": "Dataset", "route": selected_route,
                    "generated_at": datetime.now().strftime("%H:%M:%S"),
                    "stops": len(stops_list),
                    "distance_km": itinerary.get("total_distance_km", 0),
                    "routing": osrm_preview["source"],
                })
                st.success("✅ Itinerary generated!")

    # ── TAB 2: Custom Coordinates ─────────────────────────────────────────────
    with tab2:
        st.markdown("Add custom stops with coordinates for ad-hoc planning.")
        custom_text = st.text_area(
            "One stop per line: Name, Lat, Lon, Type, Priority, TimeFrom, TimeTo",
            height=160,
            placeholder="Dadar Warehouse, 19.018, 72.848, Delivery, High, 09:00, 11:00",
            key="custom_coords_input",
        )
        col1, col2 = st.columns(2)
        with col1:
            c_driver = st.text_input("Driver Name", value="Suresh Patil", key="c_driver")
            c_start  = st.text_input("Start Time", value="08:30", key="c_start")
        with col2:
            c_mode  = st.selectbox("Transport Mode", ["Road","Rail","Air","Sea"], key="c_mode")
            c_hours = st.slider("Max Hours", 4, 14, 8, key="c_hours")
            c_cap   = st.number_input("Capacity (kg)", 100, 10000, 500, step=100, key="c_cap")

        if st.button("🚀 Generate Custom Itinerary", key="gen_custom"):
            custom_stops = []
            for i, line in enumerate(custom_text.strip().split("\n")):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        custom_stops.append({
                            "stop_id": f"CS{i+1}", "location_name": parts[0],
                            "lat": float(parts[1]), "lon": float(parts[2]),
                            "stop_type":         parts[3] if len(parts) > 3 else "Delivery",
                            "priority":          parts[4] if len(parts) > 4 else "Medium",
                            "time_window_start": parts[5] if len(parts) > 5 else "08:00",
                            "time_window_end":   parts[6] if len(parts) > 6 else "18:00",
                            "notes": "",
                        })
                    except ValueError:
                        pass

            if not custom_stops:
                st.error("No valid stops parsed. Check the format.")
            else:
                constraints = {
                    "driver_name": c_driver, "start_time": c_start,
                    "transport_mode": c_mode, "max_hours": c_hours,
                    "vehicle_type": "Van", "vehicle_capacity_kg": c_cap,
                }
                with st.spinner("🛰️ Fetching real road distances via OSRM…"):
                    osrm_cs = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in custom_stops])
                st.caption(
                    "Routing: 🟢 OSRM" if osrm_cs["source"] == "osrm"
                    else "Routing: 🟡 Haversine fallback"
                )
                route_ctx = (
                    f"{len(custom_stops)} custom stops | Mode: {c_mode} | "
                    f"Road distance: {osrm_cs['total_distance_km']} km | "
                    f"Drive time: {osrm_cs['total_duration_min']} min | Source: {osrm_cs['source']}"
                )
                with st.spinner("Planning custom route…"):
                    itinerary = ai.generate_itinerary(custom_stops, constraints, route_ctx)
                st.session_state["generated_itinerary"] = itinerary
                st.session_state["last_constraints"]    = constraints
                st.session_state["itinerary_source"]    = "custom"
                st.session_state["fuel_analysis"]       = None
                st.session_state["plan_history"].append({
                    "source": "Custom Coordinates", "route": "Custom",
                    "generated_at": datetime.now().strftime("%H:%M:%S"),
                    "stops": len(custom_stops),
                    "distance_km": itinerary.get("total_distance_km", 0),
                    "routing": osrm_cs["source"],
                })
                st.success("✅ Custom itinerary generated!")

    # ── TAB 3: Natural Language ───────────────────────────────────────────────
    with tab3:
        # If the last itinerary was generated from a different tab, show a clear notice
        # but do NOT wipe session state — that's what caused the dataset tab to break.
        # Generating from this tab will replace it.
        if st.session_state.get("itinerary_source") in ("dataset", "custom"):
            st.info(
                "ℹ️ The itinerary below was generated from the **Dataset** or **Custom** tab. "
                "Describe your route and click **Generate Full Itinerary** to create a new one here."
            )

        st.markdown(
            '<div class="summary-box">'
            '<b style="color:#D4A843">💬 Describe your route in plain English.</b><br>'
            '<span style="color:#8B949E;font-size:0.85rem">'
            'Mention stops, times, priorities, and constraints naturally. '
            'Include GPS coordinates inline if you have them — '
            '<code style="background:rgba(212,168,67,0.15);padding:1px 5px;border-radius:3px">'
            'Dadar (19.018, 72.848) at 9am</code> — '
            'the AI uses them exactly and <b style="color:#3FB950">OpenStreetMap</b> '
            'resolves the real location name automatically.'
            '</span></div>',
            unsafe_allow_html=True,
        )

        with st.expander("📖 Example prompts — click to expand"):
            st.markdown("""
**Location names only (AI infers coordinates):**
> Plan a route for Ramesh. Start at 8am from Dadar Warehouse. Deliver to Andheri client by 10am (urgent), pick up cargo from Kurla depot 11am–1pm, client meeting in BKC at 2:30pm.

**With explicit coordinates → OSM resolves real names automatically:**
> Pick up from (19.018, 72.848) at 9am, deliver to (19.119, 72.847) by 11am urgent, then meeting at (19.066, 72.868) at 2:30pm. Driver Suresh, Van.

**Mixed — names and coordinates together:**
> Start from Kalyan Junction (19.193, 73.102) at 8am. Deliver to Mulund Cold Storage by 10:30am. Customs at (18.950, 72.945) before 2pm.

**Coordinates with stop context:**
> Pickup at 19.193,73.102 at 9am (urgent), delivery at 19.119,72.847 by 11am, warehouse dropoff at 19.066,72.868 at 2pm. Max 8 hours, Truck.

**Multi-constraint with mixed input:**
> Suresh needs to collect documents from Pune Hadapsar at 9am, customs meeting at (18.950, 72.945) by 1pm, then deliver to Mulund cold storage before 4pm. Refrigerated van, max 9 hours.
            """)

        nl_text = st.text_area(
            "Describe your itinerary:",
            height=150,
            placeholder=(
                "Examples:\n"
                "• Plan a route for Ramesh from Dadar Warehouse (19.018, 72.848) at 8am. "
                "Deliver to Andheri (19.119, 72.847) by 10am urgent...\n"
                "• 3 stops: pickup at 19.193,73.102 at 9am, delivery at 19.119,72.847 by 11am, "
                "meeting at BKC at 2pm.\n"
                "\nCoordinates are reverse-geocoded to real location names via OpenStreetMap."
            ),
            key="nl_input",
        )

        nl_col1, nl_col2 = st.columns(2)
        with nl_col1:
            nl_driver = st.text_input("Override driver name (optional)", value="", key="nl_driver")
            nl_mode   = st.selectbox("Override transport mode",
                                     ["(auto-detect)","Road","Rail","Air","Sea"], key="nl_mode")
        with nl_col2:
            nl_hours = st.slider("Max working hours", 4, 14, 9, key="nl_hours")
            nl_cap   = st.number_input("Vehicle capacity (kg)", 100, 10000, 500, step=100, key="nl_cap")

        if st.button("🔍 Parse Description", key="nl_parse"):
            if not nl_text.strip():
                st.error("Please describe your route first.")
            elif not ai.llm:
                st.error("LLM is offline. Natural language parsing requires a connected LLM.")
            else:
                with st.spinner("🤖 Extracting stops from your description…"):
                    parsed = ai.parse_natural_language_stops(nl_text)

                # Reverse geocode any user-provided coordinates → get real location names
                coord_stops = [
                    s for s in parsed.get("stops", [])
                    if s.get("coordinates_source") == "user_provided"
                ]
                if coord_stops:
                    n = len(coord_stops)
                    with st.spinner(f"🗺️ Reverse geocoding {n} coordinate(s) via OpenStreetMap…"):
                        parsed["stops"] = GeoEngine.enrich_stops(
                            parsed["stops"], only_coord_sourced=True
                        )

                st.session_state["nl_parsed_result"] = parsed

        parsed = st.session_state.get("nl_parsed_result")
        if parsed:
            if parsed.get("error") and not parsed.get("stops"):
                st.error(f"Parse failed: {parsed['error']}")
            else:
                stops_list = parsed.get("stops", [])
                if parsed.get("parse_notes"):
                    st.info(f"📝 AI understood: {parsed['parse_notes']}")

                if stops_list:
                    # Count coordinate sources for summary badge
                    n_user = sum(1 for s in stops_list if s.get("coordinates_source") == "user_provided")
                    n_inf  = len(stops_list) - n_user
                    n_geo  = sum(1 for s in stops_list
                                 if s.get("geocode_result") and
                                 s["geocode_result"].get("source") == "nominatim")

                    st.markdown("#### ✅ Parsed Stops")

                    # Source summary
                    parts = []
                    if n_user > 0:
                        geo_note = f" ({n_geo} geocoded via OSM)" if n_geo > 0 else ""
                        parts.append(
                            f'<b style="color:#3FB950">📍 {n_user} user coordinate(s){geo_note}</b>'
                        )
                    if n_inf > 0:
                        parts.append(
                            f'<b style="color:#D4A843">🤖 {n_inf} AI-inferred location(s)</b>'
                        )
                    if parts:
                        st.markdown(
                            f'<div style="font-size:0.8rem;color:#8B949E;margin-bottom:8px">'
                            + " &nbsp;·&nbsp; ".join(parts) + "</div>",
                            unsafe_allow_html=True,
                        )

                    # Preview table rows
                    rows = []
                    for s in stops_list:
                        geo    = s.get("geocode_result")
                        src_lbl = "📍 User coord" if s.get("coordinates_source") == "user_provided" else "🤖 AI inferred"
                        if geo and geo.get("source") == "nominatim":
                            src_lbl = "🗺️ OSM geocoded"
                        rows.append({
                            "#":           s["stop_id"],
                            "Location":    s["location_name"],
                            "Coordinates": f"{round(s.get('lat',0), 5)}, {round(s.get('lon',0), 5)}",
                            "Coord Source":src_lbl,
                            "Type":        s["stop_type"],
                            "Priority":    s["priority"],
                            "Window":      f"{s['time_window_start']} – {s['time_window_end']}",
                            "Notes":       s.get("notes", ""),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    # Show full OSM address in expander if available
                    geo_stops = [s for s in stops_list
                                 if s.get("geocode_result") and s["geocode_result"].get("full_address")]
                    if geo_stops:
                        with st.expander(f"🗺️ Full OSM addresses ({len(geo_stops)} stop(s) geocoded)"):
                            for s in geo_stops:
                                geo = s["geocode_result"]
                                st.markdown(
                                    f'**{s["stop_id"]} — {s["location_name"]}**\n\n'
                                    f'<span style="font-size:0.8rem;color:#8B949E">'
                                    f'{geo["full_address"]}</span>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown("")

                    auto_c = parsed.get("constraints", {})
                    if nl_driver.strip():          auto_c["driver_name"]      = nl_driver.strip()
                    if nl_mode != "(auto-detect)": auto_c["transport_mode"]   = nl_mode
                    auto_c["max_hours"]          = nl_hours
                    auto_c["vehicle_capacity_kg"] = nl_cap

                    cc1, cc2, cc3 = st.columns(3)
                    cc1.markdown(f"**Driver:** {auto_c.get('driver_name','—')}")
                    cc2.markdown(f"**Mode:** {auto_c.get('transport_mode','Road')}")
                    cc3.markdown(f"**Start:** {auto_c.get('start_time','08:00')}")

                    if st.button("🚀 Generate Full Itinerary", type="primary", key="nl_generate"):
                        with st.spinner("🛰️ Fetching OSRM road distances…"):
                            osrm_nl = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])
                        st.caption(
                            "Routing: 🟢 OSRM" if osrm_nl["source"] == "osrm"
                            else "Routing: 🟡 Haversine fallback"
                        )
                        ctx_nl = (
                            f"{len(stops_list)} NL-parsed stops | "
                            f"Mode: {auto_c.get('transport_mode','Road')} | "
                            f"Road distance: {osrm_nl['total_distance_km']} km | "
                            f"Drive time: {osrm_nl['total_duration_min']} min | "
                            f"Source: {osrm_nl['source']}"
                        )
                        with st.spinner("🧠 Generating optimized itinerary…"):
                            itinerary = ai.generate_itinerary(stops_list, auto_c, ctx_nl)
                        # Build stops_for_map — NL itinerary embeds lat/lon in its stops
                        nl_map_stops = []
                        for itin_s in itinerary.get("stops", []):
                            matched = next((s for s in stops_list if s["stop_id"] == itin_s["stop_id"]), None)
                            if matched:
                                nl_map_stops.append({
                                    "stop_id":       itin_s["stop_id"],
                                    "location_name": itin_s["location_name"],
                                    "lat":           matched["lat"],
                                    "lon":           matched["lon"],
                                    "stop_type":     itin_s.get("stop_type","Delivery"),
                                })

                        st.session_state["generated_itinerary"] = itinerary
                        st.session_state["last_constraints"]    = auto_c
                        st.session_state["nl_parsed_result"]    = None
                        st.session_state["itinerary_source"]    = "nl"
                        st.session_state["nl_stops_for_map"]    = nl_map_stops
                        st.session_state["fuel_analysis"]       = None
                        st.session_state["plan_history"].append({
                            "source": "Natural Language", "route": "NL-parsed",
                            "generated_at": datetime.now().strftime("%H:%M:%S"),
                            "stops": len(stops_list),
                            "distance_km": itinerary.get("total_distance_km", 0),
                            "routing": osrm_nl["source"],
                        })
                        st.success("✅ Itinerary generated from your description!")
                        st.rerun()
                else:
                    st.warning("No stops extracted. Try adding specific location names and times.")

    # ── Display Itinerary ─────────────────────────────────────────────────────
    # Render the itinerary below the tabs.
    # itinerary_source tells us which tab owns the current result.
    # We ALWAYS show it here — the per-tab isolation works because:
    #   - Generating from tab1/tab2 sets source = "dataset"/"custom"
    #   - Generating from tab3 sets source = "nl"
    #   - Switching tabs does NOT clear the result (no unconditional clearing)
    #   - The user sees the last result they generated, regardless of which tab is active
    itin   = st.session_state.get("generated_itinerary")
    source = st.session_state.get("itinerary_source")
    if itin and source:
        st.markdown("---")
        nl_map = st.session_state.get("nl_stops_for_map", []) if source == "nl" else []
        _render_itinerary(
            itin, stops_df, ai, dash,
            st.session_state.get("last_constraints", {}),
            nl_stops_override=nl_map,
        )


# ─────────────────────────────────────────────
# WEATHER PANEL  (called from _render_itinerary)
# ─────────────────────────────────────────────
def _render_weather_panel(stops_with_coords: list, itin_date_str: str = None):
    """
    Renders arrival-time-aware weather forecast for every stop.

    For each stop the forecast shows conditions at the stop's arrival_time,
    not current conditions — e.g. Stop B arriving at 13:00 shows the 13:00
    hourly forecast, not what the weather is right now.

    stops_with_coords: list of dicts with {stop_id, location_name, lat, lon,
                        arrival_time (HH:MM), stop_type, sequence}
    itin_date_str:     "YYYY-MM-DD" — the itinerary date used to resolve forecasts
    """
    st.markdown("### 🌤️ Weather Forecast Along Route")
    st.markdown(
        '<div style="font-size:0.8rem;color:#8B949E;margin-bottom:16px">'
        'Forecast shown at <b style="color:#D4A843">each stop&#39;s expected arrival time</b> '
        '— not current conditions. Powered by Open-Meteo hourly API.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not stops_with_coords:
        st.info("No coordinate data available for weather lookup.")
        return

    with st.spinner("Fetching arrival-time forecasts for each stop…"):
        weather_stops = WeatherEngine.fetch_route_at_times(stops_with_coords, itin_date_str)

    # Data source summary
    sources = [ws["weather"]["source"] for ws in weather_stops]
    live_count = sum(1 for s in sources if "open-meteo" in s)
    est_count  = sum(1 for s in sources if "seasonal"   in s)

    if live_count == len(weather_stops):
        st.caption("🟢 All forecasts from Open-Meteo hourly API (real data)")
    elif live_count > 0:
        st.caption(f"🟡 Mixed: {live_count} stops from Open-Meteo · {est_count} stops from seasonal estimate")
    else:
        st.caption(
            "🟡 **Open-Meteo unreachable** — showing seasonal estimates with diurnal variation. "
            "Values are indicative. Check network/proxy settings."
        )

    adverse_count = sum(1 for ws in weather_stops if ws["weather"]["is_adverse"])
    if adverse_count:
        st.warning(
            f"⚠️ **{adverse_count} stop(s) forecast adverse weather at arrival time** — "
            "allow extra buffer time and check vehicle load securing."
        )

    # ── Cards — 4 per row ─────────────────────────────────────────────────────
    cols_per_row = min(4, len(weather_stops))
    card_rows    = [weather_stops[i:i+cols_per_row]
                    for i in range(0, len(weather_stops), cols_per_row)]

    for card_row in card_rows:
        cols = st.columns(len(card_row))
        for col, ws in zip(cols, card_row):
            w            = ws["weather"]
            source       = w.get("source", "")
            arrival      = ws.get("arrival_time", "")
            forecast_t   = w.get("forecast_time", arrival)
            precip_prob  = w.get("precip_probability")

            border_color = "#E74C3C" if w["is_adverse"] else "#30363D"
            temp_str = f"{w['temperature_c']}°C"     if w.get("temperature_c")    is not None else "—"
            wind_str = f"{w['wind_speed_kmh']} km/h" if w.get("wind_speed_kmh")   is not None else "—"
            prec_str = f"{w['precipitation_mm']} mm" if w.get("precipitation_mm") is not None else "—"
            hum_str  = f"{w['humidity_pct']}%"       if w.get("humidity_pct")     is not None else "—"
            vis_str  = f"{w['visibility_km']} km"    if w.get("visibility_km")    is not None else "—"
            prob_str = f"{precip_prob}% rain chance" if precip_prob is not None else ""

            if "open-meteo" in source:
                src_badge = '<div style="font-size:0.58rem;color:#3FB950;margin-top:5px">🟢 Open-Meteo hourly</div>'
            else:
                src_badge = '<div style="font-size:0.58rem;color:#D4A843;margin-top:5px">🟡 Seasonal estimate</div>'

            adverse_badge = (
                '<div style="font-size:0.65rem;color:#E74C3C;margin-top:5px;font-weight:600">⚠️ Adverse conditions</div>'
                if w["is_adverse"] else ""
            )
            prob_badge = (
                f'<div style="font-size:0.65rem;color:#2EA4A4;margin-top:3px">🌂 {prob_str}</div>'
                if prob_str else ""
            )

            col.markdown(
                f'<div class="weather-card" style="border:1px solid {border_color};margin-bottom:8px">'
                # Arrival time header — the KEY info
                f'<div style="font-size:0.6rem;font-family:monospace;color:#2EA4A4;'
                f'letter-spacing:0.05em;margin-bottom:6px">🕐 ARRIVAL {arrival}</div>'
                # Icon + temp
                f'<div style="font-size:1.9rem;margin-bottom:2px">{w["icon"]}</div>'
                f'<div style="font-size:0.78rem;font-weight:700;color:#F0F6FC;margin-bottom:2px">'
                f'{ws["location_name"][:20]}</div>'
                f'<div style="font-size:1.2rem;font-weight:700;color:#D4A843;margin-bottom:2px">{temp_str}</div>'
                f'<div style="font-size:0.72rem;color:#C9D1D9">{w["description"]}</div>'
                # Stats row
                f'<div style="font-size:0.65rem;color:#8B949E;margin-top:8px;line-height:1.7">'
                f'💨 {wind_str} &nbsp;·&nbsp; 🌧️ {prec_str}<br>'
                f'💧 {hum_str} &nbsp;·&nbsp; 👁️ {vis_str}'
                f'</div>'
                f'{prob_badge}{src_badge}{adverse_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Summary data table ────────────────────────────────────────────────────
    with st.expander("📋 Full Weather Forecast Table"):
        rows_data = []
        for ws in weather_stops:
            w = ws["weather"]
            rows_data.append({
                "Stop":              ws["location_name"],
                "Arrival Time":      ws.get("arrival_time", "—"),
                "Forecast At":       w.get("forecast_time", "—"),
                "Condition":         f"{w['icon']} {w['description']}",
                "Temp (°C)":         w.get("temperature_c", "—"),
                "Wind (km/h)":       w.get("wind_speed_kmh", "—"),
                "Rain (mm)":         w.get("precipitation_mm", "—"),
                "Rain Chance (%)":   w.get("precip_probability", "—"),
                "Humidity (%)":      w.get("humidity_pct", "—"),
                "Visibility (km)":   w.get("visibility_km", "—"),
                "Source":            w.get("source", "—"),
                "⚠️ Adverse":        "Yes" if w["is_adverse"] else "—",
            })
        st.dataframe(pd.DataFrame(rows_data), use_container_width=True)


# ─────────────────────────────────────────────
# FUEL PANEL  (called from _render_itinerary)
# ─────────────────────────────────────────────
def _render_fuel_panel(stops_list: list, constraints: dict, itin: dict):
    """
    Renders the full fuel cost + route savings analysis panel.
    """
    st.markdown("### ⛽ Fuel Cost & Route Savings Analysis")

    vehicle_type = constraints.get("vehicle_type", "Van")
    total_km     = itin.get("total_distance_km", 0)

    if not total_km:
        st.info("No distance data yet. Generate an itinerary first.")
        return

    # ── User controls ─────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        city = st.selectbox(
            "📍 City (for fuel price)",
            list(FuelEngine.CITY_PRICES.keys()),
            index=list(FuelEngine.CITY_PRICES.keys()).index("Mumbai")
                  if "Mumbai" in FuelEngine.CITY_PRICES else 0,
            key="fuel_city",
        )
    with fc2:
        fuel_type_options = ["Auto (by vehicle)", "Petrol", "Diesel"]
        fuel_choice = st.selectbox("⛽ Fuel type override", fuel_type_options, key="fuel_type_choice")
    with fc3:
        custom_price = st.number_input(
            "₹/litre override (0 = use reference)",
            min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="fuel_price_override"
        )

    override_price = custom_price if custom_price > 0 else None
    fuel_type_map  = {"Petrol": "petrol", "Diesel": "diesel"}

    # If override fuel type selected, compute price accordingly
    if fuel_choice != "Auto (by vehicle)" and override_price is None:
        ft          = fuel_type_map[fuel_choice]
        override_price = FuelEngine.get_price(city, ft)

    # ── Current itinerary fuel cost ───────────────────────────────────────────
    fuel_current = FuelEngine.compute_fuel_cost(
        total_km, vehicle_type, city, override_price
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route Distance",    f"{total_km} km")
    m2.metric("Fuel Consumed",     f"{fuel_current['litres_consumed']} L")
    m3.metric("Price/Litre",       f"₹{fuel_current['price_per_litre']:.2f}")
    m4.metric("Total Fuel Cost",   f"₹{fuel_current['total_cost_inr']:,.2f}")

    st.markdown(
        f'<div style="font-size:0.75rem;color:#8B949E;margin-bottom:20px">'
        f'Vehicle: <b>{vehicle_type}</b> · '
        f'Efficiency: <b>{fuel_current["efficiency_kmpl"]} km/L</b> · '
        f'Fuel: <b>{fuel_current["fuel_type"].title()}</b> · '
        f'City reference: <b>{city}</b> '
        f'<span style="color:#30363D">(prices as of Jun 2025 — update via override)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Route savings analysis ────────────────────────────────────────────────
    st.markdown("#### 🔀 Optimized vs Un-Optimized Route Savings")

    if len(stops_list) < 3:
        st.info("Add at least 3 stops to compute route savings comparison.")
        return

    if st.button("🔍 Run Savings Analysis", key="run_savings"):
        with st.spinner("Comparing original vs NN-optimized order via OSRM…"):
            analysis = FuelEngine.savings_analysis(
                stops_list, vehicle_type, city, override_price
            )
        st.session_state["fuel_analysis"] = analysis

    analysis = st.session_state.get("fuel_analysis")
    if not analysis:
        st.caption("Click 'Run Savings Analysis' to compare route orders.")
        return

    saved_km   = analysis.get("saved_km", 0)
    saving_pct = analysis.get("saving_pct", 0)
    saved_cost = analysis.get("saved_cost_inr", 0)

    if saved_km > 0:
        st.markdown(
            f'<div class="fuel-save">'
            f'<div style="font-size:1.05rem;font-weight:700;color:#3FB950;margin-bottom:10px">'
            f'✅ You can save <b>₹{saved_cost:,.2f}</b> by reordering your stops!</div>'
            f'<div style="display:flex;gap:32px;flex-wrap:wrap">'
            f'<span style="font-size:0.85rem;color:#C9D1D9">📏 Distance saved: <b>{saved_km} km</b></span>'
            f'<span style="font-size:0.85rem;color:#C9D1D9">📉 Reduction: <b>{saving_pct}%</b></span>'
            f'<span style="font-size:0.85rem;color:#C9D1D9">⛽ Fuel saved: '
            f'<b>{round(saved_km / analysis["fuel_detail_optimized"]["efficiency_kmpl"], 2)} L</b></span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="fuel-warn">'
            '<b style="color:#D4A843">ℹ️ Your current stop order is already near-optimal.</b><br>'
            '<span style="color:#8B949E;font-size:0.82rem">The nearest-neighbor reordering did not '
            'produce a shorter total route for this set of stops.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Side-by-side comparison
    sa1, sa2 = st.columns(2)
    with sa1:
        st.markdown("**📋 Original Order**")
        for i, name in enumerate(analysis.get("original_order", []), 1):
            st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
                        f'<span style="color:#D4A843">{i}.</span> {name}</div>', unsafe_allow_html=True)
        orig_c = analysis.get("original_cost_inr", 0)
        st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
                    f'<b>Total: {analysis.get("original_km",0)} km · ₹{orig_c:,.2f}</b></div>', unsafe_allow_html=True)

    with sa2:
        st.markdown("**✅ Optimized Order**")
        for i, name in enumerate(analysis.get("optimized_order", []), 1):
            st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
                        f'<span style="color:#3FB950">{i}.</span> {name}</div>', unsafe_allow_html=True)
        opt_c = analysis.get("optimized_cost_inr", 0)
        st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
                    f'<b>Total: {analysis.get("optimized_km",0)} km · ₹{opt_c:,.2f}</b></div>', unsafe_allow_html=True)

    # Per-leg savings chart
    per_stop = analysis.get("per_stop_savings", [])
    if per_stop:
        st.markdown("#### 📊 Leg-by-Leg Distance Comparison")
        df_legs = pd.DataFrame(per_stop)
        fig = go.Figure()
        fig.add_bar(
            name="Original", x=df_legs["leg"].astype(str),
            y=df_legs["original_km"],
            marker_color="#E74C3C",
            hovertemplate="Leg %{x}: %{y:.2f} km<extra>Original</extra>",
        )
        fig.add_bar(
            name="Optimized", x=df_legs["leg"].astype(str),
            y=df_legs["optimized_km"],
            marker_color="#3FB950",
            hovertemplate="Leg %{x}: %{y:.2f} km<extra>Optimized</extra>",
        )
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1D9", height=260,
            xaxis=dict(title="Leg #", showgrid=False),
            yaxis=dict(title="km", showgrid=True, gridcolor="#21262D"),
            margin=dict(t=10, b=40, l=50, r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Optimized Route Map ──────────────────────────────────────────────────
    optimized_order = analysis.get("optimized_order", [])
    if optimized_order and len(stops_list) >= 2:
        st.markdown("#### 🗺️ Optimized Route Map")
        st.caption("Stop order resequenced by Nearest-Neighbor TSP for minimum distance.")

        # Reorder stops_list to match the optimized sequence
        nn_order      = MLEngine().nearest_neighbor_route([(s["lat"], s["lon"]) for s in stops_list])
        opt_stops     = [stops_list[i] for i in nn_order]

        # Build a lightweight itinerary shell so route_map_scatter draws the route line
        opt_itin_stub = {
            "stops": [
                {
                    "sequence": idx + 1,
                    "stop_id":  s["stop_id"],
                    "stop_type": s.get("stop_type", "Delivery"),
                }
                for idx, s in enumerate(opt_stops)
            ]
        }

        # Use Dashboard.route_map_scatter — reuse existing map function
        from plotly.subplots import make_subplots as _msp  # already imported at top
        lats   = [s["lat"]  for s in opt_stops]
        lons   = [s["lon"]  for s in opt_stops]
        names  = [s["location_name"] for s in opt_stops]
        colors_map = {
            "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
            "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
        }
        pt_colors = [colors_map.get(s.get("stop_type","Delivery"), "#3FB950") for s in opt_stops]

        import plotly.graph_objects as _go
        fig_opt = _go.Figure()
        # Route line
        fig_opt.add_trace(_go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=2.5, color="#3FB950"), name="Optimized Route",
        ))
        # Stop markers
        fig_opt.add_trace(_go.Scattergeo(
            lat=lats, lon=lons, mode="markers+text",
            marker=dict(size=13, color=pt_colors, line=dict(color="#0D1117", width=1)),
            text=[f"{i+1}. {n}" for i, n in enumerate(names)],
            textposition="top center",
            textfont=dict(size=9, color="#C9D1D9"),
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Stops",
        ))
        import numpy as _np
        fig_opt.update_geos(
            center=dict(lat=_np.mean(lats), lon=_np.mean(lons)),
            projection_scale=8,
            showland=True,       landcolor="#21262D",
            showocean=True,      oceancolor="#161B22",
            showcountries=True,  countrycolor="#30363D",
            showcoastlines=True, coastlinecolor="#30363D",
        )
        fig_opt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1D9", height=420,
            margin=dict(t=10, b=10, l=10, r=10),
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_opt, use_container_width=True, key="opt_route_map")

    # Routing source note
    rs = analysis.get("routing_source", "")
    if rs == "osrm":
        st.caption("🛰️ Savings computed using real OSRM road distances.")
    else:
        st.caption("📐 Savings computed using Haversine estimates (OSRM unreachable).")


# ─────────────────────────────────────────────
# ITINERARY RENDERER
# ─────────────────────────────────────────────
def _render_itinerary(itin, stops_df, ai, dash, constraints=None, nl_stops_override=None):
    if constraints is None:
        constraints = {}

    # Build stops_list for map + fuel panel.
    # For dataset routes: match stop_ids back to the stops_df for coordinates.
    # For NL / custom routes: nl_stops_override carries the pre-built lat/lon list.
    stops_list = []
    if nl_stops_override:
        stops_list = nl_stops_override
    elif not stops_df.empty and "stop_id" in stops_df.columns:
        for s in itin.get("stops", []):
            row = stops_df[stops_df["stop_id"] == s["stop_id"]]
            if not row.empty:
                stops_list.append({
                    "stop_id": s["stop_id"], "location_name": s["location_name"],
                    "lat": float(row.iloc[0]["lat"]), "lon": float(row.iloc[0]["lon"]),
                    "stop_type": s["stop_type"],
                })

    # Header
    st.markdown(
        f'<div class="itinerary-header">'
        f'<div style="font-family:\'Playfair Display\',serif;font-size:1.4rem;color:#F0F6FC;font-weight:700">'
        f'📋 {itin.get("itinerary_title","Optimized Itinerary")}</div>'
        f'<div style="margin-top:10px;color:#8B949E;font-size:0.82rem">'
        f'🚗 {itin.get("driver","—")} &nbsp;|&nbsp; 🚛 {itin.get("vehicle","—")} '
        f'&nbsp;|&nbsp; 📅 {itin.get("date","—")} &nbsp;|&nbsp; 🛣️ {itin.get("transport_mode","—")}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Distance",      f"{itin.get('total_distance_km', 0)} km")
    c2.metric("Total Duration",      f"{itin.get('total_duration_min', 0)} min")
    c3.metric("Efficiency Score",    f"{itin.get('efficiency_score', 0)}/100")
    c4.metric("On-Time Probability", f"{itin.get('on_time_probability', 0)}%")

    # Routing source banner
    r_src = itin.get("routing_source", "")
    if r_src == "osrm":
        st.success("🛰️ Road distances powered by **OSRM** — real road network data")
    elif r_src == "haversine_fallback":
        st.warning("📐 Straight-line distance estimates used (OSRM unreachable). Times are approximate.")

    # Map
    if stops_list:
        st.markdown("### 🗺️ Route Map")
        st.plotly_chart(dash.route_map_scatter(stops_list, itin), use_container_width=True)

    # Stop sequence
    st.markdown("### 📍 Stop Sequence")
    itin_stops = sorted(itin.get("stops", []), key=lambda x: x["sequence"])
    max_seq    = max((x["sequence"] for x in itin_stops), default=0)

    for s in itin_stops:
        risk_color = {"High":"#E74C3C","Medium":"#D4A843","Low":"#3FB950","None":"#3FB950"}.get(
            s.get("risk_flag","None"), "#3FB950"
        )
        risk_html = (
            f' &nbsp;·&nbsp; <span style="color:{risk_color}">⚠️ {s.get("risk_reason","")}</span>'
            if s.get("risk_flag","None") not in ["None",""] else ""
        )
        notes_html = (
            f'<div class="stop-detail" style="margin-top:4px;font-style:italic">{s.get("notes","")}</div>'
            if s.get("notes") else ""
        )
        connector = "" if s["sequence"] == max_seq else '<div class="connector-line"></div>'
        st.markdown(
            f'<div class="stop-card"><div class="stop-row">'
            f'<div class="stop-num">{s["sequence"]}</div>'
            f'<div style="flex:1">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<b style="color:#F0F6FC">{s["location_name"]}</b>'
            f'<span style="font-size:0.8rem;color:#8B949E">🕐 {s.get("arrival_time","—")} → {s.get("departure_time","—")}</span>'
            f'</div>'
            f'<div class="stop-detail">Type: {s.get("stop_type","—")} &nbsp;·&nbsp; '
            f'Priority: {s.get("priority","—")} &nbsp;·&nbsp; '
            f'Service: {s.get("service_duration_min","—")} min &nbsp;·&nbsp; '
            f'Travel: {s.get("travel_time_from_prev_min","—")} min &nbsp;·&nbsp; '
            f'Dist: {s.get("distance_from_prev_km","—")} km{risk_html}</div>'
            f'{notes_html}</div></div></div>{connector}',
            unsafe_allow_html=True,
        )

    if itin.get("warnings"):
        with st.expander("⚠️ Warnings"):
            for w in itin["warnings"]:
                st.warning(w)

    with st.expander("📝 Optimization Notes"):
        st.info(itin.get("optimization_notes", "No notes."))

    # ── Constraint Violation Checker ─────────────────────────────────────────
    st.markdown("### ⚡ Constraint Violation Check")
    if constraints:
        violations = ai.check_violations(itin, constraints)
        if not violations:
            st.markdown(
                '<div style="background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);'
                'border-radius:10px;padding:14px 18px;margin-bottom:12px">'
                '<b style="color:#3FB950">✅ No violations detected.</b> '
                'All stops are within time windows and driver hour limits.</div>',
                unsafe_allow_html=True,
            )
        else:
            crit = [v for v in violations if v["severity"] == "Critical"]
            warn = [v for v in violations if v["severity"] == "Warning"]
            vc1, vc2 = st.columns(2)
            vc1.metric("🔴 Critical", len(crit))
            vc2.metric("🟡 Warnings", len(warn))
            for v in violations:
                css   = "violation-critical" if v["severity"] == "Critical" else "violation-warning"
                color = "#E74C3C"            if v["severity"] == "Critical" else "#D4A843"
                icon  = "🔴"                 if v["severity"] == "Critical" else "🟡"
                label = {"time_window":"Time Window","driver_hours":"Driver Hours","capacity":"Capacity"}.get(v["type"], v["type"])
                st.markdown(
                    f'<div class="{css}">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<b style="color:#F0F6FC">{icon} Stop {v["sequence"]} — {v["location_name"]}</b>'
                    f'<span style="font-size:0.75rem;color:{color};font-weight:600">{label} · {v["severity"]}</span>'
                    f'</div><div style="font-size:0.82rem;color:#C9D1D9;margin-top:6px">{v["detail"]}</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Generate an itinerary with constraints to see violation analysis.")

    # ── Weather Panel ────────────────────────────────────────────────────────
    # Build a stops list that includes arrival_time so forecasts are time-aware.
    # Priority: dataset stops_list merged with itinerary arrival times.
    itin_date = itin.get("date", datetime.now().strftime("%Y-%m-%d"))

    # Map stop_id → arrival_time from the generated itinerary
    arrival_map = {
        s["stop_id"]: s.get("arrival_time", "08:00")
        for s in itin.get("stops", [])
    }

    weather_stops = []
    if stops_list:
        # Dataset route: merge lat/lon from stops_list + arrival_time from itinerary
        for s in stops_list:
            weather_stops.append({
                **s,
                "arrival_time": arrival_map.get(s["stop_id"], "08:00"),
                "sequence":     next(
                    (itin_s["sequence"] for itin_s in itin.get("stops", [])
                     if itin_s["stop_id"] == s["stop_id"]), 0
                ),
            })
        # Sort by sequence so cards appear in travel order
        weather_stops.sort(key=lambda x: x.get("sequence", 0))
    else:
        # NL / custom stops — coords embedded in itinerary stops
        for s in itin.get("stops", []):
            if "lat" in s and "lon" in s:
                weather_stops.append({
                    "stop_id":       s.get("stop_id", ""),
                    "location_name": s.get("location_name", ""),
                    "lat":           s["lat"],
                    "lon":           s["lon"],
                    "stop_type":     s.get("stop_type", ""),
                    "arrival_time":  s.get("arrival_time", "08:00"),
                    "sequence":      s.get("sequence", 0),
                })

    if weather_stops:
        st.markdown("---")
        _render_weather_panel(weather_stops, itin_date_str=itin_date)

    # ── Fuel & Savings Panel ─────────────────────────────────────────────────
    st.markdown("---")
    # Build stops_list for fuel engine — prefer stops_list (dataset), else NL
    fuel_stops = stops_list
    if not fuel_stops:
        for s in itin.get("stops", []):
            if "lat" in s and "lon" in s:
                fuel_stops.append({
                    "stop_id": s.get("stop_id",""),
                    "location_name": s.get("location_name",""),
                    "lat": s["lat"], "lon": s["lon"],
                    "stop_type": s.get("stop_type",""),
                })
    _render_fuel_panel(fuel_stops, constraints, itin)

    # Adjust
    st.markdown("### 🔄 Adjust Itinerary")
    change_req = st.text_area(
        "Describe the change in plain English:",
        placeholder="e.g. 'Remove stop 3', 'Add 30-min rest after stop 2', 'Traffic — delay all by 20 min'",
        height=90,
    )
    if st.button("Apply Changes"):
        if change_req.strip():
            with st.spinner("Re-planning…"):
                updated = ai.adjust_itinerary(itin, change_req)
            st.session_state["generated_itinerary"] = updated
            st.success("Itinerary updated!")
            st.rerun()
        else:
            st.error("Please describe the change.")

    # Export
    st.markdown("### 📤 Export")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button("⬇️ Download JSON", json.dumps(itin, indent=2, default=str),
                           "itinerary.json", "application/json", use_container_width=True)
    with col_e2:
        stop_rows = itin.get("stops", [])
        if stop_rows:
            st.download_button("⬇️ Download CSV", pd.DataFrame(stop_rows).to_csv(index=False),
                               "itinerary_stops.csv", "text/csv", use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: CLUSTERING
# ─────────────────────────────────────────────
def page_clustering(stops_df, ml, dash):
    st.markdown('<div class="section-title">🧩 Stop Clustering Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Unsupervised clustering to find natural groupings in your stop network</div>', unsafe_allow_html=True)

    if stops_df.empty:
        st.warning("No stops data loaded.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        n_clusters  = st.slider("Number of Clusters", 2, 8, 4)
        sample_size = st.slider("Sample Size", 20, min(len(stops_df), 200), min(len(stops_df), 60))
        if st.button("Run Clustering"):
            sample_df = stops_df.sample(min(sample_size, len(stops_df)), random_state=42)
            labels, X_2d = ml.cluster_stops(sample_df, n_clusters)
            st.session_state["cluster_labels"] = labels
            st.session_state["cluster_X2d"]   = X_2d
            st.session_state["cluster_df"]    = sample_df.reset_index(drop=True)
            st.success("Clustering complete!")

    with col2:
        if st.session_state.get("cluster_labels") is not None:
            st.markdown("**Cluster Visualization (PCA 2D)**")
            st.plotly_chart(
                dash.cluster_scatter(st.session_state["cluster_df"],
                                     st.session_state["cluster_labels"],
                                     st.session_state["cluster_X2d"]),
                use_container_width=True,
            )

    if st.session_state.get("cluster_labels") is not None:
        labels = st.session_state["cluster_labels"]
        cdf    = st.session_state["cluster_df"]

        keywords = ml.get_cluster_keywords()
        st.markdown("### Cluster Keyword Profiles")
        cols = st.columns(min(len(keywords), 5))
        for i, (cid, kws) in enumerate(keywords.items()):
            with cols[i % len(cols)]:
                st.markdown(
                    f'<div class="card card-gold"><b>Cluster {cid+1}</b><br>'
                    f'<small style="color:#8B949E">{" · ".join(kws)}</small></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Cluster Composition by Stop Type")
        cdf2 = cdf.copy()
        cdf2["cluster"] = [f"Cluster {l+1}" for l in labels]
        comp = cdf2.groupby(["cluster","stop_type"]).size().reset_index(name="count")
        fig  = px.bar(comp, x="cluster", y="count", color="stop_type",
                      color_discrete_map=dash.STOP_COLORS, barmode="stack")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1D9", height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Geographic Distribution of Clusters")
        cdf2["x"] = st.session_state["cluster_X2d"][:, 0]
        fig_geo   = px.scatter_mapbox(
            cdf2, lat="lat", lon="lon", color="cluster",
            hover_data=["location_name","stop_type","priority"], zoom=9, height=380,
        )
        fig_geo.update_layout(
            mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1D9", margin=dict(t=0,b=0,l=0,r=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_geo, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: AI ASSISTANT
# ─────────────────────────────────────────────
def page_assistant(stops_df, ai, kb_df):
    st.markdown('<div class="section-title">💬 Logistics AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Grounded in your logistics knowledge base · '
        'Only answers travel & logistics questions</div>',
        unsafe_allow_html=True,
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    for col, label, active in [
        (col_s1, "LLM",        bool(ai.llm)),
        (col_s2, "Embeddings", bool(ai.embeddings)),
        (col_s3, "RAG Index",  bool(ai.embeddings) and not kb_df.empty),
    ]:
        color  = "#2EA4A4" if active else "#E74C3C"
        status = ("🟢 Connected" if active else "🔴 Offline") if label != "RAG Index" else ("🟢 Ready" if active else "⚠️ Keyword fallback")
        col.markdown(
            f'<div class="card" style="padding:10px;text-align:center">'
            f'<span style="font-size:0.8rem;color:#8B949E">{label}</span><br>'
            f'<b style="color:{color}">{status}</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    with st.expander("ℹ️ What can I ask?", expanded=False):
        st.markdown("""
**In scope:** route planning, delivery operations, customs, fleet management, freight modes, KPIs, documentation.
**Out of scope:** questions unrelated to travel or logistics are politely declined.
        """)

    if "assistant_history" not in st.session_state:
        st.session_state["assistant_history"] = []

    for msg in st.session_state["assistant_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    _render_sources(msg["sources"])
                if msg.get("rejected"):
                    st.caption("🚫 Outside the logistics/travel domain.")
                elif msg.get("grounded"):
                    st.caption("✅ Grounded in knowledge base via semantic retrieval.")

    user_input = st.chat_input("Ask about routes, delivery windows, customs, fuel costs…")
    if user_input:
        st.session_state["assistant_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                result = ai.get_kb_answer(user_input, kb_df)
            answer_text = result["text"]
            sources     = result.get("sources", [])
            grounded    = result.get("grounded", False)
            rejected    = result.get("rejected", False)
            st.markdown(answer_text)
            if sources:   _render_sources(sources)
            if rejected:  st.caption("🚫 Outside the logistics/travel domain.")
            elif grounded: st.caption("✅ Grounded in knowledge base via semantic retrieval.")
            else:          st.caption("⚠️ Low-confidence retrieval — broader KB context used.")

        st.session_state["assistant_history"].append({
            "role": "assistant", "content": answer_text,
            "sources": sources, "grounded": grounded, "rejected": rejected,
        })

    col_a, _ = st.columns([1, 5])
    with col_a:
        if st.button("🗑️ Clear Chat"):
            st.session_state["assistant_history"] = []
            st.rerun()

    if not kb_df.empty:
        with st.expander(f"📚 Knowledge Base ({len(kb_df)} articles)"):
            st.dataframe(
                kb_df[["category","topic","content"]], use_container_width=True,
                column_config={"content": st.column_config.TextColumn("content", width="large")},
            )


def _render_sources(sources: list):
    if not sources:
        return
    seen, unique = set(), []
    for s in sources:
        key = s.get("topic","")
        if key not in seen:
            seen.add(key); unique.append(s)

    lines = []
    for s in unique:
        pct        = int(s.get("score", 0) * 100)
        bar        = "█" * (pct // 10) + "░" * (10 - pct // 10)
        lines.append(
            f"**{s.get('category','—')}** › {s.get('topic','—')} "
            f"<span style='color:#D4A843;font-family:monospace;font-size:0.75rem'>{bar} {pct}%</span>"
        )
    st.markdown(
        '<div style="background:rgba(46,164,164,0.08);border:1px solid rgba(46,164,164,0.25);'
        'border-radius:8px;padding:10px 14px;margin-top:8px">'
        '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.06em;'
        'text-transform:uppercase;margin-bottom:6px">📎 Sources retrieved</div>'
        + "".join(f'<div style="font-size:0.8rem;color:#C9D1D9;margin:3px 0">{l}</div>' for l in lines)
        + "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PAGE: PERFORMANCE
# ─────────────────────────────────────────────
def page_performance(stops_df, routes_df, ai, dash):
    st.markdown('<div class="section-title">📈 Performance Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Route efficiency, on-time delivery, fuel costs, and operational insights</div>', unsafe_allow_html=True)

    if stops_df.empty or routes_df.empty:
        st.warning("No data loaded.")
        return

    c1, c2, c3, c4 = st.columns(4)
    on_time_pct     = round(100 * (stops_df["status"] == "On Time").mean(), 1)
    avg_distance    = round(routes_df["distance_km"].mean(), 1)
    total_fuel_cost = round(routes_df["estimated_fuel_cost_inr"].sum(), 0)
    avg_stops       = round(stops_df.groupby("route_id").size().mean(), 1)
    c1.metric("On-Time Rate",       f"{on_time_pct}%")
    c2.metric("Avg Distance/Route", f"{avg_distance} km")
    c3.metric("Total Fuel Cost",    f"₹{total_fuel_cost:,.0f}")
    c4.metric("Avg Stops/Route",    avg_stops)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**On-Time % by Transport Mode (Trend)**")
        st.plotly_chart(dash.performance_timeline(routes_df), use_container_width=True)
    with col2:
        st.markdown("**On-Time Delivery Rate**")
        st.plotly_chart(dash.on_time_gauge(on_time_pct), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Avg Fuel Cost by Transport Mode**")
        st.plotly_chart(dash.fuel_cost_bar(routes_df), use_container_width=True)
    with col4:
        st.markdown("**Stops by Status**")
        sc = stops_df["status"].value_counts().reset_index()
        sc.columns = ["status","count"]
        fig = go.Figure(go.Bar(
            x=sc["status"], y=sc["count"],
            marker_color=[dash.STATUS_COLORS.get(s,"#8B949E") for s in sc["status"]],
            hovertemplate="%{x}: %{y}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1D9", height=260,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"),
            margin=dict(t=10, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    if "delay_reason" in stops_df.columns:
        st.markdown("**Delay Reasons**")
        delayed = stops_df[stops_df["delay_reason"].notna() & (stops_df["delay_reason"] != "")]
        if not delayed.empty:
            dr    = delayed["delay_reason"].value_counts().reset_index()
            dr.columns = ["reason","count"]
            fig_d = px.bar(dr, x="count", y="reason", orientation="h",
                           color_discrete_sequence=["#E74C3C"])
            fig_d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#C9D1D9", height=260,
                xaxis=dict(showgrid=True, gridcolor="#21262D"),
                yaxis=dict(showgrid=False),
                margin=dict(t=10, b=30, l=150, r=20),
            )
            st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("### 🤖 AI Route Performance Insights")
    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing route performance…"):
            insights = ai.analyze_route_performance(routes_df, stops_df)
        st.markdown(f'<div class="summary-box">{insights}</div>', unsafe_allow_html=True)

    if st.session_state.get("plan_history"):
        with st.expander("📋 Itinerary Generation History (this session)"):
            st.dataframe(pd.DataFrame(st.session_state["plan_history"]), use_container_width=True)

    with st.expander("📋 Full Stops Table"):
        st.dataframe(stops_df.sort_values("scheduled_date", ascending=False), use_container_width=True)

    with st.expander("📋 Full Routes Table"):
        st.dataframe(routes_df.sort_values("route_date", ascending=False), use_container_width=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    init_session()

    loader = DataLoader()
    ai     = AIEngine()
    ml     = MLEngine()
    dash   = Dashboard()

    stops_df  = loader.load_stops()
    routes_df = loader.load_routes()
    kb_df     = loader.load_kb()
    stats     = loader.get_summary_stats(stops_df, routes_df)

    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:20px 0 10px">'
            '<div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;color:#F0F6FC;font-weight:900">🗺️ RouteIQ</div>'
            '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.1em;text-transform:uppercase">Logistics Itinerary Planner</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["📊 Overview", "🔍 Itinerary Planner", "🧩 Clustering", "💬 AI Assistant", "📈 Performance"],
            label_visibility="collapsed",
        )

        if not stops_df.empty:
            st.markdown("---")
            st.markdown(
                '<div style="font-size:0.7rem;color:#8B949E;letter-spacing:0.1em;'
                'text-transform:uppercase;margin-bottom:8px">Quick Stats</div>',
                unsafe_allow_html=True,
            )
            for val, label, color in [
                (stats.get("total_stops",  0),              "Total Stops",        "#D4A843"),
                (stats.get("total_routes", 0),              "Active Routes",      "#2EA4A4"),
                (f"{stats.get('on_time_pct',0)}%",          "On-Time Rate",       "#3FB950"),
                (stats.get("high_priority", 0),             "High Priority Stops","#E74C3C"),
            ]:
                st.markdown(
                    f'<div class="card" style="padding:12px">'
                    f'<div style="color:{color};font-size:1.4rem;font-weight:700">{val}</div>'
                    f'<div style="color:#8B949E;font-size:0.72rem">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.75rem;color:#8B949E">'
            f'{"🟢 LLM Connected" if ai.llm else "🔴 LLM Offline (fallback)"}</div>',
            unsafe_allow_html=True,
        )

    if page == "📊 Overview":
        page_overview(stops_df, routes_df, stats, dash)
    elif page == "🔍 Itinerary Planner":
        page_planner(stops_df, ai, ml, dash)
    elif page == "🧩 Clustering":
        page_clustering(stops_df, ml, dash)
    elif page == "💬 AI Assistant":
        page_assistant(stops_df, ai, kb_df)
    elif page == "📈 Performance":
        page_performance(stops_df, routes_df, ai, dash)


if __name__ == "__main__":
    main()
    
#1.3######################
# import os

# # ── Tiktoken cache (must be set before any tiktoken / langchain import) ───────
# _tiktoken_cache_dir = os.path.abspath("./token")
# os.makedirs(_tiktoken_cache_dir, exist_ok=True)
# os.environ["TIKTOKEN_CACHE_DIR"] = _tiktoken_cache_dir

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import re
# from datetime import datetime, timedelta
# import random
# import math
# import httpx
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")

# load_dotenv()

# # ─────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="RouteIQ — Logistics Itinerary Planner",
#     page_icon="🗺️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────────────────────────
# # GLOBAL STYLES
# # ─────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

# :root {
#   --midnight: #0D1117;
#   --obsidian: #161B22;
#   --charcoal: #21262D;
#   --steel: #30363D;
#   --ink: #8B949E;
#   --silver: #C9D1D9;
#   --white: #F0F6FC;
#   --gold: #D4A843;
#   --amber: #E8873A;
#   --teal: #2EA4A4;
#   --crimson: #C0392B;
#   --sage: #3FB950;
#   --violet: #8957E5;
#   --blue: #1F6FEB;
# }

# html, body, [class*="css"] {
#   font-family: 'DM Sans', sans-serif !important;
#   background-color: var(--midnight) !important;
#   color: var(--silver) !important;
# }
# section[data-testid="stSidebar"] {
#   background: var(--obsidian) !important;
#   border-right: 1px solid var(--steel) !important;
# }
# section[data-testid="stSidebar"] .stRadio label {
#   color: var(--silver) !important;
#   font-size: 0.9rem !important;
# }
# h1, h2, h3 {
#   font-family: 'Playfair Display', serif !important;
#   color: var(--white) !important;
# }
# [data-testid="metric-container"] {
#   background: var(--obsidian) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 12px !important;
#   padding: 16px !important;
# }
# [data-testid="metric-container"] label {
#   color: var(--ink) !important;
#   font-size: 0.75rem !important;
#   letter-spacing: 0.08em !important;
#   text-transform: uppercase !important;
# }
# [data-testid="metric-container"] [data-testid="stMetricValue"] {
#   color: var(--gold) !important;
#   font-family: 'Playfair Display', serif !important;
#   font-size: 2rem !important;
# }
# details {
#   background: var(--obsidian) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 8px !important;
# }
# details summary { color: var(--teal) !important; font-weight: 500 !important; }
# [data-testid="stDataFrame"] { border: 1px solid var(--steel) !important; border-radius: 8px !important; }
# textarea, input[type="text"] {
#   background: var(--charcoal) !important;
#   color: var(--white) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 8px !important;
# }
# .stButton > button {
#   background: linear-gradient(135deg, var(--gold), var(--amber)) !important;
#   color: var(--midnight) !important;
#   font-weight: 600 !important;
#   border: none !important;
#   border-radius: 8px !important;
#   letter-spacing: 0.04em !important;
#   transition: opacity 0.2s !important;
# }
# .stButton > button:hover { opacity: 0.85 !important; }
# [data-testid="stSelectbox"] select, .stSelectbox > div {
#   background: var(--charcoal) !important;
#   color: var(--white) !important;
#   border-color: var(--steel) !important;
# }
# .stTabs [data-baseweb="tab"] { color: var(--ink) !important; border-bottom: 2px solid transparent !important; }
# .stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

# .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; }
# .badge-delivery  { background: rgba(212,168,67,0.2);  color: #D4A843; border: 1px solid rgba(212,168,67,0.4); }
# .badge-meeting   { background: rgba(46,164,164,0.2);  color: #2EA4A4; border: 1px solid rgba(46,164,164,0.4); }
# .badge-pickup    { background: rgba(63,185,80,0.2);   color: #3FB950; border: 1px solid rgba(63,185,80,0.4); }
# .badge-warehouse { background: rgba(137,87,229,0.2);  color: #8957E5; border: 1px solid rgba(137,87,229,0.4); }
# .badge-customs   { background: rgba(192,57,43,0.2);   color: #E74C3C; border: 1px solid rgba(192,57,43,0.4); }
# .badge-rest      { background: rgba(139,148,158,0.2); color: #8B949E; border: 1px solid rgba(139,148,158,0.4); }

# .card { background: var(--obsidian); border: 1px solid var(--steel); border-radius: 12px; padding: 20px; margin-bottom: 12px; }
# .card-gold  { border-left: 4px solid var(--gold); }
# .card-teal  { border-left: 4px solid var(--teal); }
# .card-red   { border-left: 4px solid var(--crimson); }
# .card-green { border-left: 4px solid var(--sage); }
# .card-blue  { border-left: 4px solid var(--blue); }

# .section-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: var(--white); margin-bottom: 4px; }
# .section-sub   { color: var(--ink); font-size: 0.85rem; margin-bottom: 20px; letter-spacing: 0.04em; }
# .hero-title    { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 900; color: var(--white); line-height: 1.1; }
# .hero-accent   { color: var(--gold); }

# .stop-card { background: var(--charcoal); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# .stop-num  { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; background: var(--gold); color: var(--midnight); border-radius: 50%; font-weight: 700; font-size: 0.85rem; margin-right: 10px; flex-shrink: 0; }
# .stop-row  { display: flex; align-items: flex-start; }
# .stop-detail { font-size: 0.82rem; color: var(--ink); margin-top: 4px; }
# .connector-line { width: 2px; height: 30px; background: linear-gradient(var(--gold), var(--teal)); margin: 0 auto 0 13px; }

# .summary-box { background: linear-gradient(135deg, rgba(212,168,67,0.08), rgba(46,164,164,0.08)); border: 1px solid rgba(212,168,67,0.3); border-radius: 12px; padding: 20px; margin: 12px 0; }
# .itinerary-header { background: linear-gradient(135deg, rgba(31,111,235,0.15), rgba(46,164,164,0.15)); border: 1px solid rgba(31,111,235,0.3); border-radius: 12px; padding: 18px 22px; margin-bottom: 18px; }
# .violation-critical { background: rgba(192,57,43,0.1); border-left: 4px solid #E74C3C; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
# .violation-warning  { background: rgba(212,168,67,0.1); border-left: 4px solid #D4A843; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
# .weather-card { background: var(--charcoal); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 16px; text-align: center; }
# .fuel-card    { background: var(--obsidian); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# .fuel-save    { background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.35); border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; }
# .fuel-warn    { background: rgba(192,57,43,0.08); border: 1px solid rgba(192,57,43,0.3); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# # SESSION STATE
# # ─────────────────────────────────────────────
# def init_session():
#     defaults = {
#         "generated_itinerary": None,
#         "batch_itineraries":   [],
#         "cluster_labels":      None,
#         "cluster_X2d":         None,
#         "cluster_df":          None,
#         "assistant_history":   [],
#         "plan_history":        [],
#         "last_constraints":    {},
#         "nl_parsed_result":    None,
#         "weather_cache":       {},      # key: "lat_lon" → weather dict
#         "fuel_analysis":       None,    # last fuel/savings analysis result
#         "itinerary_source":    None,    # "dataset" | "custom" | "nl" — which tab owns it
#         "nl_stops_for_map":    [],      # lat/lon list for NL itinerary map
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v


# # ─────────────────────────────────────────────
# # DATA LOADER
# # ─────────────────────────────────────────────
# class DataLoader:
#     STOPS_FILE  = "stops.csv"
#     ROUTES_FILE = "routes.csv"
#     KB_FILE     = "logistics_kb.csv"

#     def load_stops(self):
#         if not os.path.exists(self.STOPS_FILE):
#             self._generate_and_save()
#         return pd.read_csv(self.STOPS_FILE, parse_dates=["time_window_start", "time_window_end"])

#     def load_routes(self):
#         if not os.path.exists(self.ROUTES_FILE):
#             self._generate_and_save()
#         return pd.read_csv(self.ROUTES_FILE)

#     def load_kb(self):
#         if not os.path.exists(self.KB_FILE):
#             self._generate_and_save()
#         return pd.read_csv(self.KB_FILE)

#     def _generate_and_save(self):
#         from generate_data import generate_stops, generate_routes, generate_kb
#         stops  = generate_stops(60)
#         routes = generate_routes(stops)
#         kb     = generate_kb()
#         stops.to_csv(self.STOPS_FILE,  index=False)
#         routes.to_csv(self.ROUTES_FILE, index=False)
#         kb.to_csv(self.KB_FILE,         index=False)

#     def get_summary_stats(self, stops_df, routes_df):
#         if stops_df.empty:
#             return {}
#         return {
#             "total_stops":         len(stops_df),
#             "total_routes":        stops_df["route_id"].nunique() if "route_id" in stops_df.columns else 0,
#             "total_distance_km":   round(routes_df["distance_km"].sum(), 1) if not routes_df.empty else 0,
#             "avg_stops_per_route": round(stops_df.groupby("route_id").size().mean(), 1) if "route_id" in stops_df.columns else 0,
#             "on_time_pct":         round(100 * (stops_df["status"] == "On Time").mean(), 1) if "status" in stops_df.columns else 0,
#             "high_priority":       int((stops_df["priority"] == "High").sum()) if "priority" in stops_df.columns else 0,
#         }


# # ─────────────────────────────────────────────
# # ML ENGINE
# # ─────────────────────────────────────────────
# class MLEngine:
#     def __init__(self):
#         self.vectorizer     = None
#         self.kmeans         = None
#         self.feature_matrix = None

#     def cluster_stops(self, stops_df, n_clusters=4):
#         texts = (
#             stops_df["stop_type"].fillna("") + " " +
#             stops_df["location_name"].fillna("") + " " +
#             stops_df["notes"].fillna("")
#         )
#         self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
#         X = self.vectorizer.fit_transform(texts)
#         self.feature_matrix = X
#         self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         labels = self.kmeans.fit_predict(X)
#         pca    = PCA(n_components=2, random_state=42)
#         X_2d   = pca.fit_transform(X.toarray())
#         return labels, X_2d

#     def get_cluster_keywords(self, top_n=5):
#         if self.vectorizer is None or self.kmeans is None:
#             return {}
#         terms    = self.vectorizer.get_feature_names_out()
#         keywords = {}
#         for i, center in enumerate(self.kmeans.cluster_centers_):
#             top_idx     = center.argsort()[-top_n:][::-1]
#             keywords[i] = [terms[j] for j in top_idx]
#         return keywords

#     def nearest_neighbor_route(self, coords):
#         if len(coords) <= 1:
#             return list(range(len(coords)))
#         unvisited = list(range(1, len(coords)))
#         route     = [0]
#         while unvisited:
#             curr    = route[-1]
#             nearest = min(unvisited, key=lambda j: self._haversine(coords[curr], coords[j]))
#             route.append(nearest)
#             unvisited.remove(nearest)
#         return route

#     @staticmethod
#     def _haversine(c1, c2):
#         lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
#         lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
#         a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
#         return 6371 * 2 * math.asin(math.sqrt(a))

#     @staticmethod
#     def compute_distance_km(lat1, lon1, lat2, lon2):
#         return MLEngine._haversine((lat1, lon1), (lat2, lon2))

#     @staticmethod
#     def osrm_route(coords: list) -> dict:
#         """
#         Real road routing via public OSRM demo API.
#         Falls back to Haversine (35 km/h) if OSRM is unreachable.

#         Args:
#             coords: list of (lat, lon) tuples in stop order

#         Returns:
#             {legs, total_distance_km, total_duration_min, source}
#         """
#         if len(coords) < 2:
#             return {"legs": [], "total_distance_km": 0.0,
#                     "total_duration_min": 0.0, "source": "osrm"}

#         waypoints = ";".join(f"{lon},{lat}" for lat, lon in coords)
#         url = (
#             "http://router.project-osrm.org/route/v1/driving/" + waypoints
#             + "?overview=false&steps=false&annotations=false"
#         )
#         try:
#             resp = httpx.get(url, timeout=8.0)
#             data = resp.json()
#             if data.get("code") != "Ok":
#                 raise ValueError(f"OSRM code: {data.get('code')}")
#             legs = [
#                 {
#                     "distance_km":  round(leg["distance"] / 1000, 2),
#                     "duration_min": round(leg["duration"] / 60, 1),
#                 }
#                 for leg in data["routes"][0]["legs"]
#             ]
#             return {
#                 "legs":               legs,
#                 "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
#                 "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
#                 "source":             "osrm",
#             }
#         except Exception:
#             legs = []
#             for i in range(len(coords) - 1):
#                 d = MLEngine._haversine(coords[i], coords[i + 1])
#                 legs.append({"distance_km": round(d, 2), "duration_min": round(d / 35 * 60, 1)})
#             return {
#                 "legs":               legs,
#                 "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
#                 "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
#                 "source":             "haversine_fallback",
#             }


# # ─────────────────────────────────────────────
# # WEATHER ENGINE  (Open-Meteo — no API key)
# # ─────────────────────────────────────────────
# class WeatherEngine:
#     """
#     Arrival-time-aware weather forecasting via Open-Meteo hourly API (free, no key).

#     For each stop we fetch the HOURLY forecast for that location, then pick the
#     hour slot that matches the stop's expected arrival_time.  This means:
#       - Stop A arriving at 09:00 → forecast for 09:00
#       - Stop B arriving at 13:00 → forecast for 13:00
#       - etc.

#     Caching strategy:
#       - Full hourly payload cached by lat/lon key (one API call per unique location).
#       - Individual hour lookups extracted from the cached payload — no repeat calls.

#     Fallback chain (guaranteed to always return usable data):
#       1. Open-Meteo HTTPS (verify=False for internal proxies)
#       2. Open-Meteo HTTP (plain, bypasses TLS entirely)
#       3. Seasonal estimate (hardcoded India climate by month + latitude band)
#     """

#     WMO_CODES = {
#         0:  ("Clear sky",            "☀️"),
#         1:  ("Mainly clear",         "🌤️"),
#         2:  ("Partly cloudy",        "⛅"),
#         3:  ("Overcast",             "☁️"),
#         45: ("Fog",                  "🌫️"),
#         48: ("Icy fog",              "🌫️"),
#         51: ("Light drizzle",        "🌦️"),
#         53: ("Moderate drizzle",     "🌦️"),
#         55: ("Dense drizzle",        "🌧️"),
#         61: ("Slight rain",          "🌧️"),
#         63: ("Moderate rain",        "🌧️"),
#         65: ("Heavy rain",           "🌧️"),
#         71: ("Slight snow",          "🌨️"),
#         73: ("Moderate snow",        "❄️"),
#         75: ("Heavy snow",           "❄️"),
#         80: ("Slight showers",       "🌦️"),
#         81: ("Moderate showers",     "🌧️"),
#         82: ("Heavy showers",        "⛈️"),
#         95: ("Thunderstorm",         "⛈️"),
#         96: ("Thunderstorm + hail",  "⛈️"),
#         99: ("Severe thunderstorm",  "⛈️"),
#     }
#     ADVERSE_CODES = {63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99}

#     # ── Seasonal fallback (India, month × lat band) ───────────────────────────
#     _SEASONAL_FALLBACK = {
#         1:  {"north": (18, 60, "Mainly clear",  "🌤️", 1),  "south": (28, 65, "Partly cloudy", "⛅", 2)},
#         2:  {"north": (20, 58, "Mainly clear",  "🌤️", 1),  "south": (30, 62, "Mainly clear",  "🌤️", 1)},
#         3:  {"north": (27, 55, "Clear sky",     "☀️", 0),  "south": (33, 60, "Clear sky",     "☀️", 0)},
#         4:  {"north": (33, 50, "Clear sky",     "☀️", 0),  "south": (35, 65, "Clear sky",     "☀️", 0)},
#         5:  {"north": (37, 45, "Clear sky",     "☀️", 0),  "south": (35, 70, "Partly cloudy", "⛅", 2)},
#         6:  {"north": (34, 75, "Moderate rain", "🌧️", 63), "south": (30, 85, "Heavy rain",    "🌧️", 65)},
#         7:  {"north": (30, 82, "Heavy rain",    "🌧️", 65), "south": (28, 88, "Heavy showers", "⛈️", 82)},
#         8:  {"north": (30, 80, "Moderate rain", "🌧️", 63), "south": (28, 86, "Heavy rain",    "🌧️", 65)},
#         9:  {"north": (29, 78, "Slight rain",   "🌦️", 61), "south": (29, 82, "Moderate rain", "🌧️", 63)},
#         10: {"north": (26, 65, "Partly cloudy", "⛅", 2),  "south": (29, 72, "Partly cloudy", "⛅", 2)},
#         11: {"north": (21, 58, "Mainly clear",  "🌤️", 1),  "south": (28, 68, "Mainly clear",  "🌤️", 1)},
#         12: {"north": (16, 62, "Mainly clear",  "🌤️", 1),  "south": (27, 65, "Partly cloudy", "⛅", 2)},
#     }

#     @staticmethod
#     def _cache_key(lat: float, lon: float) -> str:
#         return f"{round(lat, 3)}_{round(lon, 3)}"

#     @classmethod
#     def _seasonal_estimate(cls, lat: float, lon: float, arrival_hour: int = None) -> dict:
#         """Plausible weather estimate based on season, location, and time of day."""
#         import random as _rnd
#         month = datetime.now().month
#         band  = "north" if lat > 20 else "south"
#         row   = cls._SEASONAL_FALLBACK.get(month, cls._SEASONAL_FALLBACK[6])
#         t, h, desc, icon, code = row[band]

#         seed = int(abs(lat * 1000 + lon * 100)) % 100
#         # Diurnal temperature variation: cooler at dawn/late evening
#         hour_offset = 0
#         if arrival_hour is not None:
#             if arrival_hour < 7:    hour_offset = -4
#             elif arrival_hour < 10: hour_offset = -2
#             elif arrival_hour < 14: hour_offset =  2
#             elif arrival_hour < 17: hour_offset =  3
#             elif arrival_hour < 20: hour_offset =  1

#         t    = round(t + hour_offset + (_rnd.Random(seed).random() - 0.5) * 2, 1)
#         wind = round(8  + _rnd.Random(seed + 1).random() * 12, 1)
#         prec = round(_rnd.Random(seed + 2).random() * (5 if code >= 61 else 0.2), 1)
#         vis  = round(6  + _rnd.Random(seed + 3).random() * 4, 1) if code >= 45 else round(9 + _rnd.Random(seed + 3).random() * 5, 1)
#         return {
#             "temperature_c":    t,
#             "wind_speed_kmh":   wind,
#             "precipitation_mm": prec,
#             "humidity_pct":     h,
#             "visibility_km":    vis,
#             "weather_code":     code,
#             "description":      desc,
#             "icon":             icon,
#             "is_adverse":       code in cls.ADVERSE_CODES,
#             "source":           "seasonal-estimate",
#             "forecast_hour":    arrival_hour,
#         }

#     # ── Core: fetch full 48-h hourly payload for a location ──────────────────
#     @classmethod
#     def _fetch_hourly_payload(cls, lat: float, lon: float) -> dict | None:
#         """
#         Fetch 48-hour hourly forecast from Open-Meteo.
#         Returns the raw API response dict, or None on failure.
#         Payload cached in session state under key "hourly_payload_{lat}_{lon}".
#         """
#         cache_key = f"hourly_payload_{cls._cache_key(lat, lon)}"
#         cache     = st.session_state.get("weather_cache", {})
#         if cache_key in cache:
#             return cache[cache_key]

#         params = (
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,"
#             f"precipitation,weather_code,wind_speed_10m,visibility"
#             f"&wind_speed_unit=kmh"
#             f"&timezone=Asia%2FKolkata"
#             f"&forecast_days=2"          # today + tomorrow — covers any same-day route
#         )
#         urls = [
#             "https://api.open-meteo.com/v1/forecast" + params,
#             "http://api.open-meteo.com/v1/forecast"  + params,
#         ]
#         for url in urls:
#             try:
#                 resp = httpx.get(url, timeout=8.0, verify=False, follow_redirects=True)
#                 if resp.status_code != 200:
#                     continue
#                 data = resp.json()
#                 if "hourly" not in data or "time" not in data.get("hourly", {}):
#                     continue
#                 cache[cache_key] = data
#                 st.session_state["weather_cache"] = cache
#                 return data
#             except Exception:
#                 continue
#         return None   # both URLs failed

#     # ── Extract one hour slot from the payload ────────────────────────────────
#     @classmethod
#     def _extract_hour(cls, payload: dict, target_dt: datetime) -> dict:
#         """
#         Given a full hourly payload, extract the slot closest to target_dt.
#         Open-Meteo returns ISO strings like "2025-06-15T09:00" in the "time" array.
#         """
#         hourly     = payload["hourly"]
#         time_strs  = hourly["time"]           # list of "YYYY-MM-DDTHH:00"
#         target_str = target_dt.strftime("%Y-%m-%dT%H:00")

#         # Find exact match first, then nearest
#         idx = None
#         if target_str in time_strs:
#             idx = time_strs.index(target_str)
#         else:
#             # Find closest hour
#             best_diff = float("inf")
#             for i, ts in enumerate(time_strs):
#                 try:
#                     dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
#                     diff = abs((dt - target_dt).total_seconds())
#                     if diff < best_diff:
#                         best_diff = diff
#                         idx = i
#                 except Exception:
#                     continue

#         if idx is None:
#             return {}

#         def _val(key, default=0):
#             arr = hourly.get(key, [])
#             v   = arr[idx] if idx < len(arr) else default
#             return v if v is not None else default

#         code       = int(_val("weather_code", 0))
#         desc, icon = cls.WMO_CODES.get(code, ("Partly cloudy", "⛅"))
#         vis_raw    = _val("visibility", 10000)

#         return {
#             "temperature_c":        round(float(_val("temperature_2m",          25)), 1),
#             "wind_speed_kmh":       round(float(_val("wind_speed_10m",          10)), 1),
#             "precipitation_mm":     round(float(_val("precipitation",             0)), 1),
#             "precip_probability":   int(_val("precipitation_probability",          0)),
#             "humidity_pct":         int(_val("relative_humidity_2m",             65)),
#             "visibility_km":        round(float(vis_raw) / 1000, 1),
#             "weather_code":         code,
#             "description":          desc,
#             "icon":                 icon,
#             "is_adverse":           code in cls.ADVERSE_CODES,
#             "source":               "open-meteo-hourly",
#         }

#     # ── Public API: fetch weather at a specific arrival time ─────────────────
#     @classmethod
#     def fetch_at_time(cls, lat: float, lon: float, arrival_time_str: str,
#                       itin_date_str: str = None) -> dict:
#         """
#         Return weather forecast for (lat, lon) at the stop's arrival time.

#         Args:
#             lat, lon:           Stop coordinates.
#             arrival_time_str:   "HH:MM" string from the itinerary stop.
#             itin_date_str:      "YYYY-MM-DD" from itin["date"]; defaults to today.

#         Returns a weather dict with an extra "forecast_time" field showing
#         the exact datetime we fetched for.
#         """
#         # Parse target datetime
#         try:
#             date_str = itin_date_str or datetime.now().strftime("%Y-%m-%d")
#             target   = datetime.strptime(f"{date_str} {arrival_time_str}", "%Y-%m-%d %H:%M")
#         except Exception:
#             target   = datetime.now()

#         # Per-stop cache key (includes the target hour)
#         stop_key  = f"stop_{cls._cache_key(lat, lon)}_{target.strftime('%Y%m%d%H')}"
#         cache     = st.session_state.get("weather_cache", {})
#         if stop_key in cache:
#             return cache[stop_key]

#         # Fetch (or reuse) the hourly payload for this location
#         payload = cls._fetch_hourly_payload(lat, lon)

#         if payload:
#             result = cls._extract_hour(payload, target)
#             if result:
#                 result["forecast_time"]   = target.strftime("%d %b %Y · %H:%M")
#                 result["arrival_time_str"] = arrival_time_str
#                 cache[stop_key] = result
#                 st.session_state["weather_cache"] = cache
#                 return result

#         # Fallback to seasonal estimate
#         result = cls._seasonal_estimate(lat, lon, arrival_hour=target.hour)
#         result["forecast_time"]   = target.strftime("%d %b %Y · %H:%M") + " (est.)"
#         result["arrival_time_str"] = arrival_time_str
#         cache[stop_key] = result
#         st.session_state["weather_cache"] = cache
#         return result

#     # ── Convenience: fetch for all stops in an itinerary ─────────────────────
#     @classmethod
#     def fetch_route_at_times(cls, stops_with_coords: list, itin_date_str: str = None) -> list:
#         """
#         Fetch arrival-time-aware forecast for every stop.

#         Args:
#             stops_with_coords: list of dicts with at minimum:
#                 {stop_id, location_name, lat, lon, arrival_time (HH:MM)}
#             itin_date_str: itinerary date "YYYY-MM-DD"

#         Returns: same list with an added "weather" key per stop.
#         """
#         results = []
#         for s in stops_with_coords:
#             arrival = s.get("arrival_time", "08:00") or "08:00"
#             w = cls.fetch_at_time(s["lat"], s["lon"], arrival, itin_date_str)
#             results.append({**s, "weather": w})
#         return results

#     # ── Legacy: fetch current weather (kept for backward compat) ─────────────
#     @classmethod
#     def fetch(cls, lat: float, lon: float) -> dict:
#         """Fetch current conditions (no arrival time). Used as generic fallback."""
#         return cls.fetch_at_time(lat, lon, datetime.now().strftime("%H:%M"))

#     @classmethod
#     def fetch_route(cls, stops_with_coords: list) -> list:
#         """Legacy: fetch current conditions for a list of stops."""
#         return cls.fetch_route_at_times(stops_with_coords)


# # ─────────────────────────────────────────────
# # FUEL ENGINE  (static reference prices — India)
# # ─────────────────────────────────────────────
# class FuelEngine:
#     """
#     Static Indian petrol/diesel reference prices (as of June 2025).
#     Vehicle efficiency lookup table.
#     Fuel savings comparisons between optimized vs un-optimized route order.

#     Reference: https://www.goodreturns.in/petrol-price.html
#     Prices in ₹ per litre. Update CITY_PRICES dict as needed.
#     """

#     # ── Static city-level prices (₹/litre) ───────────────────────────────────
#     CITY_PRICES = {
#         # Maharashtra
#         "Mumbai":     {"petrol": 103.44, "diesel": 89.97},
#         "Pune":       {"petrol": 103.57, "diesel": 90.10},
#         "Nashik":     {"petrol": 103.20, "diesel": 89.75},
#         "Nagpur":     {"petrol": 103.80, "diesel": 90.15},
#         "Aurangabad": {"petrol": 103.35, "diesel": 89.85},
#         # Delhi NCR
#         "Delhi":      {"petrol": 94.72,  "diesel": 87.62},
#         "Gurgaon":    {"petrol": 95.10,  "diesel": 88.05},
#         "Noida":      {"petrol": 94.85,  "diesel": 87.80},
#         # Karnataka
#         "Bengaluru":  {"petrol": 102.86, "diesel": 88.94},
#         "Mysuru":     {"petrol": 102.60, "diesel": 88.70},
#         # Tamil Nadu
#         "Chennai":    {"petrol": 100.75, "diesel": 92.34},
#         "Coimbatore": {"petrol": 100.50, "diesel": 92.10},
#         # Telangana / AP
#         "Hyderabad":  {"petrol": 107.41, "diesel": 95.65},
#         "Visakhapatnam": {"petrol": 106.80, "diesel": 95.20},
#         # Gujarat
#         "Ahmedabad":  {"petrol": 96.63,  "diesel": 92.38},
#         "Surat":      {"petrol": 96.50,  "diesel": 92.25},
#         # Rajasthan
#         "Jaipur":     {"petrol": 104.88, "diesel": 90.36},
#         # Uttar Pradesh
#         "Lucknow":    {"petrol": 94.65,  "diesel": 87.76},
#         "Kanpur":     {"petrol": 94.55,  "diesel": 87.65},
#         # West Bengal
#         "Kolkata":    {"petrol": 103.94, "diesel": 90.76},
#         # Default (national average)
#         "default":    {"petrol": 101.50, "diesel": 90.00},
#     }

#     # ── Vehicle fuel efficiency (km per litre) ────────────────────────────────
#     VEHICLE_EFFICIENCY = {
#         "Truck":       5.5,   # heavy truck (10-16T)
#         "Van":         12.0,  # light commercial van
#         "Tempo":       9.0,   # mini-truck / tempo
#         "Car":         15.0,  # passenger car
#         "Motorcycle":  40.0,  # two-wheeler
#         "default":     8.0,
#     }

#     # ── Fuel type by vehicle ──────────────────────────────────────────────────
#     VEHICLE_FUEL_TYPE = {
#         "Truck": "diesel", "Tempo": "diesel",
#         "Van": "diesel", "Car": "petrol", "Motorcycle": "petrol",
#     }

#     @classmethod
#     def get_price(cls, city: str, fuel_type: str = "diesel") -> float:
#         """Find the closest city match (case-insensitive substring)."""
#         city_lower = city.lower()
#         for name, prices in cls.CITY_PRICES.items():
#             if name.lower() in city_lower or city_lower in name.lower():
#                 return prices.get(fuel_type, prices["diesel"])
#         return cls.CITY_PRICES["default"].get(fuel_type, 90.0)

#     @classmethod
#     def compute_fuel_cost(
#         cls,
#         distance_km: float,
#         vehicle_type: str,
#         city: str = "Mumbai",
#         override_price: float = None,
#         override_efficiency: float = None,
#     ) -> dict:
#         """
#         Compute fuel cost for a given distance.

#         Returns:
#             {litres_consumed, price_per_litre, total_cost_inr,
#              efficiency_kmpl, fuel_type, city}
#         """
#         fuel_type  = cls.VEHICLE_FUEL_TYPE.get(vehicle_type, "diesel")
#         efficiency = override_efficiency or cls.VEHICLE_EFFICIENCY.get(vehicle_type, cls.VEHICLE_EFFICIENCY["default"])
#         price      = override_price      or cls.get_price(city, fuel_type)
#         litres     = distance_km / efficiency if efficiency > 0 else 0
#         cost       = round(litres * price, 2)
#         return {
#             "litres_consumed":  round(litres, 2),
#             "price_per_litre":  price,
#             "total_cost_inr":   cost,
#             "efficiency_kmpl":  efficiency,
#             "fuel_type":        fuel_type,
#             "city":             city,
#         }

#     @classmethod
#     def savings_analysis(
#         cls,
#         stops_list: list,
#         vehicle_type: str,
#         city: str = "Mumbai",
#         override_price: float = None,
#         override_efficiency: float = None,
#     ) -> dict:
#         """
#         Compare un-optimized (original stop order) vs NN-optimized order.

#         Returns:
#             {
#                 original_km, optimized_km, saved_km, saving_pct,
#                 original_cost, optimized_cost, saved_cost_inr,
#                 original_order, optimized_order,
#                 fuel_detail_original, fuel_detail_optimized,
#                 per_stop_savings: [{stop, original_leg_km, optimized_leg_km}]
#             }
#         """
#         if len(stops_list) < 2:
#             return {}

#         coords = [(s["lat"], s["lon"]) for s in stops_list]

#         # Original order — OSRM
#         osrm_orig = MLEngine.osrm_route(coords)
#         orig_km   = osrm_orig["total_distance_km"]

#         # NN-optimized order — OSRM
#         nn_order  = MLEngine().__class__().nearest_neighbor_route.__func__(MLEngine(), coords)
#         coords_nn = [coords[i] for i in nn_order]
#         osrm_opt  = MLEngine.osrm_route(coords_nn)
#         opt_km    = osrm_opt["total_distance_km"]

#         # Use optimized if it's actually better, else keep original
#         if opt_km >= orig_km:
#             opt_km    = orig_km
#             osrm_opt  = osrm_orig
#             nn_order  = list(range(len(stops_list)))

#         saved_km   = round(orig_km - opt_km, 2)
#         saving_pct = round(saved_km / orig_km * 100, 1) if orig_km > 0 else 0

#         fuel_orig = cls.compute_fuel_cost(orig_km, vehicle_type, city, override_price, override_efficiency)
#         fuel_opt  = cls.compute_fuel_cost(opt_km,  vehicle_type, city, override_price, override_efficiency)
#         saved_cost = round(fuel_orig["total_cost_inr"] - fuel_opt["total_cost_inr"], 2)

#         # Per-leg comparison (zip original vs optimized legs)
#         per_stop = []
#         orig_legs = osrm_orig.get("legs", [])
#         opt_legs  = osrm_opt.get("legs", [])
#         for i, s in enumerate(stops_list[1:]):
#             orig_leg = orig_legs[i]["distance_km"] if i < len(orig_legs) else 0
#             opt_s    = stops_list[nn_order[i+1]] if (i+1) < len(nn_order) else s
#             opt_leg  = opt_legs[i]["distance_km"] if i < len(opt_legs) else 0
#             per_stop.append({
#                 "leg":           i + 1,
#                 "original_stop": s["location_name"],
#                 "optimized_stop":opt_s["location_name"],
#                 "original_km":   orig_leg,
#                 "optimized_km":  opt_leg,
#                 "saved_km":      round(orig_leg - opt_leg, 2),
#             })

#         return {
#             "original_km":          orig_km,
#             "optimized_km":         opt_km,
#             "saved_km":             saved_km,
#             "saving_pct":           saving_pct,
#             "original_cost_inr":    fuel_orig["total_cost_inr"],
#             "optimized_cost_inr":   fuel_opt["total_cost_inr"],
#             "saved_cost_inr":       saved_cost,
#             "fuel_detail_original": fuel_orig,
#             "fuel_detail_optimized":fuel_opt,
#             "original_order":       [s["location_name"] for s in stops_list],
#             "optimized_order":      [stops_list[i]["location_name"] for i in nn_order],
#             "per_stop_savings":     per_stop,
#             "routing_source":       osrm_orig["source"],
#         }


# # ─────────────────────────────────────────────
# # RAG ENGINE
# # ─────────────────────────────────────────────
# class RAGEngine:
#     DOMAIN_TOPICS = [
#         # ── Core logistics ───────────────────────────────────────────────────
#         "route", "routing", "delivery", "logistics", "shipment", "freight",
#         "itinerary", "stop", "waypoint", "depot", "warehouse", "pickup",
#         "dispatch", "fleet", "vehicle", "driver", "trucking", "transport",
#         "cargo", "customs", "e-way bill", "manifest", "consignment",
#         "last mile", "first mile", "supply chain", "distribution",
#         "fuel", "mileage", "navigation", "gps", "tracking",
#         "time window", "schedule", "delay", "on-time", "eta", "arrival",
#         "temperature", "cold chain", "refrigerated", "hazmat", "dangerous goods",
#         "pod", "proof of delivery", "invoice", "bill of lading",
#         "kpi", "performance", "efficiency", "cost", "optimization",
#         "travel", "distance", "trip", "journey", "road", "highway",
#         "rail", "air freight", "sea freight", "port", "airport",
#         "tms", "wms", "erp", "telematics", "iot",
#         "breakdown", "insurance", "claim", "compliance", "regulation",
#         # ── Natural navigation / direction language ───────────────────────────
#         # Verbs people use when asking about getting from A to B
#         "go from", "going from", "get from", "getting from",
#         "travel from", "travelling from", "traveling from",
#         "reach", "reaching", "how to reach", "how do i reach",
#         "drive from", "driving from", "ride from", "riding from",
#         "commute", "commuting",
#         "best way", "fastest way", "shortest way", "quickest way",
#         "how long", "how far", "how much time",
#         "directions", "direction", "navigate", "path from", "path to",
#         "way to", "way from", "route from", "route to",
#         "from here", "to here",
#         # ── Relational / between ─────────────────────────────────────────────
#         "between",
#         # ── Movement / transit words ─────────────────────────────────────────
#         "bus", "train", "metro", "cab", "auto", "taxi", "uber", "ola",
#         "toll", "highway", "expressway", "flyover", "bridge",
#         "traffic", "congestion", "jam", "detour", "bypass",
#         "drop", "pick up", "pickup point", "drop off",
#         # ── Indian city / area names (common logistics hubs) ─────────────────
#         # Mumbai
#         "mumbai", "bombay", "bandra", "kurla", "andheri", "dadar",
#         "thane", "navi mumbai", "panvel", "borivali", "kandivali",
#         "malad", "goregaon", "jogeshwari", "vile parle", "santacruz",
#         "bkc", "nariman", "churchgate", "csmt", "colaba", "worli",
#         "lower parel", "prabhadevi", "matunga", "sion", "chembur",
#         "ghatkopar", "vikhroli", "kanjurmarg", "bhandup", "mulund",
#         "dombivli", "kalyan", "bhiwandi", "vasai", "virar", "mira road",
#         "nhava sheva", "jnpt", "nhava",
#         # Pune
#         "pune", "pimpri", "chinchwad", "hadapsar", "kothrud", "hinjewadi",
#         "wakad", "baner", "aundh", "shivajinagar", "talegaon",
#         # Other major cities
#         "delhi", "ncr", "gurgaon", "noida", "faridabad", "ghaziabad",
#         "bengaluru", "bangalore", "whitefield", "electronic city",
#         "hyderabad", "secunderabad", "cyberabad",
#         "chennai", "kolkata", "ahmedabad", "surat", "jaipur",
#         "lucknow", "chandigarh", "coimbatore", "kochi", "indore",
#         "nagpur", "nashik", "aurangabad", "visakhapatnam",
#         # ── Logistics / geographic terms ─────────────────────────────────────
#         "zone", "area", "sector", "block", "lane", "street", "nagar",
#         "colony", "society", "industrial area", "industrial estate",
#         "cargo hub", "logistics park", "cold storage", "godown",
#         "weighbridge", "octroi", "rto", "check post", "border",
#     ]
#     RELEVANCE_THRESHOLD = 0.30

#     # Regex patterns that strongly indicate a route / navigation question
#     # regardless of specific keywords — catches "from X to Y" style queries
#     _NAV_PATTERNS = [
#         r"\bfrom\b.{1,60}\bto\b",        # "from bandra to kurla"
#         r"\bgo\b.{0,40}\bto\b",           # "go to andheri"
#         r"\bget\b.{0,40}\bto\b",          # "get to the depot"
#         r"\bread?ch\b",                      # "reach" / "reaching"
#         r"\bdriv(e|ing)\b.{0,40}\bto\b",  # "drive to"
#         r"\bhow\b.{0,30}\blong\b",        # "how long does it take"
#         r"\bhow\b.{0,30}\bfar\b",         # "how far is X from Y"
#         r"\bdir?ections?\b",                 # "directions" / "direction"
#         r"\bnear(est)?\b",                   # "nearest depot"
#         r"\bway\b.{0,30}\bto\b",          # "best way to reach"
#         r"\broute\b.{0,30}\bfrom\b",      # "route from X"
#         r"\bpath\b.{0,30}\bto\b",         # "path to warehouse"
#     ]

#     def __init__(self, llm, embeddings):
#         self.llm       = llm
#         self.embeddings = embeddings
#         self._vectordb  = None

#     def _build_vectordb(self, kb_df):
#         if self._vectordb is not None or self.embeddings is None:
#             return
#         raw_docs, metadatas = [], []
#         for _, row in kb_df.iterrows():
#             raw_docs.append(f"[{row['category']} — {row['topic']}]\n{row['content']}")
#             metadatas.append({"category": row["category"], "topic": row["topic"]})
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
#         )
#         chunks, chunk_metas = [], []
#         for doc, meta in zip(raw_docs, metadatas):
#             parts = splitter.split_text(doc)
#             chunks.extend(parts)
#             chunk_metas.extend([meta] * len(parts))
#         try:
#             self._vectordb = Chroma.from_texts(
#                 chunks, self.embeddings, metadatas=chunk_metas, persist_directory="./chroma_kb"
#             )
#         except Exception:
#             self._vectordb = None

#     def _retrieve(self, question, k=5):
#         if self._vectordb is None:
#             return [], [], []
#         try:
#             results = self._vectordb.similarity_search_with_relevance_scores(question, k=k)
#             docs, scores, metas = [], [], []
#             for doc, score in results:
#                 docs.append(doc.page_content)
#                 scores.append(score)
#                 metas.append(doc.metadata)
#             return docs, scores, metas
#         except Exception:
#             return [], [], []

#     def _is_in_domain(self, question, chunks, scores):
#         """
#         Three-layer domain check — passes if ANY layer matches.

#         Layer 1 — Keyword scan: checks against expanded DOMAIN_TOPICS list.
#                   Includes natural language navigation phrases, city names,
#                   and movement verbs so "go from bandra to kurla" passes.

#         Layer 2 — Regex navigation intent: pattern-matches spatial/directional
#                   phrasing like "from X to Y", "how far", "nearest", "directions".
#                   Catches questions that contain no logistics jargon but are
#                   clearly asking about routes or locations.

#         Layer 3 — Semantic similarity: at least one retrieved KB chunk must
#                   score >= RELEVANCE_THRESHOLD (only active when embeddings online).
#         """
#         import re
#         q = question.lower()

#         # Layer 1: keyword scan
#         if any(kw in q for kw in self.DOMAIN_TOPICS):
#             return True

#         # Layer 2: navigation intent regex
#         for pattern in self._NAV_PATTERNS:
#             if re.search(pattern, q):
#                 return True

#         # Layer 3: semantic similarity
#         if scores and max(scores) >= self.RELEVANCE_THRESHOLD:
#             return True

#         return False

#     def _keyword_fallback(self, question, kb_df):
#         q = question.lower()
#         for _, row in kb_df.iterrows():
#             words = row["topic"].lower().split() + row["category"].lower().split()
#             if any(w in q for w in words):
#                 return f"**📖 {row['topic']}** *(from {row['category']} KB)*\n\n{row['content']}"
#         return (
#             "I couldn't find a matching article. "
#             "Please ask about routing, delivery, customs, fleet, or other logistics topics."
#         )

#     def answer(self, question, kb_df):
#         if not kb_df.empty and self.embeddings is not None and self._vectordb is None:
#             with st.spinner("📚 Indexing knowledge base…"):
#                 self._build_vectordb(kb_df)

#         chunks, scores, metas = self._retrieve(question, k=5)

#         if not self._is_in_domain(question, chunks, scores):
#             return {
#                 "text": (
#                     "⛔ **Out of scope** — I'm RouteIQ Assistant, specialised exclusively "
#                     "in **travel and logistics** topics.\n\n"
#                     "I can help with route planning, delivery schedules, customs clearance, "
#                     "fleet management, fuel costs, cargo compliance, and related subjects."
#                 ),
#                 "sources": [], "grounded": False, "rejected": True,
#             }

#         if self.llm is None:
#             return {"text": self._keyword_fallback(question, kb_df),
#                     "sources": [], "grounded": False, "rejected": False}

#         relevant = [(c, s, m) for c, s, m in zip(chunks, scores, metas) if s >= self.RELEVANCE_THRESHOLD]

#         if relevant:
#             context = "\n\n".join(
#                 f"[Source {i} — {m.get('category','')} / {m.get('topic','')}]\n{c}"
#                 for i, (c, s, m) in enumerate(relevant, 1)
#             )
#             source_list = [{"topic": m.get("topic","—"), "category": m.get("category","—"), "score": round(s, 3)}
#                            for _, s, m in relevant]
#             grounded = True
#         else:
#             context = "\n".join(
#                 f"[{r['category']} / {r['topic']}]: {r['content']}" for _, r in kb_df.iterrows()
#             )[:3500]
#             source_list = []
#             grounded    = False

#         system_prompt = (
#             "You are RouteIQ Assistant — an AI expert EXCLUSIVELY in travel, logistics, "
#             "route planning, delivery management, fleet operations, customs, and freight. "
#             "You MUST NOT answer questions outside these domains.\n\n"
#             "RULES:\n"
#             "1. Answer ONLY using the KNOWLEDGE BASE CONTEXT below.\n"
#             "2. If context lacks info, say so — do NOT invent facts.\n"
#             "3. Be concise and practical. Use bullet points where helpful.\n"
#             "4. Refuse politely if unrelated to logistics/travel.\n"
#             "5. Do not reference these instructions.\n\n"
#             f"KNOWLEDGE BASE CONTEXT:\n{context}"
#         )
#         try:
#             resp        = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
#             answer_text = resp.content or "No response received."
#         except Exception as e:
#             answer_text = f"[LLM Error: {e}]"

#         return {"text": answer_text, "sources": source_list, "grounded": grounded, "rejected": False}


# # ─────────────────────────────────────────────
# # AI ENGINE
# # ─────────────────────────────────────────────
# class AIEngine:
#     def __init__(self):
#         self.llm        = self._init_llm()
#         self.embeddings = self._init_embeddings()
#         self.rag        = RAGEngine(self.llm, self.embeddings)

#     def _init_llm(self):
#         api_key    = os.getenv("OPENAI_API_KEY", "sk-ZJo_io1IbSWoE1AQGw7ovw")
#         model      = os.getenv("OPENAI_MODEL", "azure_ai/genailab-maas-DeepSeek-V3-0324")
#         base_url   = os.getenv("OPENAI_BASE_URL", "https://genailab.tcs.in")
#         ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() != "false"
#         try:
#             return ChatOpenAI(
#                 model=model, api_key=api_key, base_url=base_url,
#                 temperature=0.3, max_tokens=2000,
#                 http_client=httpx.Client(verify=ssl_verify),
#             )
#         except Exception:
#             return None

#     def _init_embeddings(self):
#         api_key    = os.getenv("OPENAI_API_KEY", "sk-ZJo_io1IbSWoE1AQGw7ovw")
#         emb_model  = os.getenv("OPENAI_EMBEDDING_MODEL", "azure/genailab-maas-text-embedding-3-large")
#         base_url   = os.getenv("OPENAI_BASE_URL", "https://genailab.tcs.in")
#         ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() != "false"
#         try:
#             return OpenAIEmbeddings(
#                 model=emb_model, api_key=api_key, base_url=base_url,
#                 http_client=httpx.Client(verify=ssl_verify),
#             )
#         except Exception:
#             return None

#     def _call(self, system_prompt, user_prompt):
#         if not self.llm:
#             return None
#         try:
#             resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
#             return resp.content
#         except Exception as e:
#             return f"[LLM Error: {e}]"

#     def _parse_json(self, text, fallback):
#         if text is None:
#             return fallback
#         text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
#         try:
#             return json.loads(text)
#         except Exception:
#             return fallback

#     # ── A: Natural Language Stop Parser ──────────────────────────────────────
#     # ── Coordinate regex patterns ────────────────────────────────────────────
#     # Matches: (19.018, 72.848)  |  19.018,72.848  |  19.018 72.848
#     # Captures decimal lat/lon values typical for India (lat 8–37, lon 68–98)
#     _COORD_RE = re.compile(
#         r"""
#         \(?                              # optional opening paren
#         (?P<lat>                         # latitude group
#             [+-]?                        # optional sign
#             (?:3[0-7]|[12]\d|\d)         # 1-37 (India range, allows 8-37)
#             \.\d{2,8}                    # decimal part (at least 2 digits)
#         )
#         [\s,]+                           # separator: comma and/or space
#         (?P<lon>                         # longitude group
#             [+-]?
#             (?:9[0-8]|[7-8]\d|6[8-9])   # 68-98 (India lon range)
#             \.\d{2,8}
#         )
#         \)?                              # optional closing paren
#         """,
#         re.VERBOSE,
#     )

#     @classmethod
#     def _extract_inline_coords(cls, text: str) -> dict:
#         """
#         Pre-scan the input text for explicit coordinate pairs.
#         Returns a dict mapping approximate position in text → (lat, lon)
#         so the LLM result can be validated/corrected.

#         Supports formats:
#           - Location Name (19.018, 72.848)
#           - Location Name at 19.018,72.848
#           - coordinates: 19.018 72.848
#           - 19.018, 72.848 — stop name follows
#         """
#         coords_found = {}
#         for m in cls._COORD_RE.finditer(text):
#             try:
#                 lat = float(m.group("lat"))
#                 lon = float(m.group("lon"))
#                 # Sanity check: valid India bounding box
#                 if 6.0 <= lat <= 38.0 and 66.0 <= lon <= 100.0:
#                     coords_found[m.start()] = (lat, lon)
#             except ValueError:
#                 pass
#         return coords_found

#     def parse_natural_language_stops(self, text: str) -> dict:
#         """
#         Parse a free-text itinerary description into structured stops + constraints.
#         Supports explicit coordinates inline with stop names.

#         Coordinate formats recognised:
#             "Dadar Warehouse (19.018, 72.848) at 9am"
#             "deliver to 19.118, 72.847 by 11am"
#             "pickup from Kurla 19.072,72.879 urgent"

#         Returns: {stops, constraints, parse_notes, error}
#         """
#         # Pre-extract any explicit coordinates from the text
#         inline_coords = self._extract_inline_coords(text)
#         has_inline    = len(inline_coords) > 0

#         system_prompt = (
#             "You are a logistics data extraction agent. "
#             "Extract structured stop and constraint information from a natural language description. "
#             "The user may provide explicit GPS coordinates inline with stop names in formats like: "
#             "(19.018, 72.848), 19.018,72.848, or 19.018 72.848. "
#             "ALWAYS use those exact coordinates for that stop — never replace user-provided coordinates. "
#             "For stops without explicit coordinates, use realistic Indian GPS coordinates. "
#             "Return ONLY valid JSON — no markdown, no extra text."
#         )

#         coord_note = (
#             f"\n\nNOTE: This description contains {len(inline_coords)} explicit coordinate pair(s). "
#             "Extract and use them exactly as given for the corresponding stops."
#             if has_inline else
#             "\n\nNOTE: No explicit coordinates found — infer realistic Indian GPS coordinates from location names."
#         )

#         user_prompt = (
#             f'Extract all stops and constraints from:\n\n"{text}"{coord_note}\n\n'
#             "Return this JSON schema exactly:\n"
#             "{\n"
#             '  "stops": [\n'
#             "    {\n"
#             '      "stop_id": "NL1",\n'
#             '      "location_name": "location name (without coordinates)",\n'
#             '      "lat": <use provided coordinate if given, else infer for India>,\n'
#             '      "lon": <use provided coordinate if given, else infer for India>,\n'
#             '      "coordinates_source": "user_provided" or "inferred",\n'
#             '      "stop_type": "Delivery|Pickup|Meeting|Warehouse|Customs|Rest",\n'
#             '      "time_window_start": "HH:MM",\n'
#             '      "time_window_end": "HH:MM",\n'
#             '      "priority": "High|Medium|Low",\n'
#             '      "notes": "any special instructions"\n'
#             "    }\n"
#             "  ],\n"
#             '  "constraints": {\n'
#             '    "driver_name": "string or Unknown",\n'
#             '    "start_time": "HH:MM",\n'
#             '    "transport_mode": "Road|Rail|Air|Sea",\n'
#             '    "vehicle_type": "Truck|Van|Motorcycle|Car|Tempo",\n'
#             '    "max_hours": <number>,\n'
#             '    "vehicle_capacity_kg": <number>\n'
#             "  },\n"
#             '  "parse_notes": "brief summary — mention if coordinates were user-provided or inferred",\n'
#             '  "error": null\n'
#             "}\n\n"
#             "Coordinate extraction rules:\n"
#             "- If the user writes 'Location Name (lat, lon)' or 'Location Name lat,lon', "
#             "extract lat/lon EXACTLY as given and set coordinates_source to 'user_provided'\n"
#             "- If only a location name is given, infer realistic GPS coordinates for that "
#             "Indian city/area and set coordinates_source to 'inferred'\n"
#             "- Strip coordinates from location_name — it should contain only the place name\n"
#             "- Mixed input is fine: some stops can have user-provided coords, others inferred\n"
#             "- Validate: Indian lat range 6–37, lon range 66–100. If numbers look wrong, infer instead\n\n"
#             "Other inference rules:\n"
#             "- pick up/collect → Pickup; deliver/drop → Delivery; meeting/client → Meeting; "
#             "rest/break → Rest; customs/checkpoint → Customs; warehouse/depot → Warehouse\n"
#             "- urgent/asap/critical → High priority; 'if possible'/low → Low; default → Medium\n"
#             "- 'by 2pm' → time_window_end 14:00; 'at 9am' → time_window_start 09:00; "
#             "'between 11am–1pm' → start 11:00, end 13:00\n"
#             "- Default start_time 08:00, transport_mode Road, vehicle_type Van if not mentioned\n"
#             "- stop_id values: NL1, NL2, NL3… in order of appearance"
#         )

#         raw    = self._call(system_prompt, user_prompt)
#         result = self._parse_json(raw, {"stops": [], "constraints": {}, "parse_notes": "",
#                                         "error": "LLM offline or parse failed"})

#         if "stops"       not in result: result["stops"]       = []
#         if "constraints" not in result: result["constraints"] = {}
#         if "error"       not in result: result["error"]       = None

#         # ── Post-parse: validate LLM coordinates against inline_coords ────────
#         # If the user provided explicit coordinates and the LLM returned something
#         # different, override with the user's exact values (sequential matching).
#         if has_inline and result["stops"]:
#             coord_list = list(inline_coords.values())  # ordered by position in text
#             for i, stop in enumerate(result["stops"]):
#                 src = stop.get("coordinates_source", "inferred")
#                 if src == "user_provided" and i < len(coord_list):
#                     # Double-check: override with the regex-extracted value
#                     # in case the LLM slightly misread the numbers
#                     result["stops"][i]["lat"] = coord_list[i][0]
#                     result["stops"][i]["lon"] = coord_list[i][1]

#         return result

#     # ── B: Itinerary Generation ───────────────────────────────────────────────
#     def generate_itinerary(self, stops_list, constraints, route_context):
#         """Generate optimized itinerary JSON, then patch in real OSRM distances."""
#         system_prompt = (
#             "You are an expert logistics route planner. "
#             "Generate an optimized, feasible delivery itinerary as structured JSON. "
#             "Prioritize: (1) time window compliance, (2) High-priority stops first, "
#             "(3) shortest total distance. "
#             "Times in HH:MM 24-hour format. Durations in minutes. "
#             "Return ONLY valid JSON — no markdown."
#         )
#         user_prompt = (
#             f"CONSTRAINTS:\n{json.dumps(constraints, indent=2)}\n\n"
#             f"ROUTE CONTEXT (use these distances/times):\n{route_context}\n\n"
#             f"STOPS TO PLAN ({len(stops_list)}):\n{json.dumps(stops_list, indent=2, default=str)}\n\n"
#             "Return JSON with keys: itinerary_title, driver, vehicle, date (YYYY-MM-DD), "
#             "transport_mode, total_distance_km, total_duration_min, estimated_fuel_cost_inr, "
#             "optimization_notes, warnings (list), efficiency_score (0-100), "
#             "on_time_probability (0-100), and stops array where each stop has: "
#             "sequence, stop_id, location_name, arrival_time (HH:MM), departure_time (HH:MM), "
#             "service_duration_min, travel_time_from_prev_min, distance_from_prev_km, "
#             "stop_type, priority, status, notes, risk_flag (None/Low/Medium/High), "
#             "risk_reason, time_window_start (HH:MM), time_window_end (HH:MM)."
#         )
#         raw      = self._call(system_prompt, user_prompt)
#         result   = self._parse_json(raw, self._rule_based_itinerary(stops_list, constraints))

#         # ── Patch real OSRM leg data ─────────────────────────────────────────
#         coords     = [(s["lat"], s["lon"]) for s in stops_list]
#         osrm       = MLEngine.osrm_route(coords)
#         result["routing_source"] = osrm["source"]
#         itin_stops = result.get("stops", [])

#         if osrm["source"] == "osrm" and len(osrm["legs"]) >= len(itin_stops) - 1 and len(itin_stops) > 1:
#             for i, stop in enumerate(itin_stops):
#                 if i == 0:
#                     continue
#                 leg_idx = i - 1
#                 if leg_idx < len(osrm["legs"]):
#                     stop["distance_from_prev_km"]     = osrm["legs"][leg_idx]["distance_km"]
#                     stop["travel_time_from_prev_min"]  = osrm["legs"][leg_idx]["duration_min"]
#             result["total_distance_km"]  = osrm["total_distance_km"]
#             result["total_duration_min"] = round(
#                 osrm["total_duration_min"]
#                 + sum(s.get("service_duration_min", 0) for s in itin_stops), 1
#             )
#             result["estimated_fuel_cost_inr"] = round(osrm["total_distance_km"] * 8, 0)

#         return result

#     def _rule_based_itinerary(self, stops_list, constraints):
#         priority_order = {"High": 0, "Medium": 1, "Low": 2}
#         sorted_stops   = sorted(stops_list, key=lambda s: (
#             priority_order.get(s.get("priority", "Low"), 2),
#             s.get("time_window_start", "23:59"),
#         ))
#         current_time = datetime.strptime(constraints.get("start_time", "08:00"), "%H:%M")
#         result_stops, total_dist = [], 0
#         prev_lat = stops_list[0]["lat"] if stops_list else 19.076
#         prev_lon = stops_list[0]["lon"] if stops_list else 72.877

#         for i, s in enumerate(sorted_stops):
#             dist        = MLEngine.compute_distance_km(prev_lat, prev_lon, s["lat"], s["lon"])
#             travel_min  = max(5, int(dist / 35 * 60))
#             service_min = {"Delivery": 20, "Pickup": 15, "Meeting": 45,
#                            "Warehouse": 30, "Customs": 60, "Rest": 20}.get(s.get("stop_type", "Delivery"), 20)
#             arrival     = current_time + timedelta(minutes=travel_min)
#             departure   = arrival + timedelta(minutes=service_min)
#             result_stops.append({
#                 "sequence": i + 1, "stop_id": s.get("stop_id", f"S{i+1}"),
#                 "location_name": s.get("location_name", "Stop"),
#                 "arrival_time": arrival.strftime("%H:%M"),
#                 "departure_time": departure.strftime("%H:%M"),
#                 "service_duration_min": service_min,
#                 "travel_time_from_prev_min": travel_min,
#                 "distance_from_prev_km": round(dist, 2),
#                 "stop_type": s.get("stop_type", "Delivery"),
#                 "priority": s.get("priority", "Medium"),
#                 "status": "Scheduled", "notes": s.get("notes", ""),
#                 "risk_flag": "None", "risk_reason": "",
#                 "time_window_start": s.get("time_window_start", "08:00"),
#                 "time_window_end":   s.get("time_window_end",   "18:00"),
#             })
#             total_dist += dist
#             current_time = departure
#             prev_lat, prev_lon = s["lat"], s["lon"]

#         return {
#             "itinerary_title": "Optimized Route (Rule-based fallback)",
#             "driver": constraints.get("driver_name", "Driver"),
#             "vehicle": constraints.get("vehicle_type", "Truck"),
#             "date": datetime.now().strftime("%Y-%m-%d"),
#             "transport_mode": constraints.get("transport_mode", "Road"),
#             "total_distance_km": round(total_dist, 2),
#             "total_duration_min": sum(
#                 s["travel_time_from_prev_min"] + s["service_duration_min"] for s in result_stops
#             ),
#             "estimated_fuel_cost_inr": round(total_dist * 8, 0),
#             "stops": result_stops,
#             "optimization_notes": "Rule-based fallback: sorted by priority then time window.",
#             "warnings": ["LLM offline — using rule-based optimizer"],
#             "efficiency_score": 72, "on_time_probability": 78,
#         }

#     def adjust_itinerary(self, existing_itinerary, change_request):
#         system_prompt = (
#             "You are a logistics re-planning agent. "
#             "Return a FULLY updated itinerary JSON in the same schema. "
#             "Return ONLY valid JSON — no markdown."
#         )
#         user_prompt = (
#             f"EXISTING ITINERARY:\n{json.dumps(existing_itinerary, indent=2, default=str)}\n\n"
#             f"CHANGE REQUEST:\n{change_request}\n\n"
#             "Adjust all affected stop times. Explain the change in optimization_notes."
#         )
#         return self._parse_json(self._call(system_prompt, user_prompt), existing_itinerary)

#     # ── C: Constraint Violation Checker ──────────────────────────────────────
#     def check_violations(self, itinerary: dict, constraints: dict) -> list:
#         """
#         Check all stops for time-window breaches and driver hour overruns.
#         Returns a list of violation dicts: {sequence, location_name, type, severity, detail}
#         """
#         violations = []
#         stops      = sorted(itinerary.get("stops", []), key=lambda s: s["sequence"])
#         if not stops:
#             return violations

#         date_str  = itinerary.get("date", datetime.now().strftime("%Y-%m-%d"))
#         try:
#             base = datetime.strptime(date_str, "%Y-%m-%d")
#         except Exception:
#             base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

#         def to_dt(hhmm):
#             try:
#                 h, m = map(int, str(hhmm).strip().split(":"))
#                 return base.replace(hour=h, minute=m, second=0, microsecond=0)
#             except Exception:
#                 return base

#         max_hours = constraints.get("max_hours", 10)
#         day_end   = to_dt(constraints.get("start_time", "08:00")) + timedelta(hours=max_hours)

#         for s in stops:
#             arr          = to_dt(s.get("arrival_time",   "00:00"))
#             dep          = to_dt(s.get("departure_time",  "00:00"))
#             tw_start_raw = s.get("time_window_start", "")
#             tw_end_raw   = s.get("time_window_end",   "")

#             if tw_start_raw and tw_end_raw:
#                 tw_start = to_dt(tw_start_raw)
#                 tw_end   = to_dt(tw_end_raw)
#                 if arr < tw_start:
#                     wait = int((tw_start - arr).seconds / 60)
#                     violations.append({
#                         "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                         "type": "time_window", "severity": "Warning",
#                         "detail": (
#                             f"Arrives at {s.get('arrival_time')} but window opens at "
#                             f"{tw_start_raw}. Driver waits {wait} min."
#                         ),
#                     })
#                 elif arr > tw_end:
#                     late = int((arr - tw_end).seconds / 60)
#                     violations.append({
#                         "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                         "type": "time_window", "severity": "Critical",
#                         "detail": (
#                             f"Arrives at {s.get('arrival_time')} — window closed at "
#                             f"{tw_end_raw}. Late by {late} min. SLA breach."
#                         ),
#                     })

#             if dep > day_end:
#                 over = int((dep - day_end).seconds / 60)
#                 violations.append({
#                     "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                     "type": "driver_hours", "severity": "Critical",
#                     "detail": (
#                         f"Departure at {s.get('departure_time')} exceeds "
#                         f"{max_hours}h limit by {over} min."
#                     ),
#                 })

#         return violations

#     def get_kb_answer(self, question, kb_df):
#         return self.rag.answer(question, kb_df)

#     def analyze_route_performance(self, route_df, stops_df):
#         summary = {
#             "total_routes":      route_df["route_id"].nunique() if not route_df.empty else 0,
#             "avg_distance_km":   round(route_df["distance_km"].mean(), 1) if not route_df.empty else 0,
#             "avg_on_time_pct":   round(100 * (stops_df["status"] == "On Time").mean(), 1) if not stops_df.empty else 0,
#             "top_delay_reasons": stops_df["delay_reason"].value_counts().head(3).to_dict()
#                                   if "delay_reason" in stops_df.columns else {},
#         }
#         raw = self._call(
#             "You are a logistics analytics expert. Provide 3-5 concise actionable insights.",
#             f"Performance data:\n{json.dumps(summary, indent=2)}\n\nBullet point insights:"
#         )
#         if not raw or raw.startswith("[LLM"):
#             return (
#                 "• Review high-delay routes for recurring traffic patterns\n"
#                 "• Adjust time windows for stops that are consistently late\n"
#                 "• Prioritize High-priority stops in morning slots\n"
#                 "• Consolidate nearby stops to reduce total distance and fuel cost"
#             )
#         return raw


# # ─────────────────────────────────────────────
# # DASHBOARD
# # ─────────────────────────────────────────────
# class Dashboard:
#     STOP_COLORS = {
#         "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
#         "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
#     }
#     PRIORITY_COLORS = {"High": "#E74C3C", "Medium": "#D4A843", "Low": "#3FB950"}
#     STATUS_COLORS   = {
#         "On Time": "#3FB950", "Delayed": "#E74C3C",
#         "Scheduled": "#2EA4A4", "Cancelled": "#8B949E",
#     }

#     def _base_layout(self, height=320):
#         return dict(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=height,
#             margin=dict(t=20, b=40, l=50, r=20),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )

#     def volume_timeline(self, stops_df):
#         df2 = stops_df.copy()
#         df2["date"] = pd.to_datetime(df2["scheduled_date"])
#         daily = df2.groupby(["date","stop_type"]).size().reset_index(name="count")
#         fig   = px.bar(daily, x="date", y="count", color="stop_type",
#                        color_discrete_map=self.STOP_COLORS, barmode="stack")
#         fig.update_layout(**self._base_layout(280),
#                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"))
#         return fig

#     def stop_type_donut(self, stops_df):
#         counts = stops_df["stop_type"].value_counts().reset_index()
#         counts.columns = ["stop_type","count"]
#         fig = px.pie(counts, names="stop_type", values="count",
#                      color="stop_type", color_discrete_map=self.STOP_COLORS, hole=0.55)
#         fig.update_traces(textposition="inside", textinfo="percent+label",
#                           marker=dict(line=dict(color="#0D1117", width=2)))
#         fig.update_layout(**self._base_layout(300), showlegend=False)
#         return fig

#     def distance_by_route(self, routes_df):
#         df = routes_df.sort_values("total_distance_km", ascending=False).head(10)
#         fig = go.Figure(go.Bar(x=df["route_id"], y=df["total_distance_km"],
#                                marker_color="#D4A843",
#                                hovertemplate="%{x}: %{y:.1f} km<extra></extra>"))
#         fig.update_layout(**self._base_layout(280),
#                           xaxis=dict(showgrid=False, tickangle=-20),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="km"))
#         return fig

#     def on_time_gauge(self, pct):
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number", value=pct,
#             number={"suffix": "%", "font": {"color": "#D4A843", "size": 36}},
#             gauge={
#                 "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
#                 "bar":  {"color": "#D4A843"}, "bgcolor": "#21262D",
#                 "steps": [
#                     {"range": [0,  60],  "color": "rgba(192,57,43,0.3)"},
#                     {"range": [60, 80],  "color": "rgba(232,135,58,0.3)"},
#                     {"range": [80, 100], "color": "rgba(63,185,80,0.3)"},
#                 ],
#                 "threshold": {"line": {"color": "#3FB950", "width": 3}, "value": 85},
#             },
#         ))
#         layout = self._base_layout(220)
#         layout["margin"] = dict(t=20, b=10, l=30, r=30)

#         fig.update_layout(**layout)
#         return fig

#     def priority_bar(self, stops_df):
#         pc = stops_df["priority"].value_counts().reset_index()
#         pc.columns = ["priority","count"]
#         fig = go.Figure(go.Bar(
#             x=pc["priority"], y=pc["count"],
#             marker_color=[self.PRIORITY_COLORS.get(p,"#8B949E") for p in pc["priority"]],
#             hovertemplate="%{x}: %{y}<extra></extra>",
#         ))
#         fig.update_layout(**self._base_layout(240),
#                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"))
#         return fig

#     def route_map_scatter(self, stops_list, itinerary=None):
#         lats   = [s["lat"] for s in stops_list]
#         lons   = [s["lon"] for s in stops_list]
#         names  = [s["location_name"] for s in stops_list]
#         colors = [self.STOP_COLORS.get(s.get("stop_type","Delivery"), "#D4A843") for s in stops_list]

#         fig = go.Figure()
#         fig.add_trace(go.Scattergeo(
#             lat=lats, lon=lons, mode="markers+text",
#             marker=dict(size=12, color=colors, line=dict(color="#0D1117", width=1)),
#             text=[f"{i+1}. {n}" for i, n in enumerate(names)],
#             textposition="top center",
#             textfont=dict(size=9, color="#C9D1D9"),
#             hovertemplate="<b>%{text}</b><extra></extra>", name="Stops",
#         ))
#         if itinerary and "stops" in itinerary:
#             seq  = sorted(itinerary["stops"], key=lambda s: s["sequence"])
#             rlat, rlon = [], []
#             for ist in seq:
#                 m = next((s for s in stops_list if s["stop_id"] == ist["stop_id"]), None)
#                 if m:
#                     rlat.append(m["lat"]); rlon.append(m["lon"])
#             if rlat:
#                 fig.add_trace(go.Scattergeo(lat=rlat, lon=rlon, mode="lines",
#                                              line=dict(width=2, color="#D4A843"), name="Route"))
#         fig.update_geos(
#             center=dict(lat=np.mean(lats), lon=np.mean(lons)), projection_scale=8,
#             showland=True, landcolor="#21262D", showocean=True, oceancolor="#161B22",
#             showcountries=True, countrycolor="#30363D", showcoastlines=True, coastlinecolor="#30363D",
#         )
#         layout = self._base_layout(420)
#         layout["margin"] = dict(t=10, b=10, l=10, r=10)
#         fig.update_layout(**layout, geo=dict(bgcolor="rgba(0,0,0,0)"), showlegend=True)
#         return fig

#     def cluster_scatter(self, stops_df, labels, X_2d):
#         df = stops_df.copy()
#         df["Cluster"] = [f"Cluster {l+1}" for l in labels]
#         df["x"] = X_2d[:, 0]; df["y"] = X_2d[:, 1]
#         fig = px.scatter(df, x="x", y="y", color="Cluster",
#                          hover_data=["location_name","stop_type","priority"],
#                          color_discrete_sequence=["#D4A843","#2EA4A4","#3FB950","#8957E5","#E74C3C","#1F6FEB"])
#         fig.update_layout(**self._base_layout(380),
#                           xaxis=dict(showgrid=False, title="Component 1"),
#                           yaxis=dict(showgrid=False, title="Component 2"))
#         return fig

#     def performance_timeline(self, routes_df):
#         df = routes_df.copy(); df["date"] = pd.to_datetime(df["route_date"])
#         fig = px.line(df.sort_values("date"), x="date", y="on_time_pct", color="transport_mode",
#                       color_discrete_map={"Road":"#D4A843","Air":"#2EA4A4","Rail":"#3FB950","Sea":"#8957E5"})
#         fig.update_layout(**self._base_layout(300), xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="On-Time %", range=[0,105]))
#         return fig

#     def fuel_cost_bar(self, routes_df):
#         df = routes_df.groupby("transport_mode")["estimated_fuel_cost_inr"].mean().reset_index()
#         fig = go.Figure(go.Bar(x=df["transport_mode"], y=df["estimated_fuel_cost_inr"],
#                                marker_color=["#D4A843","#2EA4A4","#3FB950","#8957E5"],
#                                hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
#         fig.update_layout(**self._base_layout(260), xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="Avg Cost (₹)"))
#         return fig


# # ─────────────────────────────────────────────
# # PAGE: OVERVIEW
# # ─────────────────────────────────────────────
# def page_overview(stops_df, routes_df, stats, dash):
#     st.markdown('<div class="hero-title">🗺️ Route<span class="hero-accent">IQ</span></div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">AI-Powered Logistics Itinerary Planning & Route Optimization</div>', unsafe_allow_html=True)
#     st.markdown("---")

#     if stops_df.empty:
#         st.warning("No data loaded. Please ensure stops.csv and routes.csv exist.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Stops",    stats.get("total_stops", 0))
#     c2.metric("Active Routes",  stats.get("total_routes", 0))
#     c3.metric("Total Distance", f"{stats.get('total_distance_km', 0):,} km")
#     c4.metric("On-Time Rate",   f"{stats.get('on_time_pct', 0)}%")

#     st.markdown("---")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.markdown("**Daily Stop Volume by Type**")
#         st.plotly_chart(dash.volume_timeline(stops_df), use_container_width=True)
#     with col2:
#         st.markdown("**Stop Type Distribution**")
#         st.plotly_chart(dash.stop_type_donut(stops_df), use_container_width=True)

#     col3, col4 = st.columns(2)
#     with col3:
#         st.markdown("**Top Routes by Distance**")
#         st.plotly_chart(dash.distance_by_route(routes_df), use_container_width=True)
#     with col4:
#         st.markdown("**Stop Priority Breakdown**")
#         st.plotly_chart(dash.priority_bar(stops_df), use_container_width=True)

#     st.markdown("---")
#     st.markdown("### 🚨 High-Priority Open Stops")
#     high = stops_df[(stops_df["priority"] == "High") & (stops_df["status"] != "Delivered")].head(5)
#     if high.empty:
#         st.info("No high-priority open stops.")
#     else:
#         for _, row in high.iterrows():
#             type_cls = {"Delivery":"card-gold","Meeting":"card-teal","Pickup":"card-green",
#                         "Warehouse":"card-blue","Customs":"card-red"}.get(row["stop_type"],"card-gold")
#             st.markdown(
#                 f'<div class="card {type_cls}">'
#                 f'<div style="display:flex;justify-content:space-between;align-items:center">'
#                 f'<div><b style="color:#F0F6FC">{row["location_name"]}</b>'
#                 f'<span class="badge badge-delivery" style="margin-left:8px">{row["stop_type"]}</span></div>'
#                 f'<div style="text-align:right;font-size:0.8rem;color:#8B949E">'
#                 f'Route: {row.get("route_id","—")} &nbsp;|&nbsp; {row.get("scheduled_date","—")}</div></div>'
#                 f'<div class="stop-detail">Window: {row.get("time_window_start","—")} – '
#                 f'{row.get("time_window_end","—")} &nbsp;·&nbsp; {row.get("notes","")}</div></div>',
#                 unsafe_allow_html=True,
#             )


# # ─────────────────────────────────────────────
# # PAGE: ITINERARY PLANNER  (3 tabs)
# # ─────────────────────────────────────────────
# def page_planner(stops_df, ai, ml, dash):
#     st.markdown('<div class="section-title">🔍 Itinerary Planner</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Generate optimized route plans — '
#         'from your dataset, custom coordinates, or plain English</div>',
#         unsafe_allow_html=True,
#     )

#     tab1, tab2, tab3 = st.tabs(["📋 Plan from Dataset", "✏️ Custom Coordinates", "💬 Natural Language"])

#     # ── TAB 1: Plan from Dataset ──────────────────────────────────────────────
#     with tab1:
#         if stops_df.empty:
#             st.warning("No stops data loaded.")
#         else:
#             routes         = stops_df["route_id"].unique().tolist()
#             selected_route = st.selectbox("Select Route to Plan", routes)
#             route_stops    = stops_df[stops_df["route_id"] == selected_route].copy()

#             st.markdown(f"**{len(route_stops)} stops on route {selected_route}**")
#             st.dataframe(
#                 route_stops[["stop_id","location_name","stop_type","priority",
#                               "time_window_start","time_window_end","notes"]].reset_index(drop=True),
#                 use_container_width=True,
#             )

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 driver_name = st.text_input("Driver Name", value="Ramesh Kumar")
#                 start_time  = st.text_input("Start Time (HH:MM)", value="08:00")
#             with col2:
#                 transport_mode   = st.selectbox("Transport Mode", ["Road","Rail","Air","Sea"])
#                 vehicle_type     = st.selectbox("Vehicle Type", ["Truck","Van","Motorcycle","Car","Tempo"])
#             with col3:
#                 max_hours        = st.slider("Max Working Hours", 4, 14, 8)
#                 vehicle_capacity = st.number_input("Vehicle Capacity (kg)", 100, 10000, 1000, step=100)

#             optimize = st.checkbox("Apply Nearest-Neighbor Optimization", value=True)

#             if st.button("🚀 Generate Itinerary", type="primary", key="gen_dataset"):
#                 coords = list(zip(route_stops["lat"], route_stops["lon"]))
#                 if optimize and len(coords) > 2:
#                     order       = ml.nearest_neighbor_route(coords)
#                     route_stops = route_stops.iloc[order].reset_index(drop=True)

#                 stops_list = [
#                     {
#                         "stop_id":           str(r["stop_id"]),
#                         "location_name":     r["location_name"],
#                         "lat":               float(r["lat"]),
#                         "lon":               float(r["lon"]),
#                         "stop_type":         r["stop_type"],
#                         "time_window_start": str(r["time_window_start"])[-8:-3] if pd.notnull(r["time_window_start"]) else "08:00",
#                         "time_window_end":   str(r["time_window_end"])[-8:-3]   if pd.notnull(r["time_window_end"])   else "18:00",
#                         "priority":          r["priority"],
#                         "notes":             str(r.get("notes","")) if pd.notnull(r.get("notes","")) else "",
#                     }
#                     for _, r in route_stops.iterrows()
#                 ]
#                 constraints = {
#                     "driver_name": driver_name, "start_time": start_time,
#                     "transport_mode": transport_mode, "vehicle_type": vehicle_type,
#                     "max_hours": max_hours, "vehicle_capacity_kg": vehicle_capacity,
#                 }

#                 with st.spinner("🛰️ Fetching real road distances via OSRM…"):
#                     osrm_preview = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])

#                 st.caption(
#                     "Routing: 🟢 OSRM (real roads)" if osrm_preview["source"] == "osrm"
#                     else "Routing: 🟡 Haversine fallback (OSRM unreachable)"
#                 )
#                 route_context = (
#                     f"Route {selected_route} | {len(stops_list)} stops | Mode: {transport_mode} | "
#                     f"Road distance: {osrm_preview['total_distance_km']} km | "
#                     f"Drive time: {osrm_preview['total_duration_min']} min | "
#                     f"Source: {osrm_preview['source']}"
#                 )
#                 with st.spinner("🧠 AI optimizing your route…"):
#                     itinerary = ai.generate_itinerary(stops_list, constraints, route_context)

#                 st.session_state["generated_itinerary"] = itinerary
#                 st.session_state["last_constraints"]    = constraints
#                 st.session_state["itinerary_source"]    = "dataset"
#                 st.session_state["fuel_analysis"]       = None  # reset stale savings
#                 st.session_state["plan_history"].append({
#                     "source": "Dataset", "route": selected_route,
#                     "generated_at": datetime.now().strftime("%H:%M:%S"),
#                     "stops": len(stops_list),
#                     "distance_km": itinerary.get("total_distance_km", 0),
#                     "routing": osrm_preview["source"],
#                 })
#                 st.success("✅ Itinerary generated!")

#     # ── TAB 2: Custom Coordinates ─────────────────────────────────────────────
#     with tab2:
#         st.markdown("Add custom stops with coordinates for ad-hoc planning.")
#         custom_text = st.text_area(
#             "One stop per line: Name, Lat, Lon, Type, Priority, TimeFrom, TimeTo",
#             height=160,
#             placeholder="Dadar Warehouse, 19.018, 72.848, Delivery, High, 09:00, 11:00",
#             key="custom_coords_input",
#         )
#         col1, col2 = st.columns(2)
#         with col1:
#             c_driver = st.text_input("Driver Name", value="Suresh Patil", key="c_driver")
#             c_start  = st.text_input("Start Time", value="08:30", key="c_start")
#         with col2:
#             c_mode  = st.selectbox("Transport Mode", ["Road","Rail","Air","Sea"], key="c_mode")
#             c_hours = st.slider("Max Hours", 4, 14, 8, key="c_hours")
#             c_cap   = st.number_input("Capacity (kg)", 100, 10000, 500, step=100, key="c_cap")

#         if st.button("🚀 Generate Custom Itinerary", key="gen_custom"):
#             custom_stops = []
#             for i, line in enumerate(custom_text.strip().split("\n")):
#                 parts = [p.strip() for p in line.split(",")]
#                 if len(parts) >= 3:
#                     try:
#                         custom_stops.append({
#                             "stop_id": f"CS{i+1}", "location_name": parts[0],
#                             "lat": float(parts[1]), "lon": float(parts[2]),
#                             "stop_type":         parts[3] if len(parts) > 3 else "Delivery",
#                             "priority":          parts[4] if len(parts) > 4 else "Medium",
#                             "time_window_start": parts[5] if len(parts) > 5 else "08:00",
#                             "time_window_end":   parts[6] if len(parts) > 6 else "18:00",
#                             "notes": "",
#                         })
#                     except ValueError:
#                         pass

#             if not custom_stops:
#                 st.error("No valid stops parsed. Check the format.")
#             else:
#                 constraints = {
#                     "driver_name": c_driver, "start_time": c_start,
#                     "transport_mode": c_mode, "max_hours": c_hours,
#                     "vehicle_type": "Van", "vehicle_capacity_kg": c_cap,
#                 }
#                 with st.spinner("🛰️ Fetching real road distances via OSRM…"):
#                     osrm_cs = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in custom_stops])
#                 st.caption(
#                     "Routing: 🟢 OSRM" if osrm_cs["source"] == "osrm"
#                     else "Routing: 🟡 Haversine fallback"
#                 )
#                 route_ctx = (
#                     f"{len(custom_stops)} custom stops | Mode: {c_mode} | "
#                     f"Road distance: {osrm_cs['total_distance_km']} km | "
#                     f"Drive time: {osrm_cs['total_duration_min']} min | Source: {osrm_cs['source']}"
#                 )
#                 with st.spinner("Planning custom route…"):
#                     itinerary = ai.generate_itinerary(custom_stops, constraints, route_ctx)
#                 st.session_state["generated_itinerary"] = itinerary
#                 st.session_state["last_constraints"]    = constraints
#                 st.session_state["itinerary_source"]    = "custom"
#                 st.session_state["fuel_analysis"]       = None
#                 st.session_state["plan_history"].append({
#                     "source": "Custom Coordinates", "route": "Custom",
#                     "generated_at": datetime.now().strftime("%H:%M:%S"),
#                     "stops": len(custom_stops),
#                     "distance_km": itinerary.get("total_distance_km", 0),
#                     "routing": osrm_cs["source"],
#                 })
#                 st.success("✅ Custom itinerary generated!")

#     # ── TAB 3: Natural Language ───────────────────────────────────────────────
#     with tab3:
#         # If the last itinerary was generated from a different tab, show a clear notice
#         # but do NOT wipe session state — that's what caused the dataset tab to break.
#         # Generating from this tab will replace it.
#         if st.session_state.get("itinerary_source") in ("dataset", "custom"):
#             st.info(
#                 "ℹ️ The itinerary below was generated from the **Dataset** or **Custom** tab. "
#                 "Describe your route and click **Generate Full Itinerary** to create a new one here."
#             )

#         st.markdown(
#             '<div class="summary-box">'
#             '<b style="color:#D4A843">💬 Describe your route in plain English.</b><br>'
#             '<span style="color:#8B949E;font-size:0.85rem">'
#             'Mention stops, times, priorities, and constraints naturally. '
#             'You can include GPS coordinates inline — e.g. '
#             '<code style="background:rgba(212,168,67,0.15);padding:1px 5px;border-radius:3px">'
#             'Dadar Warehouse (19.018, 72.848) at 9am</code> — '
#             'and the AI will use them exactly. Mix names and coordinates freely.'
#             '</span></div>',
#             unsafe_allow_html=True,
#         )

#         with st.expander("📖 Example prompts — click to expand"):
#             st.markdown("""
# **With location names only (AI infers coordinates):**
# > Plan a route for Ramesh. Start at 8am from Dadar Warehouse. Deliver to Andheri client by 10am (urgent), pick up cargo from Kurla depot 11am–1pm, client meeting in BKC at 2:30pm.

# **With explicit coordinates (paste from stops.csv or Google Maps):**
# > Pick up from Dadar Warehouse (19.018, 72.848) at 9am, deliver to Andheri Hub (19.119, 72.847) by 11am urgent, then client meeting at BKC (19.066, 72.868) at 2pm. Driver Suresh, Van.

# **Mixed — some stops with coordinates, some without:**
# > Start from Kalyan Junction (19.193, 73.102) at 8am. Deliver to Mulund Cold Storage by 10:30am. Customs checkpoint at Nhava Sheva (18.950, 72.945) before 2pm.

# **Just coordinates (no location names):**
# > 3 stops: pickup at 19.193,73.102 at 9am, delivery at 19.119,72.847 by 11am, meeting at 19.066,72.868 at 2pm.

# **Multi-constraint:**
# > Suresh needs to collect documents from Pune Hadapsar at 9am, customs meeting in Navi Mumbai by 1pm, then deliver to Mulund cold storage (19.172, 73.002) before 4pm. Refrigerated van, max 9 hours.
#             """)

#         nl_text = st.text_area(
#             "Describe your itinerary:",
#             height=150,
#             placeholder=(
#                 "Plan a route for Ramesh starting at 8am from Dadar Warehouse (19.018, 72.848). "
#                 "Deliver to Andheri client (19.119, 72.847) by 10am urgent, "
#                 "then meeting at BKC at 2pm...\n\n"
#                 "You can mix location names and coordinates freely."
#             ),
#             key="nl_input",
#         )

#         nl_col1, nl_col2 = st.columns(2)
#         with nl_col1:
#             nl_driver = st.text_input("Override driver name (optional)", value="", key="nl_driver")
#             nl_mode   = st.selectbox("Override transport mode",
#                                      ["(auto-detect)","Road","Rail","Air","Sea"], key="nl_mode")
#         with nl_col2:
#             nl_hours = st.slider("Max working hours", 4, 14, 9, key="nl_hours")
#             nl_cap   = st.number_input("Vehicle capacity (kg)", 100, 10000, 500, step=100, key="nl_cap")

#         if st.button("🔍 Parse Description", key="nl_parse"):
#             if not nl_text.strip():
#                 st.error("Please describe your route first.")
#             elif not ai.llm:
#                 st.error("LLM is offline. Natural language parsing requires a connected LLM.")
#             else:
#                 with st.spinner("🤖 Extracting stops from your description…"):
#                     parsed = ai.parse_natural_language_stops(nl_text)
#                 st.session_state["nl_parsed_result"] = parsed

#         parsed = st.session_state.get("nl_parsed_result")
#         if parsed:
#             if parsed.get("error") and not parsed.get("stops"):
#                 st.error(f"Parse failed: {parsed['error']}")
#             else:
#                 stops_list = parsed.get("stops", [])
#                 if parsed.get("parse_notes"):
#                     st.info(f"📝 AI understood: {parsed['parse_notes']}")

#                 if stops_list:
#                     # Count user-provided vs inferred coordinates
#                     n_provided = sum(1 for s in stops_list if s.get("coordinates_source") == "user_provided")
#                     n_inferred = len(stops_list) - n_provided

#                     st.markdown("#### ✅ Parsed Stops")

#                     # Coordinate source summary
#                     if n_provided > 0 and n_inferred > 0:
#                         st.markdown(
#                             f'<div style="font-size:0.8rem;color:#8B949E;margin-bottom:8px">'
#                             f'📍 <b style="color:#3FB950">{n_provided} stop(s)</b> with user-provided coordinates &nbsp;·&nbsp; '
#                             f'<b style="color:#D4A843">{n_inferred} stop(s)</b> with AI-inferred coordinates'
#                             f'</div>',
#                             unsafe_allow_html=True,
#                         )
#                     elif n_provided > 0:
#                         st.success(f"📍 All {n_provided} stop coordinates taken from your input.")
#                     else:
#                         st.info(f"📍 All {n_inferred} stop coordinates inferred by AI from location names.")

#                     st.dataframe(
#                         pd.DataFrame([{
#                             "#":           s["stop_id"],
#                             "Location":    s["location_name"],
#                             "Coordinates": f"{round(s.get('lat',0),5)}, {round(s.get('lon',0),5)}",
#                             "Source":      "📍 User" if s.get("coordinates_source") == "user_provided" else "🤖 Inferred",
#                             "Type":        s["stop_type"],
#                             "Priority":    s["priority"],
#                             "Window":      f"{s['time_window_start']} – {s['time_window_end']}",
#                             "Notes":       s.get("notes",""),
#                         } for s in stops_list]),
#                         use_container_width=True,
#                     )

#                     auto_c = parsed.get("constraints", {})
#                     if nl_driver.strip():          auto_c["driver_name"]      = nl_driver.strip()
#                     if nl_mode != "(auto-detect)": auto_c["transport_mode"]   = nl_mode
#                     auto_c["max_hours"]          = nl_hours
#                     auto_c["vehicle_capacity_kg"] = nl_cap

#                     cc1, cc2, cc3 = st.columns(3)
#                     cc1.markdown(f"**Driver:** {auto_c.get('driver_name','—')}")
#                     cc2.markdown(f"**Mode:** {auto_c.get('transport_mode','Road')}")
#                     cc3.markdown(f"**Start:** {auto_c.get('start_time','08:00')}")

#                     if st.button("🚀 Generate Full Itinerary", type="primary", key="nl_generate"):
#                         with st.spinner("🛰️ Fetching OSRM road distances…"):
#                             osrm_nl = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])
#                         st.caption(
#                             "Routing: 🟢 OSRM" if osrm_nl["source"] == "osrm"
#                             else "Routing: 🟡 Haversine fallback"
#                         )
#                         ctx_nl = (
#                             f"{len(stops_list)} NL-parsed stops | "
#                             f"Mode: {auto_c.get('transport_mode','Road')} | "
#                             f"Road distance: {osrm_nl['total_distance_km']} km | "
#                             f"Drive time: {osrm_nl['total_duration_min']} min | "
#                             f"Source: {osrm_nl['source']}"
#                         )
#                         with st.spinner("🧠 Generating optimized itinerary…"):
#                             itinerary = ai.generate_itinerary(stops_list, auto_c, ctx_nl)
#                         # Build stops_for_map — NL itinerary embeds lat/lon in its stops
#                         nl_map_stops = []
#                         for itin_s in itinerary.get("stops", []):
#                             matched = next((s for s in stops_list if s["stop_id"] == itin_s["stop_id"]), None)
#                             if matched:
#                                 nl_map_stops.append({
#                                     "stop_id":       itin_s["stop_id"],
#                                     "location_name": itin_s["location_name"],
#                                     "lat":           matched["lat"],
#                                     "lon":           matched["lon"],
#                                     "stop_type":     itin_s.get("stop_type","Delivery"),
#                                 })

#                         st.session_state["generated_itinerary"] = itinerary
#                         st.session_state["last_constraints"]    = auto_c
#                         st.session_state["nl_parsed_result"]    = None
#                         st.session_state["itinerary_source"]    = "nl"
#                         st.session_state["nl_stops_for_map"]    = nl_map_stops
#                         st.session_state["fuel_analysis"]       = None
#                         st.session_state["plan_history"].append({
#                             "source": "Natural Language", "route": "NL-parsed",
#                             "generated_at": datetime.now().strftime("%H:%M:%S"),
#                             "stops": len(stops_list),
#                             "distance_km": itinerary.get("total_distance_km", 0),
#                             "routing": osrm_nl["source"],
#                         })
#                         st.success("✅ Itinerary generated from your description!")
#                         st.rerun()
#                 else:
#                     st.warning("No stops extracted. Try adding specific location names and times.")

#     # ── Display Itinerary ─────────────────────────────────────────────────────
#     # Render the itinerary below the tabs.
#     # itinerary_source tells us which tab owns the current result.
#     # We ALWAYS show it here — the per-tab isolation works because:
#     #   - Generating from tab1/tab2 sets source = "dataset"/"custom"
#     #   - Generating from tab3 sets source = "nl"
#     #   - Switching tabs does NOT clear the result (no unconditional clearing)
#     #   - The user sees the last result they generated, regardless of which tab is active
#     itin   = st.session_state.get("generated_itinerary")
#     source = st.session_state.get("itinerary_source")
#     if itin and source:
#         st.markdown("---")
#         nl_map = st.session_state.get("nl_stops_for_map", []) if source == "nl" else []
#         _render_itinerary(
#             itin, stops_df, ai, dash,
#             st.session_state.get("last_constraints", {}),
#             nl_stops_override=nl_map,
#         )


# # ─────────────────────────────────────────────
# # WEATHER PANEL  (called from _render_itinerary)
# # ─────────────────────────────────────────────
# def _render_weather_panel(stops_with_coords: list, itin_date_str: str = None):
#     """
#     Renders arrival-time-aware weather forecast for every stop.

#     For each stop the forecast shows conditions at the stop's arrival_time,
#     not current conditions — e.g. Stop B arriving at 13:00 shows the 13:00
#     hourly forecast, not what the weather is right now.

#     stops_with_coords: list of dicts with {stop_id, location_name, lat, lon,
#                         arrival_time (HH:MM), stop_type, sequence}
#     itin_date_str:     "YYYY-MM-DD" — the itinerary date used to resolve forecasts
#     """
#     st.markdown("### 🌤️ Weather Forecast Along Route")
#     st.markdown(
#         '<div style="font-size:0.8rem;color:#8B949E;margin-bottom:16px">'
#         'Forecast shown at <b style="color:#D4A843">each stop&#39;s expected arrival time</b> '
#         '— not current conditions. Powered by Open-Meteo hourly API.'
#         '</div>',
#         unsafe_allow_html=True,
#     )

#     if not stops_with_coords:
#         st.info("No coordinate data available for weather lookup.")
#         return

#     with st.spinner("Fetching arrival-time forecasts for each stop…"):
#         weather_stops = WeatherEngine.fetch_route_at_times(stops_with_coords, itin_date_str)

#     # Data source summary
#     sources = [ws["weather"]["source"] for ws in weather_stops]
#     live_count = sum(1 for s in sources if "open-meteo" in s)
#     est_count  = sum(1 for s in sources if "seasonal"   in s)

#     if live_count == len(weather_stops):
#         st.caption("🟢 All forecasts from Open-Meteo hourly API (real data)")
#     elif live_count > 0:
#         st.caption(f"🟡 Mixed: {live_count} stops from Open-Meteo · {est_count} stops from seasonal estimate")
#     else:
#         st.caption(
#             "🟡 **Open-Meteo unreachable** — showing seasonal estimates with diurnal variation. "
#             "Values are indicative. Check network/proxy settings."
#         )

#     adverse_count = sum(1 for ws in weather_stops if ws["weather"]["is_adverse"])
#     if adverse_count:
#         st.warning(
#             f"⚠️ **{adverse_count} stop(s) forecast adverse weather at arrival time** — "
#             "allow extra buffer time and check vehicle load securing."
#         )

#     # ── Cards — 4 per row ─────────────────────────────────────────────────────
#     cols_per_row = min(4, len(weather_stops))
#     card_rows    = [weather_stops[i:i+cols_per_row]
#                     for i in range(0, len(weather_stops), cols_per_row)]

#     for card_row in card_rows:
#         cols = st.columns(len(card_row))
#         for col, ws in zip(cols, card_row):
#             w            = ws["weather"]
#             source       = w.get("source", "")
#             arrival      = ws.get("arrival_time", "")
#             forecast_t   = w.get("forecast_time", arrival)
#             precip_prob  = w.get("precip_probability")

#             border_color = "#E74C3C" if w["is_adverse"] else "#30363D"
#             temp_str = f"{w['temperature_c']}°C"     if w.get("temperature_c")    is not None else "—"
#             wind_str = f"{w['wind_speed_kmh']} km/h" if w.get("wind_speed_kmh")   is not None else "—"
#             prec_str = f"{w['precipitation_mm']} mm" if w.get("precipitation_mm") is not None else "—"
#             hum_str  = f"{w['humidity_pct']}%"       if w.get("humidity_pct")     is not None else "—"
#             vis_str  = f"{w['visibility_km']} km"    if w.get("visibility_km")    is not None else "—"
#             prob_str = f"{precip_prob}% rain chance" if precip_prob is not None else ""

#             if "open-meteo" in source:
#                 src_badge = '<div style="font-size:0.58rem;color:#3FB950;margin-top:5px">🟢 Open-Meteo hourly</div>'
#             else:
#                 src_badge = '<div style="font-size:0.58rem;color:#D4A843;margin-top:5px">🟡 Seasonal estimate</div>'

#             adverse_badge = (
#                 '<div style="font-size:0.65rem;color:#E74C3C;margin-top:5px;font-weight:600">⚠️ Adverse conditions</div>'
#                 if w["is_adverse"] else ""
#             )
#             prob_badge = (
#                 f'<div style="font-size:0.65rem;color:#2EA4A4;margin-top:3px">🌂 {prob_str}</div>'
#                 if prob_str else ""
#             )

#             col.markdown(
#                 f'<div class="weather-card" style="border:1px solid {border_color};margin-bottom:8px">'
#                 # Arrival time header — the KEY info
#                 f'<div style="font-size:0.6rem;font-family:monospace;color:#2EA4A4;'
#                 f'letter-spacing:0.05em;margin-bottom:6px">🕐 ARRIVAL {arrival}</div>'
#                 # Icon + temp
#                 f'<div style="font-size:1.9rem;margin-bottom:2px">{w["icon"]}</div>'
#                 f'<div style="font-size:0.78rem;font-weight:700;color:#F0F6FC;margin-bottom:2px">'
#                 f'{ws["location_name"][:20]}</div>'
#                 f'<div style="font-size:1.2rem;font-weight:700;color:#D4A843;margin-bottom:2px">{temp_str}</div>'
#                 f'<div style="font-size:0.72rem;color:#C9D1D9">{w["description"]}</div>'
#                 # Stats row
#                 f'<div style="font-size:0.65rem;color:#8B949E;margin-top:8px;line-height:1.7">'
#                 f'💨 {wind_str} &nbsp;·&nbsp; 🌧️ {prec_str}<br>'
#                 f'💧 {hum_str} &nbsp;·&nbsp; 👁️ {vis_str}'
#                 f'</div>'
#                 f'{prob_badge}{src_badge}{adverse_badge}'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )

#     # ── Summary data table ────────────────────────────────────────────────────
#     with st.expander("📋 Full Weather Forecast Table"):
#         rows_data = []
#         for ws in weather_stops:
#             w = ws["weather"]
#             rows_data.append({
#                 "Stop":              ws["location_name"],
#                 "Arrival Time":      ws.get("arrival_time", "—"),
#                 "Forecast At":       w.get("forecast_time", "—"),
#                 "Condition":         f"{w['icon']} {w['description']}",
#                 "Temp (°C)":         w.get("temperature_c", "—"),
#                 "Wind (km/h)":       w.get("wind_speed_kmh", "—"),
#                 "Rain (mm)":         w.get("precipitation_mm", "—"),
#                 "Rain Chance (%)":   w.get("precip_probability", "—"),
#                 "Humidity (%)":      w.get("humidity_pct", "—"),
#                 "Visibility (km)":   w.get("visibility_km", "—"),
#                 "Source":            w.get("source", "—"),
#                 "⚠️ Adverse":        "Yes" if w["is_adverse"] else "—",
#             })
#         st.dataframe(pd.DataFrame(rows_data), use_container_width=True)


# # ─────────────────────────────────────────────
# # FUEL PANEL  (called from _render_itinerary)
# # ─────────────────────────────────────────────
# def _render_fuel_panel(stops_list: list, constraints: dict, itin: dict):
#     """
#     Renders the full fuel cost + route savings analysis panel.
#     """
#     st.markdown("### ⛽ Fuel Cost & Route Savings Analysis")

#     vehicle_type = constraints.get("vehicle_type", "Van")
#     total_km     = itin.get("total_distance_km", 0)

#     if not total_km:
#         st.info("No distance data yet. Generate an itinerary first.")
#         return

#     # ── User controls ─────────────────────────────────────────────────────────
#     fc1, fc2, fc3 = st.columns(3)
#     with fc1:
#         city = st.selectbox(
#             "📍 City (for fuel price)",
#             list(FuelEngine.CITY_PRICES.keys()),
#             index=list(FuelEngine.CITY_PRICES.keys()).index("Mumbai")
#                   if "Mumbai" in FuelEngine.CITY_PRICES else 0,
#             key="fuel_city",
#         )
#     with fc2:
#         fuel_type_options = ["Auto (by vehicle)", "Petrol", "Diesel"]
#         fuel_choice = st.selectbox("⛽ Fuel type override", fuel_type_options, key="fuel_type_choice")
#     with fc3:
#         custom_price = st.number_input(
#             "₹/litre override (0 = use reference)",
#             min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="fuel_price_override"
#         )

#     override_price = custom_price if custom_price > 0 else None
#     fuel_type_map  = {"Petrol": "petrol", "Diesel": "diesel"}

#     # If override fuel type selected, compute price accordingly
#     if fuel_choice != "Auto (by vehicle)" and override_price is None:
#         ft          = fuel_type_map[fuel_choice]
#         override_price = FuelEngine.get_price(city, ft)

#     # ── Current itinerary fuel cost ───────────────────────────────────────────
#     fuel_current = FuelEngine.compute_fuel_cost(
#         total_km, vehicle_type, city, override_price
#     )

#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("Route Distance",    f"{total_km} km")
#     m2.metric("Fuel Consumed",     f"{fuel_current['litres_consumed']} L")
#     m3.metric("Price/Litre",       f"₹{fuel_current['price_per_litre']:.2f}")
#     m4.metric("Total Fuel Cost",   f"₹{fuel_current['total_cost_inr']:,.2f}")

#     st.markdown(
#         f'<div style="font-size:0.75rem;color:#8B949E;margin-bottom:20px">'
#         f'Vehicle: <b>{vehicle_type}</b> · '
#         f'Efficiency: <b>{fuel_current["efficiency_kmpl"]} km/L</b> · '
#         f'Fuel: <b>{fuel_current["fuel_type"].title()}</b> · '
#         f'City reference: <b>{city}</b> '
#         f'<span style="color:#30363D">(prices as of Jun 2025 — update via override)</span>'
#         f'</div>',
#         unsafe_allow_html=True,
#     )

#     # ── Route savings analysis ────────────────────────────────────────────────
#     st.markdown("#### 🔀 Optimized vs Un-Optimized Route Savings")

#     if len(stops_list) < 3:
#         st.info("Add at least 3 stops to compute route savings comparison.")
#         return

#     if st.button("🔍 Run Savings Analysis", key="run_savings"):
#         with st.spinner("Comparing original vs NN-optimized order via OSRM…"):
#             analysis = FuelEngine.savings_analysis(
#                 stops_list, vehicle_type, city, override_price
#             )
#         st.session_state["fuel_analysis"] = analysis

#     analysis = st.session_state.get("fuel_analysis")
#     if not analysis:
#         st.caption("Click 'Run Savings Analysis' to compare route orders.")
#         return

#     saved_km   = analysis.get("saved_km", 0)
#     saving_pct = analysis.get("saving_pct", 0)
#     saved_cost = analysis.get("saved_cost_inr", 0)

#     if saved_km > 0:
#         st.markdown(
#             f'<div class="fuel-save">'
#             f'<div style="font-size:1.05rem;font-weight:700;color:#3FB950;margin-bottom:10px">'
#             f'✅ You can save <b>₹{saved_cost:,.2f}</b> by reordering your stops!</div>'
#             f'<div style="display:flex;gap:32px;flex-wrap:wrap">'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">📏 Distance saved: <b>{saved_km} km</b></span>'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">📉 Reduction: <b>{saving_pct}%</b></span>'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">⛽ Fuel saved: '
#             f'<b>{round(saved_km / analysis["fuel_detail_optimized"]["efficiency_kmpl"], 2)} L</b></span>'
#             f'</div></div>',
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             '<div class="fuel-warn">'
#             '<b style="color:#D4A843">ℹ️ Your current stop order is already near-optimal.</b><br>'
#             '<span style="color:#8B949E;font-size:0.82rem">The nearest-neighbor reordering did not '
#             'produce a shorter total route for this set of stops.</span>'
#             '</div>',
#             unsafe_allow_html=True,
#         )

#     # Side-by-side comparison
#     sa1, sa2 = st.columns(2)
#     with sa1:
#         st.markdown("**📋 Original Order**")
#         for i, name in enumerate(analysis.get("original_order", []), 1):
#             st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
#                         f'<span style="color:#D4A843">{i}.</span> {name}</div>', unsafe_allow_html=True)
#         orig_c = analysis.get("original_cost_inr", 0)
#         st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
#                     f'<b>Total: {analysis.get("original_km",0)} km · ₹{orig_c:,.2f}</b></div>', unsafe_allow_html=True)

#     with sa2:
#         st.markdown("**✅ Optimized Order**")
#         for i, name in enumerate(analysis.get("optimized_order", []), 1):
#             st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
#                         f'<span style="color:#3FB950">{i}.</span> {name}</div>', unsafe_allow_html=True)
#         opt_c = analysis.get("optimized_cost_inr", 0)
#         st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
#                     f'<b>Total: {analysis.get("optimized_km",0)} km · ₹{opt_c:,.2f}</b></div>', unsafe_allow_html=True)

#     # Per-leg savings chart
#     per_stop = analysis.get("per_stop_savings", [])
#     if per_stop:
#         st.markdown("#### 📊 Leg-by-Leg Distance Comparison")
#         df_legs = pd.DataFrame(per_stop)
#         fig = go.Figure()
#         fig.add_bar(
#             name="Original", x=df_legs["leg"].astype(str),
#             y=df_legs["original_km"],
#             marker_color="#E74C3C",
#             hovertemplate="Leg %{x}: %{y:.2f} km<extra>Original</extra>",
#         )
#         fig.add_bar(
#             name="Optimized", x=df_legs["leg"].astype(str),
#             y=df_legs["optimized_km"],
#             marker_color="#3FB950",
#             hovertemplate="Leg %{x}: %{y:.2f} km<extra>Optimized</extra>",
#         )
#         fig.update_layout(
#             barmode="group",
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=260,
#             xaxis=dict(title="Leg #", showgrid=False),
#             yaxis=dict(title="km", showgrid=True, gridcolor="#21262D"),
#             margin=dict(t=10, b=40, l=50, r=20),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     # ── Optimized Route Map ──────────────────────────────────────────────────
#     optimized_order = analysis.get("optimized_order", [])
#     if optimized_order and len(stops_list) >= 2:
#         st.markdown("#### 🗺️ Optimized Route Map")
#         st.caption("Stop order resequenced by Nearest-Neighbor TSP for minimum distance.")

#         # Reorder stops_list to match the optimized sequence
#         nn_order      = MLEngine().nearest_neighbor_route([(s["lat"], s["lon"]) for s in stops_list])
#         opt_stops     = [stops_list[i] for i in nn_order]

#         # Build a lightweight itinerary shell so route_map_scatter draws the route line
#         opt_itin_stub = {
#             "stops": [
#                 {
#                     "sequence": idx + 1,
#                     "stop_id":  s["stop_id"],
#                     "stop_type": s.get("stop_type", "Delivery"),
#                 }
#                 for idx, s in enumerate(opt_stops)
#             ]
#         }

#         # Use Dashboard.route_map_scatter — reuse existing map function
#         from plotly.subplots import make_subplots as _msp  # already imported at top
#         lats   = [s["lat"]  for s in opt_stops]
#         lons   = [s["lon"]  for s in opt_stops]
#         names  = [s["location_name"] for s in opt_stops]
#         colors_map = {
#             "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
#             "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
#         }
#         pt_colors = [colors_map.get(s.get("stop_type","Delivery"), "#3FB950") for s in opt_stops]

#         import plotly.graph_objects as _go
#         fig_opt = _go.Figure()
#         # Route line
#         fig_opt.add_trace(_go.Scattergeo(
#             lat=lats, lon=lons, mode="lines",
#             line=dict(width=2.5, color="#3FB950"), name="Optimized Route",
#         ))
#         # Stop markers
#         fig_opt.add_trace(_go.Scattergeo(
#             lat=lats, lon=lons, mode="markers+text",
#             marker=dict(size=13, color=pt_colors, line=dict(color="#0D1117", width=1)),
#             text=[f"{i+1}. {n}" for i, n in enumerate(names)],
#             textposition="top center",
#             textfont=dict(size=9, color="#C9D1D9"),
#             hovertemplate="<b>%{text}</b><extra></extra>",
#             name="Stops",
#         ))
#         import numpy as _np
#         fig_opt.update_geos(
#             center=dict(lat=_np.mean(lats), lon=_np.mean(lons)),
#             projection_scale=8,
#             showland=True,       landcolor="#21262D",
#             showocean=True,      oceancolor="#161B22",
#             showcountries=True,  countrycolor="#30363D",
#             showcoastlines=True, coastlinecolor="#30363D",
#         )
#         fig_opt.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=420,
#             margin=dict(t=10, b=10, l=10, r=10),
#             geo=dict(bgcolor="rgba(0,0,0,0)"),
#             showlegend=True,
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig_opt, use_container_width=True, key="opt_route_map")

#     # Routing source note
#     rs = analysis.get("routing_source", "")
#     if rs == "osrm":
#         st.caption("🛰️ Savings computed using real OSRM road distances.")
#     else:
#         st.caption("📐 Savings computed using Haversine estimates (OSRM unreachable).")


# # ─────────────────────────────────────────────
# # ITINERARY RENDERER
# # ─────────────────────────────────────────────
# def _render_itinerary(itin, stops_df, ai, dash, constraints=None, nl_stops_override=None):
#     if constraints is None:
#         constraints = {}

#     # Build stops_list for map + fuel panel.
#     # For dataset routes: match stop_ids back to the stops_df for coordinates.
#     # For NL / custom routes: nl_stops_override carries the pre-built lat/lon list.
#     stops_list = []
#     if nl_stops_override:
#         stops_list = nl_stops_override
#     elif not stops_df.empty and "stop_id" in stops_df.columns:
#         for s in itin.get("stops", []):
#             row = stops_df[stops_df["stop_id"] == s["stop_id"]]
#             if not row.empty:
#                 stops_list.append({
#                     "stop_id": s["stop_id"], "location_name": s["location_name"],
#                     "lat": float(row.iloc[0]["lat"]), "lon": float(row.iloc[0]["lon"]),
#                     "stop_type": s["stop_type"],
#                 })

#     # Header
#     st.markdown(
#         f'<div class="itinerary-header">'
#         f'<div style="font-family:\'Playfair Display\',serif;font-size:1.4rem;color:#F0F6FC;font-weight:700">'
#         f'📋 {itin.get("itinerary_title","Optimized Itinerary")}</div>'
#         f'<div style="margin-top:10px;color:#8B949E;font-size:0.82rem">'
#         f'🚗 {itin.get("driver","—")} &nbsp;|&nbsp; 🚛 {itin.get("vehicle","—")} '
#         f'&nbsp;|&nbsp; 📅 {itin.get("date","—")} &nbsp;|&nbsp; 🛣️ {itin.get("transport_mode","—")}'
#         f'</div></div>',
#         unsafe_allow_html=True,
#     )

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Distance",      f"{itin.get('total_distance_km', 0)} km")
#     c2.metric("Total Duration",      f"{itin.get('total_duration_min', 0)} min")
#     c3.metric("Efficiency Score",    f"{itin.get('efficiency_score', 0)}/100")
#     c4.metric("On-Time Probability", f"{itin.get('on_time_probability', 0)}%")

#     # Routing source banner
#     r_src = itin.get("routing_source", "")
#     if r_src == "osrm":
#         st.success("🛰️ Road distances powered by **OSRM** — real road network data")
#     elif r_src == "haversine_fallback":
#         st.warning("📐 Straight-line distance estimates used (OSRM unreachable). Times are approximate.")

#     # Map
#     if stops_list:
#         st.markdown("### 🗺️ Route Map")
#         st.plotly_chart(dash.route_map_scatter(stops_list, itin), use_container_width=True)

#     # Stop sequence
#     st.markdown("### 📍 Stop Sequence")
#     itin_stops = sorted(itin.get("stops", []), key=lambda x: x["sequence"])
#     max_seq    = max((x["sequence"] for x in itin_stops), default=0)

#     for s in itin_stops:
#         risk_color = {"High":"#E74C3C","Medium":"#D4A843","Low":"#3FB950","None":"#3FB950"}.get(
#             s.get("risk_flag","None"), "#3FB950"
#         )
#         risk_html = (
#             f' &nbsp;·&nbsp; <span style="color:{risk_color}">⚠️ {s.get("risk_reason","")}</span>'
#             if s.get("risk_flag","None") not in ["None",""] else ""
#         )
#         notes_html = (
#             f'<div class="stop-detail" style="margin-top:4px;font-style:italic">{s.get("notes","")}</div>'
#             if s.get("notes") else ""
#         )
#         connector = "" if s["sequence"] == max_seq else '<div class="connector-line"></div>'
#         st.markdown(
#             f'<div class="stop-card"><div class="stop-row">'
#             f'<div class="stop-num">{s["sequence"]}</div>'
#             f'<div style="flex:1">'
#             f'<div style="display:flex;justify-content:space-between;align-items:center">'
#             f'<b style="color:#F0F6FC">{s["location_name"]}</b>'
#             f'<span style="font-size:0.8rem;color:#8B949E">🕐 {s.get("arrival_time","—")} → {s.get("departure_time","—")}</span>'
#             f'</div>'
#             f'<div class="stop-detail">Type: {s.get("stop_type","—")} &nbsp;·&nbsp; '
#             f'Priority: {s.get("priority","—")} &nbsp;·&nbsp; '
#             f'Service: {s.get("service_duration_min","—")} min &nbsp;·&nbsp; '
#             f'Travel: {s.get("travel_time_from_prev_min","—")} min &nbsp;·&nbsp; '
#             f'Dist: {s.get("distance_from_prev_km","—")} km{risk_html}</div>'
#             f'{notes_html}</div></div></div>{connector}',
#             unsafe_allow_html=True,
#         )

#     if itin.get("warnings"):
#         with st.expander("⚠️ Warnings"):
#             for w in itin["warnings"]:
#                 st.warning(w)

#     with st.expander("📝 Optimization Notes"):
#         st.info(itin.get("optimization_notes", "No notes."))

#     # ── Constraint Violation Checker ─────────────────────────────────────────
#     st.markdown("### ⚡ Constraint Violation Check")
#     if constraints:
#         violations = ai.check_violations(itin, constraints)
#         if not violations:
#             st.markdown(
#                 '<div style="background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);'
#                 'border-radius:10px;padding:14px 18px;margin-bottom:12px">'
#                 '<b style="color:#3FB950">✅ No violations detected.</b> '
#                 'All stops are within time windows and driver hour limits.</div>',
#                 unsafe_allow_html=True,
#             )
#         else:
#             crit = [v for v in violations if v["severity"] == "Critical"]
#             warn = [v for v in violations if v["severity"] == "Warning"]
#             vc1, vc2 = st.columns(2)
#             vc1.metric("🔴 Critical", len(crit))
#             vc2.metric("🟡 Warnings", len(warn))
#             for v in violations:
#                 css   = "violation-critical" if v["severity"] == "Critical" else "violation-warning"
#                 color = "#E74C3C"            if v["severity"] == "Critical" else "#D4A843"
#                 icon  = "🔴"                 if v["severity"] == "Critical" else "🟡"
#                 label = {"time_window":"Time Window","driver_hours":"Driver Hours","capacity":"Capacity"}.get(v["type"], v["type"])
#                 st.markdown(
#                     f'<div class="{css}">'
#                     f'<div style="display:flex;justify-content:space-between">'
#                     f'<b style="color:#F0F6FC">{icon} Stop {v["sequence"]} — {v["location_name"]}</b>'
#                     f'<span style="font-size:0.75rem;color:{color};font-weight:600">{label} · {v["severity"]}</span>'
#                     f'</div><div style="font-size:0.82rem;color:#C9D1D9;margin-top:6px">{v["detail"]}</div></div>',
#                     unsafe_allow_html=True,
#                 )
#     else:
#         st.info("Generate an itinerary with constraints to see violation analysis.")

#     # ── Weather Panel ────────────────────────────────────────────────────────
#     # Build a stops list that includes arrival_time so forecasts are time-aware.
#     # Priority: dataset stops_list merged with itinerary arrival times.
#     itin_date = itin.get("date", datetime.now().strftime("%Y-%m-%d"))

#     # Map stop_id → arrival_time from the generated itinerary
#     arrival_map = {
#         s["stop_id"]: s.get("arrival_time", "08:00")
#         for s in itin.get("stops", [])
#     }

#     weather_stops = []
#     if stops_list:
#         # Dataset route: merge lat/lon from stops_list + arrival_time from itinerary
#         for s in stops_list:
#             weather_stops.append({
#                 **s,
#                 "arrival_time": arrival_map.get(s["stop_id"], "08:00"),
#                 "sequence":     next(
#                     (itin_s["sequence"] for itin_s in itin.get("stops", [])
#                      if itin_s["stop_id"] == s["stop_id"]), 0
#                 ),
#             })
#         # Sort by sequence so cards appear in travel order
#         weather_stops.sort(key=lambda x: x.get("sequence", 0))
#     else:
#         # NL / custom stops — coords embedded in itinerary stops
#         for s in itin.get("stops", []):
#             if "lat" in s and "lon" in s:
#                 weather_stops.append({
#                     "stop_id":       s.get("stop_id", ""),
#                     "location_name": s.get("location_name", ""),
#                     "lat":           s["lat"],
#                     "lon":           s["lon"],
#                     "stop_type":     s.get("stop_type", ""),
#                     "arrival_time":  s.get("arrival_time", "08:00"),
#                     "sequence":      s.get("sequence", 0),
#                 })

#     if weather_stops:
#         st.markdown("---")
#         _render_weather_panel(weather_stops, itin_date_str=itin_date)

#     # ── Fuel & Savings Panel ─────────────────────────────────────────────────
#     st.markdown("---")
#     # Build stops_list for fuel engine — prefer stops_list (dataset), else NL
#     fuel_stops = stops_list
#     if not fuel_stops:
#         for s in itin.get("stops", []):
#             if "lat" in s and "lon" in s:
#                 fuel_stops.append({
#                     "stop_id": s.get("stop_id",""),
#                     "location_name": s.get("location_name",""),
#                     "lat": s["lat"], "lon": s["lon"],
#                     "stop_type": s.get("stop_type",""),
#                 })
#     _render_fuel_panel(fuel_stops, constraints, itin)

#     # Adjust
#     st.markdown("### 🔄 Adjust Itinerary")
#     change_req = st.text_area(
#         "Describe the change in plain English:",
#         placeholder="e.g. 'Remove stop 3', 'Add 30-min rest after stop 2', 'Traffic — delay all by 20 min'",
#         height=90,
#     )
#     if st.button("Apply Changes"):
#         if change_req.strip():
#             with st.spinner("Re-planning…"):
#                 updated = ai.adjust_itinerary(itin, change_req)
#             st.session_state["generated_itinerary"] = updated
#             st.success("Itinerary updated!")
#             st.rerun()
#         else:
#             st.error("Please describe the change.")

#     # Export
#     st.markdown("### 📤 Export")
#     col_e1, col_e2 = st.columns(2)
#     with col_e1:
#         st.download_button("⬇️ Download JSON", json.dumps(itin, indent=2, default=str),
#                            "itinerary.json", "application/json", use_container_width=True)
#     with col_e2:
#         stop_rows = itin.get("stops", [])
#         if stop_rows:
#             st.download_button("⬇️ Download CSV", pd.DataFrame(stop_rows).to_csv(index=False),
#                                "itinerary_stops.csv", "text/csv", use_container_width=True)


# # ─────────────────────────────────────────────
# # PAGE: CLUSTERING
# # ─────────────────────────────────────────────
# def page_clustering(stops_df, ml, dash):
#     st.markdown('<div class="section-title">🧩 Stop Clustering Engine</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Unsupervised clustering to find natural groupings in your stop network</div>', unsafe_allow_html=True)

#     if stops_df.empty:
#         st.warning("No stops data loaded.")
#         return

#     col1, col2 = st.columns([1, 3])
#     with col1:
#         n_clusters  = st.slider("Number of Clusters", 2, 8, 4)
#         sample_size = st.slider("Sample Size", 20, min(len(stops_df), 200), min(len(stops_df), 60))
#         if st.button("Run Clustering"):
#             sample_df = stops_df.sample(min(sample_size, len(stops_df)), random_state=42)
#             labels, X_2d = ml.cluster_stops(sample_df, n_clusters)
#             st.session_state["cluster_labels"] = labels
#             st.session_state["cluster_X2d"]   = X_2d
#             st.session_state["cluster_df"]    = sample_df.reset_index(drop=True)
#             st.success("Clustering complete!")

#     with col2:
#         if st.session_state.get("cluster_labels") is not None:
#             st.markdown("**Cluster Visualization (PCA 2D)**")
#             st.plotly_chart(
#                 dash.cluster_scatter(st.session_state["cluster_df"],
#                                      st.session_state["cluster_labels"],
#                                      st.session_state["cluster_X2d"]),
#                 use_container_width=True,
#             )

#     if st.session_state.get("cluster_labels") is not None:
#         labels = st.session_state["cluster_labels"]
#         cdf    = st.session_state["cluster_df"]

#         keywords = ml.get_cluster_keywords()
#         st.markdown("### Cluster Keyword Profiles")
#         cols = st.columns(min(len(keywords), 5))
#         for i, (cid, kws) in enumerate(keywords.items()):
#             with cols[i % len(cols)]:
#                 st.markdown(
#                     f'<div class="card card-gold"><b>Cluster {cid+1}</b><br>'
#                     f'<small style="color:#8B949E">{" · ".join(kws)}</small></div>',
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("### Cluster Composition by Stop Type")
#         cdf2 = cdf.copy()
#         cdf2["cluster"] = [f"Cluster {l+1}" for l in labels]
#         comp = cdf2.groupby(["cluster","stop_type"]).size().reset_index(name="count")
#         fig  = px.bar(comp, x="cluster", y="count", color="stop_type",
#                       color_discrete_map=dash.STOP_COLORS, barmode="stack")
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=320,
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#             xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#         st.markdown("### Geographic Distribution of Clusters")
#         cdf2["x"] = st.session_state["cluster_X2d"][:, 0]
#         fig_geo   = px.scatter_mapbox(
#             cdf2, lat="lat", lon="lon", color="cluster",
#             hover_data=["location_name","stop_type","priority"], zoom=9, height=380,
#         )
#         fig_geo.update_layout(
#             mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", margin=dict(t=0,b=0,l=0,r=0),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig_geo, use_container_width=True)


# # ─────────────────────────────────────────────
# # PAGE: AI ASSISTANT
# # ─────────────────────────────────────────────
# def page_assistant(stops_df, ai, kb_df):
#     st.markdown('<div class="section-title">💬 Logistics AI Assistant</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Grounded in your logistics knowledge base · '
#         'Only answers travel & logistics questions</div>',
#         unsafe_allow_html=True,
#     )

#     col_s1, col_s2, col_s3 = st.columns(3)
#     for col, label, active in [
#         (col_s1, "LLM",        bool(ai.llm)),
#         (col_s2, "Embeddings", bool(ai.embeddings)),
#         (col_s3, "RAG Index",  bool(ai.embeddings) and not kb_df.empty),
#     ]:
#         color  = "#2EA4A4" if active else "#E74C3C"
#         status = ("🟢 Connected" if active else "🔴 Offline") if label != "RAG Index" else ("🟢 Ready" if active else "⚠️ Keyword fallback")
#         col.markdown(
#             f'<div class="card" style="padding:10px;text-align:center">'
#             f'<span style="font-size:0.8rem;color:#8B949E">{label}</span><br>'
#             f'<b style="color:{color}">{status}</b></div>',
#             unsafe_allow_html=True,
#         )

#     st.markdown("")

#     with st.expander("ℹ️ What can I ask?", expanded=False):
#         st.markdown("""
# **In scope:** route planning, delivery operations, customs, fleet management, freight modes, KPIs, documentation.
# **Out of scope:** questions unrelated to travel or logistics are politely declined.
#         """)

#     if "assistant_history" not in st.session_state:
#         st.session_state["assistant_history"] = []

#     for msg in st.session_state["assistant_history"]:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             if msg["role"] == "assistant":
#                 if msg.get("sources"):
#                     _render_sources(msg["sources"])
#                 if msg.get("rejected"):
#                     st.caption("🚫 Outside the logistics/travel domain.")
#                 elif msg.get("grounded"):
#                     st.caption("✅ Grounded in knowledge base via semantic retrieval.")

#     user_input = st.chat_input("Ask about routes, delivery windows, customs, fuel costs…")
#     if user_input:
#         st.session_state["assistant_history"].append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         with st.chat_message("assistant"):
#             with st.spinner("Searching knowledge base…"):
#                 result = ai.get_kb_answer(user_input, kb_df)
#             answer_text = result["text"]
#             sources     = result.get("sources", [])
#             grounded    = result.get("grounded", False)
#             rejected    = result.get("rejected", False)
#             st.markdown(answer_text)
#             if sources:   _render_sources(sources)
#             if rejected:  st.caption("🚫 Outside the logistics/travel domain.")
#             elif grounded: st.caption("✅ Grounded in knowledge base via semantic retrieval.")
#             else:          st.caption("⚠️ Low-confidence retrieval — broader KB context used.")

#         st.session_state["assistant_history"].append({
#             "role": "assistant", "content": answer_text,
#             "sources": sources, "grounded": grounded, "rejected": rejected,
#         })

#     col_a, _ = st.columns([1, 5])
#     with col_a:
#         if st.button("🗑️ Clear Chat"):
#             st.session_state["assistant_history"] = []
#             st.rerun()

#     if not kb_df.empty:
#         with st.expander(f"📚 Knowledge Base ({len(kb_df)} articles)"):
#             st.dataframe(
#                 kb_df[["category","topic","content"]], use_container_width=True,
#                 column_config={"content": st.column_config.TextColumn("content", width="large")},
#             )


# def _render_sources(sources: list):
#     if not sources:
#         return
#     seen, unique = set(), []
#     for s in sources:
#         key = s.get("topic","")
#         if key not in seen:
#             seen.add(key); unique.append(s)

#     lines = []
#     for s in unique:
#         pct        = int(s.get("score", 0) * 100)
#         bar        = "█" * (pct // 10) + "░" * (10 - pct // 10)
#         lines.append(
#             f"**{s.get('category','—')}** › {s.get('topic','—')} "
#             f"<span style='color:#D4A843;font-family:monospace;font-size:0.75rem'>{bar} {pct}%</span>"
#         )
#     st.markdown(
#         '<div style="background:rgba(46,164,164,0.08);border:1px solid rgba(46,164,164,0.25);'
#         'border-radius:8px;padding:10px 14px;margin-top:8px">'
#         '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.06em;'
#         'text-transform:uppercase;margin-bottom:6px">📎 Sources retrieved</div>'
#         + "".join(f'<div style="font-size:0.8rem;color:#C9D1D9;margin:3px 0">{l}</div>' for l in lines)
#         + "</div>",
#         unsafe_allow_html=True,
#     )


# # ─────────────────────────────────────────────
# # PAGE: PERFORMANCE
# # ─────────────────────────────────────────────
# def page_performance(stops_df, routes_df, ai, dash):
#     st.markdown('<div class="section-title">📈 Performance Analytics</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Route efficiency, on-time delivery, fuel costs, and operational insights</div>', unsafe_allow_html=True)

#     if stops_df.empty or routes_df.empty:
#         st.warning("No data loaded.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     on_time_pct     = round(100 * (stops_df["status"] == "On Time").mean(), 1)
#     avg_distance    = round(routes_df["distance_km"].mean(), 1)
#     total_fuel_cost = round(routes_df["estimated_fuel_cost_inr"].sum(), 0)
#     avg_stops       = round(stops_df.groupby("route_id").size().mean(), 1)
#     c1.metric("On-Time Rate",       f"{on_time_pct}%")
#     c2.metric("Avg Distance/Route", f"{avg_distance} km")
#     c3.metric("Total Fuel Cost",    f"₹{total_fuel_cost:,.0f}")
#     c4.metric("Avg Stops/Route",    avg_stops)

#     st.markdown("---")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**On-Time % by Transport Mode (Trend)**")
#         st.plotly_chart(dash.performance_timeline(routes_df), use_container_width=True)
#     with col2:
#         st.markdown("**On-Time Delivery Rate**")
#         st.plotly_chart(dash.on_time_gauge(on_time_pct), use_container_width=True)

#     col3, col4 = st.columns(2)
#     with col3:
#         st.markdown("**Avg Fuel Cost by Transport Mode**")
#         st.plotly_chart(dash.fuel_cost_bar(routes_df), use_container_width=True)
#     with col4:
#         st.markdown("**Stops by Status**")
#         sc = stops_df["status"].value_counts().reset_index()
#         sc.columns = ["status","count"]
#         fig = go.Figure(go.Bar(
#             x=sc["status"], y=sc["count"],
#             marker_color=[dash.STATUS_COLORS.get(s,"#8B949E") for s in sc["status"]],
#             hovertemplate="%{x}: %{y}<extra></extra>",
#         ))
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=260,
#             xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"),
#             margin=dict(t=10, b=40, l=40, r=20),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     if "delay_reason" in stops_df.columns:
#         st.markdown("**Delay Reasons**")
#         delayed = stops_df[stops_df["delay_reason"].notna() & (stops_df["delay_reason"] != "")]
#         if not delayed.empty:
#             dr    = delayed["delay_reason"].value_counts().reset_index()
#             dr.columns = ["reason","count"]
#             fig_d = px.bar(dr, x="count", y="reason", orientation="h",
#                            color_discrete_sequence=["#E74C3C"])
#             fig_d.update_layout(
#                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                 font_color="#C9D1D9", height=260,
#                 xaxis=dict(showgrid=True, gridcolor="#21262D"),
#                 yaxis=dict(showgrid=False),
#                 margin=dict(t=10, b=30, l=150, r=20),
#             )
#             st.plotly_chart(fig_d, use_container_width=True)

#     st.markdown("### 🤖 AI Route Performance Insights")
#     if st.button("Generate AI Insights"):
#         with st.spinner("Analyzing route performance…"):
#             insights = ai.analyze_route_performance(routes_df, stops_df)
#         st.markdown(f'<div class="summary-box">{insights}</div>', unsafe_allow_html=True)

#     if st.session_state.get("plan_history"):
#         with st.expander("📋 Itinerary Generation History (this session)"):
#             st.dataframe(pd.DataFrame(st.session_state["plan_history"]), use_container_width=True)

#     with st.expander("📋 Full Stops Table"):
#         st.dataframe(stops_df.sort_values("scheduled_date", ascending=False), use_container_width=True)

#     with st.expander("📋 Full Routes Table"):
#         st.dataframe(routes_df.sort_values("route_date", ascending=False), use_container_width=True)


# # ─────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────
# def main():
#     init_session()

#     loader = DataLoader()
#     ai     = AIEngine()
#     ml     = MLEngine()
#     dash   = Dashboard()

#     stops_df  = loader.load_stops()
#     routes_df = loader.load_routes()
#     kb_df     = loader.load_kb()
#     stats     = loader.get_summary_stats(stops_df, routes_df)

#     with st.sidebar:
#         st.markdown(
#             '<div style="text-align:center;padding:20px 0 10px">'
#             '<div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;color:#F0F6FC;font-weight:900">🗺️ RouteIQ</div>'
#             '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.1em;text-transform:uppercase">Logistics Itinerary Planner</div>'
#             '</div>',
#             unsafe_allow_html=True,
#         )
#         st.markdown("---")
#         page = st.radio(
#             "Navigation",
#             ["📊 Overview", "🔍 Itinerary Planner", "🧩 Clustering", "💬 AI Assistant", "📈 Performance"],
#             label_visibility="collapsed",
#         )

#         if not stops_df.empty:
#             st.markdown("---")
#             st.markdown(
#                 '<div style="font-size:0.7rem;color:#8B949E;letter-spacing:0.1em;'
#                 'text-transform:uppercase;margin-bottom:8px">Quick Stats</div>',
#                 unsafe_allow_html=True,
#             )
#             for val, label, color in [
#                 (stats.get("total_stops",  0),              "Total Stops",        "#D4A843"),
#                 (stats.get("total_routes", 0),              "Active Routes",      "#2EA4A4"),
#                 (f"{stats.get('on_time_pct',0)}%",          "On-Time Rate",       "#3FB950"),
#                 (stats.get("high_priority", 0),             "High Priority Stops","#E74C3C"),
#             ]:
#                 st.markdown(
#                     f'<div class="card" style="padding:12px">'
#                     f'<div style="color:{color};font-size:1.4rem;font-weight:700">{val}</div>'
#                     f'<div style="color:#8B949E;font-size:0.72rem">{label}</div></div>',
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("---")
#         st.markdown(
#             f'<div style="font-size:0.75rem;color:#8B949E">'
#             f'{"🟢 LLM Connected" if ai.llm else "🔴 LLM Offline (fallback)"}</div>',
#             unsafe_allow_html=True,
#         )

#     if page == "📊 Overview":
#         page_overview(stops_df, routes_df, stats, dash)
#     elif page == "🔍 Itinerary Planner":
#         page_planner(stops_df, ai, ml, dash)
#     elif page == "🧩 Clustering":
#         page_clustering(stops_df, ml, dash)
#     elif page == "💬 AI Assistant":
#         page_assistant(stops_df, ai, kb_df)
#     elif page == "📈 Performance":
#         page_performance(stops_df, routes_df, ai, dash)


# if __name__ == "__main__":
#     main()

# 1.1 ##############################################
# import os

# # ── Tiktoken cache (must be set before any tiktoken / langchain import) ───────
# _tiktoken_cache_dir = os.path.abspath("./token")
# os.makedirs(_tiktoken_cache_dir, exist_ok=True)
# os.environ["TIKTOKEN_CACHE_DIR"] = _tiktoken_cache_dir

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import re
# from datetime import datetime, timedelta
# import random
# import math
# import httpx
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")

# load_dotenv()

# # ─────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="RouteIQ — Logistics Itinerary Planner",
#     page_icon="🗺️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────────────────────────
# # GLOBAL STYLES
# # ─────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

# :root {
#   --midnight: #0D1117;
#   --obsidian: #161B22;
#   --charcoal: #21262D;
#   --steel: #30363D;
#   --ink: #8B949E;
#   --silver: #C9D1D9;
#   --white: #F0F6FC;
#   --gold: #D4A843;
#   --amber: #E8873A;
#   --teal: #2EA4A4;
#   --crimson: #C0392B;
#   --sage: #3FB950;
#   --violet: #8957E5;
#   --blue: #1F6FEB;
# }

# html, body, [class*="css"] {
#   font-family: 'DM Sans', sans-serif !important;
#   background-color: var(--midnight) !important;
#   color: var(--silver) !important;
# }
# section[data-testid="stSidebar"] {
#   background: var(--obsidian) !important;
#   border-right: 1px solid var(--steel) !important;
# }
# section[data-testid="stSidebar"] .stRadio label {
#   color: var(--silver) !important;
#   font-size: 0.9rem !important;
# }
# h1, h2, h3 {
#   font-family: 'Playfair Display', serif !important;
#   color: var(--white) !important;
# }
# [data-testid="metric-container"] {
#   background: var(--obsidian) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 12px !important;
#   padding: 16px !important;
# }
# [data-testid="metric-container"] label {
#   color: var(--ink) !important;
#   font-size: 0.75rem !important;
#   letter-spacing: 0.08em !important;
#   text-transform: uppercase !important;
# }
# [data-testid="metric-container"] [data-testid="stMetricValue"] {
#   color: var(--gold) !important;
#   font-family: 'Playfair Display', serif !important;
#   font-size: 2rem !important;
# }
# details {
#   background: var(--obsidian) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 8px !important;
# }
# details summary { color: var(--teal) !important; font-weight: 500 !important; }
# [data-testid="stDataFrame"] { border: 1px solid var(--steel) !important; border-radius: 8px !important; }
# textarea, input[type="text"] {
#   background: var(--charcoal) !important;
#   color: var(--white) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 8px !important;
# }
# .stButton > button {
#   background: linear-gradient(135deg, var(--gold), var(--amber)) !important;
#   color: var(--midnight) !important;
#   font-weight: 600 !important;
#   border: none !important;
#   border-radius: 8px !important;
#   letter-spacing: 0.04em !important;
#   transition: opacity 0.2s !important;
# }
# .stButton > button:hover { opacity: 0.85 !important; }
# [data-testid="stSelectbox"] select, .stSelectbox > div {
#   background: var(--charcoal) !important;
#   color: var(--white) !important;
#   border-color: var(--steel) !important;
# }
# .stTabs [data-baseweb="tab"] { color: var(--ink) !important; border-bottom: 2px solid transparent !important; }
# .stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

# .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; }
# .badge-delivery  { background: rgba(212,168,67,0.2);  color: #D4A843; border: 1px solid rgba(212,168,67,0.4); }
# .badge-meeting   { background: rgba(46,164,164,0.2);  color: #2EA4A4; border: 1px solid rgba(46,164,164,0.4); }
# .badge-pickup    { background: rgba(63,185,80,0.2);   color: #3FB950; border: 1px solid rgba(63,185,80,0.4); }
# .badge-warehouse { background: rgba(137,87,229,0.2);  color: #8957E5; border: 1px solid rgba(137,87,229,0.4); }
# .badge-customs   { background: rgba(192,57,43,0.2);   color: #E74C3C; border: 1px solid rgba(192,57,43,0.4); }
# .badge-rest      { background: rgba(139,148,158,0.2); color: #8B949E; border: 1px solid rgba(139,148,158,0.4); }

# .card { background: var(--obsidian); border: 1px solid var(--steel); border-radius: 12px; padding: 20px; margin-bottom: 12px; }
# .card-gold  { border-left: 4px solid var(--gold); }
# .card-teal  { border-left: 4px solid var(--teal); }
# .card-red   { border-left: 4px solid var(--crimson); }
# .card-green { border-left: 4px solid var(--sage); }
# .card-blue  { border-left: 4px solid var(--blue); }

# .section-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: var(--white); margin-bottom: 4px; }
# .section-sub   { color: var(--ink); font-size: 0.85rem; margin-bottom: 20px; letter-spacing: 0.04em; }
# .hero-title    { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 900; color: var(--white); line-height: 1.1; }
# .hero-accent   { color: var(--gold); }

# .stop-card { background: var(--charcoal); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# .stop-num  { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; background: var(--gold); color: var(--midnight); border-radius: 50%; font-weight: 700; font-size: 0.85rem; margin-right: 10px; flex-shrink: 0; }
# .stop-row  { display: flex; align-items: flex-start; }
# .stop-detail { font-size: 0.82rem; color: var(--ink); margin-top: 4px; }
# .connector-line { width: 2px; height: 30px; background: linear-gradient(var(--gold), var(--teal)); margin: 0 auto 0 13px; }

# .summary-box { background: linear-gradient(135deg, rgba(212,168,67,0.08), rgba(46,164,164,0.08)); border: 1px solid rgba(212,168,67,0.3); border-radius: 12px; padding: 20px; margin: 12px 0; }
# .itinerary-header { background: linear-gradient(135deg, rgba(31,111,235,0.15), rgba(46,164,164,0.15)); border: 1px solid rgba(31,111,235,0.3); border-radius: 12px; padding: 18px 22px; margin-bottom: 18px; }
# .violation-critical { background: rgba(192,57,43,0.1); border-left: 4px solid #E74C3C; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
# .violation-warning  { background: rgba(212,168,67,0.1); border-left: 4px solid #D4A843; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
# .weather-card { background: var(--charcoal); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 16px; text-align: center; }
# .fuel-card    { background: var(--obsidian); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# .fuel-save    { background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.35); border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; }
# .fuel-warn    { background: rgba(192,57,43,0.08); border: 1px solid rgba(192,57,43,0.3); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# # SESSION STATE
# # ─────────────────────────────────────────────
# def init_session():
#     defaults = {
#         "generated_itinerary": None,
#         "batch_itineraries":   [],
#         "cluster_labels":      None,
#         "cluster_X2d":         None,
#         "cluster_df":          None,
#         "assistant_history":   [],
#         "plan_history":        [],
#         "last_constraints":    {},
#         "nl_parsed_result":    None,
#         "weather_cache":       {},      # key: "lat_lon" → weather dict
#         "fuel_analysis":       None,    # last fuel/savings analysis result
#         "itinerary_source":    None,    # "dataset" | "custom" | "nl" — which tab owns it
#         "nl_stops_for_map":    [],      # lat/lon list for NL itinerary map
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v


# # ─────────────────────────────────────────────
# # DATA LOADER
# # ─────────────────────────────────────────────
# class DataLoader:
#     STOPS_FILE  = "stops.csv"
#     ROUTES_FILE = "routes.csv"
#     KB_FILE     = "logistics_kb.csv"

#     def load_stops(self):
#         if not os.path.exists(self.STOPS_FILE):
#             self._generate_and_save()
#         df = pd.read_csv(self.STOPS_FILE, parse_dates=["time_window_start", "time_window_end"])

#         # Parse "coordinates" column ("lat, lon") into separate float columns
#         # so all downstream map / OSRM / clustering / weather code keeps working.
#         if "coordinates" in df.columns and "lat" not in df.columns:
#             try:
#                 split = df["coordinates"].str.split(",", expand=True)
#                 df["lat"] = pd.to_numeric(split[0].str.strip(), errors="coerce")
#                 df["lon"] = pd.to_numeric(split[1].str.strip(), errors="coerce")
#             except Exception:
#                 df["lat"] = float("nan")
#                 df["lon"] = float("nan")

#         # If both lat/lon and coordinates exist, keep coordinates as display column
#         # and ensure lat/lon are floats (handles files that still have both).
#         if "lat" in df.columns:
#             df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
#         if "lon" in df.columns:
#             df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

#         return df

#     def load_routes(self):
#         if not os.path.exists(self.ROUTES_FILE):
#             self._generate_and_save()
#         return pd.read_csv(self.ROUTES_FILE)

#     def load_kb(self):
#         if not os.path.exists(self.KB_FILE):
#             self._generate_and_save()
#         return pd.read_csv(self.KB_FILE)

#     def _generate_and_save(self):
#         from generate_data import generate_stops, generate_routes, generate_kb
#         stops  = generate_stops(60)
#         routes = generate_routes(stops)
#         kb     = generate_kb()
#         stops.to_csv(self.STOPS_FILE,  index=False)
#         routes.to_csv(self.ROUTES_FILE, index=False)
#         kb.to_csv(self.KB_FILE,         index=False)

#     def get_summary_stats(self, stops_df, routes_df):
#         if stops_df.empty:
#             return {}
#         return {
#             "total_stops":         len(stops_df),
#             "total_routes":        stops_df["route_id"].nunique() if "route_id" in stops_df.columns else 0,
#             "total_distance_km":   round(routes_df["distance_km"].sum(), 1) if not routes_df.empty else 0,
#             "avg_stops_per_route": round(stops_df.groupby("route_id").size().mean(), 1) if "route_id" in stops_df.columns else 0,
#             "on_time_pct":         round(100 * (stops_df["status"] == "On Time").mean(), 1) if "status" in stops_df.columns else 0,
#             "high_priority":       int((stops_df["priority"].str.title() == "High").sum()) if "priority" in stops_df.columns else 0,
#         }


# # ─────────────────────────────────────────────
# # ML ENGINE
# # ─────────────────────────────────────────────
# class MLEngine:
#     def __init__(self):
#         self.vectorizer     = None
#         self.kmeans         = None
#         self.feature_matrix = None

#     def cluster_stops(self, stops_df, n_clusters=4):
#         texts = (
#             stops_df["stop_type"].fillna("") + " " +
#             stops_df["location_name"].fillna("") + " " +
#             stops_df["notes"].fillna("")
#         )
#         self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
#         X = self.vectorizer.fit_transform(texts)
#         self.feature_matrix = X
#         self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         labels = self.kmeans.fit_predict(X)
#         pca    = PCA(n_components=2, random_state=42)
#         X_2d   = pca.fit_transform(X.toarray())
#         return labels, X_2d

#     def get_cluster_keywords(self, top_n=5):
#         if self.vectorizer is None or self.kmeans is None:
#             return {}
#         terms    = self.vectorizer.get_feature_names_out()
#         keywords = {}
#         for i, center in enumerate(self.kmeans.cluster_centers_):
#             top_idx     = center.argsort()[-top_n:][::-1]
#             keywords[i] = [terms[j] for j in top_idx]
#         return keywords

#     def nearest_neighbor_route(self, coords):
#         if len(coords) <= 1:
#             return list(range(len(coords)))
#         unvisited = list(range(1, len(coords)))
#         route     = [0]
#         while unvisited:
#             curr    = route[-1]
#             nearest = min(unvisited, key=lambda j: self._haversine(coords[curr], coords[j]))
#             route.append(nearest)
#             unvisited.remove(nearest)
#         return route

#     @staticmethod
#     def _haversine(c1, c2):
#         lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
#         lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
#         a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
#         return 6371 * 2 * math.asin(math.sqrt(a))

#     @staticmethod
#     def compute_distance_km(lat1, lon1, lat2, lon2):
#         return MLEngine._haversine((lat1, lon1), (lat2, lon2))

#     @staticmethod
#     def osrm_route(coords: list) -> dict:
#         """
#         Real road routing via public OSRM demo API.
#         Falls back to Haversine (35 km/h) if OSRM is unreachable.

#         Args:
#             coords: list of (lat, lon) tuples in stop order

#         Returns:
#             {legs, total_distance_km, total_duration_min, source}
#         """
#         if len(coords) < 2:
#             return {"legs": [], "total_distance_km": 0.0,
#                     "total_duration_min": 0.0, "source": "osrm"}

#         waypoints = ";".join(f"{lon},{lat}" for lat, lon in coords)
#         url = (
#             "http://router.project-osrm.org/route/v1/driving/" + waypoints
#             + "?overview=false&steps=false&annotations=false"
#         )
#         try:
#             resp = httpx.get(url, timeout=8.0)
#             data = resp.json()
#             if data.get("code") != "Ok":
#                 raise ValueError(f"OSRM code: {data.get('code')}")
#             legs = [
#                 {
#                     "distance_km":  round(leg["distance"] / 1000, 2),
#                     "duration_min": round(leg["duration"] / 60, 1),
#                 }
#                 for leg in data["routes"][0]["legs"]
#             ]
#             return {
#                 "legs":               legs,
#                 "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
#                 "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
#                 "source":             "osrm",
#             }
#         except Exception:
#             legs = []
#             for i in range(len(coords) - 1):
#                 d = MLEngine._haversine(coords[i], coords[i + 1])
#                 legs.append({"distance_km": round(d, 2), "duration_min": round(d / 35 * 60, 1)})
#             return {
#                 "legs":               legs,
#                 "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
#                 "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
#                 "source":             "haversine_fallback",
#             }


# # ─────────────────────────────────────────────
# # WEATHER ENGINE  (Open-Meteo — no API key)
# # ─────────────────────────────────────────────
# class WeatherEngine:
#     """
#     Arrival-time-aware weather forecasting via Open-Meteo hourly API (free, no key).

#     For each stop we fetch the HOURLY forecast for that location, then pick the
#     hour slot that matches the stop's expected arrival_time.  This means:
#       - Stop A arriving at 09:00 → forecast for 09:00
#       - Stop B arriving at 13:00 → forecast for 13:00
#       - etc.

#     Caching strategy:
#       - Full hourly payload cached by lat/lon key (one API call per unique location).
#       - Individual hour lookups extracted from the cached payload — no repeat calls.

#     Fallback chain (guaranteed to always return usable data):
#       1. Open-Meteo HTTPS (verify=False for internal proxies)
#       2. Open-Meteo HTTP (plain, bypasses TLS entirely)
#       3. Seasonal estimate (hardcoded India climate by month + latitude band)
#     """

#     WMO_CODES = {
#         0:  ("Clear sky",            "☀️"),
#         1:  ("Mainly clear",         "🌤️"),
#         2:  ("Partly cloudy",        "⛅"),
#         3:  ("Overcast",             "☁️"),
#         45: ("Fog",                  "🌫️"),
#         48: ("Icy fog",              "🌫️"),
#         51: ("Light drizzle",        "🌦️"),
#         53: ("Moderate drizzle",     "🌦️"),
#         55: ("Dense drizzle",        "🌧️"),
#         61: ("Slight rain",          "🌧️"),
#         63: ("Moderate rain",        "🌧️"),
#         65: ("Heavy rain",           "🌧️"),
#         71: ("Slight snow",          "🌨️"),
#         73: ("Moderate snow",        "❄️"),
#         75: ("Heavy snow",           "❄️"),
#         80: ("Slight showers",       "🌦️"),
#         81: ("Moderate showers",     "🌧️"),
#         82: ("Heavy showers",        "⛈️"),
#         95: ("Thunderstorm",         "⛈️"),
#         96: ("Thunderstorm + hail",  "⛈️"),
#         99: ("Severe thunderstorm",  "⛈️"),
#     }
#     ADVERSE_CODES = {63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99}

#     # ── Seasonal fallback (India, month × lat band) ───────────────────────────
#     _SEASONAL_FALLBACK = {
#         1:  {"north": (18, 60, "Mainly clear",  "🌤️", 1),  "south": (28, 65, "Partly cloudy", "⛅", 2)},
#         2:  {"north": (20, 58, "Mainly clear",  "🌤️", 1),  "south": (30, 62, "Mainly clear",  "🌤️", 1)},
#         3:  {"north": (27, 55, "Clear sky",     "☀️", 0),  "south": (33, 60, "Clear sky",     "☀️", 0)},
#         4:  {"north": (33, 50, "Clear sky",     "☀️", 0),  "south": (35, 65, "Clear sky",     "☀️", 0)},
#         5:  {"north": (37, 45, "Clear sky",     "☀️", 0),  "south": (35, 70, "Partly cloudy", "⛅", 2)},
#         6:  {"north": (34, 75, "Moderate rain", "🌧️", 63), "south": (30, 85, "Heavy rain",    "🌧️", 65)},
#         7:  {"north": (30, 82, "Heavy rain",    "🌧️", 65), "south": (28, 88, "Heavy showers", "⛈️", 82)},
#         8:  {"north": (30, 80, "Moderate rain", "🌧️", 63), "south": (28, 86, "Heavy rain",    "🌧️", 65)},
#         9:  {"north": (29, 78, "Slight rain",   "🌦️", 61), "south": (29, 82, "Moderate rain", "🌧️", 63)},
#         10: {"north": (26, 65, "Partly cloudy", "⛅", 2),  "south": (29, 72, "Partly cloudy", "⛅", 2)},
#         11: {"north": (21, 58, "Mainly clear",  "🌤️", 1),  "south": (28, 68, "Mainly clear",  "🌤️", 1)},
#         12: {"north": (16, 62, "Mainly clear",  "🌤️", 1),  "south": (27, 65, "Partly cloudy", "⛅", 2)},
#     }

#     @staticmethod
#     def _cache_key(lat: float, lon: float) -> str:
#         return f"{round(lat, 3)}_{round(lon, 3)}"

#     @classmethod
#     def _seasonal_estimate(cls, lat: float, lon: float, arrival_hour: int = None) -> dict:
#         """Plausible weather estimate based on season, location, and time of day."""
#         import random as _rnd
#         month = datetime.now().month
#         band  = "north" if lat > 20 else "south"
#         row   = cls._SEASONAL_FALLBACK.get(month, cls._SEASONAL_FALLBACK[6])
#         t, h, desc, icon, code = row[band]

#         seed = int(abs(lat * 1000 + lon * 100)) % 100
#         # Diurnal temperature variation: cooler at dawn/late evening
#         hour_offset = 0
#         if arrival_hour is not None:
#             if arrival_hour < 7:    hour_offset = -4
#             elif arrival_hour < 10: hour_offset = -2
#             elif arrival_hour < 14: hour_offset =  2
#             elif arrival_hour < 17: hour_offset =  3
#             elif arrival_hour < 20: hour_offset =  1

#         t    = round(t + hour_offset + (_rnd.Random(seed).random() - 0.5) * 2, 1)
#         wind = round(8  + _rnd.Random(seed + 1).random() * 12, 1)
#         prec = round(_rnd.Random(seed + 2).random() * (5 if code >= 61 else 0.2), 1)
#         vis  = round(6  + _rnd.Random(seed + 3).random() * 4, 1) if code >= 45 else round(9 + _rnd.Random(seed + 3).random() * 5, 1)
#         return {
#             "temperature_c":    t,
#             "wind_speed_kmh":   wind,
#             "precipitation_mm": prec,
#             "humidity_pct":     h,
#             "visibility_km":    vis,
#             "weather_code":     code,
#             "description":      desc,
#             "icon":             icon,
#             "is_adverse":       code in cls.ADVERSE_CODES,
#             "source":           "seasonal-estimate",
#             "forecast_hour":    arrival_hour,
#         }

#     # ── Core: fetch full 48-h hourly payload for a location ──────────────────
#     @classmethod
#     def _fetch_hourly_payload(cls, lat: float, lon: float) -> dict | None:
#         """
#         Fetch 48-hour hourly forecast from Open-Meteo.
#         Returns the raw API response dict, or None on failure.
#         Payload cached in session state under key "hourly_payload_{lat}_{lon}".
#         """
#         cache_key = f"hourly_payload_{cls._cache_key(lat, lon)}"
#         cache     = st.session_state.get("weather_cache", {})
#         if cache_key in cache:
#             return cache[cache_key]

#         params = (
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,"
#             f"precipitation,weather_code,wind_speed_10m,visibility"
#             f"&wind_speed_unit=kmh"
#             f"&timezone=Asia%2FKolkata"
#             f"&forecast_days=2"          # today + tomorrow — covers any same-day route
#         )
#         urls = [
#             "https://api.open-meteo.com/v1/forecast" + params,
#             "http://api.open-meteo.com/v1/forecast"  + params,
#         ]
#         for url in urls:
#             try:
#                 resp = httpx.get(url, timeout=8.0, verify=False, follow_redirects=True)
#                 if resp.status_code != 200:
#                     continue
#                 data = resp.json()
#                 if "hourly" not in data or "time" not in data.get("hourly", {}):
#                     continue
#                 cache[cache_key] = data
#                 st.session_state["weather_cache"] = cache
#                 return data
#             except Exception:
#                 continue
#         return None   # both URLs failed

#     # ── Extract one hour slot from the payload ────────────────────────────────
#     @classmethod
#     def _extract_hour(cls, payload: dict, target_dt: datetime) -> dict:
#         """
#         Given a full hourly payload, extract the slot closest to target_dt.
#         Open-Meteo returns ISO strings like "2025-06-15T09:00" in the "time" array.
#         """
#         hourly     = payload["hourly"]
#         time_strs  = hourly["time"]           # list of "YYYY-MM-DDTHH:00"
#         target_str = target_dt.strftime("%Y-%m-%dT%H:00")

#         # Find exact match first, then nearest
#         idx = None
#         if target_str in time_strs:
#             idx = time_strs.index(target_str)
#         else:
#             # Find closest hour
#             best_diff = float("inf")
#             for i, ts in enumerate(time_strs):
#                 try:
#                     dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
#                     diff = abs((dt - target_dt).total_seconds())
#                     if diff < best_diff:
#                         best_diff = diff
#                         idx = i
#                 except Exception:
#                     continue

#         if idx is None:
#             return {}

#         def _val(key, default=0):
#             arr = hourly.get(key, [])
#             v   = arr[idx] if idx < len(arr) else default
#             return v if v is not None else default

#         code       = int(_val("weather_code", 0))
#         desc, icon = cls.WMO_CODES.get(code, ("Partly cloudy", "⛅"))
#         vis_raw    = _val("visibility", 10000)

#         return {
#             "temperature_c":        round(float(_val("temperature_2m",          25)), 1),
#             "wind_speed_kmh":       round(float(_val("wind_speed_10m",          10)), 1),
#             "precipitation_mm":     round(float(_val("precipitation",             0)), 1),
#             "precip_probability":   int(_val("precipitation_probability",          0)),
#             "humidity_pct":         int(_val("relative_humidity_2m",             65)),
#             "visibility_km":        round(float(vis_raw) / 1000, 1),
#             "weather_code":         code,
#             "description":          desc,
#             "icon":                 icon,
#             "is_adverse":           code in cls.ADVERSE_CODES,
#             "source":               "open-meteo-hourly",
#         }

#     # ── Public API: fetch weather at a specific arrival time ─────────────────
#     @classmethod
#     def fetch_at_time(cls, lat: float, lon: float, arrival_time_str: str,
#                       itin_date_str: str = None) -> dict:
#         """
#         Return weather forecast for (lat, lon) at the stop's arrival time.

#         Args:
#             lat, lon:           Stop coordinates.
#             arrival_time_str:   "HH:MM" string from the itinerary stop.
#             itin_date_str:      "YYYY-MM-DD" from itin["date"]; defaults to today.

#         Returns a weather dict with an extra "forecast_time" field showing
#         the exact datetime we fetched for.
#         """
#         # Parse target datetime
#         try:
#             date_str = itin_date_str or datetime.now().strftime("%Y-%m-%d")
#             target   = datetime.strptime(f"{date_str} {arrival_time_str}", "%Y-%m-%d %H:%M")
#         except Exception:
#             target   = datetime.now()

#         # Per-stop cache key (includes the target hour)
#         stop_key  = f"stop_{cls._cache_key(lat, lon)}_{target.strftime('%Y%m%d%H')}"
#         cache     = st.session_state.get("weather_cache", {})
#         if stop_key in cache:
#             return cache[stop_key]

#         # Fetch (or reuse) the hourly payload for this location
#         payload = cls._fetch_hourly_payload(lat, lon)

#         if payload:
#             result = cls._extract_hour(payload, target)
#             if result:
#                 result["forecast_time"]   = target.strftime("%d %b %Y · %H:%M")
#                 result["arrival_time_str"] = arrival_time_str
#                 cache[stop_key] = result
#                 st.session_state["weather_cache"] = cache
#                 return result

#         # Fallback to seasonal estimate
#         result = cls._seasonal_estimate(lat, lon, arrival_hour=target.hour)
#         result["forecast_time"]   = target.strftime("%d %b %Y · %H:%M") + " (est.)"
#         result["arrival_time_str"] = arrival_time_str
#         cache[stop_key] = result
#         st.session_state["weather_cache"] = cache
#         return result

#     # ── Convenience: fetch for all stops in an itinerary ─────────────────────
#     @classmethod
#     def fetch_route_at_times(cls, stops_with_coords: list, itin_date_str: str = None) -> list:
#         """
#         Fetch arrival-time-aware forecast for every stop.

#         Args:
#             stops_with_coords: list of dicts with at minimum:
#                 {stop_id, location_name, lat, lon, arrival_time (HH:MM)}
#             itin_date_str: itinerary date "YYYY-MM-DD"

#         Returns: same list with an added "weather" key per stop.
#         """
#         results = []
#         for s in stops_with_coords:
#             arrival = s.get("arrival_time", "08:00") or "08:00"
#             w = cls.fetch_at_time(s["lat"], s["lon"], arrival, itin_date_str)
#             results.append({**s, "weather": w})
#         return results

#     # ── Legacy: fetch current weather (kept for backward compat) ─────────────
#     @classmethod
#     def fetch(cls, lat: float, lon: float) -> dict:
#         """Fetch current conditions (no arrival time). Used as generic fallback."""
#         return cls.fetch_at_time(lat, lon, datetime.now().strftime("%H:%M"))

#     @classmethod
#     def fetch_route(cls, stops_with_coords: list) -> list:
#         """Legacy: fetch current conditions for a list of stops."""
#         return cls.fetch_route_at_times(stops_with_coords)


# # ─────────────────────────────────────────────
# # FUEL ENGINE  (static reference prices — India)
# # ─────────────────────────────────────────────
# class FuelEngine:
#     """
#     Static Indian petrol/diesel reference prices (as of June 2025).
#     Vehicle efficiency lookup table.
#     Fuel savings comparisons between optimized vs un-optimized route order.

#     Reference: https://www.goodreturns.in/petrol-price.html
#     Prices in ₹ per litre. Update CITY_PRICES dict as needed.
#     """

#     # ── Static city-level prices (₹/litre) ───────────────────────────────────
#     CITY_PRICES = {
#         # Maharashtra
#         "Mumbai":     {"petrol": 103.44, "diesel": 89.97},
#         "Pune":       {"petrol": 103.57, "diesel": 90.10},
#         "Nashik":     {"petrol": 103.20, "diesel": 89.75},
#         "Nagpur":     {"petrol": 103.80, "diesel": 90.15},
#         "Aurangabad": {"petrol": 103.35, "diesel": 89.85},
#         # Delhi NCR
#         "Delhi":      {"petrol": 94.72,  "diesel": 87.62},
#         "Gurgaon":    {"petrol": 95.10,  "diesel": 88.05},
#         "Noida":      {"petrol": 94.85,  "diesel": 87.80},
#         # Karnataka
#         "Bengaluru":  {"petrol": 102.86, "diesel": 88.94},
#         "Mysuru":     {"petrol": 102.60, "diesel": 88.70},
#         # Tamil Nadu
#         "Chennai":    {"petrol": 100.75, "diesel": 92.34},
#         "Coimbatore": {"petrol": 100.50, "diesel": 92.10},
#         # Telangana / AP
#         "Hyderabad":  {"petrol": 107.41, "diesel": 95.65},
#         "Visakhapatnam": {"petrol": 106.80, "diesel": 95.20},
#         # Gujarat
#         "Ahmedabad":  {"petrol": 96.63,  "diesel": 92.38},
#         "Surat":      {"petrol": 96.50,  "diesel": 92.25},
#         # Rajasthan
#         "Jaipur":     {"petrol": 104.88, "diesel": 90.36},
#         # Uttar Pradesh
#         "Lucknow":    {"petrol": 94.65,  "diesel": 87.76},
#         "Kanpur":     {"petrol": 94.55,  "diesel": 87.65},
#         # West Bengal
#         "Kolkata":    {"petrol": 103.94, "diesel": 90.76},
#         # Default (national average)
#         "default":    {"petrol": 101.50, "diesel": 90.00},
#     }

#     # ── Vehicle fuel efficiency (km per litre) ────────────────────────────────
#     VEHICLE_EFFICIENCY = {
#         "Truck":       5.5,   # heavy truck (10-16T)
#         "Van":         12.0,  # light commercial van
#         "Tempo":       9.0,   # mini-truck / tempo
#         "Car":         15.0,  # passenger car
#         "Motorcycle":  40.0,  # two-wheeler
#         "default":     8.0,
#     }

#     # ── Fuel type by vehicle ──────────────────────────────────────────────────
#     VEHICLE_FUEL_TYPE = {
#         "Truck": "diesel", "Tempo": "diesel",
#         "Van": "diesel", "Car": "petrol", "Motorcycle": "petrol",
#     }

#     @classmethod
#     def get_price(cls, city: str, fuel_type: str = "diesel") -> float:
#         """Find the closest city match (case-insensitive substring)."""
#         city_lower = city.lower()
#         for name, prices in cls.CITY_PRICES.items():
#             if name.lower() in city_lower or city_lower in name.lower():
#                 return prices.get(fuel_type, prices["diesel"])
#         return cls.CITY_PRICES["default"].get(fuel_type, 90.0)

#     @classmethod
#     def compute_fuel_cost(
#         cls,
#         distance_km: float,
#         vehicle_type: str,
#         city: str = "Mumbai",
#         override_price: float = None,
#         override_efficiency: float = None,
#     ) -> dict:
#         """
#         Compute fuel cost for a given distance.

#         Returns:
#             {litres_consumed, price_per_litre, total_cost_inr,
#              efficiency_kmpl, fuel_type, city}
#         """
#         fuel_type  = cls.VEHICLE_FUEL_TYPE.get(vehicle_type, "diesel")
#         efficiency = override_efficiency or cls.VEHICLE_EFFICIENCY.get(vehicle_type, cls.VEHICLE_EFFICIENCY["default"])
#         price      = override_price      or cls.get_price(city, fuel_type)
#         litres     = distance_km / efficiency if efficiency > 0 else 0
#         cost       = round(litres * price, 2)
#         return {
#             "litres_consumed":  round(litres, 2),
#             "price_per_litre":  price,
#             "total_cost_inr":   cost,
#             "efficiency_kmpl":  efficiency,
#             "fuel_type":        fuel_type,
#             "city":             city,
#         }

#     @classmethod
#     def savings_analysis(
#         cls,
#         stops_list: list,
#         vehicle_type: str,
#         city: str = "Mumbai",
#         override_price: float = None,
#         override_efficiency: float = None,
#     ) -> dict:
#         """
#         Compare un-optimized (original stop order) vs NN-optimized order.

#         Returns:
#             {
#                 original_km, optimized_km, saved_km, saving_pct,
#                 original_cost, optimized_cost, saved_cost_inr,
#                 original_order, optimized_order,
#                 fuel_detail_original, fuel_detail_optimized,
#                 per_stop_savings: [{stop, original_leg_km, optimized_leg_km}]
#             }
#         """
#         if len(stops_list) < 2:
#             return {}

#         coords = [(s["lat"], s["lon"]) for s in stops_list]

#         # Original order — OSRM
#         osrm_orig = MLEngine.osrm_route(coords)
#         orig_km   = osrm_orig["total_distance_km"]

#         # NN-optimized order — OSRM
#         nn_order  = MLEngine().__class__().nearest_neighbor_route.__func__(MLEngine(), coords)
#         coords_nn = [coords[i] for i in nn_order]
#         osrm_opt  = MLEngine.osrm_route(coords_nn)
#         opt_km    = osrm_opt["total_distance_km"]

#         # Use optimized if it's actually better, else keep original
#         if opt_km >= orig_km:
#             opt_km    = orig_km
#             osrm_opt  = osrm_orig
#             nn_order  = list(range(len(stops_list)))

#         saved_km   = round(orig_km - opt_km, 2)
#         saving_pct = round(saved_km / orig_km * 100, 1) if orig_km > 0 else 0

#         fuel_orig = cls.compute_fuel_cost(orig_km, vehicle_type, city, override_price, override_efficiency)
#         fuel_opt  = cls.compute_fuel_cost(opt_km,  vehicle_type, city, override_price, override_efficiency)
#         saved_cost = round(fuel_orig["total_cost_inr"] - fuel_opt["total_cost_inr"], 2)

#         # Per-leg comparison (zip original vs optimized legs)
#         per_stop = []
#         orig_legs = osrm_orig.get("legs", [])
#         opt_legs  = osrm_opt.get("legs", [])
#         for i, s in enumerate(stops_list[1:]):
#             orig_leg = orig_legs[i]["distance_km"] if i < len(orig_legs) else 0
#             opt_s    = stops_list[nn_order[i+1]] if (i+1) < len(nn_order) else s
#             opt_leg  = opt_legs[i]["distance_km"] if i < len(opt_legs) else 0
#             per_stop.append({
#                 "leg":           i + 1,
#                 "original_stop": s["location_name"],
#                 "optimized_stop":opt_s["location_name"],
#                 "original_km":   orig_leg,
#                 "optimized_km":  opt_leg,
#                 "saved_km":      round(orig_leg - opt_leg, 2),
#             })

#         return {
#             "original_km":          orig_km,
#             "optimized_km":         opt_km,
#             "saved_km":             saved_km,
#             "saving_pct":           saving_pct,
#             "original_cost_inr":    fuel_orig["total_cost_inr"],
#             "optimized_cost_inr":   fuel_opt["total_cost_inr"],
#             "saved_cost_inr":       saved_cost,
#             "fuel_detail_original": fuel_orig,
#             "fuel_detail_optimized":fuel_opt,
#             "original_order":       [s["location_name"] for s in stops_list],
#             "optimized_order":      [stops_list[i]["location_name"] for i in nn_order],
#             "per_stop_savings":     per_stop,
#             "routing_source":       osrm_orig["source"],
#         }


# # ─────────────────────────────────────────────
# # RAG ENGINE
# # ─────────────────────────────────────────────
# class RAGEngine:
#     DOMAIN_TOPICS = [
#         # ── Core logistics ───────────────────────────────────────────────────
#         "route", "routing", "delivery", "logistics", "shipment", "freight",
#         "itinerary", "stop", "waypoint", "depot", "warehouse", "pickup",
#         "dispatch", "fleet", "vehicle", "driver", "trucking", "transport",
#         "cargo", "customs", "e-way bill", "manifest", "consignment",
#         "last mile", "first mile", "supply chain", "distribution",
#         "fuel", "mileage", "navigation", "gps", "tracking",
#         "time window", "schedule", "delay", "on-time", "eta", "arrival",
#         "temperature", "cold chain", "refrigerated", "hazmat", "dangerous goods",
#         "pod", "proof of delivery", "invoice", "bill of lading",
#         "kpi", "performance", "efficiency", "cost", "optimization",
#         "travel", "distance", "trip", "journey", "road", "highway",
#         "rail", "air freight", "sea freight", "port", "airport",
#         "tms", "wms", "erp", "telematics", "iot",
#         "breakdown", "insurance", "claim", "compliance", "regulation",
#         # ── Natural navigation / direction language ───────────────────────────
#         # Verbs people use when asking about getting from A to B
#         "go from", "going from", "get from", "getting from",
#         "travel from", "travelling from", "traveling from",
#         "reach", "reaching", "how to reach", "how do i reach",
#         "drive from", "driving from", "ride from", "riding from",
#         "commute", "commuting",
#         "best way", "fastest way", "shortest way", "quickest way",
#         "how long", "how far", "how much time",
#         "directions", "direction", "navigate", "path from", "path to",
#         "way to", "way from", "route from", "route to",
#         "from here", "to here",
#         # ── Relational / between ─────────────────────────────────────────────
#         "between",
#         # ── Movement / transit words ─────────────────────────────────────────
#         "bus", "train", "metro", "cab", "auto", "taxi", "uber", "ola",
#         "toll", "highway", "expressway", "flyover", "bridge",
#         "traffic", "congestion", "jam", "detour", "bypass",
#         "drop", "pick up", "pickup point", "drop off",
#         # ── Indian city / area names (common logistics hubs) ─────────────────
#         # Mumbai
#         "mumbai", "bombay", "bandra", "kurla", "andheri", "dadar",
#         "thane", "navi mumbai", "panvel", "borivali", "kandivali",
#         "malad", "goregaon", "jogeshwari", "vile parle", "santacruz",
#         "bkc", "nariman", "churchgate", "csmt", "colaba", "worli",
#         "lower parel", "prabhadevi", "matunga", "sion", "chembur",
#         "ghatkopar", "vikhroli", "kanjurmarg", "bhandup", "mulund",
#         "dombivli", "kalyan", "bhiwandi", "vasai", "virar", "mira road",
#         "nhava sheva", "jnpt", "nhava",
#         # Pune
#         "pune", "pimpri", "chinchwad", "hadapsar", "kothrud", "hinjewadi",
#         "wakad", "baner", "aundh", "shivajinagar", "talegaon",
#         # Other major cities
#         "delhi", "ncr", "gurgaon", "noida", "faridabad", "ghaziabad",
#         "bengaluru", "bangalore", "whitefield", "electronic city",
#         "hyderabad", "secunderabad", "cyberabad",
#         "chennai", "kolkata", "ahmedabad", "surat", "jaipur",
#         "lucknow", "chandigarh", "coimbatore", "kochi", "indore",
#         "nagpur", "nashik", "aurangabad", "visakhapatnam",
#         # ── Logistics / geographic terms ─────────────────────────────────────
#         "zone", "area", "sector", "block", "lane", "street", "nagar",
#         "colony", "society", "industrial area", "industrial estate",
#         "cargo hub", "logistics park", "cold storage", "godown",
#         "weighbridge", "octroi", "rto", "check post", "border",
#     ]
#     RELEVANCE_THRESHOLD = 0.30

#     # Regex patterns that strongly indicate a route / navigation question
#     # regardless of specific keywords — catches "from X to Y" style queries
#     _NAV_PATTERNS = [
#         r"\bfrom\b.{1,60}\bto\b",        # "from bandra to kurla"
#         r"\bgo\b.{0,40}\bto\b",           # "go to andheri"
#         r"\bget\b.{0,40}\bto\b",          # "get to the depot"
#         r"\bread?ch\b",                      # "reach" / "reaching"
#         r"\bdriv(e|ing)\b.{0,40}\bto\b",  # "drive to"
#         r"\bhow\b.{0,30}\blong\b",        # "how long does it take"
#         r"\bhow\b.{0,30}\bfar\b",         # "how far is X from Y"
#         r"\bdir?ections?\b",                 # "directions" / "direction"
#         r"\bnear(est)?\b",                   # "nearest depot"
#         r"\bway\b.{0,30}\bto\b",          # "best way to reach"
#         r"\broute\b.{0,30}\bfrom\b",      # "route from X"
#         r"\bpath\b.{0,30}\bto\b",         # "path to warehouse"
#     ]

#     def __init__(self, llm, embeddings):
#         self.llm       = llm
#         self.embeddings = embeddings
#         self._vectordb  = None

#     def _build_vectordb(self, kb_df):
#         if self._vectordb is not None or self.embeddings is None:
#             return
#         raw_docs, metadatas = [], []
#         for _, row in kb_df.iterrows():
#             raw_docs.append(f"[{row['category']} — {row['topic']}]\n{row['content']}")
#             metadatas.append({"category": row["category"], "topic": row["topic"]})
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
#         )
#         chunks, chunk_metas = [], []
#         for doc, meta in zip(raw_docs, metadatas):
#             parts = splitter.split_text(doc)
#             chunks.extend(parts)
#             chunk_metas.extend([meta] * len(parts))
#         try:
#             self._vectordb = Chroma.from_texts(
#                 chunks, self.embeddings, metadatas=chunk_metas, persist_directory="./chroma_kb"
#             )
#         except Exception:
#             self._vectordb = None

#     def _retrieve(self, question, k=5):
#         if self._vectordb is None:
#             return [], [], []
#         try:
#             results = self._vectordb.similarity_search_with_relevance_scores(question, k=k)
#             docs, scores, metas = [], [], []
#             for doc, score in results:
#                 docs.append(doc.page_content)
#                 scores.append(score)
#                 metas.append(doc.metadata)
#             return docs, scores, metas
#         except Exception:
#             return [], [], []

#     def _is_in_domain(self, question, chunks, scores):
#         """
#         Three-layer domain check — passes if ANY layer matches.

#         Layer 1 — Keyword scan: checks against expanded DOMAIN_TOPICS list.
#                   Includes natural language navigation phrases, city names,
#                   and movement verbs so "go from bandra to kurla" passes.

#         Layer 2 — Regex navigation intent: pattern-matches spatial/directional
#                   phrasing like "from X to Y", "how far", "nearest", "directions".
#                   Catches questions that contain no logistics jargon but are
#                   clearly asking about routes or locations.

#         Layer 3 — Semantic similarity: at least one retrieved KB chunk must
#                   score >= RELEVANCE_THRESHOLD (only active when embeddings online).
#         """
#         import re
#         q = question.lower()

#         # Layer 1: keyword scan
#         if any(kw in q for kw in self.DOMAIN_TOPICS):
#             return True

#         # Layer 2: navigation intent regex
#         for pattern in self._NAV_PATTERNS:
#             if re.search(pattern, q):
#                 return True

#         # Layer 3: semantic similarity
#         if scores and max(scores) >= self.RELEVANCE_THRESHOLD:
#             return True

#         return False

#     def _keyword_fallback(self, question, kb_df):
#         q = question.lower()
#         for _, row in kb_df.iterrows():
#             words = row["topic"].lower().split() + row["category"].lower().split()
#             if any(w in q for w in words):
#                 return f"**📖 {row['topic']}** *(from {row['category']} KB)*\n\n{row['content']}"
#         return (
#             "I couldn't find a matching article. "
#             "Please ask about routing, delivery, customs, fleet, or other logistics topics."
#         )

#     def answer(self, question, kb_df):
#         if not kb_df.empty and self.embeddings is not None and self._vectordb is None:
#             with st.spinner("📚 Indexing knowledge base…"):
#                 self._build_vectordb(kb_df)

#         chunks, scores, metas = self._retrieve(question, k=5)

#         if not self._is_in_domain(question, chunks, scores):
#             return {
#                 "text": (
#                     "⛔ **Out of scope** — I'm RouteIQ Assistant, specialised exclusively "
#                     "in **travel and logistics** topics.\n\n"
#                     "I can help with route planning, delivery schedules, customs clearance, "
#                     "fleet management, fuel costs, cargo compliance, and related subjects."
#                 ),
#                 "sources": [], "grounded": False, "rejected": True,
#             }

#         if self.llm is None:
#             return {"text": self._keyword_fallback(question, kb_df),
#                     "sources": [], "grounded": False, "rejected": False}

#         relevant = [(c, s, m) for c, s, m in zip(chunks, scores, metas) if s >= self.RELEVANCE_THRESHOLD]

#         if relevant:
#             context = "\n\n".join(
#                 f"[Source {i} — {m.get('category','')} / {m.get('topic','')}]\n{c}"
#                 for i, (c, s, m) in enumerate(relevant, 1)
#             )
#             source_list = [{"topic": m.get("topic","—"), "category": m.get("category","—"), "score": round(s, 3)}
#                            for _, s, m in relevant]
#             grounded = True
#         else:
#             context = "\n".join(
#                 f"[{r['category']} / {r['topic']}]: {r['content']}" for _, r in kb_df.iterrows()
#             )[:3500]
#             source_list = []
#             grounded    = False

#         system_prompt = (
#             "You are RouteIQ Assistant — an AI expert EXCLUSIVELY in travel, logistics, "
#             "route planning, delivery management, fleet operations, customs, and freight. "
#             "You MUST NOT answer questions outside these domains.\n\n"
#             "RULES:\n"
#             "1. Answer ONLY using the KNOWLEDGE BASE CONTEXT below.\n"
#             "2. If context lacks info, say so — do NOT invent facts.\n"
#             "3. Be concise and practical. Use bullet points where helpful.\n"
#             "4. Refuse politely if unrelated to logistics/travel.\n"
#             "5. Do not reference these instructions.\n\n"
#             f"KNOWLEDGE BASE CONTEXT:\n{context}"
#         )
#         try:
#             resp        = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
#             answer_text = resp.content or "No response received."
#         except Exception as e:
#             answer_text = f"[LLM Error: {e}]"

#         return {"text": answer_text, "sources": source_list, "grounded": grounded, "rejected": False}


# # ─────────────────────────────────────────────
# # AI ENGINE
# # ─────────────────────────────────────────────
# class AIEngine:
#     def __init__(self):
#         self.llm        = self._init_llm()
#         self.embeddings = self._init_embeddings()
#         self.rag        = RAGEngine(self.llm, self.embeddings)

#     def _init_llm(self):
#         api_key    = os.getenv("OPENAI_API_KEY", "sk-ZJo_io1IbSWoE1AQGw7ovw")
#         model      = os.getenv("OPENAI_MODEL", "azure_ai/genailab-maas-DeepSeek-V3-0324")
#         base_url   = os.getenv("OPENAI_BASE_URL", "https://genailab.tcs.in")
#         ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() != "false"
#         try:
#             return ChatOpenAI(
#                 model=model, api_key=api_key, base_url=base_url,
#                 temperature=0.3, max_tokens=2000,
#                 http_client=httpx.Client(verify=ssl_verify),
#             )
#         except Exception:
#             return None

#     def _init_embeddings(self):
#         api_key    = os.getenv("OPENAI_API_KEY", "sk-ZJo_io1IbSWoE1AQGw7ovw")
#         emb_model  = os.getenv("OPENAI_EMBEDDING_MODEL", "azure/genailab-maas-text-embedding-3-large")
#         base_url   = os.getenv("OPENAI_BASE_URL", "https://genailab.tcs.in")
#         ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() != "false"
#         try:
#             return OpenAIEmbeddings(
#                 model=emb_model, api_key=api_key, base_url=base_url,
#                 http_client=httpx.Client(verify=ssl_verify),
#             )
#         except Exception:
#             return None

#     def _call(self, system_prompt, user_prompt):
#         if not self.llm:
#             return None
#         try:
#             resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
#             return resp.content
#         except Exception as e:
#             return f"[LLM Error: {e}]"

#     def _parse_json(self, text, fallback):
#         if text is None:
#             return fallback
#         text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
#         try:
#             return json.loads(text)
#         except Exception:
#             return fallback

#     # ── A: Natural Language Stop Parser ──────────────────────────────────────
#     def parse_natural_language_stops(self, text: str) -> dict:
#         """
#         Parse a free-text itinerary description into structured stops + constraints.
#         Returns: {stops, constraints, parse_notes, error}
#         """
#         system_prompt = (
#             "You are a logistics data extraction agent. "
#             "Extract structured stop and constraint information from a natural language description. "
#             "Use realistic Indian GPS coordinates if none are given. "
#             "Return ONLY valid JSON — no markdown, no extra text."
#         )
#         user_prompt = (
#             f'Extract all stops and constraints from:\n\n"{text}"\n\n'
#             "Return this JSON schema exactly:\n"
#             "{\n"
#             '  "stops": [\n'
#             "    {\n"
#             '      "stop_id": "NL1",\n'
#             '      "location_name": "full location name",\n'
#             '      "lat": <realistic latitude>,\n'
#             '      "lon": <realistic longitude>,\n'
#             '      "stop_type": "Delivery|Pickup|Meeting|Warehouse|Customs|Rest",\n'
#             '      "time_window_start": "HH:MM",\n'
#             '      "time_window_end": "HH:MM",\n'
#             '      "priority": "High|Medium|Low",\n'
#             '      "notes": "any special instructions"\n'
#             "    }\n"
#             "  ],\n"
#             '  "constraints": {\n'
#             '    "driver_name": "string or Unknown",\n'
#             '    "start_time": "HH:MM",\n'
#             '    "transport_mode": "Road|Rail|Air|Sea",\n'
#             '    "vehicle_type": "Truck|Van|Motorcycle|Car|Tempo",\n'
#             '    "max_hours": <number>,\n'
#             '    "vehicle_capacity_kg": <number>\n'
#             "  },\n"
#             '  "parse_notes": "brief summary of what was understood",\n'
#             '  "error": null\n'
#             "}\n\n"
#             "Inference rules:\n"
#             "- pick up/collect → Pickup; deliver/drop → Delivery; meeting/client → Meeting; "
#             "rest/break → Rest; customs/checkpoint → Customs; warehouse/depot → Warehouse\n"
#             "- urgent/asap/critical → High priority; default → Medium\n"
#             "- 'by 2pm' → time_window_end 14:00; 'at 9am' → time_window_start 09:00\n"
#             "- Default start_time 08:00, transport_mode Road if not mentioned\n"
#             "- stop_id values: NL1, NL2, NL3 in order"
#         )
#         raw    = self._call(system_prompt, user_prompt)
#         result = self._parse_json(raw, {"stops": [], "constraints": {}, "parse_notes": "",
#                                         "error": "LLM offline or parse failed"})
#         if "stops"       not in result: result["stops"]       = []
#         if "constraints" not in result: result["constraints"] = {}
#         if "error"       not in result: result["error"]       = None
#         return result

#     # ── B: Itinerary Generation ───────────────────────────────────────────────
#     def generate_itinerary(self, stops_list, constraints, route_context):
#         """Generate optimized itinerary JSON, then patch in real OSRM distances."""
#         system_prompt = (
#             "You are an expert logistics route planner. "
#             "Generate an optimized, feasible delivery itinerary as structured JSON. "
#             "Prioritize: (1) time window compliance, (2) High-priority stops first, "
#             "(3) shortest total distance. "
#             "Times in HH:MM 24-hour format. Durations in minutes. "
#             "Return ONLY valid JSON — no markdown."
#         )
#         user_prompt = (
#             f"CONSTRAINTS:\n{json.dumps(constraints, indent=2)}\n\n"
#             f"ROUTE CONTEXT (use these distances/times):\n{route_context}\n\n"
#             f"STOPS TO PLAN ({len(stops_list)}):\n{json.dumps(stops_list, indent=2, default=str)}\n\n"
#             "Return JSON with keys: itinerary_title, driver, vehicle, date (YYYY-MM-DD), "
#             "transport_mode, total_distance_km, total_duration_min, estimated_fuel_cost_inr, "
#             "optimization_notes, warnings (list), efficiency_score (0-100), "
#             "on_time_probability (0-100), and stops array where each stop has: "
#             "sequence, stop_id, location_name, arrival_time (HH:MM), departure_time (HH:MM), "
#             "service_duration_min, travel_time_from_prev_min, distance_from_prev_km, "
#             "stop_type, priority, status, notes, risk_flag (None/Low/Medium/High), "
#             "risk_reason, time_window_start (HH:MM), time_window_end (HH:MM)."
#         )
#         raw      = self._call(system_prompt, user_prompt)
#         result   = self._parse_json(raw, self._rule_based_itinerary(stops_list, constraints))

#         # ── Patch real OSRM leg data ─────────────────────────────────────────
#         coords     = [(s["lat"], s["lon"]) for s in stops_list]
#         osrm       = MLEngine.osrm_route(coords)
#         result["routing_source"] = osrm["source"]
#         itin_stops = result.get("stops", [])

#         if osrm["source"] == "osrm" and len(osrm["legs"]) >= len(itin_stops) - 1 and len(itin_stops) > 1:
#             for i, stop in enumerate(itin_stops):
#                 if i == 0:
#                     continue
#                 leg_idx = i - 1
#                 if leg_idx < len(osrm["legs"]):
#                     stop["distance_from_prev_km"]     = osrm["legs"][leg_idx]["distance_km"]
#                     stop["travel_time_from_prev_min"]  = osrm["legs"][leg_idx]["duration_min"]
#             result["total_distance_km"]  = osrm["total_distance_km"]
#             result["total_duration_min"] = round(
#                 osrm["total_duration_min"]
#                 + sum(s.get("service_duration_min", 0) for s in itin_stops), 1
#             )
#             result["estimated_fuel_cost_inr"] = round(osrm["total_distance_km"] * 8, 0)

#         return result

#     def _rule_based_itinerary(self, stops_list, constraints):
#         priority_order = {"High": 0, "Medium": 1, "Low": 2}
#         sorted_stops   = sorted(stops_list, key=lambda s: (
#             priority_order.get(s.get("priority", "Low"), 2),
#             s.get("time_window_start", "23:59"),
#         ))
#         current_time = datetime.strptime(constraints.get("start_time", "08:00"), "%H:%M")
#         result_stops, total_dist = [], 0
#         prev_lat = stops_list[0]["lat"] if stops_list else 19.076
#         prev_lon = stops_list[0]["lon"] if stops_list else 72.877

#         for i, s in enumerate(sorted_stops):
#             dist        = MLEngine.compute_distance_km(prev_lat, prev_lon, s["lat"], s["lon"])
#             travel_min  = max(5, int(dist / 35 * 60))
#             service_min = {"Delivery": 20, "Pickup": 15, "Meeting": 45,
#                            "Warehouse": 30, "Customs": 60, "Rest": 20}.get(s.get("stop_type", "Delivery"), 20)
#             arrival     = current_time + timedelta(minutes=travel_min)
#             departure   = arrival + timedelta(minutes=service_min)
#             result_stops.append({
#                 "sequence": i + 1, "stop_id": s.get("stop_id", f"S{i+1}"),
#                 "location_name": s.get("location_name", "Stop"),
#                 "arrival_time": arrival.strftime("%H:%M"),
#                 "departure_time": departure.strftime("%H:%M"),
#                 "service_duration_min": service_min,
#                 "travel_time_from_prev_min": travel_min,
#                 "distance_from_prev_km": round(dist, 2),
#                 "stop_type": s.get("stop_type", "Delivery"),
#                 "priority": s.get("priority", "Medium"),
#                 "status": "Scheduled", "notes": s.get("notes", ""),
#                 "risk_flag": "None", "risk_reason": "",
#                 "time_window_start": s.get("time_window_start", "08:00"),
#                 "time_window_end":   s.get("time_window_end",   "18:00"),
#             })
#             total_dist += dist
#             current_time = departure
#             prev_lat, prev_lon = s["lat"], s["lon"]

#         return {
#             "itinerary_title": "Optimized Route (Rule-based fallback)",
#             "driver": constraints.get("driver_name", "Driver"),
#             "vehicle": constraints.get("vehicle_type", "Truck"),
#             "date": datetime.now().strftime("%Y-%m-%d"),
#             "transport_mode": constraints.get("transport_mode", "Road"),
#             "total_distance_km": round(total_dist, 2),
#             "total_duration_min": sum(
#                 s["travel_time_from_prev_min"] + s["service_duration_min"] for s in result_stops
#             ),
#             "estimated_fuel_cost_inr": round(total_dist * 8, 0),
#             "stops": result_stops,
#             "optimization_notes": "Rule-based fallback: sorted by priority then time window.",
#             "warnings": ["LLM offline — using rule-based optimizer"],
#             "efficiency_score": 72, "on_time_probability": 78,
#         }

#     def adjust_itinerary(self, existing_itinerary, change_request):
#         system_prompt = (
#             "You are a logistics re-planning agent. "
#             "Return a FULLY updated itinerary JSON in the same schema. "
#             "Return ONLY valid JSON — no markdown."
#         )
#         user_prompt = (
#             f"EXISTING ITINERARY:\n{json.dumps(existing_itinerary, indent=2, default=str)}\n\n"
#             f"CHANGE REQUEST:\n{change_request}\n\n"
#             "Adjust all affected stop times. Explain the change in optimization_notes."
#         )
#         return self._parse_json(self._call(system_prompt, user_prompt), existing_itinerary)

#     # ── C: Constraint Violation Checker ──────────────────────────────────────
#     def check_violations(self, itinerary: dict, constraints: dict) -> list:
#         """
#         Check all stops for time-window breaches and driver hour overruns.
#         Returns a list of violation dicts: {sequence, location_name, type, severity, detail}
#         """
#         violations = []
#         stops      = sorted(itinerary.get("stops", []), key=lambda s: s["sequence"])
#         if not stops:
#             return violations

#         date_str  = itinerary.get("date", datetime.now().strftime("%Y-%m-%d"))
#         try:
#             base = datetime.strptime(date_str, "%Y-%m-%d")
#         except Exception:
#             base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

#         def to_dt(hhmm):
#             try:
#                 h, m = map(int, str(hhmm).strip().split(":"))
#                 return base.replace(hour=h, minute=m, second=0, microsecond=0)
#             except Exception:
#                 return base

#         max_hours = constraints.get("max_hours", 10)
#         day_end   = to_dt(constraints.get("start_time", "08:00")) + timedelta(hours=max_hours)

#         for s in stops:
#             arr          = to_dt(s.get("arrival_time",   "00:00"))
#             dep          = to_dt(s.get("departure_time",  "00:00"))
#             tw_start_raw = s.get("time_window_start", "")
#             tw_end_raw   = s.get("time_window_end",   "")

#             if tw_start_raw and tw_end_raw:
#                 tw_start = to_dt(tw_start_raw)
#                 tw_end   = to_dt(tw_end_raw)
#                 if arr < tw_start:
#                     wait = int((tw_start - arr).seconds / 60)
#                     violations.append({
#                         "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                         "type": "time_window", "severity": "Warning",
#                         "detail": (
#                             f"Arrives at {s.get('arrival_time')} but window opens at "
#                             f"{tw_start_raw}. Driver waits {wait} min."
#                         ),
#                     })
#                 elif arr > tw_end:
#                     late = int((arr - tw_end).seconds / 60)
#                     violations.append({
#                         "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                         "type": "time_window", "severity": "Critical",
#                         "detail": (
#                             f"Arrives at {s.get('arrival_time')} — window closed at "
#                             f"{tw_end_raw}. Late by {late} min. SLA breach."
#                         ),
#                     })

#             if dep > day_end:
#                 over = int((dep - day_end).seconds / 60)
#                 violations.append({
#                     "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                     "type": "driver_hours", "severity": "Critical",
#                     "detail": (
#                         f"Departure at {s.get('departure_time')} exceeds "
#                         f"{max_hours}h limit by {over} min."
#                     ),
#                 })

#         return violations

#     def get_kb_answer(self, question, kb_df):
#         return self.rag.answer(question, kb_df)

#     def analyze_route_performance(self, route_df, stops_df):
#         summary = {
#             "total_routes":      route_df["route_id"].nunique() if not route_df.empty else 0,
#             "avg_distance_km":   round(route_df["distance_km"].mean(), 1) if not route_df.empty else 0,
#             "avg_on_time_pct":   round(100 * (stops_df["status"] == "On Time").mean(), 1) if not stops_df.empty else 0,
#             "top_delay_reasons": stops_df["delay_reason"].value_counts().head(3).to_dict()
#                                   if "delay_reason" in stops_df.columns else {},
#         }
#         raw = self._call(
#             "You are a logistics analytics expert. Provide 3-5 concise actionable insights.",
#             f"Performance data:\n{json.dumps(summary, indent=2)}\n\nBullet point insights:"
#         )
#         if not raw or raw.startswith("[LLM"):
#             return (
#                 "• Review high-delay routes for recurring traffic patterns\n"
#                 "• Adjust time windows for stops that are consistently late\n"
#                 "• Prioritize High-priority stops in morning slots\n"
#                 "• Consolidate nearby stops to reduce total distance and fuel cost"
#             )
#         return raw


# # ─────────────────────────────────────────────
# # DASHBOARD
# # ─────────────────────────────────────────────
# class Dashboard:
#     STOP_COLORS = {
#         "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
#         "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
#     }
#     PRIORITY_COLORS = {"High": "#E74C3C", "Medium": "#D4A843", "Low": "#3FB950"}
#     STATUS_COLORS   = {
#         "On Time": "#3FB950", "Delayed": "#E74C3C",
#         "Scheduled": "#2EA4A4", "Cancelled": "#8B949E",
#     }

#     def _base_layout(self, height=320):
#         return dict(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=height,
#             margin=dict(t=20, b=40, l=50, r=20),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )

#     def volume_timeline(self, stops_df):
#         df2 = stops_df.copy()
#         df2["date"] = pd.to_datetime(df2["scheduled_date"])
#         daily = df2.groupby(["date","stop_type"]).size().reset_index(name="count")
#         fig   = px.bar(daily, x="date", y="count", color="stop_type",
#                        color_discrete_map=self.STOP_COLORS, barmode="stack")
#         fig.update_layout(**self._base_layout(280),
#                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"))
#         return fig

#     def stop_type_donut(self, stops_df):
#         counts = stops_df["stop_type"].value_counts().reset_index()
#         counts.columns = ["stop_type","count"]
#         fig = px.pie(counts, names="stop_type", values="count",
#                      color="stop_type", color_discrete_map=self.STOP_COLORS, hole=0.55)
#         fig.update_traces(textposition="inside", textinfo="percent+label",
#                           marker=dict(line=dict(color="#0D1117", width=2)))
#         fig.update_layout(**self._base_layout(300), showlegend=False)
#         return fig

#     def distance_by_route(self, routes_df):
#         df = routes_df.sort_values("total_distance_km", ascending=False).head(10)
#         fig = go.Figure(go.Bar(x=df["route_id"], y=df["total_distance_km"],
#                                marker_color="#D4A843",
#                                hovertemplate="%{x}: %{y:.1f} km<extra></extra>"))
#         fig.update_layout(**self._base_layout(280),
#                           xaxis=dict(showgrid=False, tickangle=-20),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="km"))
#         return fig

#     def on_time_gauge(self, pct):
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number", value=pct,
#             number={"suffix": "%", "font": {"color": "#D4A843", "size": 36}},
#             gauge={
#                 "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
#                 "bar":  {"color": "#D4A843"}, "bgcolor": "#21262D",
#                 "steps": [
#                     {"range": [0,  60],  "color": "rgba(192,57,43,0.3)"},
#                     {"range": [60, 80],  "color": "rgba(232,135,58,0.3)"},
#                     {"range": [80, 100], "color": "rgba(63,185,80,0.3)"},
#                 ],
#                 "threshold": {"line": {"color": "#3FB950", "width": 3}, "value": 85},
#             },
#         ))
#         layout = self._base_layout(220)
#         layout["margin"] = dict(t=20, b=10, l=30, r=30)

#         fig.update_layout(**layout)
#         return fig

#     def priority_bar(self, stops_df):
#         pc = stops_df["priority"].value_counts().reset_index()
#         pc.columns = ["priority","count"]
#         fig = go.Figure(go.Bar(
#             x=pc["priority"], y=pc["count"],
#             marker_color=[self.PRIORITY_COLORS.get(p,"#8B949E") for p in pc["priority"]],
#             hovertemplate="%{x}: %{y}<extra></extra>",
#         ))
#         fig.update_layout(**self._base_layout(240),
#                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"))
#         return fig

#     def route_map_scatter(self, stops_list, itinerary=None):
#         lats   = [s["lat"] for s in stops_list]
#         lons   = [s["lon"] for s in stops_list]
#         names  = [s["location_name"] for s in stops_list]
#         colors = [self.STOP_COLORS.get(s.get("stop_type","Delivery"), "#D4A843") for s in stops_list]

#         fig = go.Figure()
#         fig.add_trace(go.Scattergeo(
#             lat=lats, lon=lons, mode="markers+text",
#             marker=dict(size=12, color=colors, line=dict(color="#0D1117", width=1)),
#             text=[f"{i+1}. {n}" for i, n in enumerate(names)],
#             textposition="top center",
#             textfont=dict(size=9, color="#C9D1D9"),
#             hovertemplate="<b>%{text}</b><extra></extra>", name="Stops",
#         ))
#         if itinerary and "stops" in itinerary:
#             seq  = sorted(itinerary["stops"], key=lambda s: s["sequence"])
#             rlat, rlon = [], []
#             for ist in seq:
#                 m = next((s for s in stops_list if s["stop_id"] == ist["stop_id"]), None)
#                 if m:
#                     rlat.append(m["lat"]); rlon.append(m["lon"])
#             if rlat:
#                 fig.add_trace(go.Scattergeo(lat=rlat, lon=rlon, mode="lines",
#                                              line=dict(width=2, color="#D4A843"), name="Route"))
#         fig.update_geos(
#             center=dict(lat=np.mean(lats), lon=np.mean(lons)), projection_scale=8,
#             showland=True, landcolor="#21262D", showocean=True, oceancolor="#161B22",
#             showcountries=True, countrycolor="#30363D", showcoastlines=True, coastlinecolor="#30363D",
#         )
#         layout = self._base_layout(420)
#         layout["margin"] = dict(t=10, b=10, l=10, r=10)
#         fig.update_layout(**layout, geo=dict(bgcolor="rgba(0,0,0,0)"), showlegend=True)
#         return fig

#     def cluster_scatter(self, stops_df, labels, X_2d):
#         df = stops_df.copy()
#         df["Cluster"] = [f"Cluster {l+1}" for l in labels]
#         df["x"] = X_2d[:, 0]; df["y"] = X_2d[:, 1]
#         fig = px.scatter(df, x="x", y="y", color="Cluster",
#                          hover_data=["location_name","stop_type","priority"],
#                          color_discrete_sequence=["#D4A843","#2EA4A4","#3FB950","#8957E5","#E74C3C","#1F6FEB"])
#         fig.update_layout(**self._base_layout(380),
#                           xaxis=dict(showgrid=False, title="Component 1"),
#                           yaxis=dict(showgrid=False, title="Component 2"))
#         return fig

#     def performance_timeline(self, routes_df):
#         df = routes_df.copy(); df["date"] = pd.to_datetime(df["route_date"])
#         fig = px.line(df.sort_values("date"), x="date", y="on_time_pct", color="transport_mode",
#                       color_discrete_map={"Road":"#D4A843","Air":"#2EA4A4","Rail":"#3FB950","Sea":"#8957E5"})
#         fig.update_layout(**self._base_layout(300), xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="On-Time %", range=[0,105]))
#         return fig

#     def fuel_cost_bar(self, routes_df):
#         df = routes_df.groupby("transport_mode")["estimated_fuel_cost_inr"].mean().reset_index()
#         fig = go.Figure(go.Bar(x=df["transport_mode"], y=df["estimated_fuel_cost_inr"],
#                                marker_color=["#D4A843","#2EA4A4","#3FB950","#8957E5"],
#                                hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
#         fig.update_layout(**self._base_layout(260), xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="Avg Cost (₹)"))
#         return fig


# # ─────────────────────────────────────────────
# # PAGE: OVERVIEW
# # ─────────────────────────────────────────────
# def page_overview(stops_df, routes_df, stats, dash):
#     st.markdown('<div class="hero-title">🗺️ Route<span class="hero-accent">IQ</span></div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">AI-Powered Logistics Itinerary Planning & Route Optimization</div>', unsafe_allow_html=True)
#     st.markdown("---")

#     if stops_df.empty:
#         st.warning("No data loaded. Please ensure stops.csv and routes.csv exist.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Stops",    stats.get("total_stops", 0))
#     c2.metric("Active Routes",  stats.get("total_routes", 0))
#     c3.metric("Total Distance", f"{stats.get('total_distance_km', 0):,} km")
#     c4.metric("On-Time Rate",   f"{stats.get('on_time_pct', 0)}%")

#     st.markdown("---")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.markdown("**Daily Stop Volume by Type**")
#         st.plotly_chart(dash.volume_timeline(stops_df), use_container_width=True)
#     with col2:
#         st.markdown("**Stop Type Distribution**")
#         st.plotly_chart(dash.stop_type_donut(stops_df), use_container_width=True)

#     col3, col4 = st.columns(2)
#     with col3:
#         st.markdown("**Top Routes by Distance**")
#         st.plotly_chart(dash.distance_by_route(routes_df), use_container_width=True)
#     with col4:
#         st.markdown("**Stop Priority Breakdown**")
#         st.plotly_chart(dash.priority_bar(stops_df), use_container_width=True)

#     st.markdown("---")
#     st.markdown("### 🚨 High-Priority Open Stops")
#     high = stops_df[(stops_df["priority"] == "High") & (stops_df["status"].isin(["Scheduled", "Delayed", "Failed"]))].head(5)
#     if high.empty:
#         st.info("No high-priority open stops.")
#     else:
#         for _, row in high.iterrows():
#             type_cls = {"Delivery":"card-gold","Meeting":"card-teal","Pickup":"card-green",
#                         "Warehouse":"card-blue","Customs":"card-red"}.get(row["stop_type"],"card-gold")
#             st.markdown(
#                 f'<div class="card {type_cls}">'
#                 f'<div style="display:flex;justify-content:space-between;align-items:center">'
#                 f'<div><b style="color:#F0F6FC">{row["location_name"]}</b>'
#                 f'<span class="badge badge-delivery" style="margin-left:8px">{row["stop_type"]}</span></div>'
#                 f'<div style="text-align:right;font-size:0.8rem;color:#8B949E">'
#                 f'Route: {row.get("route_id","—")} &nbsp;|&nbsp; {row.get("scheduled_date","—")}</div></div>'
#                 f'<div class="stop-detail">Window: {row.get("time_window_start","—")} – '
#                 f'{row.get("time_window_end","—")} &nbsp;·&nbsp; {row.get("notes","")}</div></div>',
#                 unsafe_allow_html=True,
#             )


# # ─────────────────────────────────────────────
# # PAGE: ITINERARY PLANNER  (3 tabs)
# # ─────────────────────────────────────────────
# def page_planner(stops_df, ai, ml, dash):
#     st.markdown('<div class="section-title">🔍 Itinerary Planner</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Generate optimized route plans — '
#         'from your dataset, custom coordinates, or plain English</div>',
#         unsafe_allow_html=True,
#     )

#     tab1, tab2, tab3 = st.tabs(["📋 Plan from Dataset", "✏️ Custom Coordinates", "💬 Natural Language"])

#     # ── TAB 1: Plan from Dataset ──────────────────────────────────────────────
#     with tab1:
#         if stops_df.empty:
#             st.warning("No stops data loaded.")
#         else:
#             routes         = stops_df["route_id"].unique().tolist()
#             selected_route = st.selectbox("Select Route to Plan", routes)
#             route_stops    = stops_df[stops_df["route_id"] == selected_route].copy()

#             st.markdown(f"**{len(route_stops)} stops on route {selected_route}**")
#             # Build display columns — always show coordinates; lat/lon are internal
#             display_cols = ["stop_id", "location_name", "stop_type", "priority",
#                              "time_window_start", "time_window_end"]
#             if "coordinates" in route_stops.columns:
#                 display_cols.append("coordinates")
#             if "notes" in route_stops.columns:
#                 display_cols.append("notes")
#             st.dataframe(
#                 route_stops[display_cols].reset_index(drop=True),
#                 use_container_width=True,
#             )

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 driver_name = st.text_input("Driver Name", value="Ramesh Kumar")
#                 start_time  = st.text_input("Start Time (HH:MM)", value="08:00")
#             with col2:
#                 transport_mode   = st.selectbox("Transport Mode", ["Road","Rail","Air","Sea"])
#                 vehicle_type     = st.selectbox("Vehicle Type", ["Truck","Van","Motorcycle","Car","Tempo"])
#             with col3:
#                 max_hours        = st.slider("Max Working Hours", 4, 14, 8)
#                 vehicle_capacity = st.number_input("Vehicle Capacity (kg)", 100, 10000, 1000, step=100)

#             optimize = st.checkbox("Apply Nearest-Neighbor Optimization", value=True)

#             if st.button("🚀 Generate Itinerary", type="primary", key="gen_dataset"):
#                 # lat/lon parsed from coordinates column in load_stops()
#                 route_stops = route_stops.dropna(subset=["lat", "lon"]).reset_index(drop=True)
#                 coords = list(zip(route_stops["lat"], route_stops["lon"]))
#                 if optimize and len(coords) > 2:
#                     order       = ml.nearest_neighbor_route(coords)
#                     route_stops = route_stops.iloc[order].reset_index(drop=True)

#                 stops_list = [
#                     {
#                         "stop_id":           str(r["stop_id"]),
#                         "location_name":     r["location_name"],
#                         "lat":               float(r["lat"]),
#                         "lon":               float(r["lon"]),
#                         "stop_type":         r["stop_type"],
#                         "time_window_start": str(r["time_window_start"])[-8:-3] if pd.notnull(r["time_window_start"]) else "08:00",
#                         "time_window_end":   str(r["time_window_end"])[-8:-3]   if pd.notnull(r["time_window_end"])   else "18:00",
#                         "priority":          r["priority"],
#                         "notes":             str(r.get("notes","")) if pd.notnull(r.get("notes","")) else "",
#                     }
#                     for _, r in route_stops.iterrows()
#                 ]
#                 constraints = {
#                     "driver_name": driver_name, "start_time": start_time,
#                     "transport_mode": transport_mode, "vehicle_type": vehicle_type,
#                     "max_hours": max_hours, "vehicle_capacity_kg": vehicle_capacity,
#                 }

#                 with st.spinner("🛰️ Fetching real road distances via OSRM…"):
#                     osrm_preview = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])

#                 st.caption(
#                     "Routing: 🟢 OSRM (real roads)" if osrm_preview["source"] == "osrm"
#                     else "Routing: 🟡 Haversine fallback (OSRM unreachable)"
#                 )
#                 route_context = (
#                     f"Route {selected_route} | {len(stops_list)} stops | Mode: {transport_mode} | "
#                     f"Road distance: {osrm_preview['total_distance_km']} km | "
#                     f"Drive time: {osrm_preview['total_duration_min']} min | "
#                     f"Source: {osrm_preview['source']}"
#                 )
#                 with st.spinner("🧠 AI optimizing your route…"):
#                     itinerary = ai.generate_itinerary(stops_list, constraints, route_context)

#                 st.session_state["generated_itinerary"] = itinerary
#                 st.session_state["last_constraints"]    = constraints
#                 st.session_state["itinerary_source"]    = "dataset"
#                 st.session_state["fuel_analysis"]       = None  # reset stale savings
#                 st.session_state["plan_history"].append({
#                     "source": "Dataset", "route": selected_route,
#                     "generated_at": datetime.now().strftime("%H:%M:%S"),
#                     "stops": len(stops_list),
#                     "distance_km": itinerary.get("total_distance_km", 0),
#                     "routing": osrm_preview["source"],
#                 })
#                 st.success("✅ Itinerary generated!")

#     # ── TAB 2: Custom Coordinates ─────────────────────────────────────────────
#     with tab2:
#         st.markdown("Add custom stops with coordinates for ad-hoc planning.")
#         custom_text = st.text_area(
#             "One stop per line: Name, Lat, Lon, Type, Priority, TimeFrom, TimeTo",
#             height=160,
#             placeholder="Dadar Warehouse, 19.018, 72.848, Delivery, High, 09:00, 11:00",
#             key="custom_coords_input",
#         )
#         col1, col2 = st.columns(2)
#         with col1:
#             c_driver = st.text_input("Driver Name", value="Suresh Patil", key="c_driver")
#             c_start  = st.text_input("Start Time", value="08:30", key="c_start")
#         with col2:
#             c_mode  = st.selectbox("Transport Mode", ["Road","Rail","Air","Sea"], key="c_mode")
#             c_hours = st.slider("Max Hours", 4, 14, 8, key="c_hours")
#             c_cap   = st.number_input("Capacity (kg)", 100, 10000, 500, step=100, key="c_cap")

#         if st.button("🚀 Generate Custom Itinerary", key="gen_custom"):
#             custom_stops = []
#             for i, line in enumerate(custom_text.strip().split("\n")):
#                 parts = [p.strip() for p in line.split(",")]
#                 if len(parts) >= 3:
#                     try:
#                         custom_stops.append({
#                             "stop_id": f"CS{i+1}", "location_name": parts[0],
#                             "lat": float(parts[1]), "lon": float(parts[2]),
#                             "stop_type":         parts[3] if len(parts) > 3 else "Delivery",
#                             "priority":          parts[4] if len(parts) > 4 else "Medium",
#                             "time_window_start": parts[5] if len(parts) > 5 else "08:00",
#                             "time_window_end":   parts[6] if len(parts) > 6 else "18:00",
#                             "notes": "",
#                         })
#                     except ValueError:
#                         pass

#             if not custom_stops:
#                 st.error("No valid stops parsed. Check the format.")
#             else:
#                 constraints = {
#                     "driver_name": c_driver, "start_time": c_start,
#                     "transport_mode": c_mode, "max_hours": c_hours,
#                     "vehicle_type": "Van", "vehicle_capacity_kg": c_cap,
#                 }
#                 with st.spinner("🛰️ Fetching real road distances via OSRM…"):
#                     osrm_cs = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in custom_stops])
#                 st.caption(
#                     "Routing: 🟢 OSRM" if osrm_cs["source"] == "osrm"
#                     else "Routing: 🟡 Haversine fallback"
#                 )
#                 route_ctx = (
#                     f"{len(custom_stops)} custom stops | Mode: {c_mode} | "
#                     f"Road distance: {osrm_cs['total_distance_km']} km | "
#                     f"Drive time: {osrm_cs['total_duration_min']} min | Source: {osrm_cs['source']}"
#                 )
#                 with st.spinner("Planning custom route…"):
#                     itinerary = ai.generate_itinerary(custom_stops, constraints, route_ctx)
#                 st.session_state["generated_itinerary"] = itinerary
#                 st.session_state["last_constraints"]    = constraints
#                 st.session_state["itinerary_source"]    = "custom"
#                 st.session_state["fuel_analysis"]       = None
#                 st.session_state["plan_history"].append({
#                     "source": "Custom Coordinates", "route": "Custom",
#                     "generated_at": datetime.now().strftime("%H:%M:%S"),
#                     "stops": len(custom_stops),
#                     "distance_km": itinerary.get("total_distance_km", 0),
#                     "routing": osrm_cs["source"],
#                 })
#                 st.success("✅ Custom itinerary generated!")

#     # ── TAB 3: Natural Language ───────────────────────────────────────────────
#     with tab3:
#         # If the last itinerary was generated from a different tab, show a clear notice
#         # but do NOT wipe session state — that's what caused the dataset tab to break.
#         # Generating from this tab will replace it.
#         if st.session_state.get("itinerary_source") in ("dataset", "custom"):
#             st.info(
#                 "ℹ️ The itinerary below was generated from the **Dataset** or **Custom** tab. "
#                 "Describe your route and click **Generate Full Itinerary** to create a new one here."
#             )

#         st.markdown(
#             '<div class="summary-box">'
#             '<b style="color:#D4A843">💬 Describe your route in plain English.</b><br>'
#             '<span style="color:#8B949E;font-size:0.85rem">'
#             'Mention stops, times, priorities, and constraints naturally — '
#             'the AI extracts the structure and builds a full optimized itinerary.'
#             '</span></div>',
#             unsafe_allow_html=True,
#         )

#         with st.expander("📖 Example prompts"):
#             st.markdown("""
# **Simple run:**
# > Plan a route for Ramesh. Start at 8am from Dadar Warehouse. Deliver to Andheri client by 10am (urgent), pick up cargo from Kurla depot 11am–1pm, client meeting in BKC at 2:30pm.

# **Multi-constraint:**
# > Suresh needs to collect documents from Pune Hadapsar at 9am, customs meeting in Navi Mumbai by 1pm, then deliver to Mulund cold storage before 4pm. Refrigerated van, max 9 hours.

# **Minimal:**
# > 3 stops: Thane pickup 9am, Andheri delivery by 11am, Bandra meeting at 2pm.
#             """)

#         nl_text = st.text_area(
#             "Describe your itinerary:",
#             height=140,
#             placeholder=(
#                 "Plan a route for Ramesh starting at 8am from Dadar Warehouse. "
#                 "Deliver to Andheri client by 10am (urgent)..."
#             ),
#             key="nl_input",
#         )

#         nl_col1, nl_col2 = st.columns(2)
#         with nl_col1:
#             nl_driver = st.text_input("Override driver name (optional)", value="", key="nl_driver")
#             nl_mode   = st.selectbox("Override transport mode",
#                                      ["(auto-detect)","Road","Rail","Air","Sea"], key="nl_mode")
#         with nl_col2:
#             nl_hours = st.slider("Max working hours", 4, 14, 9, key="nl_hours")
#             nl_cap   = st.number_input("Vehicle capacity (kg)", 100, 10000, 500, step=100, key="nl_cap")

#         if st.button("🔍 Parse Description", key="nl_parse"):
#             if not nl_text.strip():
#                 st.error("Please describe your route first.")
#             elif not ai.llm:
#                 st.error("LLM is offline. Natural language parsing requires a connected LLM.")
#             else:
#                 with st.spinner("🤖 Extracting stops from your description…"):
#                     parsed = ai.parse_natural_language_stops(nl_text)
#                 st.session_state["nl_parsed_result"] = parsed

#         parsed = st.session_state.get("nl_parsed_result")
#         if parsed:
#             if parsed.get("error") and not parsed.get("stops"):
#                 st.error(f"Parse failed: {parsed['error']}")
#             else:
#                 stops_list = parsed.get("stops", [])
#                 if parsed.get("parse_notes"):
#                     st.info(f"📝 AI understood: {parsed['parse_notes']}")

#                 if stops_list:
#                     st.markdown("#### ✅ Parsed Stops")
#                     st.dataframe(
#                         pd.DataFrame([{
#                             "#": s["stop_id"], "Location": s["location_name"],
#                             "Type": s["stop_type"], "Priority": s["priority"],
#                             "Window": f"{s['time_window_start']} – {s['time_window_end']}",
#                             "Notes": s.get("notes",""),
#                         } for s in stops_list]),
#                         use_container_width=True,
#                     )

#                     auto_c = parsed.get("constraints", {})
#                     if nl_driver.strip():          auto_c["driver_name"]      = nl_driver.strip()
#                     if nl_mode != "(auto-detect)": auto_c["transport_mode"]   = nl_mode
#                     auto_c["max_hours"]          = nl_hours
#                     auto_c["vehicle_capacity_kg"] = nl_cap

#                     cc1, cc2, cc3 = st.columns(3)
#                     cc1.markdown(f"**Driver:** {auto_c.get('driver_name','—')}")
#                     cc2.markdown(f"**Mode:** {auto_c.get('transport_mode','Road')}")
#                     cc3.markdown(f"**Start:** {auto_c.get('start_time','08:00')}")

#                     if st.button("🚀 Generate Full Itinerary", type="primary", key="nl_generate"):
#                         with st.spinner("🛰️ Fetching OSRM road distances…"):
#                             osrm_nl = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])
#                         st.caption(
#                             "Routing: 🟢 OSRM" if osrm_nl["source"] == "osrm"
#                             else "Routing: 🟡 Haversine fallback"
#                         )
#                         ctx_nl = (
#                             f"{len(stops_list)} NL-parsed stops | "
#                             f"Mode: {auto_c.get('transport_mode','Road')} | "
#                             f"Road distance: {osrm_nl['total_distance_km']} km | "
#                             f"Drive time: {osrm_nl['total_duration_min']} min | "
#                             f"Source: {osrm_nl['source']}"
#                         )
#                         with st.spinner("🧠 Generating optimized itinerary…"):
#                             itinerary = ai.generate_itinerary(stops_list, auto_c, ctx_nl)
#                         # Build stops_for_map — NL itinerary embeds lat/lon in its stops
#                         nl_map_stops = []
#                         for itin_s in itinerary.get("stops", []):
#                             matched = next((s for s in stops_list if s["stop_id"] == itin_s["stop_id"]), None)
#                             if matched:
#                                 nl_map_stops.append({
#                                     "stop_id":       itin_s["stop_id"],
#                                     "location_name": itin_s["location_name"],
#                                     "lat":           matched["lat"],
#                                     "lon":           matched["lon"],
#                                     "stop_type":     itin_s.get("stop_type","Delivery"),
#                                 })

#                         st.session_state["generated_itinerary"] = itinerary
#                         st.session_state["last_constraints"]    = auto_c
#                         st.session_state["nl_parsed_result"]    = None
#                         st.session_state["itinerary_source"]    = "nl"
#                         st.session_state["nl_stops_for_map"]    = nl_map_stops
#                         st.session_state["fuel_analysis"]       = None
#                         st.session_state["plan_history"].append({
#                             "source": "Natural Language", "route": "NL-parsed",
#                             "generated_at": datetime.now().strftime("%H:%M:%S"),
#                             "stops": len(stops_list),
#                             "distance_km": itinerary.get("total_distance_km", 0),
#                             "routing": osrm_nl["source"],
#                         })
#                         st.success("✅ Itinerary generated from your description!")
#                         st.rerun()
#                 else:
#                     st.warning("No stops extracted. Try adding specific location names and times.")

#     # ── Display Itinerary ─────────────────────────────────────────────────────
#     # Render the itinerary below the tabs.
#     # itinerary_source tells us which tab owns the current result.
#     # We ALWAYS show it here — the per-tab isolation works because:
#     #   - Generating from tab1/tab2 sets source = "dataset"/"custom"
#     #   - Generating from tab3 sets source = "nl"
#     #   - Switching tabs does NOT clear the result (no unconditional clearing)
#     #   - The user sees the last result they generated, regardless of which tab is active
#     itin   = st.session_state.get("generated_itinerary")
#     source = st.session_state.get("itinerary_source")
#     if itin and source:
#         st.markdown("---")
#         nl_map = st.session_state.get("nl_stops_for_map", []) if source == "nl" else []
#         _render_itinerary(
#             itin, stops_df, ai, dash,
#             st.session_state.get("last_constraints", {}),
#             nl_stops_override=nl_map,
#         )


# # ─────────────────────────────────────────────
# # WEATHER PANEL  (called from _render_itinerary)
# # ─────────────────────────────────────────────
# def _render_weather_panel(stops_with_coords: list, itin_date_str: str = None):
#     """
#     Renders arrival-time-aware weather forecast for every stop.

#     For each stop the forecast shows conditions at the stop's arrival_time,
#     not current conditions — e.g. Stop B arriving at 13:00 shows the 13:00
#     hourly forecast, not what the weather is right now.

#     stops_with_coords: list of dicts with {stop_id, location_name, lat, lon,
#                         arrival_time (HH:MM), stop_type, sequence}
#     itin_date_str:     "YYYY-MM-DD" — the itinerary date used to resolve forecasts
#     """
#     st.markdown("### 🌤️ Weather Forecast Along Route")
#     st.markdown(
#         '<div style="font-size:0.8rem;color:#8B949E;margin-bottom:16px">'
#         'Forecast shown at <b style="color:#D4A843">each stop&#39;s expected arrival time</b> '
#         '— not current conditions. Powered by Open-Meteo hourly API.'
#         '</div>',
#         unsafe_allow_html=True,
#     )

#     if not stops_with_coords:
#         st.info("No coordinate data available for weather lookup.")
#         return

#     with st.spinner("Fetching arrival-time forecasts for each stop…"):
#         weather_stops = WeatherEngine.fetch_route_at_times(stops_with_coords, itin_date_str)

#     # Data source summary
#     sources = [ws["weather"]["source"] for ws in weather_stops]
#     live_count = sum(1 for s in sources if "open-meteo" in s)
#     est_count  = sum(1 for s in sources if "seasonal"   in s)

#     if live_count == len(weather_stops):
#         st.caption("🟢 All forecasts from Open-Meteo hourly API (real data)")
#     elif live_count > 0:
#         st.caption(f"🟡 Mixed: {live_count} stops from Open-Meteo · {est_count} stops from seasonal estimate")
#     else:
#         st.caption(
#             "🟡 **Open-Meteo unreachable** — showing seasonal estimates with diurnal variation. "
#             "Values are indicative. Check network/proxy settings."
#         )

#     adverse_count = sum(1 for ws in weather_stops if ws["weather"]["is_adverse"])
#     if adverse_count:
#         st.warning(
#             f"⚠️ **{adverse_count} stop(s) forecast adverse weather at arrival time** — "
#             "allow extra buffer time and check vehicle load securing."
#         )

#     # ── Cards — 4 per row ─────────────────────────────────────────────────────
#     cols_per_row = min(4, len(weather_stops))
#     card_rows    = [weather_stops[i:i+cols_per_row]
#                     for i in range(0, len(weather_stops), cols_per_row)]

#     for card_row in card_rows:
#         cols = st.columns(len(card_row))
#         for col, ws in zip(cols, card_row):
#             w            = ws["weather"]
#             source       = w.get("source", "")
#             arrival      = ws.get("arrival_time", "")
#             forecast_t   = w.get("forecast_time", arrival)
#             precip_prob  = w.get("precip_probability")

#             border_color = "#E74C3C" if w["is_adverse"] else "#30363D"
#             temp_str = f"{w['temperature_c']}°C"     if w.get("temperature_c")    is not None else "—"
#             wind_str = f"{w['wind_speed_kmh']} km/h" if w.get("wind_speed_kmh")   is not None else "—"
#             prec_str = f"{w['precipitation_mm']} mm" if w.get("precipitation_mm") is not None else "—"
#             hum_str  = f"{w['humidity_pct']}%"       if w.get("humidity_pct")     is not None else "—"
#             vis_str  = f"{w['visibility_km']} km"    if w.get("visibility_km")    is not None else "—"
#             prob_str = f"{precip_prob}% rain chance" if precip_prob is not None else ""

#             if "open-meteo" in source:
#                 src_badge = '<div style="font-size:0.58rem;color:#3FB950;margin-top:5px">🟢 Open-Meteo hourly</div>'
#             else:
#                 src_badge = '<div style="font-size:0.58rem;color:#D4A843;margin-top:5px">🟡 Seasonal estimate</div>'

#             adverse_badge = (
#                 '<div style="font-size:0.65rem;color:#E74C3C;margin-top:5px;font-weight:600">⚠️ Adverse conditions</div>'
#                 if w["is_adverse"] else ""
#             )
#             prob_badge = (
#                 f'<div style="font-size:0.65rem;color:#2EA4A4;margin-top:3px">🌂 {prob_str}</div>'
#                 if prob_str else ""
#             )

#             col.markdown(
#                 f'<div class="weather-card" style="border:1px solid {border_color};margin-bottom:8px">'
#                 # Arrival time header — the KEY info
#                 f'<div style="font-size:0.6rem;font-family:monospace;color:#2EA4A4;'
#                 f'letter-spacing:0.05em;margin-bottom:6px">🕐 ARRIVAL {arrival}</div>'
#                 # Icon + temp
#                 f'<div style="font-size:1.9rem;margin-bottom:2px">{w["icon"]}</div>'
#                 f'<div style="font-size:0.78rem;font-weight:700;color:#F0F6FC;margin-bottom:2px">'
#                 f'{ws["location_name"][:20]}</div>'
#                 f'<div style="font-size:1.2rem;font-weight:700;color:#D4A843;margin-bottom:2px">{temp_str}</div>'
#                 f'<div style="font-size:0.72rem;color:#C9D1D9">{w["description"]}</div>'
#                 # Stats row
#                 f'<div style="font-size:0.65rem;color:#8B949E;margin-top:8px;line-height:1.7">'
#                 f'💨 {wind_str} &nbsp;·&nbsp; 🌧️ {prec_str}<br>'
#                 f'💧 {hum_str} &nbsp;·&nbsp; 👁️ {vis_str}'
#                 f'</div>'
#                 f'{prob_badge}{src_badge}{adverse_badge}'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )

#     # ── Summary data table ────────────────────────────────────────────────────
#     with st.expander("📋 Full Weather Forecast Table"):
#         rows_data = []
#         for ws in weather_stops:
#             w = ws["weather"]
#             rows_data.append({
#                 "Stop":              ws["location_name"],
#                 "Arrival Time":      ws.get("arrival_time", "—"),
#                 "Forecast At":       w.get("forecast_time", "—"),
#                 "Condition":         f"{w['icon']} {w['description']}",
#                 "Temp (°C)":         w.get("temperature_c", "—"),
#                 "Wind (km/h)":       w.get("wind_speed_kmh", "—"),
#                 "Rain (mm)":         w.get("precipitation_mm", "—"),
#                 "Rain Chance (%)":   w.get("precip_probability", "—"),
#                 "Humidity (%)":      w.get("humidity_pct", "—"),
#                 "Visibility (km)":   w.get("visibility_km", "—"),
#                 "Source":            w.get("source", "—"),
#                 "⚠️ Adverse":        "Yes" if w["is_adverse"] else "—",
#             })
#         st.dataframe(pd.DataFrame(rows_data), use_container_width=True)


# # ─────────────────────────────────────────────
# # FUEL PANEL  (called from _render_itinerary)
# # ─────────────────────────────────────────────
# def _render_fuel_panel(stops_list: list, constraints: dict, itin: dict):
#     """
#     Renders the full fuel cost + route savings analysis panel.
#     """
#     st.markdown("### ⛽ Fuel Cost & Route Savings Analysis")

#     vehicle_type = constraints.get("vehicle_type", "Van")
#     total_km     = itin.get("total_distance_km", 0)

#     if not total_km:
#         st.info("No distance data yet. Generate an itinerary first.")
#         return

#     # ── User controls ─────────────────────────────────────────────────────────
#     fc1, fc2, fc3 = st.columns(3)
#     with fc1:
#         city = st.selectbox(
#             "📍 City (for fuel price)",
#             list(FuelEngine.CITY_PRICES.keys()),
#             index=list(FuelEngine.CITY_PRICES.keys()).index("Mumbai")
#                   if "Mumbai" in FuelEngine.CITY_PRICES else 0,
#             key="fuel_city",
#         )
#     with fc2:
#         fuel_type_options = ["Auto (by vehicle)", "Petrol", "Diesel"]
#         fuel_choice = st.selectbox("⛽ Fuel type override", fuel_type_options, key="fuel_type_choice")
#     with fc3:
#         custom_price = st.number_input(
#             "₹/litre override (0 = use reference)",
#             min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="fuel_price_override"
#         )

#     override_price = custom_price if custom_price > 0 else None
#     fuel_type_map  = {"Petrol": "petrol", "Diesel": "diesel"}

#     # If override fuel type selected, compute price accordingly
#     if fuel_choice != "Auto (by vehicle)" and override_price is None:
#         ft          = fuel_type_map[fuel_choice]
#         override_price = FuelEngine.get_price(city, ft)

#     # ── Current itinerary fuel cost ───────────────────────────────────────────
#     fuel_current = FuelEngine.compute_fuel_cost(
#         total_km, vehicle_type, city, override_price
#     )

#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("Route Distance",    f"{total_km} km")
#     m2.metric("Fuel Consumed",     f"{fuel_current['litres_consumed']} L")
#     m3.metric("Price/Litre",       f"₹{fuel_current['price_per_litre']:.2f}")
#     m4.metric("Total Fuel Cost",   f"₹{fuel_current['total_cost_inr']:,.2f}")

#     st.markdown(
#         f'<div style="font-size:0.75rem;color:#8B949E;margin-bottom:20px">'
#         f'Vehicle: <b>{vehicle_type}</b> · '
#         f'Efficiency: <b>{fuel_current["efficiency_kmpl"]} km/L</b> · '
#         f'Fuel: <b>{fuel_current["fuel_type"].title()}</b> · '
#         f'City reference: <b>{city}</b> '
#         f'<span style="color:#30363D">(prices as of Jun 2025 — update via override)</span>'
#         f'</div>',
#         unsafe_allow_html=True,
#     )

#     # ── Route savings analysis ────────────────────────────────────────────────
#     st.markdown("#### 🔀 Optimized vs Un-Optimized Route Savings")

#     if len(stops_list) < 3:
#         st.info("Add at least 3 stops to compute route savings comparison.")
#         return

#     if st.button("🔍 Run Savings Analysis", key="run_savings"):
#         with st.spinner("Comparing original vs NN-optimized order via OSRM…"):
#             analysis = FuelEngine.savings_analysis(
#                 stops_list, vehicle_type, city, override_price
#             )
#         st.session_state["fuel_analysis"] = analysis

#     analysis = st.session_state.get("fuel_analysis")
#     if not analysis:
#         st.caption("Click 'Run Savings Analysis' to compare route orders.")
#         return

#     saved_km   = analysis.get("saved_km", 0)
#     saving_pct = analysis.get("saving_pct", 0)
#     saved_cost = analysis.get("saved_cost_inr", 0)

#     if saved_km > 0:
#         st.markdown(
#             f'<div class="fuel-save">'
#             f'<div style="font-size:1.05rem;font-weight:700;color:#3FB950;margin-bottom:10px">'
#             f'✅ You can save <b>₹{saved_cost:,.2f}</b> by reordering your stops!</div>'
#             f'<div style="display:flex;gap:32px;flex-wrap:wrap">'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">📏 Distance saved: <b>{saved_km} km</b></span>'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">📉 Reduction: <b>{saving_pct}%</b></span>'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">⛽ Fuel saved: '
#             f'<b>{round(saved_km / analysis["fuel_detail_optimized"]["efficiency_kmpl"], 2)} L</b></span>'
#             f'</div></div>',
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             '<div class="fuel-warn">'
#             '<b style="color:#D4A843">ℹ️ Your current stop order is already near-optimal.</b><br>'
#             '<span style="color:#8B949E;font-size:0.82rem">The nearest-neighbor reordering did not '
#             'produce a shorter total route for this set of stops.</span>'
#             '</div>',
#             unsafe_allow_html=True,
#         )

#     # Side-by-side comparison
#     sa1, sa2 = st.columns(2)
#     with sa1:
#         st.markdown("**📋 Original Order**")
#         for i, name in enumerate(analysis.get("original_order", []), 1):
#             st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
#                         f'<span style="color:#D4A843">{i}.</span> {name}</div>', unsafe_allow_html=True)
#         orig_c = analysis.get("original_cost_inr", 0)
#         st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
#                     f'<b>Total: {analysis.get("original_km",0)} km · ₹{orig_c:,.2f}</b></div>', unsafe_allow_html=True)

#     with sa2:
#         st.markdown("**✅ Optimized Order**")
#         for i, name in enumerate(analysis.get("optimized_order", []), 1):
#             st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
#                         f'<span style="color:#3FB950">{i}.</span> {name}</div>', unsafe_allow_html=True)
#         opt_c = analysis.get("optimized_cost_inr", 0)
#         st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
#                     f'<b>Total: {analysis.get("optimized_km",0)} km · ₹{opt_c:,.2f}</b></div>', unsafe_allow_html=True)

#     # Per-leg savings chart
#     per_stop = analysis.get("per_stop_savings", [])
#     if per_stop:
#         st.markdown("#### 📊 Leg-by-Leg Distance Comparison")
#         df_legs = pd.DataFrame(per_stop)
#         fig = go.Figure()
#         fig.add_bar(
#             name="Original", x=df_legs["leg"].astype(str),
#             y=df_legs["original_km"],
#             marker_color="#E74C3C",
#             hovertemplate="Leg %{x}: %{y:.2f} km<extra>Original</extra>",
#         )
#         fig.add_bar(
#             name="Optimized", x=df_legs["leg"].astype(str),
#             y=df_legs["optimized_km"],
#             marker_color="#3FB950",
#             hovertemplate="Leg %{x}: %{y:.2f} km<extra>Optimized</extra>",
#         )
#         fig.update_layout(
#             barmode="group",
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=260,
#             xaxis=dict(title="Leg #", showgrid=False),
#             yaxis=dict(title="km", showgrid=True, gridcolor="#21262D"),
#             margin=dict(t=10, b=40, l=50, r=20),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     # ── Optimized Route Map ──────────────────────────────────────────────────
#     optimized_order = analysis.get("optimized_order", [])
#     if optimized_order and len(stops_list) >= 2:
#         st.markdown("#### 🗺️ Optimized Route Map")
#         st.caption("Stop order resequenced by Nearest-Neighbor TSP for minimum distance.")

#         # Reorder stops_list to match the optimized sequence
#         nn_order      = MLEngine().nearest_neighbor_route([(s["lat"], s["lon"]) for s in stops_list])
#         opt_stops     = [stops_list[i] for i in nn_order]

#         # Build a lightweight itinerary shell so route_map_scatter draws the route line
#         opt_itin_stub = {
#             "stops": [
#                 {
#                     "sequence": idx + 1,
#                     "stop_id":  s["stop_id"],
#                     "stop_type": s.get("stop_type", "Delivery"),
#                 }
#                 for idx, s in enumerate(opt_stops)
#             ]
#         }

#         # Use Dashboard.route_map_scatter — reuse existing map function
#         from plotly.subplots import make_subplots as _msp  # already imported at top
#         lats   = [s["lat"]  for s in opt_stops]
#         lons   = [s["lon"]  for s in opt_stops]
#         names  = [s["location_name"] for s in opt_stops]
#         colors_map = {
#             "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
#             "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
#         }
#         pt_colors = [colors_map.get(s.get("stop_type","Delivery"), "#3FB950") for s in opt_stops]

#         import plotly.graph_objects as _go
#         fig_opt = _go.Figure()
#         # Route line
#         fig_opt.add_trace(_go.Scattergeo(
#             lat=lats, lon=lons, mode="lines",
#             line=dict(width=2.5, color="#3FB950"), name="Optimized Route",
#         ))
#         # Stop markers
#         fig_opt.add_trace(_go.Scattergeo(
#             lat=lats, lon=lons, mode="markers+text",
#             marker=dict(size=13, color=pt_colors, line=dict(color="#0D1117", width=1)),
#             text=[f"{i+1}. {n}" for i, n in enumerate(names)],
#             textposition="top center",
#             textfont=dict(size=9, color="#C9D1D9"),
#             hovertemplate="<b>%{text}</b><extra></extra>",
#             name="Stops",
#         ))
#         import numpy as _np
#         fig_opt.update_geos(
#             center=dict(lat=_np.mean(lats), lon=_np.mean(lons)),
#             projection_scale=8,
#             showland=True,       landcolor="#21262D",
#             showocean=True,      oceancolor="#161B22",
#             showcountries=True,  countrycolor="#30363D",
#             showcoastlines=True, coastlinecolor="#30363D",
#         )
#         fig_opt.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=420,
#             margin=dict(t=10, b=10, l=10, r=10),
#             geo=dict(bgcolor="rgba(0,0,0,0)"),
#             showlegend=True,
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig_opt, use_container_width=True, key="opt_route_map")

#     # Routing source note
#     rs = analysis.get("routing_source", "")
#     if rs == "osrm":
#         st.caption("🛰️ Savings computed using real OSRM road distances.")
#     else:
#         st.caption("📐 Savings computed using Haversine estimates (OSRM unreachable).")


# # ─────────────────────────────────────────────
# # ITINERARY RENDERER
# # ─────────────────────────────────────────────
# def _render_itinerary(itin, stops_df, ai, dash, constraints=None, nl_stops_override=None):
#     if constraints is None:
#         constraints = {}

#     # Build stops_list for map + fuel panel.
#     # For dataset routes: match stop_ids back to the stops_df for coordinates.
#     # For NL / custom routes: nl_stops_override carries the pre-built lat/lon list.
#     stops_list = []
#     if nl_stops_override:
#         stops_list = nl_stops_override
#     elif not stops_df.empty and "stop_id" in stops_df.columns:
#         # lat/lon are guaranteed present — parsed from "coordinates" in load_stops()
#         has_coords = "lat" in stops_df.columns and "lon" in stops_df.columns
#         for s in itin.get("stops", []):
#             row = stops_df[stops_df["stop_id"] == s["stop_id"]]
#             if not row.empty and has_coords:
#                 lat_val = row.iloc[0]["lat"]
#                 lon_val = row.iloc[0]["lon"]
#                 if pd.notna(lat_val) and pd.notna(lon_val):
#                     stops_list.append({
#                         "stop_id":       s["stop_id"],
#                         "location_name": s["location_name"],
#                         "lat":           float(lat_val),
#                         "lon":           float(lon_val),
#                         "stop_type":     s["stop_type"],
#                     })

#     # Header
#     st.markdown(
#         f'<div class="itinerary-header">'
#         f'<div style="font-family:\'Playfair Display\',serif;font-size:1.4rem;color:#F0F6FC;font-weight:700">'
#         f'📋 {itin.get("itinerary_title","Optimized Itinerary")}</div>'
#         f'<div style="margin-top:10px;color:#8B949E;font-size:0.82rem">'
#         f'🚗 {itin.get("driver","—")} &nbsp;|&nbsp; 🚛 {itin.get("vehicle","—")} '
#         f'&nbsp;|&nbsp; 📅 {itin.get("date","—")} &nbsp;|&nbsp; 🛣️ {itin.get("transport_mode","—")}'
#         f'</div></div>',
#         unsafe_allow_html=True,
#     )

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Distance",      f"{itin.get('total_distance_km', 0)} km")
#     c2.metric("Total Duration",      f"{itin.get('total_duration_min', 0)} min")
#     c3.metric("Efficiency Score",    f"{itin.get('efficiency_score', 0)}/100")
#     c4.metric("On-Time Probability", f"{itin.get('on_time_probability', 0)}%")

#     # Routing source banner
#     r_src = itin.get("routing_source", "")
#     if r_src == "osrm":
#         st.success("🛰️ Road distances powered by **OSRM** — real road network data")
#     elif r_src == "haversine_fallback":
#         st.warning("📐 Straight-line distance estimates used (OSRM unreachable). Times are approximate.")

#     # Map
#     if stops_list:
#         st.markdown("### 🗺️ Route Map")
#         st.plotly_chart(dash.route_map_scatter(stops_list, itin), use_container_width=True)

#     # Stop sequence
#     st.markdown("### 📍 Stop Sequence")
#     itin_stops = sorted(itin.get("stops", []), key=lambda x: x["sequence"])
#     max_seq    = max((x["sequence"] for x in itin_stops), default=0)

#     for s in itin_stops:
#         risk_color = {"High":"#E74C3C","Medium":"#D4A843","Low":"#3FB950","None":"#3FB950"}.get(
#             s.get("risk_flag","None"), "#3FB950"
#         )
#         risk_html = (
#             f' &nbsp;·&nbsp; <span style="color:{risk_color}">⚠️ {s.get("risk_reason","")}</span>'
#             if s.get("risk_flag","None") not in ["None",""] else ""
#         )
#         notes_html = (
#             f'<div class="stop-detail" style="margin-top:4px;font-style:italic">{s.get("notes","")}</div>'
#             if s.get("notes") else ""
#         )
#         connector = "" if s["sequence"] == max_seq else '<div class="connector-line"></div>'
#         st.markdown(
#             f'<div class="stop-card"><div class="stop-row">'
#             f'<div class="stop-num">{s["sequence"]}</div>'
#             f'<div style="flex:1">'
#             f'<div style="display:flex;justify-content:space-between;align-items:center">'
#             f'<b style="color:#F0F6FC">{s["location_name"]}</b>'
#             f'<span style="font-size:0.8rem;color:#8B949E">🕐 {s.get("arrival_time","—")} → {s.get("departure_time","—")}</span>'
#             f'</div>'
#             f'<div class="stop-detail">Type: {s.get("stop_type","—")} &nbsp;·&nbsp; '
#             f'Priority: {s.get("priority","—")} &nbsp;·&nbsp; '
#             f'Service: {s.get("service_duration_min","—")} min &nbsp;·&nbsp; '
#             f'Travel: {s.get("travel_time_from_prev_min","—")} min &nbsp;·&nbsp; '
#             f'Dist: {s.get("distance_from_prev_km","—")} km{risk_html}</div>'
#             f'{notes_html}</div></div></div>{connector}',
#             unsafe_allow_html=True,
#         )

#     if itin.get("warnings"):
#         with st.expander("⚠️ Warnings"):
#             for w in itin["warnings"]:
#                 st.warning(w)

#     with st.expander("📝 Optimization Notes"):
#         st.info(itin.get("optimization_notes", "No notes."))

#     # ── Constraint Violation Checker ─────────────────────────────────────────
#     st.markdown("### ⚡ Constraint Violation Check")
#     if constraints:
#         violations = ai.check_violations(itin, constraints)
#         if not violations:
#             st.markdown(
#                 '<div style="background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);'
#                 'border-radius:10px;padding:14px 18px;margin-bottom:12px">'
#                 '<b style="color:#3FB950">✅ No violations detected.</b> '
#                 'All stops are within time windows and driver hour limits.</div>',
#                 unsafe_allow_html=True,
#             )
#         else:
#             crit = [v for v in violations if v["severity"] == "Critical"]
#             warn = [v for v in violations if v["severity"] == "Warning"]
#             vc1, vc2 = st.columns(2)
#             vc1.metric("🔴 Critical", len(crit))
#             vc2.metric("🟡 Warnings", len(warn))
#             for v in violations:
#                 css   = "violation-critical" if v["severity"] == "Critical" else "violation-warning"
#                 color = "#E74C3C"            if v["severity"] == "Critical" else "#D4A843"
#                 icon  = "🔴"                 if v["severity"] == "Critical" else "🟡"
#                 label = {"time_window":"Time Window","driver_hours":"Driver Hours","capacity":"Capacity"}.get(v["type"], v["type"])
#                 st.markdown(
#                     f'<div class="{css}">'
#                     f'<div style="display:flex;justify-content:space-between">'
#                     f'<b style="color:#F0F6FC">{icon} Stop {v["sequence"]} — {v["location_name"]}</b>'
#                     f'<span style="font-size:0.75rem;color:{color};font-weight:600">{label} · {v["severity"]}</span>'
#                     f'</div><div style="font-size:0.82rem;color:#C9D1D9;margin-top:6px">{v["detail"]}</div></div>',
#                     unsafe_allow_html=True,
#                 )
#     else:
#         st.info("Generate an itinerary with constraints to see violation analysis.")

#     # ── Weather Panel ────────────────────────────────────────────────────────
#     # Build a stops list that includes arrival_time so forecasts are time-aware.
#     # Priority: dataset stops_list merged with itinerary arrival times.
#     itin_date = itin.get("date", datetime.now().strftime("%Y-%m-%d"))

#     # Map stop_id → arrival_time from the generated itinerary
#     arrival_map = {
#         s["stop_id"]: s.get("arrival_time", "08:00")
#         for s in itin.get("stops", [])
#     }

#     weather_stops = []
#     if stops_list:
#         # Dataset route: merge lat/lon from stops_list + arrival_time from itinerary
#         for s in stops_list:
#             weather_stops.append({
#                 **s,
#                 "arrival_time": arrival_map.get(s["stop_id"], "08:00"),
#                 "sequence":     next(
#                     (itin_s["sequence"] for itin_s in itin.get("stops", [])
#                      if itin_s["stop_id"] == s["stop_id"]), 0
#                 ),
#             })
#         # Sort by sequence so cards appear in travel order
#         weather_stops.sort(key=lambda x: x.get("sequence", 0))
#     else:
#         # NL / custom stops — coords embedded in itinerary stops
#         for s in itin.get("stops", []):
#             if "lat" in s and "lon" in s:
#                 weather_stops.append({
#                     "stop_id":       s.get("stop_id", ""),
#                     "location_name": s.get("location_name", ""),
#                     "lat":           s["lat"],
#                     "lon":           s["lon"],
#                     "stop_type":     s.get("stop_type", ""),
#                     "arrival_time":  s.get("arrival_time", "08:00"),
#                     "sequence":      s.get("sequence", 0),
#                 })

#     if weather_stops:
#         st.markdown("---")
#         _render_weather_panel(weather_stops, itin_date_str=itin_date)

#     # ── Fuel & Savings Panel ─────────────────────────────────────────────────
#     st.markdown("---")
#     # Build stops_list for fuel engine — prefer stops_list (dataset), else NL
#     fuel_stops = stops_list
#     if not fuel_stops:
#         for s in itin.get("stops", []):
#             if "lat" in s and "lon" in s:
#                 fuel_stops.append({
#                     "stop_id": s.get("stop_id",""),
#                     "location_name": s.get("location_name",""),
#                     "lat": s["lat"], "lon": s["lon"],
#                     "stop_type": s.get("stop_type",""),
#                 })
#     _render_fuel_panel(fuel_stops, constraints, itin)

#     # Adjust
#     st.markdown("### 🔄 Adjust Itinerary")
#     change_req = st.text_area(
#         "Describe the change in plain English:",
#         placeholder="e.g. 'Remove stop 3', 'Add 30-min rest after stop 2', 'Traffic — delay all by 20 min'",
#         height=90,
#     )
#     if st.button("Apply Changes"):
#         if change_req.strip():
#             with st.spinner("Re-planning…"):
#                 updated = ai.adjust_itinerary(itin, change_req)
#             st.session_state["generated_itinerary"] = updated
#             st.success("Itinerary updated!")
#             st.rerun()
#         else:
#             st.error("Please describe the change.")

#     # Export
#     st.markdown("### 📤 Export")
#     col_e1, col_e2 = st.columns(2)
#     with col_e1:
#         st.download_button("⬇️ Download JSON", json.dumps(itin, indent=2, default=str),
#                            "itinerary.json", "application/json", use_container_width=True)
#     with col_e2:
#         stop_rows = itin.get("stops", [])
#         if stop_rows:
#             st.download_button("⬇️ Download CSV", pd.DataFrame(stop_rows).to_csv(index=False),
#                                "itinerary_stops.csv", "text/csv", use_container_width=True)


# # ─────────────────────────────────────────────
# # PAGE: CLUSTERING
# # ─────────────────────────────────────────────
# def page_clustering(stops_df, ml, dash):
#     st.markdown('<div class="section-title">🧩 Stop Clustering Engine</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Unsupervised clustering to find natural groupings in your stop network</div>', unsafe_allow_html=True)

#     if stops_df.empty:
#         st.warning("No stops data loaded.")
#         return

#     col1, col2 = st.columns([1, 3])
#     with col1:
#         n_clusters  = st.slider("Number of Clusters", 2, 8, 4)
#         sample_size = st.slider("Sample Size", 20, min(len(stops_df), 200), min(len(stops_df), 60))
#         if st.button("Run Clustering"):
#             sample_df = stops_df.sample(min(sample_size, len(stops_df)), random_state=42)
#             labels, X_2d = ml.cluster_stops(sample_df, n_clusters)
#             st.session_state["cluster_labels"] = labels
#             st.session_state["cluster_X2d"]   = X_2d
#             st.session_state["cluster_df"]    = sample_df.reset_index(drop=True)
#             st.success("Clustering complete!")

#     with col2:
#         if st.session_state.get("cluster_labels") is not None:
#             st.markdown("**Cluster Visualization (PCA 2D)**")
#             st.plotly_chart(
#                 dash.cluster_scatter(st.session_state["cluster_df"],
#                                      st.session_state["cluster_labels"],
#                                      st.session_state["cluster_X2d"]),
#                 use_container_width=True,
#             )

#     if st.session_state.get("cluster_labels") is not None:
#         labels = st.session_state["cluster_labels"]
#         cdf    = st.session_state["cluster_df"]

#         keywords = ml.get_cluster_keywords()
#         st.markdown("### Cluster Keyword Profiles")
#         cols = st.columns(min(len(keywords), 5))
#         for i, (cid, kws) in enumerate(keywords.items()):
#             with cols[i % len(cols)]:
#                 st.markdown(
#                     f'<div class="card card-gold"><b>Cluster {cid+1}</b><br>'
#                     f'<small style="color:#8B949E">{" · ".join(kws)}</small></div>',
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("### Cluster Composition by Stop Type")
#         cdf2 = cdf.copy()
#         cdf2["cluster"] = [f"Cluster {l+1}" for l in labels]
#         comp = cdf2.groupby(["cluster","stop_type"]).size().reset_index(name="count")
#         fig  = px.bar(comp, x="cluster", y="count", color="stop_type",
#                       color_discrete_map=dash.STOP_COLORS, barmode="stack")
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=320,
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#             xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#         st.markdown("### Geographic Distribution of Clusters")
#         cdf2["x"] = st.session_state["cluster_X2d"][:, 0]
#         # lat/lon available from coordinates parsing in load_stops()
#         fig_geo   = px.scatter_mapbox(
#             cdf2.dropna(subset=["lat","lon"]), lat="lat", lon="lon", color="cluster",
#             hover_data=["location_name","stop_type","priority"], zoom=9, height=380,
#         )
#         fig_geo.update_layout(
#             mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", margin=dict(t=0,b=0,l=0,r=0),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig_geo, use_container_width=True)


# # ─────────────────────────────────────────────
# # PAGE: AI ASSISTANT
# # ─────────────────────────────────────────────
# def page_assistant(stops_df, ai, kb_df):
#     st.markdown('<div class="section-title">💬 Logistics AI Assistant</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Grounded in your logistics knowledge base · '
#         'Only answers travel & logistics questions</div>',
#         unsafe_allow_html=True,
#     )

#     col_s1, col_s2, col_s3 = st.columns(3)
#     for col, label, active in [
#         (col_s1, "LLM",        bool(ai.llm)),
#         (col_s2, "Embeddings", bool(ai.embeddings)),
#         (col_s3, "RAG Index",  bool(ai.embeddings) and not kb_df.empty),
#     ]:
#         color  = "#2EA4A4" if active else "#E74C3C"
#         status = ("🟢 Connected" if active else "🔴 Offline") if label != "RAG Index" else ("🟢 Ready" if active else "⚠️ Keyword fallback")
#         col.markdown(
#             f'<div class="card" style="padding:10px;text-align:center">'
#             f'<span style="font-size:0.8rem;color:#8B949E">{label}</span><br>'
#             f'<b style="color:{color}">{status}</b></div>',
#             unsafe_allow_html=True,
#         )

#     st.markdown("")

#     with st.expander("ℹ️ What can I ask?", expanded=False):
#         st.markdown("""
# **In scope:** route planning, delivery operations, customs, fleet management, freight modes, KPIs, documentation.
# **Out of scope:** questions unrelated to travel or logistics are politely declined.
#         """)

#     if "assistant_history" not in st.session_state:
#         st.session_state["assistant_history"] = []

#     for msg in st.session_state["assistant_history"]:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             if msg["role"] == "assistant":
#                 if msg.get("sources"):
#                     _render_sources(msg["sources"])
#                 if msg.get("rejected"):
#                     st.caption("🚫 Outside the logistics/travel domain.")
#                 elif msg.get("grounded"):
#                     st.caption("✅ Grounded in knowledge base via semantic retrieval.")

#     user_input = st.chat_input("Ask about routes, delivery windows, customs, fuel costs…")
#     if user_input:
#         st.session_state["assistant_history"].append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         with st.chat_message("assistant"):
#             with st.spinner("Searching knowledge base…"):
#                 result = ai.get_kb_answer(user_input, kb_df)
#             answer_text = result["text"]
#             sources     = result.get("sources", [])
#             grounded    = result.get("grounded", False)
#             rejected    = result.get("rejected", False)
#             st.markdown(answer_text)
#             if sources:   _render_sources(sources)
#             if rejected:  st.caption("🚫 Outside the logistics/travel domain.")
#             elif grounded: st.caption("✅ Grounded in knowledge base via semantic retrieval.")
#             else:          st.caption("⚠️ Low-confidence retrieval — broader KB context used.")

#         st.session_state["assistant_history"].append({
#             "role": "assistant", "content": answer_text,
#             "sources": sources, "grounded": grounded, "rejected": rejected,
#         })

#     col_a, _ = st.columns([1, 5])
#     with col_a:
#         if st.button("🗑️ Clear Chat"):
#             st.session_state["assistant_history"] = []
#             st.rerun()

#     if not kb_df.empty:
#         with st.expander(f"📚 Knowledge Base ({len(kb_df)} articles)"):
#             st.dataframe(
#                 kb_df[["category","topic","content"]], use_container_width=True,
#                 column_config={"content": st.column_config.TextColumn("content", width="large")},
#             )


# def _render_sources(sources: list):
#     if not sources:
#         return
#     seen, unique = set(), []
#     for s in sources:
#         key = s.get("topic","")
#         if key not in seen:
#             seen.add(key); unique.append(s)

#     lines = []
#     for s in unique:
#         pct        = int(s.get("score", 0) * 100)
#         bar        = "█" * (pct // 10) + "░" * (10 - pct // 10)
#         lines.append(
#             f"**{s.get('category','—')}** › {s.get('topic','—')} "
#             f"<span style='color:#D4A843;font-family:monospace;font-size:0.75rem'>{bar} {pct}%</span>"
#         )
#     st.markdown(
#         '<div style="background:rgba(46,164,164,0.08);border:1px solid rgba(46,164,164,0.25);'
#         'border-radius:8px;padding:10px 14px;margin-top:8px">'
#         '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.06em;'
#         'text-transform:uppercase;margin-bottom:6px">📎 Sources retrieved</div>'
#         + "".join(f'<div style="font-size:0.8rem;color:#C9D1D9;margin:3px 0">{l}</div>' for l in lines)
#         + "</div>",
#         unsafe_allow_html=True,
#     )


# # ─────────────────────────────────────────────
# # PAGE: PERFORMANCE
# # ─────────────────────────────────────────────
# def page_performance(stops_df, routes_df, ai, dash):
#     st.markdown('<div class="section-title">📈 Performance Analytics</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Route efficiency, on-time delivery, fuel costs, and operational insights</div>', unsafe_allow_html=True)

#     if stops_df.empty or routes_df.empty:
#         st.warning("No data loaded.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     on_time_pct     = round(100 * (stops_df["status"] == "On Time").mean(), 1)
#     avg_distance    = round(routes_df["distance_km"].mean(), 1) if not routes_df.empty else 0
#     total_fuel_cost = round(routes_df["estimated_fuel_cost_inr"].sum(), 0) if not routes_df.empty else 0
#     avg_stops       = round(stops_df.groupby("route_id").size().mean(), 1) if "route_id" in stops_df.columns else 0
#     c1.metric("On-Time Rate",       f"{on_time_pct}%")
#     c2.metric("Avg Distance/Route", f"{avg_distance} km")
#     c3.metric("Total Fuel Cost",    f"₹{total_fuel_cost:,.0f}")
#     c4.metric("Avg Stops/Route",    avg_stops)

#     st.markdown("---")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**On-Time % by Transport Mode (Trend)**")
#         st.plotly_chart(dash.performance_timeline(routes_df), use_container_width=True)
#     with col2:
#         st.markdown("**On-Time Delivery Rate**")
#         st.plotly_chart(dash.on_time_gauge(on_time_pct), use_container_width=True)

#     col3, col4 = st.columns(2)
#     with col3:
#         st.markdown("**Avg Fuel Cost by Transport Mode**")
#         st.plotly_chart(dash.fuel_cost_bar(routes_df), use_container_width=True)
#     with col4:
#         st.markdown("**Stops by Status**")
#         sc = stops_df["status"].value_counts().reset_index()
#         sc.columns = ["status","count"]
#         fig = go.Figure(go.Bar(
#             x=sc["status"], y=sc["count"],
#             marker_color=[dash.STATUS_COLORS.get(s,"#8B949E") for s in sc["status"]],
#             hovertemplate="%{x}: %{y}<extra></extra>",
#         ))
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=260,
#             xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"),
#             margin=dict(t=10, b=40, l=40, r=20),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     if "delay_reason" in stops_df.columns:
#         st.markdown("**Delay Reasons**")
#         delayed = stops_df[stops_df["delay_reason"].notna() & (stops_df["delay_reason"] != "")]
#         if not delayed.empty:
#             dr    = delayed["delay_reason"].value_counts().reset_index()
#             dr.columns = ["reason","count"]
#             fig_d = px.bar(dr, x="count", y="reason", orientation="h",
#                            color_discrete_sequence=["#E74C3C"])
#             fig_d.update_layout(
#                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                 font_color="#C9D1D9", height=260,
#                 xaxis=dict(showgrid=True, gridcolor="#21262D"),
#                 yaxis=dict(showgrid=False),
#                 margin=dict(t=10, b=30, l=150, r=20),
#             )
#             st.plotly_chart(fig_d, use_container_width=True)

#     st.markdown("### 🤖 AI Route Performance Insights")
#     if st.button("Generate AI Insights"):
#         with st.spinner("Analyzing route performance…"):
#             insights = ai.analyze_route_performance(routes_df, stops_df)
#         st.markdown(f'<div class="summary-box">{insights}</div>', unsafe_allow_html=True)

#     if st.session_state.get("plan_history"):
#         with st.expander("📋 Itinerary Generation History (this session)"):
#             st.dataframe(pd.DataFrame(st.session_state["plan_history"]), use_container_width=True)

#     with st.expander("📋 Full Stops Table"):
#         # Show coordinates column; hide raw lat/lon if coordinates present
#         display_df = stops_df.sort_values("scheduled_date", ascending=False).copy()
#         hide_cols  = []
#         if "coordinates" in display_df.columns:
#             if "lat" in display_df.columns: hide_cols.append("lat")
#             if "lon" in display_df.columns: hide_cols.append("lon")
#         if hide_cols:
#             display_df = display_df.drop(columns=hide_cols)
#         st.dataframe(display_df, use_container_width=True)

#     with st.expander("📋 Full Routes Table"):
#         st.dataframe(routes_df.sort_values("route_date", ascending=False), use_container_width=True)


# # ─────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────
# def main():
#     init_session()

#     loader = DataLoader()
#     ai     = AIEngine()
#     ml     = MLEngine()
#     dash   = Dashboard()

#     stops_df  = loader.load_stops()
#     routes_df = loader.load_routes()
#     kb_df     = loader.load_kb()
#     stats     = loader.get_summary_stats(stops_df, routes_df)

#     with st.sidebar:
#         st.markdown(
#             '<div style="text-align:center;padding:20px 0 10px">'
#             '<div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;color:#F0F6FC;font-weight:900">🗺️ RouteIQ</div>'
#             '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.1em;text-transform:uppercase">Logistics Itinerary Planner</div>'
#             '</div>',
#             unsafe_allow_html=True,
#         )
#         st.markdown("---")
#         page = st.radio(
#             "Navigation",
#             ["📊 Overview", "🔍 Itinerary Planner", "🧩 Clustering", "💬 AI Assistant", "📈 Performance"],
#             label_visibility="collapsed",
#         )

#         if not stops_df.empty:
#             st.markdown("---")
#             st.markdown(
#                 '<div style="font-size:0.7rem;color:#8B949E;letter-spacing:0.1em;'
#                 'text-transform:uppercase;margin-bottom:8px">Quick Stats</div>',
#                 unsafe_allow_html=True,
#             )
#             for val, label, color in [
#                 (stats.get("total_stops",  0),              "Total Stops",        "#D4A843"),
#                 (stats.get("total_routes", 0),              "Active Routes",      "#2EA4A4"),
#                 (f"{stats.get('on_time_pct',0)}%",          "On-Time Rate",       "#3FB950"),
#                 (stats.get("high_priority", 0),             "High Priority Stops","#E74C3C"),
#             ]:
#                 st.markdown(
#                     f'<div class="card" style="padding:12px">'
#                     f'<div style="color:{color};font-size:1.4rem;font-weight:700">{val}</div>'
#                     f'<div style="color:#8B949E;font-size:0.72rem">{label}</div></div>',
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("---")
#         st.markdown(
#             f'<div style="font-size:0.75rem;color:#8B949E">'
#             f'{"🟢 LLM Connected" if ai.llm else "🔴 LLM Offline (fallback)"}</div>',
#             unsafe_allow_html=True,
#         )

#     if page == "📊 Overview":
#         page_overview(stops_df, routes_df, stats, dash)
#     elif page == "🔍 Itinerary Planner":
#         page_planner(stops_df, ai, ml, dash)
#     elif page == "🧩 Clustering":
#         page_clustering(stops_df, ml, dash)
#     elif page == "💬 AI Assistant":
#         page_assistant(stops_df, ai, kb_df)
#     elif page == "📈 Performance":
#         page_performance(stops_df, routes_df, ai, dash)


# if __name__ == "__main__":
#     main()

# 1.0 ########################################
# import os

# # ── Tiktoken cache (must be set before any tiktoken / langchain import) ───────
# _tiktoken_cache_dir = os.path.abspath("./token")
# os.makedirs(_tiktoken_cache_dir, exist_ok=True)
# os.environ["TIKTOKEN_CACHE_DIR"] = _tiktoken_cache_dir

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import re
# from datetime import datetime, timedelta
# import random
# import math
# import httpx
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")

# load_dotenv()

# # ─────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="RouteIQ — Logistics Itinerary Planner",
#     page_icon="🗺️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────────────────────────
# # GLOBAL STYLES
# # ─────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

# :root {
#   --midnight: #0D1117;
#   --obsidian: #161B22;
#   --charcoal: #21262D;
#   --steel: #30363D;
#   --ink: #8B949E;
#   --silver: #C9D1D9;
#   --white: #F0F6FC;
#   --gold: #D4A843;
#   --amber: #E8873A;
#   --teal: #2EA4A4;
#   --crimson: #C0392B;
#   --sage: #3FB950;
#   --violet: #8957E5;
#   --blue: #1F6FEB;
# }

# html, body, [class*="css"] {
#   font-family: 'DM Sans', sans-serif !important;
#   background-color: var(--midnight) !important;
#   color: var(--silver) !important;
# }
# section[data-testid="stSidebar"] {
#   background: var(--obsidian) !important;
#   border-right: 1px solid var(--steel) !important;
# }
# section[data-testid="stSidebar"] .stRadio label {
#   color: var(--silver) !important;
#   font-size: 0.9rem !important;
# }
# h1, h2, h3 {
#   font-family: 'Playfair Display', serif !important;
#   color: var(--white) !important;
# }
# [data-testid="metric-container"] {
#   background: var(--obsidian) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 12px !important;
#   padding: 16px !important;
# }
# [data-testid="metric-container"] label {
#   color: var(--ink) !important;
#   font-size: 0.75rem !important;
#   letter-spacing: 0.08em !important;
#   text-transform: uppercase !important;
# }
# [data-testid="metric-container"] [data-testid="stMetricValue"] {
#   color: var(--gold) !important;
#   font-family: 'Playfair Display', serif !important;
#   font-size: 2rem !important;
# }
# details {
#   background: var(--obsidian) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 8px !important;
# }
# details summary { color: var(--teal) !important; font-weight: 500 !important; }
# [data-testid="stDataFrame"] { border: 1px solid var(--steel) !important; border-radius: 8px !important; }
# textarea, input[type="text"] {
#   background: var(--charcoal) !important;
#   color: var(--white) !important;
#   border: 1px solid var(--steel) !important;
#   border-radius: 8px !important;
# }
# .stButton > button {
#   background: linear-gradient(135deg, var(--gold), var(--amber)) !important;
#   color: var(--midnight) !important;
#   font-weight: 600 !important;
#   border: none !important;
#   border-radius: 8px !important;
#   letter-spacing: 0.04em !important;
#   transition: opacity 0.2s !important;
# }
# .stButton > button:hover { opacity: 0.85 !important; }
# [data-testid="stSelectbox"] select, .stSelectbox > div {
#   background: var(--charcoal) !important;
#   color: var(--white) !important;
#   border-color: var(--steel) !important;
# }
# .stTabs [data-baseweb="tab"] { color: var(--ink) !important; border-bottom: 2px solid transparent !important; }
# .stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

# .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; }
# .badge-delivery  { background: rgba(212,168,67,0.2);  color: #D4A843; border: 1px solid rgba(212,168,67,0.4); }
# .badge-meeting   { background: rgba(46,164,164,0.2);  color: #2EA4A4; border: 1px solid rgba(46,164,164,0.4); }
# .badge-pickup    { background: rgba(63,185,80,0.2);   color: #3FB950; border: 1px solid rgba(63,185,80,0.4); }
# .badge-warehouse { background: rgba(137,87,229,0.2);  color: #8957E5; border: 1px solid rgba(137,87,229,0.4); }
# .badge-customs   { background: rgba(192,57,43,0.2);   color: #E74C3C; border: 1px solid rgba(192,57,43,0.4); }
# .badge-rest      { background: rgba(139,148,158,0.2); color: #8B949E; border: 1px solid rgba(139,148,158,0.4); }

# .card { background: var(--obsidian); border: 1px solid var(--steel); border-radius: 12px; padding: 20px; margin-bottom: 12px; }
# .card-gold  { border-left: 4px solid var(--gold); }
# .card-teal  { border-left: 4px solid var(--teal); }
# .card-red   { border-left: 4px solid var(--crimson); }
# .card-green { border-left: 4px solid var(--sage); }
# .card-blue  { border-left: 4px solid var(--blue); }

# .section-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: var(--white); margin-bottom: 4px; }
# .section-sub   { color: var(--ink); font-size: 0.85rem; margin-bottom: 20px; letter-spacing: 0.04em; }
# .hero-title    { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 900; color: var(--white); line-height: 1.1; }
# .hero-accent   { color: var(--gold); }

# .stop-card { background: var(--charcoal); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# .stop-num  { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; background: var(--gold); color: var(--midnight); border-radius: 50%; font-weight: 700; font-size: 0.85rem; margin-right: 10px; flex-shrink: 0; }
# .stop-row  { display: flex; align-items: flex-start; }
# .stop-detail { font-size: 0.82rem; color: var(--ink); margin-top: 4px; }
# .connector-line { width: 2px; height: 30px; background: linear-gradient(var(--gold), var(--teal)); margin: 0 auto 0 13px; }

# .summary-box { background: linear-gradient(135deg, rgba(212,168,67,0.08), rgba(46,164,164,0.08)); border: 1px solid rgba(212,168,67,0.3); border-radius: 12px; padding: 20px; margin: 12px 0; }
# .itinerary-header { background: linear-gradient(135deg, rgba(31,111,235,0.15), rgba(46,164,164,0.15)); border: 1px solid rgba(31,111,235,0.3); border-radius: 12px; padding: 18px 22px; margin-bottom: 18px; }
# .violation-critical { background: rgba(192,57,43,0.1); border-left: 4px solid #E74C3C; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
# .violation-warning  { background: rgba(212,168,67,0.1); border-left: 4px solid #D4A843; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
# .weather-card { background: var(--charcoal); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 16px; text-align: center; }
# .fuel-card    { background: var(--obsidian); border: 1px solid var(--steel); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# .fuel-save    { background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.35); border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; }
# .fuel-warn    { background: rgba(192,57,43,0.08); border: 1px solid rgba(192,57,43,0.3); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# # SESSION STATE
# # ─────────────────────────────────────────────
# def init_session():
#     defaults = {
#         "generated_itinerary": None,
#         "batch_itineraries":   [],
#         "cluster_labels":      None,
#         "cluster_X2d":         None,
#         "cluster_df":          None,
#         "assistant_history":   [],
#         "plan_history":        [],
#         "last_constraints":    {},
#         "nl_parsed_result":    None,
#         "weather_cache":       {},      # key: "lat_lon" → weather dict
#         "fuel_analysis":       None,    # last fuel/savings analysis result
#         "itinerary_source":    None,    # "dataset" | "custom" | "nl" — which tab owns it
#         "nl_stops_for_map":    [],      # lat/lon list for NL itinerary map
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v


# # ─────────────────────────────────────────────
# # DATA LOADER
# # ─────────────────────────────────────────────
# class DataLoader:
#     STOPS_FILE  = "stops.csv"
#     ROUTES_FILE = "routes.csv"
#     KB_FILE     = "logistics_kb.csv"

#     def load_stops(self):
#         if not os.path.exists(self.STOPS_FILE):
#             self._generate_and_save()
#         return pd.read_csv(self.STOPS_FILE, parse_dates=["time_window_start", "time_window_end"])

#     def load_routes(self):
#         if not os.path.exists(self.ROUTES_FILE):
#             self._generate_and_save()
#         return pd.read_csv(self.ROUTES_FILE)

#     def load_kb(self):
#         if not os.path.exists(self.KB_FILE):
#             self._generate_and_save()
#         return pd.read_csv(self.KB_FILE)

#     def _generate_and_save(self):
#         from generate_data import generate_stops, generate_routes, generate_kb
#         stops  = generate_stops(60)
#         routes = generate_routes(stops)
#         kb     = generate_kb()
#         stops.to_csv(self.STOPS_FILE,  index=False)
#         routes.to_csv(self.ROUTES_FILE, index=False)
#         kb.to_csv(self.KB_FILE,         index=False)

#     def get_summary_stats(self, stops_df, routes_df):
#         if stops_df.empty:
#             return {}
#         return {
#             "total_stops":         len(stops_df),
#             "total_routes":        stops_df["route_id"].nunique() if "route_id" in stops_df.columns else 0,
#             "total_distance_km":   round(routes_df["distance_km"].sum(), 1) if not routes_df.empty else 0,
#             "avg_stops_per_route": round(stops_df.groupby("route_id").size().mean(), 1) if "route_id" in stops_df.columns else 0,
#             "on_time_pct":         round(100 * (stops_df["status"] == "On Time").mean(), 1) if "status" in stops_df.columns else 0,
#             "high_priority":       int((stops_df["priority"] == "High").sum()) if "priority" in stops_df.columns else 0,
#         }


# # ─────────────────────────────────────────────
# # ML ENGINE
# # ─────────────────────────────────────────────
# class MLEngine:
#     def __init__(self):
#         self.vectorizer     = None
#         self.kmeans         = None
#         self.feature_matrix = None

#     def cluster_stops(self, stops_df, n_clusters=4):
#         texts = (
#             stops_df["stop_type"].fillna("") + " " +
#             stops_df["location_name"].fillna("") + " " +
#             stops_df["notes"].fillna("")
#         )
#         self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
#         X = self.vectorizer.fit_transform(texts)
#         self.feature_matrix = X
#         self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         labels = self.kmeans.fit_predict(X)
#         pca    = PCA(n_components=2, random_state=42)
#         X_2d   = pca.fit_transform(X.toarray())
#         return labels, X_2d

#     def get_cluster_keywords(self, top_n=5):
#         if self.vectorizer is None or self.kmeans is None:
#             return {}
#         terms    = self.vectorizer.get_feature_names_out()
#         keywords = {}
#         for i, center in enumerate(self.kmeans.cluster_centers_):
#             top_idx     = center.argsort()[-top_n:][::-1]
#             keywords[i] = [terms[j] for j in top_idx]
#         return keywords

#     def nearest_neighbor_route(self, coords):
#         if len(coords) <= 1:
#             return list(range(len(coords)))
#         unvisited = list(range(1, len(coords)))
#         route     = [0]
#         while unvisited:
#             curr    = route[-1]
#             nearest = min(unvisited, key=lambda j: self._haversine(coords[curr], coords[j]))
#             route.append(nearest)
#             unvisited.remove(nearest)
#         return route

#     @staticmethod
#     def _haversine(c1, c2):
#         lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
#         lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
#         a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
#         return 6371 * 2 * math.asin(math.sqrt(a))

#     @staticmethod
#     def compute_distance_km(lat1, lon1, lat2, lon2):
#         return MLEngine._haversine((lat1, lon1), (lat2, lon2))

#     @staticmethod
#     def osrm_route(coords: list) -> dict:
#         """
#         Real road routing via public OSRM demo API.
#         Falls back to Haversine (35 km/h) if OSRM is unreachable.

#         Args:
#             coords: list of (lat, lon) tuples in stop order

#         Returns:
#             {legs, total_distance_km, total_duration_min, source}
#         """
#         if len(coords) < 2:
#             return {"legs": [], "total_distance_km": 0.0,
#                     "total_duration_min": 0.0, "source": "osrm"}

#         waypoints = ";".join(f"{lon},{lat}" for lat, lon in coords)
#         url = (
#             "http://router.project-osrm.org/route/v1/driving/" + waypoints
#             + "?overview=false&steps=false&annotations=false"
#         )
#         try:
#             resp = httpx.get(url, timeout=8.0)
#             data = resp.json()
#             if data.get("code") != "Ok":
#                 raise ValueError(f"OSRM code: {data.get('code')}")
#             legs = [
#                 {
#                     "distance_km":  round(leg["distance"] / 1000, 2),
#                     "duration_min": round(leg["duration"] / 60, 1),
#                 }
#                 for leg in data["routes"][0]["legs"]
#             ]
#             return {
#                 "legs":               legs,
#                 "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
#                 "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
#                 "source":             "osrm",
#             }
#         except Exception:
#             legs = []
#             for i in range(len(coords) - 1):
#                 d = MLEngine._haversine(coords[i], coords[i + 1])
#                 legs.append({"distance_km": round(d, 2), "duration_min": round(d / 35 * 60, 1)})
#             return {
#                 "legs":               legs,
#                 "total_distance_km":  round(sum(l["distance_km"]  for l in legs), 2),
#                 "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
#                 "source":             "haversine_fallback",
#             }


# # ─────────────────────────────────────────────
# # WEATHER ENGINE  (Open-Meteo — no API key)
# # ─────────────────────────────────────────────
# class WeatherEngine:
#     """
#     Arrival-time-aware weather forecasting via Open-Meteo hourly API (free, no key).

#     For each stop we fetch the HOURLY forecast for that location, then pick the
#     hour slot that matches the stop's expected arrival_time.  This means:
#       - Stop A arriving at 09:00 → forecast for 09:00
#       - Stop B arriving at 13:00 → forecast for 13:00
#       - etc.

#     Caching strategy:
#       - Full hourly payload cached by lat/lon key (one API call per unique location).
#       - Individual hour lookups extracted from the cached payload — no repeat calls.

#     Fallback chain (guaranteed to always return usable data):
#       1. Open-Meteo HTTPS (verify=False for internal proxies)
#       2. Open-Meteo HTTP (plain, bypasses TLS entirely)
#       3. Seasonal estimate (hardcoded India climate by month + latitude band)
#     """

#     WMO_CODES = {
#         0:  ("Clear sky",            "☀️"),
#         1:  ("Mainly clear",         "🌤️"),
#         2:  ("Partly cloudy",        "⛅"),
#         3:  ("Overcast",             "☁️"),
#         45: ("Fog",                  "🌫️"),
#         48: ("Icy fog",              "🌫️"),
#         51: ("Light drizzle",        "🌦️"),
#         53: ("Moderate drizzle",     "🌦️"),
#         55: ("Dense drizzle",        "🌧️"),
#         61: ("Slight rain",          "🌧️"),
#         63: ("Moderate rain",        "🌧️"),
#         65: ("Heavy rain",           "🌧️"),
#         71: ("Slight snow",          "🌨️"),
#         73: ("Moderate snow",        "❄️"),
#         75: ("Heavy snow",           "❄️"),
#         80: ("Slight showers",       "🌦️"),
#         81: ("Moderate showers",     "🌧️"),
#         82: ("Heavy showers",        "⛈️"),
#         95: ("Thunderstorm",         "⛈️"),
#         96: ("Thunderstorm + hail",  "⛈️"),
#         99: ("Severe thunderstorm",  "⛈️"),
#     }
#     ADVERSE_CODES = {63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99}

#     # ── Seasonal fallback (India, month × lat band) ───────────────────────────
#     _SEASONAL_FALLBACK = {
#         1:  {"north": (18, 60, "Mainly clear",  "🌤️", 1),  "south": (28, 65, "Partly cloudy", "⛅", 2)},
#         2:  {"north": (20, 58, "Mainly clear",  "🌤️", 1),  "south": (30, 62, "Mainly clear",  "🌤️", 1)},
#         3:  {"north": (27, 55, "Clear sky",     "☀️", 0),  "south": (33, 60, "Clear sky",     "☀️", 0)},
#         4:  {"north": (33, 50, "Clear sky",     "☀️", 0),  "south": (35, 65, "Clear sky",     "☀️", 0)},
#         5:  {"north": (37, 45, "Clear sky",     "☀️", 0),  "south": (35, 70, "Partly cloudy", "⛅", 2)},
#         6:  {"north": (34, 75, "Moderate rain", "🌧️", 63), "south": (30, 85, "Heavy rain",    "🌧️", 65)},
#         7:  {"north": (30, 82, "Heavy rain",    "🌧️", 65), "south": (28, 88, "Heavy showers", "⛈️", 82)},
#         8:  {"north": (30, 80, "Moderate rain", "🌧️", 63), "south": (28, 86, "Heavy rain",    "🌧️", 65)},
#         9:  {"north": (29, 78, "Slight rain",   "🌦️", 61), "south": (29, 82, "Moderate rain", "🌧️", 63)},
#         10: {"north": (26, 65, "Partly cloudy", "⛅", 2),  "south": (29, 72, "Partly cloudy", "⛅", 2)},
#         11: {"north": (21, 58, "Mainly clear",  "🌤️", 1),  "south": (28, 68, "Mainly clear",  "🌤️", 1)},
#         12: {"north": (16, 62, "Mainly clear",  "🌤️", 1),  "south": (27, 65, "Partly cloudy", "⛅", 2)},
#     }

#     @staticmethod
#     def _cache_key(lat: float, lon: float) -> str:
#         return f"{round(lat, 3)}_{round(lon, 3)}"

#     @classmethod
#     def _seasonal_estimate(cls, lat: float, lon: float, arrival_hour: int = None) -> dict:
#         """Plausible weather estimate based on season, location, and time of day."""
#         import random as _rnd
#         month = datetime.now().month
#         band  = "north" if lat > 20 else "south"
#         row   = cls._SEASONAL_FALLBACK.get(month, cls._SEASONAL_FALLBACK[6])
#         t, h, desc, icon, code = row[band]

#         seed = int(abs(lat * 1000 + lon * 100)) % 100
#         # Diurnal temperature variation: cooler at dawn/late evening
#         hour_offset = 0
#         if arrival_hour is not None:
#             if arrival_hour < 7:    hour_offset = -4
#             elif arrival_hour < 10: hour_offset = -2
#             elif arrival_hour < 14: hour_offset =  2
#             elif arrival_hour < 17: hour_offset =  3
#             elif arrival_hour < 20: hour_offset =  1

#         t    = round(t + hour_offset + (_rnd.Random(seed).random() - 0.5) * 2, 1)
#         wind = round(8  + _rnd.Random(seed + 1).random() * 12, 1)
#         prec = round(_rnd.Random(seed + 2).random() * (5 if code >= 61 else 0.2), 1)
#         vis  = round(6  + _rnd.Random(seed + 3).random() * 4, 1) if code >= 45 else round(9 + _rnd.Random(seed + 3).random() * 5, 1)
#         return {
#             "temperature_c":    t,
#             "wind_speed_kmh":   wind,
#             "precipitation_mm": prec,
#             "humidity_pct":     h,
#             "visibility_km":    vis,
#             "weather_code":     code,
#             "description":      desc,
#             "icon":             icon,
#             "is_adverse":       code in cls.ADVERSE_CODES,
#             "source":           "seasonal-estimate",
#             "forecast_hour":    arrival_hour,
#         }

#     # ── Core: fetch full 48-h hourly payload for a location ──────────────────
#     @classmethod
#     def _fetch_hourly_payload(cls, lat: float, lon: float) -> dict | None:
#         """
#         Fetch 48-hour hourly forecast from Open-Meteo.
#         Returns the raw API response dict, or None on failure.
#         Payload cached in session state under key "hourly_payload_{lat}_{lon}".
#         """
#         cache_key = f"hourly_payload_{cls._cache_key(lat, lon)}"
#         cache     = st.session_state.get("weather_cache", {})
#         if cache_key in cache:
#             return cache[cache_key]

#         params = (
#             f"?latitude={lat}&longitude={lon}"
#             f"&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,"
#             f"precipitation,weather_code,wind_speed_10m,visibility"
#             f"&wind_speed_unit=kmh"
#             f"&timezone=Asia%2FKolkata"
#             f"&forecast_days=2"          # today + tomorrow — covers any same-day route
#         )
#         urls = [
#             "https://api.open-meteo.com/v1/forecast" + params,
#             "http://api.open-meteo.com/v1/forecast"  + params,
#         ]
#         for url in urls:
#             try:
#                 resp = httpx.get(url, timeout=8.0, verify=False, follow_redirects=True)
#                 if resp.status_code != 200:
#                     continue
#                 data = resp.json()
#                 if "hourly" not in data or "time" not in data.get("hourly", {}):
#                     continue
#                 cache[cache_key] = data
#                 st.session_state["weather_cache"] = cache
#                 return data
#             except Exception:
#                 continue
#         return None   # both URLs failed

#     # ── Extract one hour slot from the payload ────────────────────────────────
#     @classmethod
#     def _extract_hour(cls, payload: dict, target_dt: datetime) -> dict:
#         """
#         Given a full hourly payload, extract the slot closest to target_dt.
#         Open-Meteo returns ISO strings like "2025-06-15T09:00" in the "time" array.
#         """
#         hourly     = payload["hourly"]
#         time_strs  = hourly["time"]           # list of "YYYY-MM-DDTHH:00"
#         target_str = target_dt.strftime("%Y-%m-%dT%H:00")

#         # Find exact match first, then nearest
#         idx = None
#         if target_str in time_strs:
#             idx = time_strs.index(target_str)
#         else:
#             # Find closest hour
#             best_diff = float("inf")
#             for i, ts in enumerate(time_strs):
#                 try:
#                     dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
#                     diff = abs((dt - target_dt).total_seconds())
#                     if diff < best_diff:
#                         best_diff = diff
#                         idx = i
#                 except Exception:
#                     continue

#         if idx is None:
#             return {}

#         def _val(key, default=0):
#             arr = hourly.get(key, [])
#             v   = arr[idx] if idx < len(arr) else default
#             return v if v is not None else default

#         code       = int(_val("weather_code", 0))
#         desc, icon = cls.WMO_CODES.get(code, ("Partly cloudy", "⛅"))
#         vis_raw    = _val("visibility", 10000)

#         return {
#             "temperature_c":        round(float(_val("temperature_2m",          25)), 1),
#             "wind_speed_kmh":       round(float(_val("wind_speed_10m",          10)), 1),
#             "precipitation_mm":     round(float(_val("precipitation",             0)), 1),
#             "precip_probability":   int(_val("precipitation_probability",          0)),
#             "humidity_pct":         int(_val("relative_humidity_2m",             65)),
#             "visibility_km":        round(float(vis_raw) / 1000, 1),
#             "weather_code":         code,
#             "description":          desc,
#             "icon":                 icon,
#             "is_adverse":           code in cls.ADVERSE_CODES,
#             "source":               "open-meteo-hourly",
#         }

#     # ── Public API: fetch weather at a specific arrival time ─────────────────
#     @classmethod
#     def fetch_at_time(cls, lat: float, lon: float, arrival_time_str: str,
#                       itin_date_str: str = None) -> dict:
#         """
#         Return weather forecast for (lat, lon) at the stop's arrival time.

#         Args:
#             lat, lon:           Stop coordinates.
#             arrival_time_str:   "HH:MM" string from the itinerary stop.
#             itin_date_str:      "YYYY-MM-DD" from itin["date"]; defaults to today.

#         Returns a weather dict with an extra "forecast_time" field showing
#         the exact datetime we fetched for.
#         """
#         # Parse target datetime
#         try:
#             date_str = itin_date_str or datetime.now().strftime("%Y-%m-%d")
#             target   = datetime.strptime(f"{date_str} {arrival_time_str}", "%Y-%m-%d %H:%M")
#         except Exception:
#             target   = datetime.now()

#         # Per-stop cache key (includes the target hour)
#         stop_key  = f"stop_{cls._cache_key(lat, lon)}_{target.strftime('%Y%m%d%H')}"
#         cache     = st.session_state.get("weather_cache", {})
#         if stop_key in cache:
#             return cache[stop_key]

#         # Fetch (or reuse) the hourly payload for this location
#         payload = cls._fetch_hourly_payload(lat, lon)

#         if payload:
#             result = cls._extract_hour(payload, target)
#             if result:
#                 result["forecast_time"]   = target.strftime("%d %b %Y · %H:%M")
#                 result["arrival_time_str"] = arrival_time_str
#                 cache[stop_key] = result
#                 st.session_state["weather_cache"] = cache
#                 return result

#         # Fallback to seasonal estimate
#         result = cls._seasonal_estimate(lat, lon, arrival_hour=target.hour)
#         result["forecast_time"]   = target.strftime("%d %b %Y · %H:%M") + " (est.)"
#         result["arrival_time_str"] = arrival_time_str
#         cache[stop_key] = result
#         st.session_state["weather_cache"] = cache
#         return result

#     # ── Convenience: fetch for all stops in an itinerary ─────────────────────
#     @classmethod
#     def fetch_route_at_times(cls, stops_with_coords: list, itin_date_str: str = None) -> list:
#         """
#         Fetch arrival-time-aware forecast for every stop.

#         Args:
#             stops_with_coords: list of dicts with at minimum:
#                 {stop_id, location_name, lat, lon, arrival_time (HH:MM)}
#             itin_date_str: itinerary date "YYYY-MM-DD"

#         Returns: same list with an added "weather" key per stop.
#         """
#         results = []
#         for s in stops_with_coords:
#             arrival = s.get("arrival_time", "08:00") or "08:00"
#             w = cls.fetch_at_time(s["lat"], s["lon"], arrival, itin_date_str)
#             results.append({**s, "weather": w})
#         return results

#     # ── Legacy: fetch current weather (kept for backward compat) ─────────────
#     @classmethod
#     def fetch(cls, lat: float, lon: float) -> dict:
#         """Fetch current conditions (no arrival time). Used as generic fallback."""
#         return cls.fetch_at_time(lat, lon, datetime.now().strftime("%H:%M"))

#     @classmethod
#     def fetch_route(cls, stops_with_coords: list) -> list:
#         """Legacy: fetch current conditions for a list of stops."""
#         return cls.fetch_route_at_times(stops_with_coords)


# # ─────────────────────────────────────────────
# # FUEL ENGINE  (static reference prices — India)
# # ─────────────────────────────────────────────
# class FuelEngine:
#     """
#     Static Indian petrol/diesel reference prices (as of June 2025).
#     Vehicle efficiency lookup table.
#     Fuel savings comparisons between optimized vs un-optimized route order.

#     Reference: https://www.goodreturns.in/petrol-price.html
#     Prices in ₹ per litre. Update CITY_PRICES dict as needed.
#     """

#     # ── Static city-level prices (₹/litre) ───────────────────────────────────
#     CITY_PRICES = {
#         # Maharashtra
#         "Mumbai":     {"petrol": 103.44, "diesel": 89.97},
#         "Pune":       {"petrol": 103.57, "diesel": 90.10},
#         "Nashik":     {"petrol": 103.20, "diesel": 89.75},
#         "Nagpur":     {"petrol": 103.80, "diesel": 90.15},
#         "Aurangabad": {"petrol": 103.35, "diesel": 89.85},
#         # Delhi NCR
#         "Delhi":      {"petrol": 94.72,  "diesel": 87.62},
#         "Gurgaon":    {"petrol": 95.10,  "diesel": 88.05},
#         "Noida":      {"petrol": 94.85,  "diesel": 87.80},
#         # Karnataka
#         "Bengaluru":  {"petrol": 102.86, "diesel": 88.94},
#         "Mysuru":     {"petrol": 102.60, "diesel": 88.70},
#         # Tamil Nadu
#         "Chennai":    {"petrol": 100.75, "diesel": 92.34},
#         "Coimbatore": {"petrol": 100.50, "diesel": 92.10},
#         # Telangana / AP
#         "Hyderabad":  {"petrol": 107.41, "diesel": 95.65},
#         "Visakhapatnam": {"petrol": 106.80, "diesel": 95.20},
#         # Gujarat
#         "Ahmedabad":  {"petrol": 96.63,  "diesel": 92.38},
#         "Surat":      {"petrol": 96.50,  "diesel": 92.25},
#         # Rajasthan
#         "Jaipur":     {"petrol": 104.88, "diesel": 90.36},
#         # Uttar Pradesh
#         "Lucknow":    {"petrol": 94.65,  "diesel": 87.76},
#         "Kanpur":     {"petrol": 94.55,  "diesel": 87.65},
#         # West Bengal
#         "Kolkata":    {"petrol": 103.94, "diesel": 90.76},
#         # Default (national average)
#         "default":    {"petrol": 101.50, "diesel": 90.00},
#     }

#     # ── Vehicle fuel efficiency (km per litre) ────────────────────────────────
#     VEHICLE_EFFICIENCY = {
#         "Truck":       5.5,   # heavy truck (10-16T)
#         "Van":         12.0,  # light commercial van
#         "Tempo":       9.0,   # mini-truck / tempo
#         "Car":         15.0,  # passenger car
#         "Motorcycle":  40.0,  # two-wheeler
#         "default":     8.0,
#     }

#     # ── Fuel type by vehicle ──────────────────────────────────────────────────
#     VEHICLE_FUEL_TYPE = {
#         "Truck": "diesel", "Tempo": "diesel",
#         "Van": "diesel", "Car": "petrol", "Motorcycle": "petrol",
#     }

#     @classmethod
#     def get_price(cls, city: str, fuel_type: str = "diesel") -> float:
#         """Find the closest city match (case-insensitive substring)."""
#         city_lower = city.lower()
#         for name, prices in cls.CITY_PRICES.items():
#             if name.lower() in city_lower or city_lower in name.lower():
#                 return prices.get(fuel_type, prices["diesel"])
#         return cls.CITY_PRICES["default"].get(fuel_type, 90.0)

#     @classmethod
#     def compute_fuel_cost(
#         cls,
#         distance_km: float,
#         vehicle_type: str,
#         city: str = "Mumbai",
#         override_price: float = None,
#         override_efficiency: float = None,
#     ) -> dict:
#         """
#         Compute fuel cost for a given distance.

#         Returns:
#             {litres_consumed, price_per_litre, total_cost_inr,
#              efficiency_kmpl, fuel_type, city}
#         """
#         fuel_type  = cls.VEHICLE_FUEL_TYPE.get(vehicle_type, "diesel")
#         efficiency = override_efficiency or cls.VEHICLE_EFFICIENCY.get(vehicle_type, cls.VEHICLE_EFFICIENCY["default"])
#         price      = override_price      or cls.get_price(city, fuel_type)
#         litres     = distance_km / efficiency if efficiency > 0 else 0
#         cost       = round(litres * price, 2)
#         return {
#             "litres_consumed":  round(litres, 2),
#             "price_per_litre":  price,
#             "total_cost_inr":   cost,
#             "efficiency_kmpl":  efficiency,
#             "fuel_type":        fuel_type,
#             "city":             city,
#         }

#     @classmethod
#     def savings_analysis(
#         cls,
#         stops_list: list,
#         vehicle_type: str,
#         city: str = "Mumbai",
#         override_price: float = None,
#         override_efficiency: float = None,
#     ) -> dict:
#         """
#         Compare un-optimized (original stop order) vs NN-optimized order.

#         Returns:
#             {
#                 original_km, optimized_km, saved_km, saving_pct,
#                 original_cost, optimized_cost, saved_cost_inr,
#                 original_order, optimized_order,
#                 fuel_detail_original, fuel_detail_optimized,
#                 per_stop_savings: [{stop, original_leg_km, optimized_leg_km}]
#             }
#         """
#         if len(stops_list) < 2:
#             return {}

#         coords = [(s["lat"], s["lon"]) for s in stops_list]

#         # Original order — OSRM
#         osrm_orig = MLEngine.osrm_route(coords)
#         orig_km   = osrm_orig["total_distance_km"]

#         # NN-optimized order — OSRM
#         nn_order  = MLEngine().__class__().nearest_neighbor_route.__func__(MLEngine(), coords)
#         coords_nn = [coords[i] for i in nn_order]
#         osrm_opt  = MLEngine.osrm_route(coords_nn)
#         opt_km    = osrm_opt["total_distance_km"]

#         # Use optimized if it's actually better, else keep original
#         if opt_km >= orig_km:
#             opt_km    = orig_km
#             osrm_opt  = osrm_orig
#             nn_order  = list(range(len(stops_list)))

#         saved_km   = round(orig_km - opt_km, 2)
#         saving_pct = round(saved_km / orig_km * 100, 1) if orig_km > 0 else 0

#         fuel_orig = cls.compute_fuel_cost(orig_km, vehicle_type, city, override_price, override_efficiency)
#         fuel_opt  = cls.compute_fuel_cost(opt_km,  vehicle_type, city, override_price, override_efficiency)
#         saved_cost = round(fuel_orig["total_cost_inr"] - fuel_opt["total_cost_inr"], 2)

#         # Per-leg comparison (zip original vs optimized legs)
#         per_stop = []
#         orig_legs = osrm_orig.get("legs", [])
#         opt_legs  = osrm_opt.get("legs", [])
#         for i, s in enumerate(stops_list[1:]):
#             orig_leg = orig_legs[i]["distance_km"] if i < len(orig_legs) else 0
#             opt_s    = stops_list[nn_order[i+1]] if (i+1) < len(nn_order) else s
#             opt_leg  = opt_legs[i]["distance_km"] if i < len(opt_legs) else 0
#             per_stop.append({
#                 "leg":           i + 1,
#                 "original_stop": s["location_name"],
#                 "optimized_stop":opt_s["location_name"],
#                 "original_km":   orig_leg,
#                 "optimized_km":  opt_leg,
#                 "saved_km":      round(orig_leg - opt_leg, 2),
#             })

#         return {
#             "original_km":          orig_km,
#             "optimized_km":         opt_km,
#             "saved_km":             saved_km,
#             "saving_pct":           saving_pct,
#             "original_cost_inr":    fuel_orig["total_cost_inr"],
#             "optimized_cost_inr":   fuel_opt["total_cost_inr"],
#             "saved_cost_inr":       saved_cost,
#             "fuel_detail_original": fuel_orig,
#             "fuel_detail_optimized":fuel_opt,
#             "original_order":       [s["location_name"] for s in stops_list],
#             "optimized_order":      [stops_list[i]["location_name"] for i in nn_order],
#             "per_stop_savings":     per_stop,
#             "routing_source":       osrm_orig["source"],
#         }


# # ─────────────────────────────────────────────
# # RAG ENGINE
# # ─────────────────────────────────────────────
# class RAGEngine:
#     DOMAIN_TOPICS = [
#         # ── Core logistics ───────────────────────────────────────────────────
#         "route", "routing", "delivery", "logistics", "shipment", "freight",
#         "itinerary", "stop", "waypoint", "depot", "warehouse", "pickup",
#         "dispatch", "fleet", "vehicle", "driver", "trucking", "transport",
#         "cargo", "customs", "e-way bill", "manifest", "consignment",
#         "last mile", "first mile", "supply chain", "distribution",
#         "fuel", "mileage", "navigation", "gps", "tracking",
#         "time window", "schedule", "delay", "on-time", "eta", "arrival",
#         "temperature", "cold chain", "refrigerated", "hazmat", "dangerous goods",
#         "pod", "proof of delivery", "invoice", "bill of lading",
#         "kpi", "performance", "efficiency", "cost", "optimization",
#         "travel", "distance", "trip", "journey", "road", "highway",
#         "rail", "air freight", "sea freight", "port", "airport",
#         "tms", "wms", "erp", "telematics", "iot",
#         "breakdown", "insurance", "claim", "compliance", "regulation",
#         # ── Natural navigation / direction language ───────────────────────────
#         # Verbs people use when asking about getting from A to B
#         "go from", "going from", "get from", "getting from",
#         "travel from", "travelling from", "traveling from",
#         "reach", "reaching", "how to reach", "how do i reach",
#         "drive from", "driving from", "ride from", "riding from",
#         "commute", "commuting",
#         "best way", "fastest way", "shortest way", "quickest way",
#         "how long", "how far", "how much time",
#         "directions", "direction", "navigate", "path from", "path to",
#         "way to", "way from", "route from", "route to",
#         "from here", "to here",
#         # ── Relational / between ─────────────────────────────────────────────
#         "between",
#         # ── Movement / transit words ─────────────────────────────────────────
#         "bus", "train", "metro", "cab", "auto", "taxi", "uber", "ola",
#         "toll", "highway", "expressway", "flyover", "bridge",
#         "traffic", "congestion", "jam", "detour", "bypass",
#         "drop", "pick up", "pickup point", "drop off",
#         # ── Indian city / area names (common logistics hubs) ─────────────────
#         # Mumbai
#         "mumbai", "bombay", "bandra", "kurla", "andheri", "dadar",
#         "thane", "navi mumbai", "panvel", "borivali", "kandivali",
#         "malad", "goregaon", "jogeshwari", "vile parle", "santacruz",
#         "bkc", "nariman", "churchgate", "csmt", "colaba", "worli",
#         "lower parel", "prabhadevi", "matunga", "sion", "chembur",
#         "ghatkopar", "vikhroli", "kanjurmarg", "bhandup", "mulund",
#         "dombivli", "kalyan", "bhiwandi", "vasai", "virar", "mira road",
#         "nhava sheva", "jnpt", "nhava",
#         # Pune
#         "pune", "pimpri", "chinchwad", "hadapsar", "kothrud", "hinjewadi",
#         "wakad", "baner", "aundh", "shivajinagar", "talegaon",
#         # Other major cities
#         "delhi", "ncr", "gurgaon", "noida", "faridabad", "ghaziabad",
#         "bengaluru", "bangalore", "whitefield", "electronic city",
#         "hyderabad", "secunderabad", "cyberabad",
#         "chennai", "kolkata", "ahmedabad", "surat", "jaipur",
#         "lucknow", "chandigarh", "coimbatore", "kochi", "indore",
#         "nagpur", "nashik", "aurangabad", "visakhapatnam",
#         # ── Logistics / geographic terms ─────────────────────────────────────
#         "zone", "area", "sector", "block", "lane", "street", "nagar",
#         "colony", "society", "industrial area", "industrial estate",
#         "cargo hub", "logistics park", "cold storage", "godown",
#         "weighbridge", "octroi", "rto", "check post", "border",
#     ]
#     RELEVANCE_THRESHOLD = 0.30

#     # Regex patterns that strongly indicate a route / navigation question
#     # regardless of specific keywords — catches "from X to Y" style queries
#     _NAV_PATTERNS = [
#         r"\bfrom\b.{1,60}\bto\b",        # "from bandra to kurla"
#         r"\bgo\b.{0,40}\bto\b",           # "go to andheri"
#         r"\bget\b.{0,40}\bto\b",          # "get to the depot"
#         r"\bread?ch\b",                      # "reach" / "reaching"
#         r"\bdriv(e|ing)\b.{0,40}\bto\b",  # "drive to"
#         r"\bhow\b.{0,30}\blong\b",        # "how long does it take"
#         r"\bhow\b.{0,30}\bfar\b",         # "how far is X from Y"
#         r"\bdir?ections?\b",                 # "directions" / "direction"
#         r"\bnear(est)?\b",                   # "nearest depot"
#         r"\bway\b.{0,30}\bto\b",          # "best way to reach"
#         r"\broute\b.{0,30}\bfrom\b",      # "route from X"
#         r"\bpath\b.{0,30}\bto\b",         # "path to warehouse"
#     ]

#     def __init__(self, llm, embeddings):
#         self.llm       = llm
#         self.embeddings = embeddings
#         self._vectordb  = None

#     def _build_vectordb(self, kb_df):
#         if self._vectordb is not None or self.embeddings is None:
#             return
#         raw_docs, metadatas = [], []
#         for _, row in kb_df.iterrows():
#             raw_docs.append(f"[{row['category']} — {row['topic']}]\n{row['content']}")
#             metadatas.append({"category": row["category"], "topic": row["topic"]})
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
#         )
#         chunks, chunk_metas = [], []
#         for doc, meta in zip(raw_docs, metadatas):
#             parts = splitter.split_text(doc)
#             chunks.extend(parts)
#             chunk_metas.extend([meta] * len(parts))
#         try:
#             self._vectordb = Chroma.from_texts(
#                 chunks, self.embeddings, metadatas=chunk_metas, persist_directory="./chroma_kb"
#             )
#         except Exception:
#             self._vectordb = None

#     def _retrieve(self, question, k=5):
#         if self._vectordb is None:
#             return [], [], []
#         try:
#             results = self._vectordb.similarity_search_with_relevance_scores(question, k=k)
#             docs, scores, metas = [], [], []
#             for doc, score in results:
#                 docs.append(doc.page_content)
#                 scores.append(score)
#                 metas.append(doc.metadata)
#             return docs, scores, metas
#         except Exception:
#             return [], [], []

#     def _is_in_domain(self, question, chunks, scores):
#         """
#         Three-layer domain check — passes if ANY layer matches.

#         Layer 1 — Keyword scan: checks against expanded DOMAIN_TOPICS list.
#                   Includes natural language navigation phrases, city names,
#                   and movement verbs so "go from bandra to kurla" passes.

#         Layer 2 — Regex navigation intent: pattern-matches spatial/directional
#                   phrasing like "from X to Y", "how far", "nearest", "directions".
#                   Catches questions that contain no logistics jargon but are
#                   clearly asking about routes or locations.

#         Layer 3 — Semantic similarity: at least one retrieved KB chunk must
#                   score >= RELEVANCE_THRESHOLD (only active when embeddings online).
#         """
#         import re
#         q = question.lower()

#         # Layer 1: keyword scan
#         if any(kw in q for kw in self.DOMAIN_TOPICS):
#             return True

#         # Layer 2: navigation intent regex
#         for pattern in self._NAV_PATTERNS:
#             if re.search(pattern, q):
#                 return True

#         # Layer 3: semantic similarity
#         if scores and max(scores) >= self.RELEVANCE_THRESHOLD:
#             return True

#         return False

#     def _keyword_fallback(self, question, kb_df):
#         q = question.lower()
#         for _, row in kb_df.iterrows():
#             words = row["topic"].lower().split() + row["category"].lower().split()
#             if any(w in q for w in words):
#                 return f"**📖 {row['topic']}** *(from {row['category']} KB)*\n\n{row['content']}"
#         return (
#             "I couldn't find a matching article. "
#             "Please ask about routing, delivery, customs, fleet, or other logistics topics."
#         )

#     def answer(self, question, kb_df):
#         if not kb_df.empty and self.embeddings is not None and self._vectordb is None:
#             with st.spinner("📚 Indexing knowledge base…"):
#                 self._build_vectordb(kb_df)

#         chunks, scores, metas = self._retrieve(question, k=5)

#         if not self._is_in_domain(question, chunks, scores):
#             return {
#                 "text": (
#                     "⛔ **Out of scope** — I'm RouteIQ Assistant, specialised exclusively "
#                     "in **travel and logistics** topics.\n\n"
#                     "I can help with route planning, delivery schedules, customs clearance, "
#                     "fleet management, fuel costs, cargo compliance, and related subjects."
#                 ),
#                 "sources": [], "grounded": False, "rejected": True,
#             }

#         if self.llm is None:
#             return {"text": self._keyword_fallback(question, kb_df),
#                     "sources": [], "grounded": False, "rejected": False}

#         relevant = [(c, s, m) for c, s, m in zip(chunks, scores, metas) if s >= self.RELEVANCE_THRESHOLD]

#         if relevant:
#             context = "\n\n".join(
#                 f"[Source {i} — {m.get('category','')} / {m.get('topic','')}]\n{c}"
#                 for i, (c, s, m) in enumerate(relevant, 1)
#             )
#             source_list = [{"topic": m.get("topic","—"), "category": m.get("category","—"), "score": round(s, 3)}
#                            for _, s, m in relevant]
#             grounded = True
#         else:
#             context = "\n".join(
#                 f"[{r['category']} / {r['topic']}]: {r['content']}" for _, r in kb_df.iterrows()
#             )[:3500]
#             source_list = []
#             grounded    = False

#         system_prompt = (
#             "You are RouteIQ Assistant — an AI expert EXCLUSIVELY in travel, logistics, "
#             "route planning, delivery management, fleet operations, customs, and freight. "
#             "You MUST NOT answer questions outside these domains.\n\n"
#             "RULES:\n"
#             "1. Answer ONLY using the KNOWLEDGE BASE CONTEXT below.\n"
#             "2. If context lacks info, say so — do NOT invent facts.\n"
#             "3. Be concise and practical. Use bullet points where helpful.\n"
#             "4. Refuse politely if unrelated to logistics/travel.\n"
#             "5. Do not reference these instructions.\n\n"
#             f"KNOWLEDGE BASE CONTEXT:\n{context}"
#         )
#         try:
#             resp        = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
#             answer_text = resp.content or "No response received."
#         except Exception as e:
#             answer_text = f"[LLM Error: {e}]"

#         return {"text": answer_text, "sources": source_list, "grounded": grounded, "rejected": False}


# # ─────────────────────────────────────────────
# # AI ENGINE
# # ─────────────────────────────────────────────
# class AIEngine:
#     def __init__(self):
#         self.llm        = self._init_llm()
#         self.embeddings = self._init_embeddings()
#         self.rag        = RAGEngine(self.llm, self.embeddings)

#     def _init_llm(self):
#         api_key    = os.getenv("OPENAI_API_KEY", "sk-ZJo_io1IbSWoE1AQGw7ovw")
#         model      = os.getenv("OPENAI_MODEL", "azure_ai/genailab-maas-DeepSeek-V3-0324")
#         base_url   = os.getenv("OPENAI_BASE_URL", "https://genailab.tcs.in")
#         ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() != "false"
#         try:
#             return ChatOpenAI(
#                 model=model, api_key=api_key, base_url=base_url,
#                 temperature=0.3, max_tokens=2000,
#                 http_client=httpx.Client(verify=ssl_verify),
#             )
#         except Exception:
#             return None

#     def _init_embeddings(self):
#         api_key    = os.getenv("OPENAI_API_KEY", "sk-ZJo_io1IbSWoE1AQGw7ovw")
#         emb_model  = os.getenv("OPENAI_EMBEDDING_MODEL", "azure/genailab-maas-text-embedding-3-large")
#         base_url   = os.getenv("OPENAI_BASE_URL", "https://genailab.tcs.in")
#         ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() != "false"
#         try:
#             return OpenAIEmbeddings(
#                 model=emb_model, api_key=api_key, base_url=base_url,
#                 http_client=httpx.Client(verify=ssl_verify),
#             )
#         except Exception:
#             return None

#     def _call(self, system_prompt, user_prompt):
#         if not self.llm:
#             return None
#         try:
#             resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
#             return resp.content
#         except Exception as e:
#             return f"[LLM Error: {e}]"

#     def _parse_json(self, text, fallback):
#         if text is None:
#             return fallback
#         text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
#         try:
#             return json.loads(text)
#         except Exception:
#             return fallback

#     # ── A: Natural Language Stop Parser ──────────────────────────────────────
#     def parse_natural_language_stops(self, text: str) -> dict:
#         """
#         Parse a free-text itinerary description into structured stops + constraints.
#         Returns: {stops, constraints, parse_notes, error}
#         """
#         system_prompt = (
#             "You are a logistics data extraction agent. "
#             "Extract structured stop and constraint information from a natural language description. "
#             "Use realistic Indian GPS coordinates if none are given. "
#             "Return ONLY valid JSON — no markdown, no extra text."
#         )
#         user_prompt = (
#             f'Extract all stops and constraints from:\n\n"{text}"\n\n'
#             "Return this JSON schema exactly:\n"
#             "{\n"
#             '  "stops": [\n'
#             "    {\n"
#             '      "stop_id": "NL1",\n'
#             '      "location_name": "full location name",\n'
#             '      "lat": <realistic latitude>,\n'
#             '      "lon": <realistic longitude>,\n'
#             '      "stop_type": "Delivery|Pickup|Meeting|Warehouse|Customs|Rest",\n'
#             '      "time_window_start": "HH:MM",\n'
#             '      "time_window_end": "HH:MM",\n'
#             '      "priority": "High|Medium|Low",\n'
#             '      "notes": "any special instructions"\n'
#             "    }\n"
#             "  ],\n"
#             '  "constraints": {\n'
#             '    "driver_name": "string or Unknown",\n'
#             '    "start_time": "HH:MM",\n'
#             '    "transport_mode": "Road|Rail|Air|Sea",\n'
#             '    "vehicle_type": "Truck|Van|Motorcycle|Car|Tempo",\n'
#             '    "max_hours": <number>,\n'
#             '    "vehicle_capacity_kg": <number>\n'
#             "  },\n"
#             '  "parse_notes": "brief summary of what was understood",\n'
#             '  "error": null\n'
#             "}\n\n"
#             "Inference rules:\n"
#             "- pick up/collect → Pickup; deliver/drop → Delivery; meeting/client → Meeting; "
#             "rest/break → Rest; customs/checkpoint → Customs; warehouse/depot → Warehouse\n"
#             "- urgent/asap/critical → High priority; default → Medium\n"
#             "- 'by 2pm' → time_window_end 14:00; 'at 9am' → time_window_start 09:00\n"
#             "- Default start_time 08:00, transport_mode Road if not mentioned\n"
#             "- stop_id values: NL1, NL2, NL3 in order"
#         )
#         raw    = self._call(system_prompt, user_prompt)
#         result = self._parse_json(raw, {"stops": [], "constraints": {}, "parse_notes": "",
#                                         "error": "LLM offline or parse failed"})
#         if "stops"       not in result: result["stops"]       = []
#         if "constraints" not in result: result["constraints"] = {}
#         if "error"       not in result: result["error"]       = None
#         return result

#     # ── B: Itinerary Generation ───────────────────────────────────────────────
#     def generate_itinerary(self, stops_list, constraints, route_context):
#         """Generate optimized itinerary JSON, then patch in real OSRM distances."""
#         system_prompt = (
#             "You are an expert logistics route planner. "
#             "Generate an optimized, feasible delivery itinerary as structured JSON. "
#             "Prioritize: (1) time window compliance, (2) High-priority stops first, "
#             "(3) shortest total distance. "
#             "Times in HH:MM 24-hour format. Durations in minutes. "
#             "Return ONLY valid JSON — no markdown."
#         )
#         user_prompt = (
#             f"CONSTRAINTS:\n{json.dumps(constraints, indent=2)}\n\n"
#             f"ROUTE CONTEXT (use these distances/times):\n{route_context}\n\n"
#             f"STOPS TO PLAN ({len(stops_list)}):\n{json.dumps(stops_list, indent=2, default=str)}\n\n"
#             "Return JSON with keys: itinerary_title, driver, vehicle, date (YYYY-MM-DD), "
#             "transport_mode, total_distance_km, total_duration_min, estimated_fuel_cost_inr, "
#             "optimization_notes, warnings (list), efficiency_score (0-100), "
#             "on_time_probability (0-100), and stops array where each stop has: "
#             "sequence, stop_id, location_name, arrival_time (HH:MM), departure_time (HH:MM), "
#             "service_duration_min, travel_time_from_prev_min, distance_from_prev_km, "
#             "stop_type, priority, status, notes, risk_flag (None/Low/Medium/High), "
#             "risk_reason, time_window_start (HH:MM), time_window_end (HH:MM)."
#         )
#         raw      = self._call(system_prompt, user_prompt)
#         result   = self._parse_json(raw, self._rule_based_itinerary(stops_list, constraints))

#         # ── Patch real OSRM leg data ─────────────────────────────────────────
#         coords     = [(s["lat"], s["lon"]) for s in stops_list]
#         osrm       = MLEngine.osrm_route(coords)
#         result["routing_source"] = osrm["source"]
#         itin_stops = result.get("stops", [])

#         if osrm["source"] == "osrm" and len(osrm["legs"]) >= len(itin_stops) - 1 and len(itin_stops) > 1:
#             for i, stop in enumerate(itin_stops):
#                 if i == 0:
#                     continue
#                 leg_idx = i - 1
#                 if leg_idx < len(osrm["legs"]):
#                     stop["distance_from_prev_km"]     = osrm["legs"][leg_idx]["distance_km"]
#                     stop["travel_time_from_prev_min"]  = osrm["legs"][leg_idx]["duration_min"]
#             result["total_distance_km"]  = osrm["total_distance_km"]
#             result["total_duration_min"] = round(
#                 osrm["total_duration_min"]
#                 + sum(s.get("service_duration_min", 0) for s in itin_stops), 1
#             )
#             result["estimated_fuel_cost_inr"] = round(osrm["total_distance_km"] * 8, 0)

#         return result

#     def _rule_based_itinerary(self, stops_list, constraints):
#         priority_order = {"High": 0, "Medium": 1, "Low": 2}
#         sorted_stops   = sorted(stops_list, key=lambda s: (
#             priority_order.get(s.get("priority", "Low"), 2),
#             s.get("time_window_start", "23:59"),
#         ))
#         current_time = datetime.strptime(constraints.get("start_time", "08:00"), "%H:%M")
#         result_stops, total_dist = [], 0
#         prev_lat = stops_list[0]["lat"] if stops_list else 19.076
#         prev_lon = stops_list[0]["lon"] if stops_list else 72.877

#         for i, s in enumerate(sorted_stops):
#             dist        = MLEngine.compute_distance_km(prev_lat, prev_lon, s["lat"], s["lon"])
#             travel_min  = max(5, int(dist / 35 * 60))
#             service_min = {"Delivery": 20, "Pickup": 15, "Meeting": 45,
#                            "Warehouse": 30, "Customs": 60, "Rest": 20}.get(s.get("stop_type", "Delivery"), 20)
#             arrival     = current_time + timedelta(minutes=travel_min)
#             departure   = arrival + timedelta(minutes=service_min)
#             result_stops.append({
#                 "sequence": i + 1, "stop_id": s.get("stop_id", f"S{i+1}"),
#                 "location_name": s.get("location_name", "Stop"),
#                 "arrival_time": arrival.strftime("%H:%M"),
#                 "departure_time": departure.strftime("%H:%M"),
#                 "service_duration_min": service_min,
#                 "travel_time_from_prev_min": travel_min,
#                 "distance_from_prev_km": round(dist, 2),
#                 "stop_type": s.get("stop_type", "Delivery"),
#                 "priority": s.get("priority", "Medium"),
#                 "status": "Scheduled", "notes": s.get("notes", ""),
#                 "risk_flag": "None", "risk_reason": "",
#                 "time_window_start": s.get("time_window_start", "08:00"),
#                 "time_window_end":   s.get("time_window_end",   "18:00"),
#             })
#             total_dist += dist
#             current_time = departure
#             prev_lat, prev_lon = s["lat"], s["lon"]

#         return {
#             "itinerary_title": "Optimized Route (Rule-based fallback)",
#             "driver": constraints.get("driver_name", "Driver"),
#             "vehicle": constraints.get("vehicle_type", "Truck"),
#             "date": datetime.now().strftime("%Y-%m-%d"),
#             "transport_mode": constraints.get("transport_mode", "Road"),
#             "total_distance_km": round(total_dist, 2),
#             "total_duration_min": sum(
#                 s["travel_time_from_prev_min"] + s["service_duration_min"] for s in result_stops
#             ),
#             "estimated_fuel_cost_inr": round(total_dist * 8, 0),
#             "stops": result_stops,
#             "optimization_notes": "Rule-based fallback: sorted by priority then time window.",
#             "warnings": ["LLM offline — using rule-based optimizer"],
#             "efficiency_score": 72, "on_time_probability": 78,
#         }

#     def adjust_itinerary(self, existing_itinerary, change_request):
#         system_prompt = (
#             "You are a logistics re-planning agent. "
#             "Return a FULLY updated itinerary JSON in the same schema. "
#             "Return ONLY valid JSON — no markdown."
#         )
#         user_prompt = (
#             f"EXISTING ITINERARY:\n{json.dumps(existing_itinerary, indent=2, default=str)}\n\n"
#             f"CHANGE REQUEST:\n{change_request}\n\n"
#             "Adjust all affected stop times. Explain the change in optimization_notes."
#         )
#         return self._parse_json(self._call(system_prompt, user_prompt), existing_itinerary)

#     # ── C: Constraint Violation Checker ──────────────────────────────────────
#     def check_violations(self, itinerary: dict, constraints: dict) -> list:
#         """
#         Check all stops for time-window breaches and driver hour overruns.
#         Returns a list of violation dicts: {sequence, location_name, type, severity, detail}
#         """
#         violations = []
#         stops      = sorted(itinerary.get("stops", []), key=lambda s: s["sequence"])
#         if not stops:
#             return violations

#         date_str  = itinerary.get("date", datetime.now().strftime("%Y-%m-%d"))
#         try:
#             base = datetime.strptime(date_str, "%Y-%m-%d")
#         except Exception:
#             base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

#         def to_dt(hhmm):
#             try:
#                 h, m = map(int, str(hhmm).strip().split(":"))
#                 return base.replace(hour=h, minute=m, second=0, microsecond=0)
#             except Exception:
#                 return base

#         max_hours = constraints.get("max_hours", 10)
#         day_end   = to_dt(constraints.get("start_time", "08:00")) + timedelta(hours=max_hours)

#         for s in stops:
#             arr          = to_dt(s.get("arrival_time",   "00:00"))
#             dep          = to_dt(s.get("departure_time",  "00:00"))
#             tw_start_raw = s.get("time_window_start", "")
#             tw_end_raw   = s.get("time_window_end",   "")

#             if tw_start_raw and tw_end_raw:
#                 tw_start = to_dt(tw_start_raw)
#                 tw_end   = to_dt(tw_end_raw)
#                 if arr < tw_start:
#                     wait = int((tw_start - arr).seconds / 60)
#                     violations.append({
#                         "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                         "type": "time_window", "severity": "Warning",
#                         "detail": (
#                             f"Arrives at {s.get('arrival_time')} but window opens at "
#                             f"{tw_start_raw}. Driver waits {wait} min."
#                         ),
#                     })
#                 elif arr > tw_end:
#                     late = int((arr - tw_end).seconds / 60)
#                     violations.append({
#                         "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                         "type": "time_window", "severity": "Critical",
#                         "detail": (
#                             f"Arrives at {s.get('arrival_time')} — window closed at "
#                             f"{tw_end_raw}. Late by {late} min. SLA breach."
#                         ),
#                     })

#             if dep > day_end:
#                 over = int((dep - day_end).seconds / 60)
#                 violations.append({
#                     "sequence": s["sequence"], "location_name": s.get("location_name",""),
#                     "type": "driver_hours", "severity": "Critical",
#                     "detail": (
#                         f"Departure at {s.get('departure_time')} exceeds "
#                         f"{max_hours}h limit by {over} min."
#                     ),
#                 })

#         return violations

#     def get_kb_answer(self, question, kb_df):
#         return self.rag.answer(question, kb_df)

#     def analyze_route_performance(self, route_df, stops_df):
#         summary = {
#             "total_routes":      route_df["route_id"].nunique() if not route_df.empty else 0,
#             "avg_distance_km":   round(route_df["distance_km"].mean(), 1) if not route_df.empty else 0,
#             "avg_on_time_pct":   round(100 * (stops_df["status"] == "On Time").mean(), 1) if not stops_df.empty else 0,
#             "top_delay_reasons": stops_df["delay_reason"].value_counts().head(3).to_dict()
#                                   if "delay_reason" in stops_df.columns else {},
#         }
#         raw = self._call(
#             "You are a logistics analytics expert. Provide 3-5 concise actionable insights.",
#             f"Performance data:\n{json.dumps(summary, indent=2)}\n\nBullet point insights:"
#         )
#         if not raw or raw.startswith("[LLM"):
#             return (
#                 "• Review high-delay routes for recurring traffic patterns\n"
#                 "• Adjust time windows for stops that are consistently late\n"
#                 "• Prioritize High-priority stops in morning slots\n"
#                 "• Consolidate nearby stops to reduce total distance and fuel cost"
#             )
#         return raw


# # ─────────────────────────────────────────────
# # DASHBOARD
# # ─────────────────────────────────────────────
# class Dashboard:
#     STOP_COLORS = {
#         "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
#         "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
#     }
#     PRIORITY_COLORS = {"High": "#E74C3C", "Medium": "#D4A843", "Low": "#3FB950"}
#     STATUS_COLORS   = {
#         "On Time": "#3FB950", "Delayed": "#E74C3C",
#         "Scheduled": "#2EA4A4", "Cancelled": "#8B949E",
#     }

#     def _base_layout(self, height=320):
#         return dict(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=height,
#             margin=dict(t=20, b=40, l=50, r=20),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )

#     def volume_timeline(self, stops_df):
#         df2 = stops_df.copy()
#         df2["date"] = pd.to_datetime(df2["scheduled_date"])
#         daily = df2.groupby(["date","stop_type"]).size().reset_index(name="count")
#         fig   = px.bar(daily, x="date", y="count", color="stop_type",
#                        color_discrete_map=self.STOP_COLORS, barmode="stack")
#         fig.update_layout(**self._base_layout(280),
#                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"))
#         return fig

#     def stop_type_donut(self, stops_df):
#         counts = stops_df["stop_type"].value_counts().reset_index()
#         counts.columns = ["stop_type","count"]
#         fig = px.pie(counts, names="stop_type", values="count",
#                      color="stop_type", color_discrete_map=self.STOP_COLORS, hole=0.55)
#         fig.update_traces(textposition="inside", textinfo="percent+label",
#                           marker=dict(line=dict(color="#0D1117", width=2)))
#         fig.update_layout(**self._base_layout(300), showlegend=False)
#         return fig

#     def distance_by_route(self, routes_df):
#         df = routes_df.sort_values("total_distance_km", ascending=False).head(10)
#         fig = go.Figure(go.Bar(x=df["route_id"], y=df["total_distance_km"],
#                                marker_color="#D4A843",
#                                hovertemplate="%{x}: %{y:.1f} km<extra></extra>"))
#         fig.update_layout(**self._base_layout(280),
#                           xaxis=dict(showgrid=False, tickangle=-20),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="km"))
#         return fig

#     def on_time_gauge(self, pct):
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number", value=pct,
#             number={"suffix": "%", "font": {"color": "#D4A843", "size": 36}},
#             gauge={
#                 "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
#                 "bar":  {"color": "#D4A843"}, "bgcolor": "#21262D",
#                 "steps": [
#                     {"range": [0,  60],  "color": "rgba(192,57,43,0.3)"},
#                     {"range": [60, 80],  "color": "rgba(232,135,58,0.3)"},
#                     {"range": [80, 100], "color": "rgba(63,185,80,0.3)"},
#                 ],
#                 "threshold": {"line": {"color": "#3FB950", "width": 3}, "value": 85},
#             },
#         ))
#         layout = self._base_layout(220)
#         layout["margin"] = dict(t=20, b=10, l=30, r=30)

#         fig.update_layout(**layout)
#         return fig

#     def priority_bar(self, stops_df):
#         pc = stops_df["priority"].value_counts().reset_index()
#         pc.columns = ["priority","count"]
#         fig = go.Figure(go.Bar(
#             x=pc["priority"], y=pc["count"],
#             marker_color=[self.PRIORITY_COLORS.get(p,"#8B949E") for p in pc["priority"]],
#             hovertemplate="%{x}: %{y}<extra></extra>",
#         ))
#         fig.update_layout(**self._base_layout(240),
#                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"))
#         return fig

#     def route_map_scatter(self, stops_list, itinerary=None):
#         lats   = [s["lat"] for s in stops_list]
#         lons   = [s["lon"] for s in stops_list]
#         names  = [s["location_name"] for s in stops_list]
#         colors = [self.STOP_COLORS.get(s.get("stop_type","Delivery"), "#D4A843") for s in stops_list]

#         fig = go.Figure()
#         fig.add_trace(go.Scattergeo(
#             lat=lats, lon=lons, mode="markers+text",
#             marker=dict(size=12, color=colors, line=dict(color="#0D1117", width=1)),
#             text=[f"{i+1}. {n}" for i, n in enumerate(names)],
#             textposition="top center",
#             textfont=dict(size=9, color="#C9D1D9"),
#             hovertemplate="<b>%{text}</b><extra></extra>", name="Stops",
#         ))
#         if itinerary and "stops" in itinerary:
#             seq  = sorted(itinerary["stops"], key=lambda s: s["sequence"])
#             rlat, rlon = [], []
#             for ist in seq:
#                 m = next((s for s in stops_list if s["stop_id"] == ist["stop_id"]), None)
#                 if m:
#                     rlat.append(m["lat"]); rlon.append(m["lon"])
#             if rlat:
#                 fig.add_trace(go.Scattergeo(lat=rlat, lon=rlon, mode="lines",
#                                              line=dict(width=2, color="#D4A843"), name="Route"))
#         fig.update_geos(
#             center=dict(lat=np.mean(lats), lon=np.mean(lons)), projection_scale=8,
#             showland=True, landcolor="#21262D", showocean=True, oceancolor="#161B22",
#             showcountries=True, countrycolor="#30363D", showcoastlines=True, coastlinecolor="#30363D",
#         )
#         layout = self._base_layout(420)
#         layout["margin"] = dict(t=10, b=10, l=10, r=10)
#         fig.update_layout(**layout, geo=dict(bgcolor="rgba(0,0,0,0)"), showlegend=True)
#         return fig

#     def cluster_scatter(self, stops_df, labels, X_2d):
#         df = stops_df.copy()
#         df["Cluster"] = [f"Cluster {l+1}" for l in labels]
#         df["x"] = X_2d[:, 0]; df["y"] = X_2d[:, 1]
#         fig = px.scatter(df, x="x", y="y", color="Cluster",
#                          hover_data=["location_name","stop_type","priority"],
#                          color_discrete_sequence=["#D4A843","#2EA4A4","#3FB950","#8957E5","#E74C3C","#1F6FEB"])
#         fig.update_layout(**self._base_layout(380),
#                           xaxis=dict(showgrid=False, title="Component 1"),
#                           yaxis=dict(showgrid=False, title="Component 2"))
#         return fig

#     def performance_timeline(self, routes_df):
#         df = routes_df.copy(); df["date"] = pd.to_datetime(df["route_date"])
#         fig = px.line(df.sort_values("date"), x="date", y="on_time_pct", color="transport_mode",
#                       color_discrete_map={"Road":"#D4A843","Air":"#2EA4A4","Rail":"#3FB950","Sea":"#8957E5"})
#         fig.update_layout(**self._base_layout(300), xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="On-Time %", range=[0,105]))
#         return fig

#     def fuel_cost_bar(self, routes_df):
#         df = routes_df.groupby("transport_mode")["estimated_fuel_cost_inr"].mean().reset_index()
#         fig = go.Figure(go.Bar(x=df["transport_mode"], y=df["estimated_fuel_cost_inr"],
#                                marker_color=["#D4A843","#2EA4A4","#3FB950","#8957E5"],
#                                hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>"))
#         fig.update_layout(**self._base_layout(260), xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=True, gridcolor="#21262D", title="Avg Cost (₹)"))
#         return fig


# # ─────────────────────────────────────────────
# # PAGE: OVERVIEW
# # ─────────────────────────────────────────────
# def page_overview(stops_df, routes_df, stats, dash):
#     st.markdown('<div class="hero-title">🗺️ Route<span class="hero-accent">IQ</span></div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">AI-Powered Logistics Itinerary Planning & Route Optimization</div>', unsafe_allow_html=True)
#     st.markdown("---")

#     if stops_df.empty:
#         st.warning("No data loaded. Please ensure stops.csv and routes.csv exist.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Stops",    stats.get("total_stops", 0))
#     c2.metric("Active Routes",  stats.get("total_routes", 0))
#     c3.metric("Total Distance", f"{stats.get('total_distance_km', 0):,} km")
#     c4.metric("On-Time Rate",   f"{stats.get('on_time_pct', 0)}%")

#     st.markdown("---")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.markdown("**Daily Stop Volume by Type**")
#         st.plotly_chart(dash.volume_timeline(stops_df), use_container_width=True)
#     with col2:
#         st.markdown("**Stop Type Distribution**")
#         st.plotly_chart(dash.stop_type_donut(stops_df), use_container_width=True)

#     col3, col4 = st.columns(2)
#     with col3:
#         st.markdown("**Top Routes by Distance**")
#         st.plotly_chart(dash.distance_by_route(routes_df), use_container_width=True)
#     with col4:
#         st.markdown("**Stop Priority Breakdown**")
#         st.plotly_chart(dash.priority_bar(stops_df), use_container_width=True)

#     st.markdown("---")
#     st.markdown("### 🚨 High-Priority Open Stops")
#     high = stops_df[(stops_df["priority"] == "High") & (stops_df["status"] != "Delivered")].head(5)
#     if high.empty:
#         st.info("No high-priority open stops.")
#     else:
#         for _, row in high.iterrows():
#             type_cls = {"Delivery":"card-gold","Meeting":"card-teal","Pickup":"card-green",
#                         "Warehouse":"card-blue","Customs":"card-red"}.get(row["stop_type"],"card-gold")
#             st.markdown(
#                 f'<div class="card {type_cls}">'
#                 f'<div style="display:flex;justify-content:space-between;align-items:center">'
#                 f'<div><b style="color:#F0F6FC">{row["location_name"]}</b>'
#                 f'<span class="badge badge-delivery" style="margin-left:8px">{row["stop_type"]}</span></div>'
#                 f'<div style="text-align:right;font-size:0.8rem;color:#8B949E">'
#                 f'Route: {row.get("route_id","—")} &nbsp;|&nbsp; {row.get("scheduled_date","—")}</div></div>'
#                 f'<div class="stop-detail">Window: {row.get("time_window_start","—")} – '
#                 f'{row.get("time_window_end","—")} &nbsp;·&nbsp; {row.get("notes","")}</div></div>',
#                 unsafe_allow_html=True,
#             )


# # ─────────────────────────────────────────────
# # PAGE: ITINERARY PLANNER  (3 tabs)
# # ─────────────────────────────────────────────
# def page_planner(stops_df, ai, ml, dash):
#     st.markdown('<div class="section-title">🔍 Itinerary Planner</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Generate optimized route plans — '
#         'from your dataset, custom coordinates, or plain English</div>',
#         unsafe_allow_html=True,
#     )

#     tab1, tab2, tab3 = st.tabs(["📋 Plan from Dataset", "✏️ Custom Coordinates", "💬 Natural Language"])

#     # ── TAB 1: Plan from Dataset ──────────────────────────────────────────────
#     with tab1:
#         if stops_df.empty:
#             st.warning("No stops data loaded.")
#         else:
#             routes         = stops_df["route_id"].unique().tolist()
#             selected_route = st.selectbox("Select Route to Plan", routes)
#             route_stops    = stops_df[stops_df["route_id"] == selected_route].copy()

#             st.markdown(f"**{len(route_stops)} stops on route {selected_route}**")
#             st.dataframe(
#                 route_stops[["stop_id","location_name","stop_type","priority",
#                               "time_window_start","time_window_end","notes"]].reset_index(drop=True),
#                 use_container_width=True,
#             )

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 driver_name = st.text_input("Driver Name", value="Ramesh Kumar")
#                 start_time  = st.text_input("Start Time (HH:MM)", value="08:00")
#             with col2:
#                 transport_mode   = st.selectbox("Transport Mode", ["Road","Rail","Air","Sea"])
#                 vehicle_type     = st.selectbox("Vehicle Type", ["Truck","Van","Motorcycle","Car","Tempo"])
#             with col3:
#                 max_hours        = st.slider("Max Working Hours", 4, 14, 8)
#                 vehicle_capacity = st.number_input("Vehicle Capacity (kg)", 100, 10000, 1000, step=100)

#             optimize = st.checkbox("Apply Nearest-Neighbor Optimization", value=True)

#             if st.button("🚀 Generate Itinerary", type="primary", key="gen_dataset"):
#                 coords = list(zip(route_stops["lat"], route_stops["lon"]))
#                 if optimize and len(coords) > 2:
#                     order       = ml.nearest_neighbor_route(coords)
#                     route_stops = route_stops.iloc[order].reset_index(drop=True)

#                 stops_list = [
#                     {
#                         "stop_id":           str(r["stop_id"]),
#                         "location_name":     r["location_name"],
#                         "lat":               float(r["lat"]),
#                         "lon":               float(r["lon"]),
#                         "stop_type":         r["stop_type"],
#                         "time_window_start": str(r["time_window_start"])[-8:-3] if pd.notnull(r["time_window_start"]) else "08:00",
#                         "time_window_end":   str(r["time_window_end"])[-8:-3]   if pd.notnull(r["time_window_end"])   else "18:00",
#                         "priority":          r["priority"],
#                         "notes":             str(r.get("notes","")) if pd.notnull(r.get("notes","")) else "",
#                     }
#                     for _, r in route_stops.iterrows()
#                 ]
#                 constraints = {
#                     "driver_name": driver_name, "start_time": start_time,
#                     "transport_mode": transport_mode, "vehicle_type": vehicle_type,
#                     "max_hours": max_hours, "vehicle_capacity_kg": vehicle_capacity,
#                 }

#                 with st.spinner("🛰️ Fetching real road distances via OSRM…"):
#                     osrm_preview = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])

#                 st.caption(
#                     "Routing: 🟢 OSRM (real roads)" if osrm_preview["source"] == "osrm"
#                     else "Routing: 🟡 Haversine fallback (OSRM unreachable)"
#                 )
#                 route_context = (
#                     f"Route {selected_route} | {len(stops_list)} stops | Mode: {transport_mode} | "
#                     f"Road distance: {osrm_preview['total_distance_km']} km | "
#                     f"Drive time: {osrm_preview['total_duration_min']} min | "
#                     f"Source: {osrm_preview['source']}"
#                 )
#                 with st.spinner("🧠 AI optimizing your route…"):
#                     itinerary = ai.generate_itinerary(stops_list, constraints, route_context)

#                 st.session_state["generated_itinerary"] = itinerary
#                 st.session_state["last_constraints"]    = constraints
#                 st.session_state["itinerary_source"]    = "dataset"
#                 st.session_state["fuel_analysis"]       = None  # reset stale savings
#                 st.session_state["plan_history"].append({
#                     "source": "Dataset", "route": selected_route,
#                     "generated_at": datetime.now().strftime("%H:%M:%S"),
#                     "stops": len(stops_list),
#                     "distance_km": itinerary.get("total_distance_km", 0),
#                     "routing": osrm_preview["source"],
#                 })
#                 st.success("✅ Itinerary generated!")

#     # ── TAB 2: Custom Coordinates ─────────────────────────────────────────────
#     with tab2:
#         st.markdown("Add custom stops with coordinates for ad-hoc planning.")
#         custom_text = st.text_area(
#             "One stop per line: Name, Lat, Lon, Type, Priority, TimeFrom, TimeTo",
#             height=160,
#             placeholder="Dadar Warehouse, 19.018, 72.848, Delivery, High, 09:00, 11:00",
#             key="custom_coords_input",
#         )
#         col1, col2 = st.columns(2)
#         with col1:
#             c_driver = st.text_input("Driver Name", value="Suresh Patil", key="c_driver")
#             c_start  = st.text_input("Start Time", value="08:30", key="c_start")
#         with col2:
#             c_mode  = st.selectbox("Transport Mode", ["Road","Rail","Air","Sea"], key="c_mode")
#             c_hours = st.slider("Max Hours", 4, 14, 8, key="c_hours")
#             c_cap   = st.number_input("Capacity (kg)", 100, 10000, 500, step=100, key="c_cap")

#         if st.button("🚀 Generate Custom Itinerary", key="gen_custom"):
#             custom_stops = []
#             for i, line in enumerate(custom_text.strip().split("\n")):
#                 parts = [p.strip() for p in line.split(",")]
#                 if len(parts) >= 3:
#                     try:
#                         custom_stops.append({
#                             "stop_id": f"CS{i+1}", "location_name": parts[0],
#                             "lat": float(parts[1]), "lon": float(parts[2]),
#                             "stop_type":         parts[3] if len(parts) > 3 else "Delivery",
#                             "priority":          parts[4] if len(parts) > 4 else "Medium",
#                             "time_window_start": parts[5] if len(parts) > 5 else "08:00",
#                             "time_window_end":   parts[6] if len(parts) > 6 else "18:00",
#                             "notes": "",
#                         })
#                     except ValueError:
#                         pass

#             if not custom_stops:
#                 st.error("No valid stops parsed. Check the format.")
#             else:
#                 constraints = {
#                     "driver_name": c_driver, "start_time": c_start,
#                     "transport_mode": c_mode, "max_hours": c_hours,
#                     "vehicle_type": "Van", "vehicle_capacity_kg": c_cap,
#                 }
#                 with st.spinner("🛰️ Fetching real road distances via OSRM…"):
#                     osrm_cs = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in custom_stops])
#                 st.caption(
#                     "Routing: 🟢 OSRM" if osrm_cs["source"] == "osrm"
#                     else "Routing: 🟡 Haversine fallback"
#                 )
#                 route_ctx = (
#                     f"{len(custom_stops)} custom stops | Mode: {c_mode} | "
#                     f"Road distance: {osrm_cs['total_distance_km']} km | "
#                     f"Drive time: {osrm_cs['total_duration_min']} min | Source: {osrm_cs['source']}"
#                 )
#                 with st.spinner("Planning custom route…"):
#                     itinerary = ai.generate_itinerary(custom_stops, constraints, route_ctx)
#                 st.session_state["generated_itinerary"] = itinerary
#                 st.session_state["last_constraints"]    = constraints
#                 st.session_state["itinerary_source"]    = "custom"
#                 st.session_state["fuel_analysis"]       = None
#                 st.session_state["plan_history"].append({
#                     "source": "Custom Coordinates", "route": "Custom",
#                     "generated_at": datetime.now().strftime("%H:%M:%S"),
#                     "stops": len(custom_stops),
#                     "distance_km": itinerary.get("total_distance_km", 0),
#                     "routing": osrm_cs["source"],
#                 })
#                 st.success("✅ Custom itinerary generated!")

#     # ── TAB 3: Natural Language ───────────────────────────────────────────────
#     with tab3:
#         # If the last itinerary was generated from a different tab, show a clear notice
#         # but do NOT wipe session state — that's what caused the dataset tab to break.
#         # Generating from this tab will replace it.
#         if st.session_state.get("itinerary_source") in ("dataset", "custom"):
#             st.info(
#                 "ℹ️ The itinerary below was generated from the **Dataset** or **Custom** tab. "
#                 "Describe your route and click **Generate Full Itinerary** to create a new one here."
#             )

#         st.markdown(
#             '<div class="summary-box">'
#             '<b style="color:#D4A843">💬 Describe your route in plain English.</b><br>'
#             '<span style="color:#8B949E;font-size:0.85rem">'
#             'Mention stops, times, priorities, and constraints naturally — '
#             'the AI extracts the structure and builds a full optimized itinerary.'
#             '</span></div>',
#             unsafe_allow_html=True,
#         )

#         with st.expander("📖 Example prompts"):
#             st.markdown("""
# **Simple run:**
# > Plan a route for Ramesh. Start at 8am from Dadar Warehouse. Deliver to Andheri client by 10am (urgent), pick up cargo from Kurla depot 11am–1pm, client meeting in BKC at 2:30pm.

# **Multi-constraint:**
# > Suresh needs to collect documents from Pune Hadapsar at 9am, customs meeting in Navi Mumbai by 1pm, then deliver to Mulund cold storage before 4pm. Refrigerated van, max 9 hours.

# **Minimal:**
# > 3 stops: Thane pickup 9am, Andheri delivery by 11am, Bandra meeting at 2pm.
#             """)

#         nl_text = st.text_area(
#             "Describe your itinerary:",
#             height=140,
#             placeholder=(
#                 "Plan a route for Ramesh starting at 8am from Dadar Warehouse. "
#                 "Deliver to Andheri client by 10am (urgent)..."
#             ),
#             key="nl_input",
#         )

#         nl_col1, nl_col2 = st.columns(2)
#         with nl_col1:
#             nl_driver = st.text_input("Override driver name (optional)", value="", key="nl_driver")
#             nl_mode   = st.selectbox("Override transport mode",
#                                      ["(auto-detect)","Road","Rail","Air","Sea"], key="nl_mode")
#         with nl_col2:
#             nl_hours = st.slider("Max working hours", 4, 14, 9, key="nl_hours")
#             nl_cap   = st.number_input("Vehicle capacity (kg)", 100, 10000, 500, step=100, key="nl_cap")

#         if st.button("🔍 Parse Description", key="nl_parse"):
#             if not nl_text.strip():
#                 st.error("Please describe your route first.")
#             elif not ai.llm:
#                 st.error("LLM is offline. Natural language parsing requires a connected LLM.")
#             else:
#                 with st.spinner("🤖 Extracting stops from your description…"):
#                     parsed = ai.parse_natural_language_stops(nl_text)
#                 st.session_state["nl_parsed_result"] = parsed

#         parsed = st.session_state.get("nl_parsed_result")
#         if parsed:
#             if parsed.get("error") and not parsed.get("stops"):
#                 st.error(f"Parse failed: {parsed['error']}")
#             else:
#                 stops_list = parsed.get("stops", [])
#                 if parsed.get("parse_notes"):
#                     st.info(f"📝 AI understood: {parsed['parse_notes']}")

#                 if stops_list:
#                     st.markdown("#### ✅ Parsed Stops")
#                     st.dataframe(
#                         pd.DataFrame([{
#                             "#": s["stop_id"], "Location": s["location_name"],
#                             "Type": s["stop_type"], "Priority": s["priority"],
#                             "Window": f"{s['time_window_start']} – {s['time_window_end']}",
#                             "Notes": s.get("notes",""),
#                         } for s in stops_list]),
#                         use_container_width=True,
#                     )

#                     auto_c = parsed.get("constraints", {})
#                     if nl_driver.strip():          auto_c["driver_name"]      = nl_driver.strip()
#                     if nl_mode != "(auto-detect)": auto_c["transport_mode"]   = nl_mode
#                     auto_c["max_hours"]          = nl_hours
#                     auto_c["vehicle_capacity_kg"] = nl_cap

#                     cc1, cc2, cc3 = st.columns(3)
#                     cc1.markdown(f"**Driver:** {auto_c.get('driver_name','—')}")
#                     cc2.markdown(f"**Mode:** {auto_c.get('transport_mode','Road')}")
#                     cc3.markdown(f"**Start:** {auto_c.get('start_time','08:00')}")

#                     if st.button("🚀 Generate Full Itinerary", type="primary", key="nl_generate"):
#                         with st.spinner("🛰️ Fetching OSRM road distances…"):
#                             osrm_nl = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])
#                         st.caption(
#                             "Routing: 🟢 OSRM" if osrm_nl["source"] == "osrm"
#                             else "Routing: 🟡 Haversine fallback"
#                         )
#                         ctx_nl = (
#                             f"{len(stops_list)} NL-parsed stops | "
#                             f"Mode: {auto_c.get('transport_mode','Road')} | "
#                             f"Road distance: {osrm_nl['total_distance_km']} km | "
#                             f"Drive time: {osrm_nl['total_duration_min']} min | "
#                             f"Source: {osrm_nl['source']}"
#                         )
#                         with st.spinner("🧠 Generating optimized itinerary…"):
#                             itinerary = ai.generate_itinerary(stops_list, auto_c, ctx_nl)
#                         # Build stops_for_map — NL itinerary embeds lat/lon in its stops
#                         nl_map_stops = []
#                         for itin_s in itinerary.get("stops", []):
#                             matched = next((s for s in stops_list if s["stop_id"] == itin_s["stop_id"]), None)
#                             if matched:
#                                 nl_map_stops.append({
#                                     "stop_id":       itin_s["stop_id"],
#                                     "location_name": itin_s["location_name"],
#                                     "lat":           matched["lat"],
#                                     "lon":           matched["lon"],
#                                     "stop_type":     itin_s.get("stop_type","Delivery"),
#                                 })

#                         st.session_state["generated_itinerary"] = itinerary
#                         st.session_state["last_constraints"]    = auto_c
#                         st.session_state["nl_parsed_result"]    = None
#                         st.session_state["itinerary_source"]    = "nl"
#                         st.session_state["nl_stops_for_map"]    = nl_map_stops
#                         st.session_state["fuel_analysis"]       = None
#                         st.session_state["plan_history"].append({
#                             "source": "Natural Language", "route": "NL-parsed",
#                             "generated_at": datetime.now().strftime("%H:%M:%S"),
#                             "stops": len(stops_list),
#                             "distance_km": itinerary.get("total_distance_km", 0),
#                             "routing": osrm_nl["source"],
#                         })
#                         st.success("✅ Itinerary generated from your description!")
#                         st.rerun()
#                 else:
#                     st.warning("No stops extracted. Try adding specific location names and times.")

#     # ── Display Itinerary ─────────────────────────────────────────────────────
#     # Render the itinerary below the tabs.
#     # itinerary_source tells us which tab owns the current result.
#     # We ALWAYS show it here — the per-tab isolation works because:
#     #   - Generating from tab1/tab2 sets source = "dataset"/"custom"
#     #   - Generating from tab3 sets source = "nl"
#     #   - Switching tabs does NOT clear the result (no unconditional clearing)
#     #   - The user sees the last result they generated, regardless of which tab is active
#     itin   = st.session_state.get("generated_itinerary")
#     source = st.session_state.get("itinerary_source")
#     if itin and source:
#         st.markdown("---")
#         nl_map = st.session_state.get("nl_stops_for_map", []) if source == "nl" else []
#         _render_itinerary(
#             itin, stops_df, ai, dash,
#             st.session_state.get("last_constraints", {}),
#             nl_stops_override=nl_map,
#         )


# # ─────────────────────────────────────────────
# # WEATHER PANEL  (called from _render_itinerary)
# # ─────────────────────────────────────────────
# def _render_weather_panel(stops_with_coords: list, itin_date_str: str = None):
#     """
#     Renders arrival-time-aware weather forecast for every stop.

#     For each stop the forecast shows conditions at the stop's arrival_time,
#     not current conditions — e.g. Stop B arriving at 13:00 shows the 13:00
#     hourly forecast, not what the weather is right now.

#     stops_with_coords: list of dicts with {stop_id, location_name, lat, lon,
#                         arrival_time (HH:MM), stop_type, sequence}
#     itin_date_str:     "YYYY-MM-DD" — the itinerary date used to resolve forecasts
#     """
#     st.markdown("### 🌤️ Weather Forecast Along Route")
#     st.markdown(
#         '<div style="font-size:0.8rem;color:#8B949E;margin-bottom:16px">'
#         'Forecast shown at <b style="color:#D4A843">each stop&#39;s expected arrival time</b> '
#         '— not current conditions. Powered by Open-Meteo hourly API.'
#         '</div>',
#         unsafe_allow_html=True,
#     )

#     if not stops_with_coords:
#         st.info("No coordinate data available for weather lookup.")
#         return

#     with st.spinner("Fetching arrival-time forecasts for each stop…"):
#         weather_stops = WeatherEngine.fetch_route_at_times(stops_with_coords, itin_date_str)

#     # Data source summary
#     sources = [ws["weather"]["source"] for ws in weather_stops]
#     live_count = sum(1 for s in sources if "open-meteo" in s)
#     est_count  = sum(1 for s in sources if "seasonal"   in s)

#     if live_count == len(weather_stops):
#         st.caption("🟢 All forecasts from Open-Meteo hourly API (real data)")
#     elif live_count > 0:
#         st.caption(f"🟡 Mixed: {live_count} stops from Open-Meteo · {est_count} stops from seasonal estimate")
#     else:
#         st.caption(
#             "🟡 **Open-Meteo unreachable** — showing seasonal estimates with diurnal variation. "
#             "Values are indicative. Check network/proxy settings."
#         )

#     adverse_count = sum(1 for ws in weather_stops if ws["weather"]["is_adverse"])
#     if adverse_count:
#         st.warning(
#             f"⚠️ **{adverse_count} stop(s) forecast adverse weather at arrival time** — "
#             "allow extra buffer time and check vehicle load securing."
#         )

#     # ── Cards — 4 per row ─────────────────────────────────────────────────────
#     cols_per_row = min(4, len(weather_stops))
#     card_rows    = [weather_stops[i:i+cols_per_row]
#                     for i in range(0, len(weather_stops), cols_per_row)]

#     for card_row in card_rows:
#         cols = st.columns(len(card_row))
#         for col, ws in zip(cols, card_row):
#             w            = ws["weather"]
#             source       = w.get("source", "")
#             arrival      = ws.get("arrival_time", "")
#             forecast_t   = w.get("forecast_time", arrival)
#             precip_prob  = w.get("precip_probability")

#             border_color = "#E74C3C" if w["is_adverse"] else "#30363D"
#             temp_str = f"{w['temperature_c']}°C"     if w.get("temperature_c")    is not None else "—"
#             wind_str = f"{w['wind_speed_kmh']} km/h" if w.get("wind_speed_kmh")   is not None else "—"
#             prec_str = f"{w['precipitation_mm']} mm" if w.get("precipitation_mm") is not None else "—"
#             hum_str  = f"{w['humidity_pct']}%"       if w.get("humidity_pct")     is not None else "—"
#             vis_str  = f"{w['visibility_km']} km"    if w.get("visibility_km")    is not None else "—"
#             prob_str = f"{precip_prob}% rain chance" if precip_prob is not None else ""

#             if "open-meteo" in source:
#                 src_badge = '<div style="font-size:0.58rem;color:#3FB950;margin-top:5px">🟢 Open-Meteo hourly</div>'
#             else:
#                 src_badge = '<div style="font-size:0.58rem;color:#D4A843;margin-top:5px">🟡 Seasonal estimate</div>'

#             adverse_badge = (
#                 '<div style="font-size:0.65rem;color:#E74C3C;margin-top:5px;font-weight:600">⚠️ Adverse conditions</div>'
#                 if w["is_adverse"] else ""
#             )
#             prob_badge = (
#                 f'<div style="font-size:0.65rem;color:#2EA4A4;margin-top:3px">🌂 {prob_str}</div>'
#                 if prob_str else ""
#             )

#             col.markdown(
#                 f'<div class="weather-card" style="border:1px solid {border_color};margin-bottom:8px">'
#                 # Arrival time header — the KEY info
#                 f'<div style="font-size:0.6rem;font-family:monospace;color:#2EA4A4;'
#                 f'letter-spacing:0.05em;margin-bottom:6px">🕐 ARRIVAL {arrival}</div>'
#                 # Icon + temp
#                 f'<div style="font-size:1.9rem;margin-bottom:2px">{w["icon"]}</div>'
#                 f'<div style="font-size:0.78rem;font-weight:700;color:#F0F6FC;margin-bottom:2px">'
#                 f'{ws["location_name"][:20]}</div>'
#                 f'<div style="font-size:1.2rem;font-weight:700;color:#D4A843;margin-bottom:2px">{temp_str}</div>'
#                 f'<div style="font-size:0.72rem;color:#C9D1D9">{w["description"]}</div>'
#                 # Stats row
#                 f'<div style="font-size:0.65rem;color:#8B949E;margin-top:8px;line-height:1.7">'
#                 f'💨 {wind_str} &nbsp;·&nbsp; 🌧️ {prec_str}<br>'
#                 f'💧 {hum_str} &nbsp;·&nbsp; 👁️ {vis_str}'
#                 f'</div>'
#                 f'{prob_badge}{src_badge}{adverse_badge}'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )

#     # ── Summary data table ────────────────────────────────────────────────────
#     with st.expander("📋 Full Weather Forecast Table"):
#         rows_data = []
#         for ws in weather_stops:
#             w = ws["weather"]
#             rows_data.append({
#                 "Stop":              ws["location_name"],
#                 "Arrival Time":      ws.get("arrival_time", "—"),
#                 "Forecast At":       w.get("forecast_time", "—"),
#                 "Condition":         f"{w['icon']} {w['description']}",
#                 "Temp (°C)":         w.get("temperature_c", "—"),
#                 "Wind (km/h)":       w.get("wind_speed_kmh", "—"),
#                 "Rain (mm)":         w.get("precipitation_mm", "—"),
#                 "Rain Chance (%)":   w.get("precip_probability", "—"),
#                 "Humidity (%)":      w.get("humidity_pct", "—"),
#                 "Visibility (km)":   w.get("visibility_km", "—"),
#                 "Source":            w.get("source", "—"),
#                 "⚠️ Adverse":        "Yes" if w["is_adverse"] else "—",
#             })
#         st.dataframe(pd.DataFrame(rows_data), use_container_width=True)


# # ─────────────────────────────────────────────
# # FUEL PANEL  (called from _render_itinerary)
# # ─────────────────────────────────────────────
# def _render_fuel_panel(stops_list: list, constraints: dict, itin: dict):
#     """
#     Renders the full fuel cost + route savings analysis panel.
#     """
#     st.markdown("### ⛽ Fuel Cost & Route Savings Analysis")

#     vehicle_type = constraints.get("vehicle_type", "Van")
#     total_km     = itin.get("total_distance_km", 0)

#     if not total_km:
#         st.info("No distance data yet. Generate an itinerary first.")
#         return

#     # ── User controls ─────────────────────────────────────────────────────────
#     fc1, fc2, fc3 = st.columns(3)
#     with fc1:
#         city = st.selectbox(
#             "📍 City (for fuel price)",
#             list(FuelEngine.CITY_PRICES.keys()),
#             index=list(FuelEngine.CITY_PRICES.keys()).index("Mumbai")
#                   if "Mumbai" in FuelEngine.CITY_PRICES else 0,
#             key="fuel_city",
#         )
#     with fc2:
#         fuel_type_options = ["Auto (by vehicle)", "Petrol", "Diesel"]
#         fuel_choice = st.selectbox("⛽ Fuel type override", fuel_type_options, key="fuel_type_choice")
#     with fc3:
#         custom_price = st.number_input(
#             "₹/litre override (0 = use reference)",
#             min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="fuel_price_override"
#         )

#     override_price = custom_price if custom_price > 0 else None
#     fuel_type_map  = {"Petrol": "petrol", "Diesel": "diesel"}

#     # If override fuel type selected, compute price accordingly
#     if fuel_choice != "Auto (by vehicle)" and override_price is None:
#         ft          = fuel_type_map[fuel_choice]
#         override_price = FuelEngine.get_price(city, ft)

#     # ── Current itinerary fuel cost ───────────────────────────────────────────
#     fuel_current = FuelEngine.compute_fuel_cost(
#         total_km, vehicle_type, city, override_price
#     )

#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("Route Distance",    f"{total_km} km")
#     m2.metric("Fuel Consumed",     f"{fuel_current['litres_consumed']} L")
#     m3.metric("Price/Litre",       f"₹{fuel_current['price_per_litre']:.2f}")
#     m4.metric("Total Fuel Cost",   f"₹{fuel_current['total_cost_inr']:,.2f}")

#     st.markdown(
#         f'<div style="font-size:0.75rem;color:#8B949E;margin-bottom:20px">'
#         f'Vehicle: <b>{vehicle_type}</b> · '
#         f'Efficiency: <b>{fuel_current["efficiency_kmpl"]} km/L</b> · '
#         f'Fuel: <b>{fuel_current["fuel_type"].title()}</b> · '
#         f'City reference: <b>{city}</b> '
#         f'<span style="color:#30363D">(prices as of Jun 2025 — update via override)</span>'
#         f'</div>',
#         unsafe_allow_html=True,
#     )

#     # ── Route savings analysis ────────────────────────────────────────────────
#     st.markdown("#### 🔀 Optimized vs Un-Optimized Route Savings")

#     if len(stops_list) < 3:
#         st.info("Add at least 3 stops to compute route savings comparison.")
#         return

#     if st.button("🔍 Run Savings Analysis", key="run_savings"):
#         with st.spinner("Comparing original vs NN-optimized order via OSRM…"):
#             analysis = FuelEngine.savings_analysis(
#                 stops_list, vehicle_type, city, override_price
#             )
#         st.session_state["fuel_analysis"] = analysis

#     analysis = st.session_state.get("fuel_analysis")
#     if not analysis:
#         st.caption("Click 'Run Savings Analysis' to compare route orders.")
#         return

#     saved_km   = analysis.get("saved_km", 0)
#     saving_pct = analysis.get("saving_pct", 0)
#     saved_cost = analysis.get("saved_cost_inr", 0)

#     if saved_km > 0:
#         st.markdown(
#             f'<div class="fuel-save">'
#             f'<div style="font-size:1.05rem;font-weight:700;color:#3FB950;margin-bottom:10px">'
#             f'✅ You can save <b>₹{saved_cost:,.2f}</b> by reordering your stops!</div>'
#             f'<div style="display:flex;gap:32px;flex-wrap:wrap">'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">📏 Distance saved: <b>{saved_km} km</b></span>'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">📉 Reduction: <b>{saving_pct}%</b></span>'
#             f'<span style="font-size:0.85rem;color:#C9D1D9">⛽ Fuel saved: '
#             f'<b>{round(saved_km / analysis["fuel_detail_optimized"]["efficiency_kmpl"], 2)} L</b></span>'
#             f'</div></div>',
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             '<div class="fuel-warn">'
#             '<b style="color:#D4A843">ℹ️ Your current stop order is already near-optimal.</b><br>'
#             '<span style="color:#8B949E;font-size:0.82rem">The nearest-neighbor reordering did not '
#             'produce a shorter total route for this set of stops.</span>'
#             '</div>',
#             unsafe_allow_html=True,
#         )

#     # Side-by-side comparison
#     sa1, sa2 = st.columns(2)
#     with sa1:
#         st.markdown("**📋 Original Order**")
#         for i, name in enumerate(analysis.get("original_order", []), 1):
#             st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
#                         f'<span style="color:#D4A843">{i}.</span> {name}</div>', unsafe_allow_html=True)
#         orig_c = analysis.get("original_cost_inr", 0)
#         st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
#                     f'<b>Total: {analysis.get("original_km",0)} km · ₹{orig_c:,.2f}</b></div>', unsafe_allow_html=True)

#     with sa2:
#         st.markdown("**✅ Optimized Order**")
#         for i, name in enumerate(analysis.get("optimized_order", []), 1):
#             st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
#                         f'<span style="color:#3FB950">{i}.</span> {name}</div>', unsafe_allow_html=True)
#         opt_c = analysis.get("optimized_cost_inr", 0)
#         st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
#                     f'<b>Total: {analysis.get("optimized_km",0)} km · ₹{opt_c:,.2f}</b></div>', unsafe_allow_html=True)

#     # Per-leg savings chart
#     per_stop = analysis.get("per_stop_savings", [])
#     if per_stop:
#         st.markdown("#### 📊 Leg-by-Leg Distance Comparison")
#         df_legs = pd.DataFrame(per_stop)
#         fig = go.Figure()
#         fig.add_bar(
#             name="Original", x=df_legs["leg"].astype(str),
#             y=df_legs["original_km"],
#             marker_color="#E74C3C",
#             hovertemplate="Leg %{x}: %{y:.2f} km<extra>Original</extra>",
#         )
#         fig.add_bar(
#             name="Optimized", x=df_legs["leg"].astype(str),
#             y=df_legs["optimized_km"],
#             marker_color="#3FB950",
#             hovertemplate="Leg %{x}: %{y:.2f} km<extra>Optimized</extra>",
#         )
#         fig.update_layout(
#             barmode="group",
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=260,
#             xaxis=dict(title="Leg #", showgrid=False),
#             yaxis=dict(title="km", showgrid=True, gridcolor="#21262D"),
#             margin=dict(t=10, b=40, l=50, r=20),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     # ── Optimized Route Map ──────────────────────────────────────────────────
#     optimized_order = analysis.get("optimized_order", [])
#     if optimized_order and len(stops_list) >= 2:
#         st.markdown("#### 🗺️ Optimized Route Map")
#         st.caption("Stop order resequenced by Nearest-Neighbor TSP for minimum distance.")

#         # Reorder stops_list to match the optimized sequence
#         nn_order      = MLEngine().nearest_neighbor_route([(s["lat"], s["lon"]) for s in stops_list])
#         opt_stops     = [stops_list[i] for i in nn_order]

#         # Build a lightweight itinerary shell so route_map_scatter draws the route line
#         opt_itin_stub = {
#             "stops": [
#                 {
#                     "sequence": idx + 1,
#                     "stop_id":  s["stop_id"],
#                     "stop_type": s.get("stop_type", "Delivery"),
#                 }
#                 for idx, s in enumerate(opt_stops)
#             ]
#         }

#         # Use Dashboard.route_map_scatter — reuse existing map function
#         from plotly.subplots import make_subplots as _msp  # already imported at top
#         lats   = [s["lat"]  for s in opt_stops]
#         lons   = [s["lon"]  for s in opt_stops]
#         names  = [s["location_name"] for s in opt_stops]
#         colors_map = {
#             "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
#             "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
#         }
#         pt_colors = [colors_map.get(s.get("stop_type","Delivery"), "#3FB950") for s in opt_stops]

#         import plotly.graph_objects as _go
#         fig_opt = _go.Figure()
#         # Route line
#         fig_opt.add_trace(_go.Scattergeo(
#             lat=lats, lon=lons, mode="lines",
#             line=dict(width=2.5, color="#3FB950"), name="Optimized Route",
#         ))
#         # Stop markers
#         fig_opt.add_trace(_go.Scattergeo(
#             lat=lats, lon=lons, mode="markers+text",
#             marker=dict(size=13, color=pt_colors, line=dict(color="#0D1117", width=1)),
#             text=[f"{i+1}. {n}" for i, n in enumerate(names)],
#             textposition="top center",
#             textfont=dict(size=9, color="#C9D1D9"),
#             hovertemplate="<b>%{text}</b><extra></extra>",
#             name="Stops",
#         ))
#         import numpy as _np
#         fig_opt.update_geos(
#             center=dict(lat=_np.mean(lats), lon=_np.mean(lons)),
#             projection_scale=8,
#             showland=True,       landcolor="#21262D",
#             showocean=True,      oceancolor="#161B22",
#             showcountries=True,  countrycolor="#30363D",
#             showcoastlines=True, coastlinecolor="#30363D",
#         )
#         fig_opt.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=420,
#             margin=dict(t=10, b=10, l=10, r=10),
#             geo=dict(bgcolor="rgba(0,0,0,0)"),
#             showlegend=True,
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig_opt, use_container_width=True, key="opt_route_map")

#     # Routing source note
#     rs = analysis.get("routing_source", "")
#     if rs == "osrm":
#         st.caption("🛰️ Savings computed using real OSRM road distances.")
#     else:
#         st.caption("📐 Savings computed using Haversine estimates (OSRM unreachable).")


# # ─────────────────────────────────────────────
# # ITINERARY RENDERER
# # ─────────────────────────────────────────────
# def _render_itinerary(itin, stops_df, ai, dash, constraints=None, nl_stops_override=None):
#     if constraints is None:
#         constraints = {}

#     # Build stops_list for map + fuel panel.
#     # For dataset routes: match stop_ids back to the stops_df for coordinates.
#     # For NL / custom routes: nl_stops_override carries the pre-built lat/lon list.
#     stops_list = []
#     if nl_stops_override:
#         stops_list = nl_stops_override
#     elif not stops_df.empty and "stop_id" in stops_df.columns:
#         for s in itin.get("stops", []):
#             row = stops_df[stops_df["stop_id"] == s["stop_id"]]
#             if not row.empty:
#                 stops_list.append({
#                     "stop_id": s["stop_id"], "location_name": s["location_name"],
#                     "lat": float(row.iloc[0]["lat"]), "lon": float(row.iloc[0]["lon"]),
#                     "stop_type": s["stop_type"],
#                 })

#     # Header
#     st.markdown(
#         f'<div class="itinerary-header">'
#         f'<div style="font-family:\'Playfair Display\',serif;font-size:1.4rem;color:#F0F6FC;font-weight:700">'
#         f'📋 {itin.get("itinerary_title","Optimized Itinerary")}</div>'
#         f'<div style="margin-top:10px;color:#8B949E;font-size:0.82rem">'
#         f'🚗 {itin.get("driver","—")} &nbsp;|&nbsp; 🚛 {itin.get("vehicle","—")} '
#         f'&nbsp;|&nbsp; 📅 {itin.get("date","—")} &nbsp;|&nbsp; 🛣️ {itin.get("transport_mode","—")}'
#         f'</div></div>',
#         unsafe_allow_html=True,
#     )

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Distance",      f"{itin.get('total_distance_km', 0)} km")
#     c2.metric("Total Duration",      f"{itin.get('total_duration_min', 0)} min")
#     c3.metric("Efficiency Score",    f"{itin.get('efficiency_score', 0)}/100")
#     c4.metric("On-Time Probability", f"{itin.get('on_time_probability', 0)}%")

#     # Routing source banner
#     r_src = itin.get("routing_source", "")
#     if r_src == "osrm":
#         st.success("🛰️ Road distances powered by **OSRM** — real road network data")
#     elif r_src == "haversine_fallback":
#         st.warning("📐 Straight-line distance estimates used (OSRM unreachable). Times are approximate.")

#     # Map
#     if stops_list:
#         st.markdown("### 🗺️ Route Map")
#         st.plotly_chart(dash.route_map_scatter(stops_list, itin), use_container_width=True)

#     # Stop sequence
#     st.markdown("### 📍 Stop Sequence")
#     itin_stops = sorted(itin.get("stops", []), key=lambda x: x["sequence"])
#     max_seq    = max((x["sequence"] for x in itin_stops), default=0)

#     for s in itin_stops:
#         risk_color = {"High":"#E74C3C","Medium":"#D4A843","Low":"#3FB950","None":"#3FB950"}.get(
#             s.get("risk_flag","None"), "#3FB950"
#         )
#         risk_html = (
#             f' &nbsp;·&nbsp; <span style="color:{risk_color}">⚠️ {s.get("risk_reason","")}</span>'
#             if s.get("risk_flag","None") not in ["None",""] else ""
#         )
#         notes_html = (
#             f'<div class="stop-detail" style="margin-top:4px;font-style:italic">{s.get("notes","")}</div>'
#             if s.get("notes") else ""
#         )
#         connector = "" if s["sequence"] == max_seq else '<div class="connector-line"></div>'
#         st.markdown(
#             f'<div class="stop-card"><div class="stop-row">'
#             f'<div class="stop-num">{s["sequence"]}</div>'
#             f'<div style="flex:1">'
#             f'<div style="display:flex;justify-content:space-between;align-items:center">'
#             f'<b style="color:#F0F6FC">{s["location_name"]}</b>'
#             f'<span style="font-size:0.8rem;color:#8B949E">🕐 {s.get("arrival_time","—")} → {s.get("departure_time","—")}</span>'
#             f'</div>'
#             f'<div class="stop-detail">Type: {s.get("stop_type","—")} &nbsp;·&nbsp; '
#             f'Priority: {s.get("priority","—")} &nbsp;·&nbsp; '
#             f'Service: {s.get("service_duration_min","—")} min &nbsp;·&nbsp; '
#             f'Travel: {s.get("travel_time_from_prev_min","—")} min &nbsp;·&nbsp; '
#             f'Dist: {s.get("distance_from_prev_km","—")} km{risk_html}</div>'
#             f'{notes_html}</div></div></div>{connector}',
#             unsafe_allow_html=True,
#         )

#     if itin.get("warnings"):
#         with st.expander("⚠️ Warnings"):
#             for w in itin["warnings"]:
#                 st.warning(w)

#     with st.expander("📝 Optimization Notes"):
#         st.info(itin.get("optimization_notes", "No notes."))

#     # ── Constraint Violation Checker ─────────────────────────────────────────
#     st.markdown("### ⚡ Constraint Violation Check")
#     if constraints:
#         violations = ai.check_violations(itin, constraints)
#         if not violations:
#             st.markdown(
#                 '<div style="background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);'
#                 'border-radius:10px;padding:14px 18px;margin-bottom:12px">'
#                 '<b style="color:#3FB950">✅ No violations detected.</b> '
#                 'All stops are within time windows and driver hour limits.</div>',
#                 unsafe_allow_html=True,
#             )
#         else:
#             crit = [v for v in violations if v["severity"] == "Critical"]
#             warn = [v for v in violations if v["severity"] == "Warning"]
#             vc1, vc2 = st.columns(2)
#             vc1.metric("🔴 Critical", len(crit))
#             vc2.metric("🟡 Warnings", len(warn))
#             for v in violations:
#                 css   = "violation-critical" if v["severity"] == "Critical" else "violation-warning"
#                 color = "#E74C3C"            if v["severity"] == "Critical" else "#D4A843"
#                 icon  = "🔴"                 if v["severity"] == "Critical" else "🟡"
#                 label = {"time_window":"Time Window","driver_hours":"Driver Hours","capacity":"Capacity"}.get(v["type"], v["type"])
#                 st.markdown(
#                     f'<div class="{css}">'
#                     f'<div style="display:flex;justify-content:space-between">'
#                     f'<b style="color:#F0F6FC">{icon} Stop {v["sequence"]} — {v["location_name"]}</b>'
#                     f'<span style="font-size:0.75rem;color:{color};font-weight:600">{label} · {v["severity"]}</span>'
#                     f'</div><div style="font-size:0.82rem;color:#C9D1D9;margin-top:6px">{v["detail"]}</div></div>',
#                     unsafe_allow_html=True,
#                 )
#     else:
#         st.info("Generate an itinerary with constraints to see violation analysis.")

#     # ── Weather Panel ────────────────────────────────────────────────────────
#     # Build a stops list that includes arrival_time so forecasts are time-aware.
#     # Priority: dataset stops_list merged with itinerary arrival times.
#     itin_date = itin.get("date", datetime.now().strftime("%Y-%m-%d"))

#     # Map stop_id → arrival_time from the generated itinerary
#     arrival_map = {
#         s["stop_id"]: s.get("arrival_time", "08:00")
#         for s in itin.get("stops", [])
#     }

#     weather_stops = []
#     if stops_list:
#         # Dataset route: merge lat/lon from stops_list + arrival_time from itinerary
#         for s in stops_list:
#             weather_stops.append({
#                 **s,
#                 "arrival_time": arrival_map.get(s["stop_id"], "08:00"),
#                 "sequence":     next(
#                     (itin_s["sequence"] for itin_s in itin.get("stops", [])
#                      if itin_s["stop_id"] == s["stop_id"]), 0
#                 ),
#             })
#         # Sort by sequence so cards appear in travel order
#         weather_stops.sort(key=lambda x: x.get("sequence", 0))
#     else:
#         # NL / custom stops — coords embedded in itinerary stops
#         for s in itin.get("stops", []):
#             if "lat" in s and "lon" in s:
#                 weather_stops.append({
#                     "stop_id":       s.get("stop_id", ""),
#                     "location_name": s.get("location_name", ""),
#                     "lat":           s["lat"],
#                     "lon":           s["lon"],
#                     "stop_type":     s.get("stop_type", ""),
#                     "arrival_time":  s.get("arrival_time", "08:00"),
#                     "sequence":      s.get("sequence", 0),
#                 })

#     if weather_stops:
#         st.markdown("---")
#         _render_weather_panel(weather_stops, itin_date_str=itin_date)

#     # ── Fuel & Savings Panel ─────────────────────────────────────────────────
#     st.markdown("---")
#     # Build stops_list for fuel engine — prefer stops_list (dataset), else NL
#     fuel_stops = stops_list
#     if not fuel_stops:
#         for s in itin.get("stops", []):
#             if "lat" in s and "lon" in s:
#                 fuel_stops.append({
#                     "stop_id": s.get("stop_id",""),
#                     "location_name": s.get("location_name",""),
#                     "lat": s["lat"], "lon": s["lon"],
#                     "stop_type": s.get("stop_type",""),
#                 })
#     _render_fuel_panel(fuel_stops, constraints, itin)

#     # Adjust
#     st.markdown("### 🔄 Adjust Itinerary")
#     change_req = st.text_area(
#         "Describe the change in plain English:",
#         placeholder="e.g. 'Remove stop 3', 'Add 30-min rest after stop 2', 'Traffic — delay all by 20 min'",
#         height=90,
#     )
#     if st.button("Apply Changes"):
#         if change_req.strip():
#             with st.spinner("Re-planning…"):
#                 updated = ai.adjust_itinerary(itin, change_req)
#             st.session_state["generated_itinerary"] = updated
#             st.success("Itinerary updated!")
#             st.rerun()
#         else:
#             st.error("Please describe the change.")

#     # Export
#     st.markdown("### 📤 Export")
#     col_e1, col_e2 = st.columns(2)
#     with col_e1:
#         st.download_button("⬇️ Download JSON", json.dumps(itin, indent=2, default=str),
#                            "itinerary.json", "application/json", use_container_width=True)
#     with col_e2:
#         stop_rows = itin.get("stops", [])
#         if stop_rows:
#             st.download_button("⬇️ Download CSV", pd.DataFrame(stop_rows).to_csv(index=False),
#                                "itinerary_stops.csv", "text/csv", use_container_width=True)


# # ─────────────────────────────────────────────
# # PAGE: CLUSTERING
# # ─────────────────────────────────────────────
# def page_clustering(stops_df, ml, dash):
#     st.markdown('<div class="section-title">🧩 Stop Clustering Engine</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Unsupervised clustering to find natural groupings in your stop network</div>', unsafe_allow_html=True)

#     if stops_df.empty:
#         st.warning("No stops data loaded.")
#         return

#     col1, col2 = st.columns([1, 3])
#     with col1:
#         n_clusters  = st.slider("Number of Clusters", 2, 8, 4)
#         sample_size = st.slider("Sample Size", 20, min(len(stops_df), 200), min(len(stops_df), 60))
#         if st.button("Run Clustering"):
#             sample_df = stops_df.sample(min(sample_size, len(stops_df)), random_state=42)
#             labels, X_2d = ml.cluster_stops(sample_df, n_clusters)
#             st.session_state["cluster_labels"] = labels
#             st.session_state["cluster_X2d"]   = X_2d
#             st.session_state["cluster_df"]    = sample_df.reset_index(drop=True)
#             st.success("Clustering complete!")

#     with col2:
#         if st.session_state.get("cluster_labels") is not None:
#             st.markdown("**Cluster Visualization (PCA 2D)**")
#             st.plotly_chart(
#                 dash.cluster_scatter(st.session_state["cluster_df"],
#                                      st.session_state["cluster_labels"],
#                                      st.session_state["cluster_X2d"]),
#                 use_container_width=True,
#             )

#     if st.session_state.get("cluster_labels") is not None:
#         labels = st.session_state["cluster_labels"]
#         cdf    = st.session_state["cluster_df"]

#         keywords = ml.get_cluster_keywords()
#         st.markdown("### Cluster Keyword Profiles")
#         cols = st.columns(min(len(keywords), 5))
#         for i, (cid, kws) in enumerate(keywords.items()):
#             with cols[i % len(cols)]:
#                 st.markdown(
#                     f'<div class="card card-gold"><b>Cluster {cid+1}</b><br>'
#                     f'<small style="color:#8B949E">{" · ".join(kws)}</small></div>',
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("### Cluster Composition by Stop Type")
#         cdf2 = cdf.copy()
#         cdf2["cluster"] = [f"Cluster {l+1}" for l in labels]
#         comp = cdf2.groupby(["cluster","stop_type"]).size().reset_index(name="count")
#         fig  = px.bar(comp, x="cluster", y="count", color="stop_type",
#                       color_discrete_map=dash.STOP_COLORS, barmode="stack")
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=320,
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#             xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#         st.markdown("### Geographic Distribution of Clusters")
#         cdf2["x"] = st.session_state["cluster_X2d"][:, 0]
#         fig_geo   = px.scatter_mapbox(
#             cdf2, lat="lat", lon="lon", color="cluster",
#             hover_data=["location_name","stop_type","priority"], zoom=9, height=380,
#         )
#         fig_geo.update_layout(
#             mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", margin=dict(t=0,b=0,l=0,r=0),
#             legend=dict(bgcolor="rgba(0,0,0,0)"),
#         )
#         st.plotly_chart(fig_geo, use_container_width=True)


# # ─────────────────────────────────────────────
# # PAGE: AI ASSISTANT
# # ─────────────────────────────────────────────
# def page_assistant(stops_df, ai, kb_df):
#     st.markdown('<div class="section-title">💬 Logistics AI Assistant</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Grounded in your logistics knowledge base · '
#         'Only answers travel & logistics questions</div>',
#         unsafe_allow_html=True,
#     )

#     col_s1, col_s2, col_s3 = st.columns(3)
#     for col, label, active in [
#         (col_s1, "LLM",        bool(ai.llm)),
#         (col_s2, "Embeddings", bool(ai.embeddings)),
#         (col_s3, "RAG Index",  bool(ai.embeddings) and not kb_df.empty),
#     ]:
#         color  = "#2EA4A4" if active else "#E74C3C"
#         status = ("🟢 Connected" if active else "🔴 Offline") if label != "RAG Index" else ("🟢 Ready" if active else "⚠️ Keyword fallback")
#         col.markdown(
#             f'<div class="card" style="padding:10px;text-align:center">'
#             f'<span style="font-size:0.8rem;color:#8B949E">{label}</span><br>'
#             f'<b style="color:{color}">{status}</b></div>',
#             unsafe_allow_html=True,
#         )

#     st.markdown("")

#     with st.expander("ℹ️ What can I ask?", expanded=False):
#         st.markdown("""
# **In scope:** route planning, delivery operations, customs, fleet management, freight modes, KPIs, documentation.
# **Out of scope:** questions unrelated to travel or logistics are politely declined.
#         """)

#     if "assistant_history" not in st.session_state:
#         st.session_state["assistant_history"] = []

#     for msg in st.session_state["assistant_history"]:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             if msg["role"] == "assistant":
#                 if msg.get("sources"):
#                     _render_sources(msg["sources"])
#                 if msg.get("rejected"):
#                     st.caption("🚫 Outside the logistics/travel domain.")
#                 elif msg.get("grounded"):
#                     st.caption("✅ Grounded in knowledge base via semantic retrieval.")

#     user_input = st.chat_input("Ask about routes, delivery windows, customs, fuel costs…")
#     if user_input:
#         st.session_state["assistant_history"].append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         with st.chat_message("assistant"):
#             with st.spinner("Searching knowledge base…"):
#                 result = ai.get_kb_answer(user_input, kb_df)
#             answer_text = result["text"]
#             sources     = result.get("sources", [])
#             grounded    = result.get("grounded", False)
#             rejected    = result.get("rejected", False)
#             st.markdown(answer_text)
#             if sources:   _render_sources(sources)
#             if rejected:  st.caption("🚫 Outside the logistics/travel domain.")
#             elif grounded: st.caption("✅ Grounded in knowledge base via semantic retrieval.")
#             else:          st.caption("⚠️ Low-confidence retrieval — broader KB context used.")

#         st.session_state["assistant_history"].append({
#             "role": "assistant", "content": answer_text,
#             "sources": sources, "grounded": grounded, "rejected": rejected,
#         })

#     col_a, _ = st.columns([1, 5])
#     with col_a:
#         if st.button("🗑️ Clear Chat"):
#             st.session_state["assistant_history"] = []
#             st.rerun()

#     if not kb_df.empty:
#         with st.expander(f"📚 Knowledge Base ({len(kb_df)} articles)"):
#             st.dataframe(
#                 kb_df[["category","topic","content"]], use_container_width=True,
#                 column_config={"content": st.column_config.TextColumn("content", width="large")},
#             )


# def _render_sources(sources: list):
#     if not sources:
#         return
#     seen, unique = set(), []
#     for s in sources:
#         key = s.get("topic","")
#         if key not in seen:
#             seen.add(key); unique.append(s)

#     lines = []
#     for s in unique:
#         pct        = int(s.get("score", 0) * 100)
#         bar        = "█" * (pct // 10) + "░" * (10 - pct // 10)
#         lines.append(
#             f"**{s.get('category','—')}** › {s.get('topic','—')} "
#             f"<span style='color:#D4A843;font-family:monospace;font-size:0.75rem'>{bar} {pct}%</span>"
#         )
#     st.markdown(
#         '<div style="background:rgba(46,164,164,0.08);border:1px solid rgba(46,164,164,0.25);'
#         'border-radius:8px;padding:10px 14px;margin-top:8px">'
#         '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.06em;'
#         'text-transform:uppercase;margin-bottom:6px">📎 Sources retrieved</div>'
#         + "".join(f'<div style="font-size:0.8rem;color:#C9D1D9;margin:3px 0">{l}</div>' for l in lines)
#         + "</div>",
#         unsafe_allow_html=True,
#     )


# # ─────────────────────────────────────────────
# # PAGE: PERFORMANCE
# # ─────────────────────────────────────────────
# def page_performance(stops_df, routes_df, ai, dash):
#     st.markdown('<div class="section-title">📈 Performance Analytics</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Route efficiency, on-time delivery, fuel costs, and operational insights</div>', unsafe_allow_html=True)

#     if stops_df.empty or routes_df.empty:
#         st.warning("No data loaded.")
#         return

#     c1, c2, c3, c4 = st.columns(4)
#     on_time_pct     = round(100 * (stops_df["status"] == "On Time").mean(), 1)
#     avg_distance    = round(routes_df["distance_km"].mean(), 1)
#     total_fuel_cost = round(routes_df["estimated_fuel_cost_inr"].sum(), 0)
#     avg_stops       = round(stops_df.groupby("route_id").size().mean(), 1)
#     c1.metric("On-Time Rate",       f"{on_time_pct}%")
#     c2.metric("Avg Distance/Route", f"{avg_distance} km")
#     c3.metric("Total Fuel Cost",    f"₹{total_fuel_cost:,.0f}")
#     c4.metric("Avg Stops/Route",    avg_stops)

#     st.markdown("---")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**On-Time % by Transport Mode (Trend)**")
#         st.plotly_chart(dash.performance_timeline(routes_df), use_container_width=True)
#     with col2:
#         st.markdown("**On-Time Delivery Rate**")
#         st.plotly_chart(dash.on_time_gauge(on_time_pct), use_container_width=True)

#     col3, col4 = st.columns(2)
#     with col3:
#         st.markdown("**Avg Fuel Cost by Transport Mode**")
#         st.plotly_chart(dash.fuel_cost_bar(routes_df), use_container_width=True)
#     with col4:
#         st.markdown("**Stops by Status**")
#         sc = stops_df["status"].value_counts().reset_index()
#         sc.columns = ["status","count"]
#         fig = go.Figure(go.Bar(
#             x=sc["status"], y=sc["count"],
#             marker_color=[dash.STATUS_COLORS.get(s,"#8B949E") for s in sc["status"]],
#             hovertemplate="%{x}: %{y}<extra></extra>",
#         ))
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#C9D1D9", height=260,
#             xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262D"),
#             margin=dict(t=10, b=40, l=40, r=20),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     if "delay_reason" in stops_df.columns:
#         st.markdown("**Delay Reasons**")
#         delayed = stops_df[stops_df["delay_reason"].notna() & (stops_df["delay_reason"] != "")]
#         if not delayed.empty:
#             dr    = delayed["delay_reason"].value_counts().reset_index()
#             dr.columns = ["reason","count"]
#             fig_d = px.bar(dr, x="count", y="reason", orientation="h",
#                            color_discrete_sequence=["#E74C3C"])
#             fig_d.update_layout(
#                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                 font_color="#C9D1D9", height=260,
#                 xaxis=dict(showgrid=True, gridcolor="#21262D"),
#                 yaxis=dict(showgrid=False),
#                 margin=dict(t=10, b=30, l=150, r=20),
#             )
#             st.plotly_chart(fig_d, use_container_width=True)

#     st.markdown("### 🤖 AI Route Performance Insights")
#     if st.button("Generate AI Insights"):
#         with st.spinner("Analyzing route performance…"):
#             insights = ai.analyze_route_performance(routes_df, stops_df)
#         st.markdown(f'<div class="summary-box">{insights}</div>', unsafe_allow_html=True)

#     if st.session_state.get("plan_history"):
#         with st.expander("📋 Itinerary Generation History (this session)"):
#             st.dataframe(pd.DataFrame(st.session_state["plan_history"]), use_container_width=True)

#     with st.expander("📋 Full Stops Table"):
#         st.dataframe(stops_df.sort_values("scheduled_date", ascending=False), use_container_width=True)

#     with st.expander("📋 Full Routes Table"):
#         st.dataframe(routes_df.sort_values("route_date", ascending=False), use_container_width=True)


# # ─────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────
# def main():
#     init_session()

#     loader = DataLoader()
#     ai     = AIEngine()
#     ml     = MLEngine()
#     dash   = Dashboard()

#     stops_df  = loader.load_stops()
#     routes_df = loader.load_routes()
#     kb_df     = loader.load_kb()
#     stats     = loader.get_summary_stats(stops_df, routes_df)

#     with st.sidebar:
#         st.markdown(
#             '<div style="text-align:center;padding:20px 0 10px">'
#             '<div style="font-family:\'Playfair Display\',serif;font-size:1.5rem;color:#F0F6FC;font-weight:900">🗺️ RouteIQ</div>'
#             '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.1em;text-transform:uppercase">Logistics Itinerary Planner</div>'
#             '</div>',
#             unsafe_allow_html=True,
#         )
#         st.markdown("---")
#         page = st.radio(
#             "Navigation",
#             ["📊 Overview", "🔍 Itinerary Planner", "🧩 Clustering", "💬 AI Assistant", "📈 Performance"],
#             label_visibility="collapsed",
#         )

#         if not stops_df.empty:
#             st.markdown("---")
#             st.markdown(
#                 '<div style="font-size:0.7rem;color:#8B949E;letter-spacing:0.1em;'
#                 'text-transform:uppercase;margin-bottom:8px">Quick Stats</div>',
#                 unsafe_allow_html=True,
#             )
#             for val, label, color in [
#                 (stats.get("total_stops",  0),              "Total Stops",        "#D4A843"),
#                 (stats.get("total_routes", 0),              "Active Routes",      "#2EA4A4"),
#                 (f"{stats.get('on_time_pct',0)}%",          "On-Time Rate",       "#3FB950"),
#                 (stats.get("high_priority", 0),             "High Priority Stops","#E74C3C"),
#             ]:
#                 st.markdown(
#                     f'<div class="card" style="padding:12px">'
#                     f'<div style="color:{color};font-size:1.4rem;font-weight:700">{val}</div>'
#                     f'<div style="color:#8B949E;font-size:0.72rem">{label}</div></div>',
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("---")
#         st.markdown(
#             f'<div style="font-size:0.75rem;color:#8B949E">'
#             f'{"🟢 LLM Connected" if ai.llm else "🔴 LLM Offline (fallback)"}</div>',
#             unsafe_allow_html=True,
#         )

#     if page == "📊 Overview":
#         page_overview(stops_df, routes_df, stats, dash)
#     elif page == "🔍 Itinerary Planner":
#         page_planner(stops_df, ai, ml, dash)
#     elif page == "🧩 Clustering":
#         page_clustering(stops_df, ml, dash)
#     elif page == "💬 AI Assistant":
#         page_assistant(stops_df, ai, kb_df)
#     elif page == "📈 Performance":
#         page_performance(stops_df, routes_df, ai, dash)


# if __name__ == "__main__":
#     main()