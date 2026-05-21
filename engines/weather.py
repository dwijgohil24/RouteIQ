import streamlit as st
from datetime import datetime

from core.http_client import get_http_adapter


class WeatherEngine:
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
        import random as _rnd
        month = datetime.now().month
        band  = "north" if lat > 20 else "south"
        row   = cls._SEASONAL_FALLBACK.get(month, cls._SEASONAL_FALLBACK[6])
        t, h, desc, icon, code = row[band]

        seed = int(abs(lat * 1000 + lon * 100)) % 100
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

    @classmethod
    def _fetch_hourly_payload(cls, lat: float, lon: float) -> dict | None:
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
            f"&forecast_days=2"
        )
        urls = [
            "https://api.open-meteo.com/v1/forecast" + params,
            "http://api.open-meteo.com/v1/forecast"  + params,
        ]
        data = get_http_adapter().get(urls)
        if data and "hourly" in data and "time" in data.get("hourly", {}):
            cache[cache_key] = data
            st.session_state["weather_cache"] = cache
            return data
        return None

    @classmethod
    def _extract_hour(cls, payload: dict, target_dt: datetime) -> dict:
        hourly     = payload["hourly"]
        time_strs  = hourly["time"]
        target_str = target_dt.strftime("%Y-%m-%dT%H:00")

        idx = None
        if target_str in time_strs:
            idx = time_strs.index(target_str)
        else:
            best_diff = float("inf")
            for i, ts in enumerate(time_strs):
                try:
                    dt   = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
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
            "temperature_c":      round(float(_val("temperature_2m",         25)), 1),
            "wind_speed_kmh":     round(float(_val("wind_speed_10m",         10)), 1),
            "precipitation_mm":   round(float(_val("precipitation",           0)), 1),
            "precip_probability": int(_val("precipitation_probability",        0)),
            "humidity_pct":       int(_val("relative_humidity_2m",           65)),
            "visibility_km":      round(float(vis_raw) / 1000, 1),
            "weather_code":       code,
            "description":        desc,
            "icon":               icon,
            "is_adverse":         code in cls.ADVERSE_CODES,
            "source":             "open-meteo-hourly",
        }

    @classmethod
    def fetch_at_time(cls, lat: float, lon: float, arrival_time_str: str,
                      itin_date_str: str = None) -> dict:
        try:
            date_str = itin_date_str or datetime.now().strftime("%Y-%m-%d")
            target   = datetime.strptime(f"{date_str} {arrival_time_str}", "%Y-%m-%d %H:%M")
        except Exception:
            target   = datetime.now()

        stop_key = f"stop_{cls._cache_key(lat, lon)}_{target.strftime('%Y%m%d%H')}"
        cache    = st.session_state.get("weather_cache", {})
        if stop_key in cache:
            return cache[stop_key]

        payload = cls._fetch_hourly_payload(lat, lon)

        if payload:
            result = cls._extract_hour(payload, target)
            if result:
                result["forecast_time"]    = target.strftime("%d %b %Y · %H:%M")
                result["arrival_time_str"] = arrival_time_str
                cache[stop_key] = result
                st.session_state["weather_cache"] = cache
                return result

        result = cls._seasonal_estimate(lat, lon, arrival_hour=target.hour)
        result["forecast_time"]    = target.strftime("%d %b %Y · %H:%M") + " (est.)"
        result["arrival_time_str"] = arrival_time_str
        cache[stop_key] = result
        st.session_state["weather_cache"] = cache
        return result

    @classmethod
    def fetch_route_at_times(cls, stops_with_coords: list, itin_date_str: str = None) -> list:
        results = []
        for s in stops_with_coords:
            arrival = s.get("arrival_time", "08:00") or "08:00"
            w = cls.fetch_at_time(s["lat"], s["lon"], arrival, itin_date_str)
            results.append({**s, "weather": w})
        return results

    @classmethod
    def fetch(cls, lat: float, lon: float) -> dict:
        return cls.fetch_at_time(lat, lon, datetime.now().strftime("%H:%M"))

    @classmethod
    def fetch_route(cls, stops_with_coords: list) -> list:
        return cls.fetch_route_at_times(stops_with_coords)
