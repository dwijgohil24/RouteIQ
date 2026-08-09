import time
import streamlit as st

from core.http_client import get_http_adapter


class GeoEngine:
    BASE_URL  = "https://nominatim.openstreetmap.org/reverse"
    CACHE_KEY = "geocode_cache"

    @classmethod
    def _cache_key(cls, lat: float, lon: float) -> str:
        return f"{round(lat, 4)}_{round(lon, 4)}"

    @classmethod
    def reverse_geocode(cls, lat: float, lon: float) -> dict:
        key   = cls._cache_key(lat, lon)
        cache = st.session_state.get(cls.CACHE_KEY, {})
        if key in cache:
            return cache[key]

        params = {
            "lat": lat, "lon": lon, "format": "json",
            "zoom": 16, "addressdetails": 1,
        }
        headers = {"User-Agent": "RouteIQ-Logistics/1.0"}

        urls   = [f"{s}://nominatim.openstreetmap.org/reverse" for s in ("https", "http")]
        data   = get_http_adapter().get(urls, params=params, headers=headers)
        result = None

        if data and "error" not in data:
            addr  = data.get("address", {})
            parts = []
            for key_try in ("amenity", "shop", "building", "tourism",
                            "road", "neighbourhood", "suburb",
                            "village", "town", "city_district", "city"):
                val = addr.get(key_try, "")
                if val and val not in parts:
                    parts.append(val)
                if len(parts) >= 2:
                    break
            city = addr.get("city") or addr.get("town") or addr.get("village") or ""
            if city and city not in parts:
                parts.append(city)
            display = ", ".join(parts) if parts else data.get("display_name", "Unknown location")
            if len(display) > 60:
                display = display[:57] + "…"
            result = {
                "display_name": display,
                "full_address": data.get("display_name", display),
                "source":       "nominatim",
            }

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

            if geo["source"] == "nominatim" and geo["display_name"]:
                stop["location_name"] = geo["display_name"]

            enriched.append(stop)

            if i < len(stops) - 1:
                time.sleep(1.1)

        return enriched
