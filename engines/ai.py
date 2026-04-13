import os
import re
import json
from datetime import datetime, timedelta

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from .ml import MLEngine
from .rag import RAGEngine


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

    # ── Coordinate regex (India bounding box) ────────────────────────────────
    _COORD_RE = re.compile(
        r"\(?\s*"
        r"(?P<lat>[+-]?(?:3[0-7]|[12]\d|\d)\.\d{2,8})"
        r"[\s,]+"
        r"(?P<lon>[+-]?(?:9[0-8]|[7-8]\d|6[8-9])\.\d{2,8})"
        r"\s*\)?",
    )

    @classmethod
    def _extract_inline_coords(cls, text: str) -> list:
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

        if has_inline and result["stops"]:
            coord_idx = 0
            for stop in result["stops"]:
                if stop.get("coordinates_source") == "user_provided":
                    if coord_idx < len(inline_coords):
                        stop["lat"] = inline_coords[coord_idx][0]
                        stop["lon"] = inline_coords[coord_idx][1]
                        coord_idx += 1

        return result

    def generate_itinerary(self, stops_list, constraints, route_context):
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
        raw    = self._call(system_prompt, user_prompt)
        result = self._parse_json(raw, self._rule_based_itinerary(stops_list, constraints))

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
                    stop["distance_from_prev_km"]    = osrm["legs"][leg_idx]["distance_km"]
                    stop["travel_time_from_prev_min"] = osrm["legs"][leg_idx]["duration_min"]
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

    def check_violations(self, itinerary: dict, constraints: dict) -> list:
        violations = []
        stops      = sorted(itinerary.get("stops", []), key=lambda s: s["sequence"])
        if not stops:
            return violations

        date_str = itinerary.get("date", datetime.now().strftime("%Y-%m-%d"))
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
