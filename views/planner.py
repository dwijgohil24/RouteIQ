import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from datetime import datetime

from engines.ml import MLEngine
from engines.weather import WeatherEngine
from engines.fuel import FuelEngine
from engines.geo import GeoEngine
from engines.map_renderer import render_animated_map_in_streamlit
from engines.pdf_exporter import generate_itinerary_pdf


# ─────────────────────────────────────────────
# WEATHER PANEL
# ─────────────────────────────────────────────
def _render_weather_panel(stops_with_coords: list, itin_date_str: str = None):
    st.markdown("### 🌤️ Weather Forecast Along Route")
    st.markdown(
        '<div style="font-size:0.8rem;color:#8B949E;margin-bottom:16px">'
        'Forecast shown at <b style="color:#D4A843">each stop\'s expected arrival time</b> '
        '— not current conditions. Powered by Open-Meteo hourly API.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not stops_with_coords:
        st.info("No coordinate data available for weather lookup.")
        return

    with st.spinner("Fetching arrival-time forecasts for each stop…"):
        weather_stops = WeatherEngine.fetch_route_at_times(stops_with_coords, itin_date_str)

    sources    = [ws["weather"]["source"] for ws in weather_stops]
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

    cols_per_row = min(4, len(weather_stops))
    card_rows    = [weather_stops[i:i+cols_per_row]
                    for i in range(0, len(weather_stops), cols_per_row)]

    for card_row in card_rows:
        cols = st.columns(len(card_row))
        for col, ws in zip(cols, card_row):
            w           = ws["weather"]
            source      = w.get("source", "")
            arrival     = ws.get("arrival_time", "")
            precip_prob = w.get("precip_probability")

            border_color = "#E74C3C" if w["is_adverse"] else "#30363D"
            temp_str = f"{w['temperature_c']}°C"     if w.get("temperature_c")    is not None else "—"
            wind_str = f"{w['wind_speed_kmh']} km/h" if w.get("wind_speed_kmh")   is not None else "—"
            prec_str = f"{w['precipitation_mm']} mm" if w.get("precipitation_mm") is not None else "—"
            hum_str  = f"{w['humidity_pct']}%"       if w.get("humidity_pct")     is not None else "—"
            vis_str  = f"{w['visibility_km']} km"    if w.get("visibility_km")    is not None else "—"
            prob_str = f"{precip_prob}% rain chance" if precip_prob is not None else ""

            src_badge = (
                '<div style="font-size:0.58rem;color:#3FB950;margin-top:5px">🟢 Open-Meteo hourly</div>'
                if "open-meteo" in source
                else '<div style="font-size:0.58rem;color:#D4A843;margin-top:5px">🟡 Seasonal estimate</div>'
            )
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
                f'<div style="font-size:0.6rem;font-family:monospace;color:#2EA4A4;'
                f'letter-spacing:0.05em;margin-bottom:6px">🕐 ARRIVAL {arrival}</div>'
                f'<div style="font-size:1.9rem;margin-bottom:2px">{w["icon"]}</div>'
                f'<div style="font-size:0.78rem;font-weight:700;color:#F0F6FC;margin-bottom:2px">'
                f'{ws["location_name"][:20]}</div>'
                f'<div style="font-size:1.2rem;font-weight:700;color:#D4A843;margin-bottom:2px">{temp_str}</div>'
                f'<div style="font-size:0.72rem;color:#C9D1D9">{w["description"]}</div>'
                f'<div style="font-size:0.65rem;color:#8B949E;margin-top:8px;line-height:1.7">'
                f'💨 {wind_str} &nbsp;·&nbsp; 🌧️ {prec_str}<br>'
                f'💧 {hum_str} &nbsp;·&nbsp; 👁️ {vis_str}'
                f'</div>'
                f'{prob_badge}{src_badge}{adverse_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with st.expander("📋 Full Weather Forecast Table"):
        rows_data = []
        for ws in weather_stops:
            w = ws["weather"]
            rows_data.append({
                "Stop":            ws["location_name"],
                "Arrival Time":    ws.get("arrival_time", "—"),
                "Forecast At":     w.get("forecast_time", "—"),
                "Condition":       f"{w['icon']} {w['description']}",
                "Temp (°C)":       w.get("temperature_c", "—"),
                "Wind (km/h)":     w.get("wind_speed_kmh", "—"),
                "Rain (mm)":       w.get("precipitation_mm", "—"),
                "Rain Chance (%)": w.get("precip_probability", "—"),
                "Humidity (%)":    w.get("humidity_pct", "—"),
                "Visibility (km)": w.get("visibility_km", "—"),
                "Source":          w.get("source", "—"),
                "⚠️ Adverse":      "Yes" if w["is_adverse"] else "—",
            })
        st.dataframe(pd.DataFrame(rows_data), use_container_width=True)


# ─────────────────────────────────────────────
# FUEL PANEL
# ─────────────────────────────────────────────
def _render_fuel_panel(stops_list: list, constraints: dict, itin: dict):
    st.markdown("### ⛽ Fuel Cost & Route Savings Analysis")

    vehicle_type = constraints.get("vehicle_type", "Van")
    total_km     = itin.get("total_distance_km", 0)

    if not total_km:
        st.info("No distance data yet. Generate an itinerary first.")
        return

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
        fuel_choice = st.selectbox("⛽ Fuel type override",
                                   ["Auto (by vehicle)", "Petrol", "Diesel"],
                                   key="fuel_type_choice")
    with fc3:
        custom_price = st.number_input(
            "₹/litre override (0 = use reference)",
            min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="fuel_price_override"
        )

    override_price = custom_price if custom_price > 0 else None
    if fuel_choice != "Auto (by vehicle)" and override_price is None:
        ft             = {"Petrol": "petrol", "Diesel": "diesel"}[fuel_choice]
        override_price = FuelEngine.get_price(city, ft)

    fuel_current = FuelEngine.compute_fuel_cost(total_km, vehicle_type, city, override_price)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route Distance",  f"{total_km} km")
    m2.metric("Fuel Consumed",   f"{fuel_current['litres_consumed']} L")
    m3.metric("Price/Litre",     f"₹{fuel_current['price_per_litre']:.2f}")
    m4.metric("Total Fuel Cost", f"₹{fuel_current['total_cost_inr']:,.2f}")

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

    st.markdown("#### 🔀 Optimized vs Un-Optimized Route Savings")

    if len(stops_list) < 3:
        st.info("Add at least 3 stops to compute route savings comparison.")
        return

    if st.button("🔍 Run Savings Analysis", key="run_savings"):
        with st.spinner("Comparing original vs NN-optimized order via OSRM…"):
            analysis = FuelEngine.savings_analysis(stops_list, vehicle_type, city, override_price)
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

    sa1, sa2 = st.columns(2)
    with sa1:
        st.markdown("**📋 Original Order**")
        for i, name in enumerate(analysis.get("original_order", []), 1):
            st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
                        f'<span style="color:#D4A843">{i}.</span> {name}</div>', unsafe_allow_html=True)
        orig_c = analysis.get("original_cost_inr", 0)
        st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
                    f'<b>Total: {analysis.get("original_km",0)} km · ₹{orig_c:,.2f}</b></div>',
                    unsafe_allow_html=True)
    with sa2:
        st.markdown("**✅ Optimized Order**")
        for i, name in enumerate(analysis.get("optimized_order", []), 1):
            st.markdown(f'<div style="font-size:0.8rem;color:#8B949E;padding:3px 0">'
                        f'<span style="color:#3FB950">{i}.</span> {name}</div>', unsafe_allow_html=True)
        opt_c = analysis.get("optimized_cost_inr", 0)
        st.markdown(f'<div style="margin-top:10px;font-size:0.85rem;color:#C9D1D9">'
                    f'<b>Total: {analysis.get("optimized_km",0)} km · ₹{opt_c:,.2f}</b></div>',
                    unsafe_allow_html=True)

    per_stop = analysis.get("per_stop_savings", [])
    if per_stop:
        st.markdown("#### 📊 Leg-by-Leg Distance Comparison")
        df_legs = pd.DataFrame(per_stop)
        fig = go.Figure()
        fig.add_bar(name="Original",  x=df_legs["leg"].astype(str), y=df_legs["original_km"],
                    marker_color="#E74C3C",
                    hovertemplate="Leg %{x}: %{y:.2f} km<extra>Original</extra>")
        fig.add_bar(name="Optimized", x=df_legs["leg"].astype(str), y=df_legs["optimized_km"],
                    marker_color="#3FB950",
                    hovertemplate="Leg %{x}: %{y:.2f} km<extra>Optimized</extra>")
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

    optimized_order = analysis.get("optimized_order", [])
    if optimized_order and len(stops_list) >= 2:
        st.markdown("#### 🗺️ Optimized Route Map")
        st.caption("Stop order resequenced by Nearest-Neighbor TSP for minimum distance.")

        nn_order  = MLEngine().nearest_neighbor_route([(s["lat"], s["lon"]) for s in stops_list])
        opt_stops = [stops_list[i] for i in nn_order]

        use_anim_opt = st.checkbox("🚗 Animate optimized route", value=True, key="anim_opt_toggle")
        if use_anim_opt:
            render_animated_map_in_streamlit(
                stops_list=opt_stops,
                vehicle_type="Car",
                animation_duration_s=4.0,
                height=420,
            )
        else:
            lats      = [s["lat"]  for s in opt_stops]
            lons      = [s["lon"]  for s in opt_stops]
            names     = [s["location_name"] for s in opt_stops]
            colors_map = {
                "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
                "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
            }
            pt_colors = [colors_map.get(s.get("stop_type", "Delivery"), "#3FB950") for s in opt_stops]

            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scattergeo(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=2.5, color="#3FB950"), name="Optimized Route",
            ))
            fig_opt.add_trace(go.Scattergeo(
                lat=lats, lon=lons, mode="markers+text",
                marker=dict(size=13, color=pt_colors, line=dict(color="#0D1117", width=1)),
                text=[f"{i+1}. {n}" for i, n in enumerate(names)],
                textposition="top center",
                textfont=dict(size=9, color="#C9D1D9"),
                hovertemplate="<b>%{text}</b><extra></extra>",
                name="Stops",
            ))
            fig_opt.update_geos(
                center=dict(lat=np.mean(lats), lon=np.mean(lons)), projection_scale=8,
                showland=True, landcolor="#21262D", showocean=True, oceancolor="#161B22",
                showcountries=True, countrycolor="#30363D", showcoastlines=True, coastlinecolor="#30363D",
            )
            fig_opt.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#C9D1D9", height=420,
                margin=dict(t=10, b=10, l=10, r=10),
                geo=dict(bgcolor="rgba(0,0,0,0)"),
                showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_opt, use_container_width=True, key="opt_route_map")

    rs = analysis.get("routing_source", "")
    if rs == "osrm":
        st.caption("🛰️ Savings computed using real OSRM road distances.")
    else:
        st.caption("📐 Savings computed using Haversine estimates (OSRM unreachable).")


# ─────────────────────────────────────────────
# ITINERARY RENDERER
# ─────────────────────────────────────────────
def _render_itinerary(itin, stops_df, ai, dash, constraints=None, nl_stops_override=None, route_geometry=None):
    if constraints is None:
        constraints = {}

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

    r_src = itin.get("routing_source", "")
    if r_src == "osrm":
        st.success("🛰️ Road distances powered by **OSRM** — real road network data")
    elif r_src == "haversine_fallback":
        st.warning("📐 Straight-line distance estimates used (OSRM unreachable). Times are approximate.")

    if stops_list:
        st.markdown("### 🗺️ Route Map")

        # Determine vehicle type from constraints for the animated icon
        _vehicle = constraints.get("vehicle_type", "Car") if constraints else "Car"

        # Toggle: animated vs static map
        use_animated = st.checkbox("🚗 Animate vehicle on route", value=True, key="anim_map_toggle")

        if use_animated:
            render_animated_map_in_streamlit(
                stops_list=stops_list,
                itinerary=itin,
                vehicle_type=_vehicle,
                animation_duration_s=4.0,
                height=500,
                route_geometry=route_geometry,
            )
            st.caption(
                f"🎬 {_vehicle} animation · Tap stops for details · Tap **↻ Replay** to rewatch"
            )
        else:
            st.plotly_chart(dash.route_map_scatter(stops_list, itin), use_container_width=True)

    st.markdown("### 📍 Stop Sequence")
    itin_stops = sorted(itin.get("stops", []), key=lambda x: x["sequence"])
    max_seq    = max((x["sequence"] for x in itin_stops), default=0)

    for s in itin_stops:
        risk_color = {"High": "#E74C3C", "Medium": "#D4A843", "Low": "#3FB950", "None": "#3FB950"}.get(
            s.get("risk_flag", "None"), "#3FB950"
        )
        risk_html = (
            f' &nbsp;·&nbsp; <span style="color:{risk_color}">⚠️ {s.get("risk_reason","")}</span>'
            if s.get("risk_flag", "None") not in ["None", ""] else ""
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
                label = {"time_window": "Time Window", "driver_hours": "Driver Hours",
                         "capacity": "Capacity"}.get(v["type"], v["type"])
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

    itin_date   = itin.get("date", datetime.now().strftime("%Y-%m-%d"))
    arrival_map = {s["stop_id"]: s.get("arrival_time", "08:00") for s in itin.get("stops", [])}

    weather_stops = []
    if stops_list:
        for s in stops_list:
            weather_stops.append({
                **s,
                "arrival_time": arrival_map.get(s["stop_id"], "08:00"),
                "sequence": next(
                    (itin_s["sequence"] for itin_s in itin.get("stops", [])
                     if itin_s["stop_id"] == s["stop_id"]), 0
                ),
            })
        weather_stops.sort(key=lambda x: x.get("sequence", 0))
    else:
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

    st.markdown("---")
    fuel_stops = stops_list
    if not fuel_stops:
        for s in itin.get("stops", []):
            if "lat" in s and "lon" in s:
                fuel_stops.append({
                    "stop_id": s.get("stop_id", ""), "location_name": s.get("location_name", ""),
                    "lat": s["lat"], "lon": s["lon"], "stop_type": s.get("stop_type", ""),
                })
    _render_fuel_panel(fuel_stops, constraints, itin)

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

    st.markdown("### 📤 Export")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.download_button("⬇️ Download JSON", json.dumps(itin, indent=2, default=str),
                           "itinerary.json", "application/json", use_container_width=True)
    with col_e2:
        stop_rows = itin.get("stops", [])
        if stop_rows:
            st.download_button("⬇️ Download CSV", pd.DataFrame(stop_rows).to_csv(index=False),
                               "itinerary_stops.csv", "text/csv", use_container_width=True)
    with col_e3:
        default_name = itin.get("driver") or "Ramesh"
        pdf_author = st.text_input(
            "Prepared by (PDF)",
            value=default_name,
            key="pdf_prepared_by",
            placeholder="Your name…",
        )
        try:
            pdf_bytes = generate_itinerary_pdf(
                itin, constraints,
                prepared_by=(pdf_author.strip() or "Ramesh"),
            )
            fname = f"routeiq_itinerary_{itin.get('date', 'export')}.pdf"
            st.download_button("⬇️ Download PDF", pdf_bytes,
                               fname, "application/pdf", use_container_width=True)
        except Exception as _pdf_err:
            st.error(f"PDF generation failed: {_pdf_err}")


# ─────────────────────────────────────────────
# MAIN PLANNER PAGE
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
            route_options = {}
            for rid in sorted(stops_df["route_id"].unique()):
                rdf   = stops_df[stops_df["route_id"] == rid]
                n     = len(rdf)
                first = rdf.iloc[0]["location_name"]
                last  = rdf.iloc[-1]["location_name"]
                route_options[f"{rid}  ({n} stops: {first} → {last})"] = rid
            selected_label = st.selectbox("Select Route to Plan", list(route_options.keys()))
            selected_route = route_options[selected_label]
            route_stops    = stops_df[stops_df["route_id"] == selected_route].copy()

            st.markdown(f"**{len(route_stops)} stops on route {selected_route}**")
            st.dataframe(
                route_stops[["stop_id", "location_name", "stop_type", "priority",
                              "time_window_start", "time_window_end", "notes"]].reset_index(drop=True),
                use_container_width=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                driver_name = st.text_input("Driver Name", value="Ramesh Kumar")
                start_time  = st.text_input("Start Time (HH:MM)", value="08:00")
            with col2:
                transport_mode   = st.selectbox("Transport Mode", ["Road", "Rail", "Air", "Sea"])
                vehicle_type     = st.selectbox("Vehicle Type", ["Truck", "Van", "Motorcycle", "Car", "Tempo"])
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
                        "notes":             str(r.get("notes", "")) if pd.notnull(r.get("notes", "")) else "",
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
                st.session_state["road_geometry"] = osrm_preview.get("route_geometry")
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
                st.session_state["fuel_analysis"]       = None
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
            c_mode  = st.selectbox("Transport Mode", ["Road", "Rail", "Air", "Sea"], key="c_mode")
            c_hours = st.slider("Max Hours", 4, 14, 8, key="c_hours")
            c_cap   = st.number_input("Capacity (kg)", 100, 10000, 500, step=100, key="c_cap")

        if st.button("🚀 Generate Custom Itinerary", key="gen_custom"):
            custom_stops = []
            parse_errors = []
            for i, line in enumerate(custom_text.strip().split("\n")):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    parse_errors.append(f"Line {i+1}: need at least Name, Lat, Lon — got: `{line}`")
                    continue
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
                    parse_errors.append(f"Line {i+1}: lat/lon must be numbers — got `{parts[1]}`, `{parts[2]}`")

            if parse_errors:
                for msg in parse_errors:
                    st.warning(f"⚠️ {msg}")

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
                st.session_state["road_geometry"] = osrm_cs.get("route_geometry")
                st.caption("Routing: 🟢 OSRM" if osrm_cs["source"] == "osrm" else "Routing: 🟡 Haversine fallback")
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
                                     ["(auto-detect)", "Road", "Rail", "Air", "Sea"], key="nl_mode")
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
                    n_user = sum(1 for s in stops_list if s.get("coordinates_source") == "user_provided")
                    n_inf  = len(stops_list) - n_user
                    n_geo  = sum(1 for s in stops_list
                                 if s.get("geocode_result") and
                                 s["geocode_result"].get("source") == "nominatim")

                    st.markdown("#### ✅ Parsed Stops")
                    parts = []
                    if n_user > 0:
                        geo_note = f" ({n_geo} geocoded via OSM)" if n_geo > 0 else ""
                        parts.append(f'<b style="color:#3FB950">📍 {n_user} user coordinate(s){geo_note}</b>')
                    if n_inf > 0:
                        parts.append(f'<b style="color:#D4A843">🤖 {n_inf} AI-inferred location(s)</b>')
                    if parts:
                        st.markdown(
                            f'<div style="font-size:0.8rem;color:#8B949E;margin-bottom:8px">'
                            + " &nbsp;·&nbsp; ".join(parts) + "</div>",
                            unsafe_allow_html=True,
                        )

                    rows = []
                    for s in stops_list:
                        geo     = s.get("geocode_result")
                        src_lbl = "📍 User coord" if s.get("coordinates_source") == "user_provided" else "🤖 AI inferred"
                        if geo and geo.get("source") == "nominatim":
                            src_lbl = "🗺️ OSM geocoded"
                        rows.append({
                            "#":            s["stop_id"],
                            "Location":     s["location_name"],
                            "Coordinates":  f"{round(s.get('lat',0), 5)}, {round(s.get('lon',0), 5)}",
                            "Coord Source": src_lbl,
                            "Type":         s["stop_type"],
                            "Priority":     s["priority"],
                            "Window":       f"{s['time_window_start']} – {s['time_window_end']}",
                            "Notes":        s.get("notes", ""),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

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
                    if nl_driver.strip():          auto_c["driver_name"]       = nl_driver.strip()
                    if nl_mode != "(auto-detect)": auto_c["transport_mode"]    = nl_mode
                    auto_c["max_hours"]           = nl_hours
                    auto_c["vehicle_capacity_kg"] = nl_cap

                    cc1, cc2, cc3 = st.columns(3)
                    cc1.markdown(f"**Driver:** {auto_c.get('driver_name','—')}")
                    cc2.markdown(f"**Mode:** {auto_c.get('transport_mode','Road')}")
                    cc3.markdown(f"**Start:** {auto_c.get('start_time','08:00')}")

                    if st.button("🚀 Generate Full Itinerary", type="primary", key="nl_generate"):
                        with st.spinner("🛰️ Fetching OSRM road distances…"):
                            osrm_nl = MLEngine.osrm_route([(s["lat"], s["lon"]) for s in stops_list])
                        st.session_state["road_geometry"] = osrm_nl.get("route_geometry")
                        st.caption("Routing: 🟢 OSRM" if osrm_nl["source"] == "osrm" else "Routing: 🟡 Haversine fallback")
                        ctx_nl = (
                            f"{len(stops_list)} NL-parsed stops | "
                            f"Mode: {auto_c.get('transport_mode','Road')} | "
                            f"Road distance: {osrm_nl['total_distance_km']} km | "
                            f"Drive time: {osrm_nl['total_duration_min']} min | "
                            f"Source: {osrm_nl['source']}"
                        )
                        with st.spinner("🧠 Generating optimized itinerary…"):
                            itinerary = ai.generate_itinerary(stops_list, auto_c, ctx_nl)
                        nl_map_stops = []
                        for itin_s in itinerary.get("stops", []):
                            matched = next((s for s in stops_list if s["stop_id"] == itin_s["stop_id"]), None)
                            if matched:
                                nl_map_stops.append({
                                    "stop_id":       itin_s["stop_id"],
                                    "location_name": itin_s["location_name"],
                                    "lat":           matched["lat"],
                                    "lon":           matched["lon"],
                                    "stop_type":     itin_s.get("stop_type", "Delivery"),
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

    # ── Render itinerary below tabs ───────────────────────────────────────────
    itin   = st.session_state.get("generated_itinerary")
    source = st.session_state.get("itinerary_source")
    if itin and source:
        st.markdown("---")
        nl_map = st.session_state.get("nl_stops_for_map", []) if source == "nl" else []
        _render_itinerary(
            itin, stops_df, ai, dash,
            st.session_state.get("last_constraints", {}),
            nl_stops_override=nl_map,
            route_geometry=st.session_state.get("road_geometry"),
        )
