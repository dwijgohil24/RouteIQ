import streamlit as st


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
            type_cls = {"Delivery": "card-gold", "Meeting": "card-teal", "Pickup": "card-green",
                        "Warehouse": "card-blue", "Customs": "card-red"}.get(row["stop_type"], "card-gold")
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
