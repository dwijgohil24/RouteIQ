import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
        sc.columns = ["status", "count"]
        fig = go.Figure(go.Bar(
            x=sc["status"], y=sc["count"],
            marker_color=[dash.STATUS_COLORS.get(s, "#8B949E") for s in sc["status"]],
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
            dr.columns = ["reason", "count"]
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
