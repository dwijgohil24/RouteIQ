import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
        df2   = stops_df.copy()
        df2["date"] = pd.to_datetime(df2["scheduled_date"])
        daily = df2.groupby(["date", "stop_type"]).size().reset_index(name="count")
        fig   = px.bar(daily, x="date", y="count", color="stop_type",
                       color_discrete_map=self.STOP_COLORS, barmode="stack")
        fig.update_layout(**self._base_layout(280),
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True, gridcolor="#21262D"))
        return fig

    def stop_type_donut(self, stops_df):
        counts = stops_df["stop_type"].value_counts().reset_index()
        counts.columns = ["stop_type", "count"]
        fig = px.pie(counts, names="stop_type", values="count",
                     color="stop_type", color_discrete_map=self.STOP_COLORS, hole=0.55)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker=dict(line=dict(color="#0D1117", width=2)))
        fig.update_layout(**self._base_layout(300), showlegend=False)
        return fig

    def distance_by_route(self, routes_df):
        df  = routes_df.sort_values("total_distance_km", ascending=False).head(10)
        fig = go.Figure(go.Bar(
            x=df["route_id"], y=df["total_distance_km"],
            marker_color="#D4A843",
            hovertemplate="%{x}: %{y:.1f} km<extra></extra>",
        ))
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
        pc.columns = ["priority", "count"]
        fig = go.Figure(go.Bar(
            x=pc["priority"], y=pc["count"],
            marker_color=[self.PRIORITY_COLORS.get(p, "#8B949E") for p in pc["priority"]],
            hovertemplate="%{x}: %{y}<extra></extra>",
        ))
        fig.update_layout(**self._base_layout(240),
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True, gridcolor="#21262D"))
        return fig

    def route_map_scatter(self, stops_list, itinerary=None):
        lats   = [s["lat"] for s in stops_list]
        lons   = [s["lon"] for s in stops_list]
        names  = [s["location_name"] for s in stops_list]
        colors = [self.STOP_COLORS.get(s.get("stop_type", "Delivery"), "#D4A843") for s in stops_list]

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
                         hover_data=["location_name", "stop_type", "priority"],
                         color_discrete_sequence=["#D4A843", "#2EA4A4", "#3FB950", "#8957E5", "#E74C3C", "#1F6FEB"])
        fig.update_layout(**self._base_layout(380),
                          xaxis=dict(showgrid=False, title="Component 1"),
                          yaxis=dict(showgrid=False, title="Component 2"))
        return fig

    def performance_timeline(self, routes_df):
        df = routes_df.copy(); df["date"] = pd.to_datetime(df["route_date"])
        fig = px.line(df.sort_values("date"), x="date", y="on_time_pct", color="transport_mode",
                      color_discrete_map={"Road": "#D4A843", "Air": "#2EA4A4", "Rail": "#3FB950", "Sea": "#8957E5"})
        fig.update_layout(**self._base_layout(300), xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True, gridcolor="#21262D", title="On-Time %", range=[0, 105]))
        return fig

    def fuel_cost_bar(self, routes_df):
        df = routes_df.groupby("transport_mode")["estimated_fuel_cost_inr"].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=df["transport_mode"], y=df["estimated_fuel_cost_inr"],
            marker_color=["#D4A843", "#2EA4A4", "#3FB950", "#8957E5"],
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(**self._base_layout(260), xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True, gridcolor="#21262D", title="Avg Cost (₹)"))
        return fig
