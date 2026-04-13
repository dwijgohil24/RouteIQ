import streamlit as st
import plotly.express as px


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
            sample_df    = stops_df.sample(min(sample_size, len(stops_df)), random_state=42)
            labels, X_2d = ml.cluster_stops(sample_df, n_clusters)
            st.session_state["cluster_labels"] = labels
            st.session_state["cluster_X2d"]    = X_2d
            st.session_state["cluster_df"]     = sample_df.reset_index(drop=True)
            st.success("Clustering complete!")

    with col2:
        if st.session_state.get("cluster_labels") is not None:
            st.markdown("**Cluster Visualization (PCA 2D)**")
            st.plotly_chart(
                dash.cluster_scatter(
                    st.session_state["cluster_df"],
                    st.session_state["cluster_labels"],
                    st.session_state["cluster_X2d"],
                ),
                use_container_width=True,
            )

    if st.session_state.get("cluster_labels") is not None:
        labels   = st.session_state["cluster_labels"]
        cdf      = st.session_state["cluster_df"]
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
        comp = cdf2.groupby(["cluster", "stop_type"]).size().reset_index(name="count")
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
            hover_data=["location_name", "stop_type", "priority"], zoom=9, height=380,
        )
        fig_geo.update_layout(
            mapbox_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#C9D1D9", margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_geo, use_container_width=True)
