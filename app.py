import os

# Tiktoken cache must be set before any langchain/tiktoken import
_tiktoken_cache_dir = os.path.abspath("./token")
os.makedirs(_tiktoken_cache_dir, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = _tiktoken_cache_dir

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be the first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="RouteIQ — Logistics Itinerary Planner",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import inject_css, init_session
from data import DataLoader
from dashboard import Dashboard
from engines import MLEngine, AIEngine
from views import (
    page_overview, page_planner, page_clustering,
    page_assistant, page_performance,
)

inject_css()


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

        page = st.radio(
            "Navigation",
            ["📊 Overview", "🔍 Itinerary Planner", "🧩 Clustering",
             "💬 AI Assistant", "📈 Performance"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;color:#8B949E;letter-spacing:0.1em;'
            'text-transform:uppercase;margin-bottom:8px">Quick Stats</div>',
            unsafe_allow_html=True,
        )
        for val, label, color in [
            (stats.get("total_stops",  0),             "Total Stops",         "#D4A843"),
            (stats.get("total_routes", 0),             "Active Routes",       "#2EA4A4"),
            (f"{stats.get('on_time_pct',0)}%",         "On-Time Rate",        "#3FB950"),
            (stats.get("high_priority", 0),            "High Priority Stops", "#E74C3C"),
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
