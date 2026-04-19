import streamlit as st

APP_CSS = """
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
  font-size: 1.4rem !important;
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

.section-title { font-family: 'Playfair Display', serif; font-size: 1.1rem; color: var(--white); margin-bottom: 4px; }
.section-sub   { color: var(--ink); font-size: 0.8rem; margin-bottom: 20px; letter-spacing: 0.04em; }
.hero-title    { font-family: 'Playfair Display', serif; font-size: 2.0rem; font-weight: 900; color: var(--white); line-height: 1.1; }
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
"""


def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


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
        "geocode_cache":       {},
        "weather_cache":       {},
        "fuel_analysis":       None,
        "itinerary_source":    None,
        "nl_stops_for_map":    [],
        "road_geometry":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
