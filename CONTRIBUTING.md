# Contributing to RouteIQ

Thanks for your interest in contributing! This guide covers how to set up, what to work on, and how to get your changes merged.

---

## Getting Started

1. **Fork** the repo on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RouteIQ.git
   cd RouteIQ
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up the environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
5. **Create a `.env` file**:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```
6. **Run the app** to verify everything works:
   ```bash
   streamlit run app.py
   ```

---

## Project Structure

```
RouteIQ/
├── app.py               — Entry point, page routing
├── config.py            — CSS injection, session state init
├── data.py              — DataLoader for CSV files
├── dashboard.py         — Plotly chart factories
├── generate_data.py     — Regenerates stops.csv / routes.csv from fixed location set
│
├── engines/
│   ├── ai.py            — LLM itinerary generation (Groq/Llama)
│   ├── ml.py            — Clustering, TSP, OSRM routing (road geometry)
│   ├── rag.py           — ChromaDB vector store for KB Q&A
│   ├── fuel.py          — Fuel price lookup and cost calculation
│   ├── weather.py       — Open-Meteo weather forecasts
│   ├── geo.py           — Nominatim reverse geocoding
│   ├── map_renderer.py  — Animated Folium/Leaflet map (road polylines, overlay anchoring)
│   └── pdf_exporter.py  — Branded PDF export (reportlab)
│
├── views/
│   ├── overview.py      — Overview dashboard page
│   ├── planner.py       — Itinerary Planner page (all three input modes)
│   ├── clustering.py    — Clustering page
│   ├── assistant.py     — AI Assistant page
│   └── performance.py   — Performance Analytics page
│
├── stops.csv            — Auto-generated from generate_data.py
├── routes.csv           — Auto-generated from generate_data.py
├── logistics_kb.csv     — Auto-generated on first run
└── requirements.txt
```

---

## How to Contribute

### Bug fixes
- Open an issue first describing the bug and how to reproduce it
- Reference the issue number in your PR (`Fixes #42`)

### New features
- Check the **Extension Ideas** section in the README for inspiration
- Open an issue to discuss your approach before building; avoid surprise large PRs
- Keep new features opt-in where possible — don't break existing flows

### Map / routing changes
- OSRM geometry is fetched in `engines/ml.py` (`osrm_route`) and threaded through `views/planner.py` → `engines/map_renderer.py`
- The animated map uses road-following polylines when OSRM is reachable and falls back to Haversine straight lines when it isn't; preserve this fallback in any changes
- Overlay elements (legend, replay button) are anchored inside `map.getContainer()` — keep them there so they stay visible at all iframe sizes

### PDF export changes
- The PDF template lives in `engines/pdf_exporter.py`
- Header, footer, and about-page copy are part of the RouteIQ brand — keep them intact
- The `generate_itinerary_pdf(itin, constraints)` public API returns `bytes`; don't change its signature without updating the caller in `views/planner.py`

### Data regeneration
- If you change location names or coordinates, run `python generate_data.py` to rebuild `stops.csv` and `routes.csv` from the canonical `LOCATIONS` set
- This ensures every location name maps to exactly one set of coordinates (avoids same-name/different-coordinate bugs)

### Code style
- Follow the existing patterns in the codebase
- Use type hints for function parameters and return values
- Keep `except Exception` blocks narrow — catch the actual error type where known
- Use descriptive variable names; avoid single-letter names outside of tight loops

### Commits
- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `style:`
- Keep commits focused — one logical change per commit
- Reference issue numbers in commit messages where applicable

---

## Pull Request Process

1. Make sure `streamlit run app.py` starts without errors
2. Test your changes across all affected pages
3. If you added or changed a dependency, update `requirements.txt`
4. Update the README if you added new features, dependencies, or changed external API usage
5. Open a PR using the [PR template](.github/pull_request_template.md) — fill in all sections
6. Include screenshots or GIFs for any UI changes

---

## Need Help?

Open an issue with the `question` label and we'll get back to you.
