# Contributing to RouteIQ

Thanks for your interest in contributing to RouteIQ! This guide will help you get started.

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
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. **Create a `.env` file** (see README for details):
   ```env
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```
6. **Run the app** to verify everything works:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
RouteIQ/
├── app.py               — Entry point, page routing
├── config.py            — CSS injection, session state init
├── data.py              — DataLoader for CSV files
├── dashboard.py         — Plotly chart factories
├── engines/             — Core logic
│   ├── ai.py            — LLM itinerary generation (Groq/Llama)
│   ├── ml.py            — Clustering, TSP, OSRM routing
│   ├── rag.py           — ChromaDB vector store for KB Q&A
│   ├── fuel.py          — Fuel price lookup and cost calculation
│   ├── weather.py       — Open-Meteo weather forecasts
│   ├── geo.py           — Nominatim reverse geocoding
│   └── map_renderer.py  — Animated Folium/Leaflet route maps
├── views/               — Streamlit page renderers
│   ├── overview.py
│   ├── planner.py
│   ├── clustering.py
│   ├── assistant.py
│   └── performance.py
└── data_60/             — Sample datasets
```

## How to Contribute

### Bug fixes
- Open an issue first describing the bug
- Reference the issue number in your PR

### New features
- Check the **Extension Ideas** in the README for inspiration
- Open an issue to discuss the approach before building
- Keep new features behind toggles or optional flags when possible
- Don't remove existing functionality — add alternatives

### Code style
- Follow the existing patterns in the codebase
- Use type hints for function parameters and return values
- Keep `except Exception` blocks specific — catch the actual error types
- Use descriptive variable names

### Commits
- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`
- Keep commits focused — one logical change per commit

## Pull Request Process

1. Make sure the app runs without errors
2. Test your changes across all affected pages
3. Update the README if you added new features or dependencies
4. Open a PR with a clear description of what changed and why
5. Include screenshots or GIFs for UI changes

## Need Help?

Open an issue with the `question` label and we'll help you out.
