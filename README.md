# RouteIQ — AI-Powered Logistics Itinerary Planner

> A generative AI agent that accepts delivery/travel constraints and outputs optimized, step-by-step itineraries with real road distances, animated route maps, weather forecasts, fuel cost analysis, risk flags, and live re-planning — all inside a polished Streamlit dashboard.

---

## Features

| Module | What it does |
|---|---|
| **Overview Dashboard** | Daily stop volume, type distribution donut, route distance rankings, priority breakdown, high-priority open stops feed |
| **Itinerary Planner** | Three input modes (dataset, custom coordinates, natural language) → AI-generated optimized itinerary with sequence, times, real road distances, risk flags, weather forecast per stop, and fuel cost analysis |
| **Animated Route Map** | Interactive Folium/Leaflet map with a vehicle icon that animates along road-following polylines (OSRM geometry); legend and replay button anchored to the visible map area |
| **Clustering Engine** | TF-IDF + KMeans unsupervised clustering of stops, PCA 2D scatter, keyword profiles, geographic scatter map |
| **AI Assistant** | Conversational chatbot grounded in a logistics knowledge base via RAG (semantic retrieval + ChromaDB); out-of-scope questions are politely declined |
| **Performance Analytics** | On-time rate gauge, fuel cost by mode, delay reason analysis, transport mode trends, AI-generated insights |

---

## Architecture

```
RouteIQ/
├── app.py               — entry point: Streamlit config, caching, page routing
├── config.py            — CSS injection, session state initialisation
├── data.py              — DataLoader: loads stops.csv, routes.csv, logistics_kb.csv (auto-generates if missing)
├── dashboard.py         — Dashboard: all Plotly figure factories
│
├── engines/
│   ├── ai.py            — AIEngine: itinerary generation, re-planning, NL parsing, violation checks
│   ├── ml.py            — MLEngine: TF-IDF, KMeans, PCA, Haversine, Nearest-Neighbor TSP, OSRM routing
│   ├── rag.py           — RAGEngine: ChromaDB vector store, semantic retrieval, KB Q&A
│   ├── fuel.py          — FuelEngine: city fuel prices, vehicle efficiency, savings analysis
│   ├── weather.py       — WeatherEngine: Open-Meteo hourly forecasts, seasonal fallback
│   ├── geo.py           — GeoEngine: Nominatim reverse geocoding
│   ├── map_renderer.py  — Animated Folium/Leaflet route map with road geometry & overlay anchoring
│   └── pdf_exporter.py  — Branded PDF export: per-page header/footer, stops table, about page
│
├── views/
│   ├── overview.py      — page_overview()
│   ├── planner.py       — page_planner(), weather panel, fuel panel, itinerary renderer
│   ├── clustering.py    — page_clustering()
│   ├── assistant.py     — page_assistant()
│   └── performance.py   — page_performance()
│
├── stops.csv            ← auto-generated on first run
├── routes.csv           ← auto-generated on first run
├── logistics_kb.csv     ← auto-generated on first run
├── requirements.txt
└── .env
```

### AI Itinerary Output Schema

```json
{
  "itinerary_title": "Optimized Route RT-100",
  "driver": "Ramesh Kumar",
  "vehicle": "Truck",
  "date": "2025-06-15",
  "transport_mode": "Road",
  "total_distance_km": 84.3,
  "total_duration_min": 312,
  "estimated_fuel_cost_inr": 1382.46,
  "routing_source": "osrm",
  "efficiency_score": 87,
  "on_time_probability": 91,
  "optimization_notes": "Prioritized High-urgency stops in AM slots; clustered Andheri stops to minimize backtracking.",
  "warnings": [],
  "stops": [
    {
      "sequence": 1,
      "stop_id": "STP-001",
      "location_name": "Dadar Warehouse",
      "arrival_time": "08:00",
      "departure_time": "08:20",
      "service_duration_min": 20,
      "travel_time_from_prev_min": 0,
      "distance_from_prev_km": 0,
      "stop_type": "Delivery",
      "priority": "High",
      "status": "Scheduled",
      "notes": "Call 30 min before arrival",
      "risk_flag": "None",
      "risk_reason": "",
      "time_window_start": "08:00",
      "time_window_end": "10:00"
    }
  ]
}
```

---

## Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RouteIQ.git
cd RouteIQ
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the HuggingFace embedding model (~90 MB) automatically — internet required once.

### 4. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

> **No API key?** The app runs fully in offline/fallback mode — rule-based Nearest-Neighbor optimizer, keyword-matching KB answers, and all charts/clustering remain functional. Only AI itinerary generation and the chat assistant require a key.

### 5. Launch

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

> Data files (`stops.csv`, `routes.csv`, `logistics_kb.csv`) are auto-generated on first run if missing — no separate generation step needed.

---

## Pages Walkthrough

### Overview
Landing dashboard. Volume timeline by stop type, type distribution donut, top routes by distance, priority breakdown, and a live feed of high-priority open stops.

### Itinerary Planner

Three input modes:

**Dataset mode** — Select any route from the loaded stop data. The dropdown shows `RT-100  (12 stops: Dadar Warehouse → Thane Depot)` style labels for quick context. Configure driver/vehicle/time constraints and optionally apply Nearest-Neighbor optimization. Road distances and road-following polylines are fetched from the public OSRM API (Haversine straight-line fallback if unreachable).

**Custom Coordinates mode** — Paste comma-delimited stops (`Name, Lat, Lon, Type, Priority, TimeFrom, TimeTo`) for ad-hoc planning without a dataset. Invalid lines are reported individually with the exact parse error.

**Natural Language mode** — Describe your route in plain English. The AI extracts stops, priorities, time windows, and constraints. Inline GPS coordinates (`Dadar (19.018, 72.848) at 9am`) are detected and reverse-geocoded to real location names via OpenStreetMap/Nominatim automatically.

**After generation (all modes):**
- Animated route map: vehicle icon travels along road-following polylines (not straight lines), with a color-coded legend and replay button anchored to the map's bottom edge
- Constraint violation checker (time windows, driver hours, vehicle capacity)
- Per-stop arrival-time weather forecast via Open-Meteo
- Fuel cost panel with city price lookup and optimized vs. un-optimized route savings
- Re-planning: describe any change in plain English and the AI updates the full itinerary
- Export as JSON, CSV, or branded PDF

### PDF Export

The PDF export produces a multi-page branded document:
- **Header** (every page): RouteIQ logo + title on a dark background bar
- **Footer** (every page): proprietary notice + gold page number
- **Page 1**: itinerary summary — driver/vehicle/date, key metrics (distance, duration, efficiency, on-time probability)
- **Stop table**: full 11-column paginated table with color-coded stop types, priorities, and risk levels
- **Warnings / Notes**: optimization notes and constraint violations (omitted if empty)
- **About page**: RouteIQ brand description and disclaimer

### Clustering
Choose cluster count (2–8), run KMeans + PCA on stop text features. Includes 2D scatter, keyword profiles, stacked composition bar, and a geographic scatter map.

### AI Assistant
Persistent chat interface grounded in the logistics knowledge base via RAG (ChromaDB semantic retrieval). Answers only travel and logistics questions — out-of-scope questions are politely declined.

Example questions: *"What documents do I need for customs clearance?"*, *"How do I handle a breakdown mid-route?"*, *"What is the e-way bill validity?"*

### Performance
On-time gauge, transport mode trend line, fuel cost comparison, delay reason breakdown, AI-generated performance insights, and full sortable data tables.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required for AI features) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local HuggingFace embedding model for RAG |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `langchain` + `langchain-groq` | LLM orchestration (Groq / Llama 3.3) |
| `langchain-community` + `langchain-huggingface` | HuggingFace embeddings, ChromaDB vector store |
| `sentence-transformers` | Local embedding model (no API key needed) |
| `chromadb` | Vector database for RAG retrieval |
| `scikit-learn` | TF-IDF, KMeans, PCA |
| `pandas` | Data manipulation |
| `plotly` | Interactive charts (Scattergeo, gauge, bar, line) |
| `folium` | Interactive Leaflet.js maps with vehicle animation |
| `reportlab` | Branded PDF generation |
| `httpx` | HTTP client for OSRM, Open-Meteo, Nominatim APIs |
| `python-dotenv` | `.env` config loading |
| `numpy` | Numerical operations |

---

## External APIs Used (all free, no key required)

| API | Purpose | Fallback |
|---|---|---|
| [OSRM](http://router.project-osrm.org) | Real road distances, drive times, and road-following geometry | Haversine straight-line estimate |
| [Open-Meteo](https://open-meteo.com) | Hourly weather forecast per stop at arrival time | Seasonal estimate with diurnal variation |
| [Nominatim / OSM](https://nominatim.openstreetmap.org) | Reverse geocoding of GPS coordinates to location names | `"Location @ lat, lon"` label |

---

## Extension Ideas

- **VRPTW solver** (OR-Tools) — replace Nearest-Neighbor with a proper Vehicle Routing Problem solver for fleets of 5+ vehicles
- **Live traffic integration** — HereMaps or TomTom API for dynamic ETA updates
- **WhatsApp / SMS notifications** — send delivery ETAs to customers automatically
- **IoT cold-chain monitoring** — integrate temperature sensor data into stop status
- **Multi-day itineraries** — extend the planner to span multiple dates with driver rest constraints
- **User authentication** — Streamlit Auth or Cognito to support per-driver dashboards
- **Fleet dispatch view** — assign multiple routes to multiple drivers in a single planning session

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
