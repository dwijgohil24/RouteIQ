# 🗺️ RouteIQ — AI-Powered Logistics Itinerary Planner

> A generative AI agent that accepts delivery/travel constraints and outputs optimized, step-by-step itineraries with timing, routing, risk flags, and re-planning capabilities — all inside a polished Streamlit dashboard.

---

## ✨ Features

| Module | What it does |
|---|---|
| **Overview Dashboard** | Daily stop volume, type distribution donut, route distance rankings, priority breakdown, high-priority open stops feed |
| **Itinerary Planner** | Dataset-driven or custom stop input → AI-generated optimized itinerary with sequence, times, distances, risk flags, and interactive map; supports live re-planning from natural language change requests |
| **Clustering Engine** | TF-IDF + KMeans unsupervised clustering of stops, PCA 2D scatter, keyword profiles, geographic scatter map |
| **AI Assistant** | Conversational chatbot grounded in a 20-article logistics knowledge base (routing, customs, compliance, fleet) |
| **Performance Analytics** | On-time rate gauge, fuel cost by mode, delay reason analysis, transport mode trends, AI-generated insights |

---

## 🏗️ Architecture

```
app.py
├── DataLoader       — loads & caches CSVs; computes summary stats
├── MLEngine         — TF-IDF vectorization, KMeans, PCA, Haversine distance, Nearest-Neighbor TSP
├── AIEngine         — LangChain + OpenAI: itinerary generation, re-planning, KB chat, performance insights
└── Dashboard        — all Plotly figure factories (Scattergeo, bar, pie, gauge, scatter, line)

generate_data.py     — generates stops.csv (60 rows), routes.csv, logistics_kb.csv
stops.csv            — 60 synthetic logistics stops across Mumbai/Pune metro
routes.csv           — route-level summaries (distance, fuel cost, on-time %)
logistics_kb.csv     — 20 logistics knowledge articles
```

### AI Itinerary Output Schema

```json
{
  "itinerary_title": "Optimized Route RT-100",
  "driver": "Ramesh Kumar",
  "vehicle": "Truck",
  "date": "2024-03-15",
  "transport_mode": "Road",
  "total_distance_km": 84.3,
  "total_duration_min": 312,
  "estimated_fuel_cost_inr": 674.4,
  "stops": [
    {
      "sequence": 1,
      "stop_id": "STP-2024001",
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
      "risk_reason": ""
    }
  ],
  "optimization_notes": "Prioritized High-urgency stops in AM slots; clustered Andheri stops to minimize backtracking.",
  "warnings": [],
  "efficiency_score": 87,
  "on_time_probability": 91
}
```

---

## 🚀 Setup & Run

### 1. Clone / copy project files

```
logistics_ai/
├── app.py
├── generate_data.py
├── requirements.txt
├── .env.example
├── stops.csv           ← auto-generated
├── routes.csv          ← auto-generated
└── logistics_kb.csv    ← auto-generated
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

> **No API key?** The app runs fully in offline/fallback mode using the rule-based Nearest-Neighbor optimizer and keyword-matching KB answers. All charts, clustering, and data tables remain functional.

### 5. Generate data (if CSVs missing)

```bash
python generate_data.py
```

### 6. Launch

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 📊 Pages Walkthrough

### 📊 Overview
Landing dashboard. Volume timeline by stop type, type distribution donut, top routes by distance, priority breakdown, and a live feed of high-priority open stops.

### 🔍 Itinerary Planner
**Dataset mode**: Select any route from your stop data, configure driver/vehicle/time constraints, and optionally apply Nearest-Neighbor optimization before sending to the AI for a full structured itinerary — complete with an interactive Scattergeo map, color-coded stop sequence, arrival/departure times, risk flags, and fuel cost.

**Custom mode**: Paste comma-delimited stop coordinates directly for ad-hoc planning without a dataset.

**Re-planning**: Describe any change in plain English ("Add 30-min break after stop 3", "Traffic on Western Express — delay all by 20 min") and the AI updates the full itinerary in seconds.

**Export**: Download the itinerary as JSON.

### 🧩 Clustering
Choose cluster count (2–8), run KMeans + PCA on stop text features. Includes 2D scatter, keyword profiles, stacked composition bar, and a geographic mapbox scatter.

### 💬 AI Assistant
Persistent chat interface grounded in the logistics knowledge base. Ask: "What documents do I need for customs clearance?", "How do I handle a breakdown mid-route?", "What is the e-way bill validity?".

### 📈 Performance
On-time gauge, transport mode trend line, fuel cost comparison, delay reason breakdown, AI-generated performance insights, and full sortable data tables.

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required for AI features) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint (change for Azure/proxy) |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `langchain` + `langchain-openai` | LLM orchestration |
| `scikit-learn` | TF-IDF, KMeans, PCA |
| `pandas` | Data manipulation |
| `plotly` | Interactive charts (Scattergeo, mapbox, gauge) |
| `python-dotenv` | `.env` config loading |
| `numpy` | Numerical operations |

---

## 💡 Extension Ideas

- **OSRM / Google Maps API** — replace Haversine estimates with real road distances and durations
- **VRPTW solver** (OR-Tools) — replace Nearest-Neighbor with proper Vehicle Routing Problem solver for fleets of 5+ vehicles
- **Live traffic integration** — HereMaps or TomTom API for dynamic ETA updates
- **WhatsApp / SMS notifications** — send delivery ETAs to customers automatically
- **IoT cold-chain monitoring** — integrate temperature sensor data into stop status
- **SLA breach alerting** — flag stops where time-window compliance is at risk
