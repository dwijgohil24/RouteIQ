# Decouple.md — Design Patterns & Decoupling Decisions

> Documents every pattern applied, the problem it solved, the tradeoff accepted, and a minimal code snippet showing the before/after.

---

## 1. Strategy Pattern — Routing Algorithm

**File:** `engines/routing.py`  
**Consumers:** `engines/ml.py` → `MLEngine.osrm_route()`

### Problem
`MLEngine.osrm_route()` embedded the full OSRM HTTP call **and** the Haversine fallback in one static method. Adding a new provider (Google Maps, HERE) meant editing `ml.py` — a class that also owns clustering and nearest-neighbor logic. Testing required mocking `httpx` inside `MLEngine`.

### Pattern applied
Abstract `RoutingStrategy` with two concrete strategies: `OSRMStrategy` (delegates to `HaversineStrategy` on failure) and `HaversineStrategy` standalone. A module-level default is swappable at runtime without touching any call site.

```python
# Before — algorithm baked into MLEngine
@staticmethod
def osrm_route(coords):
    waypoints = ";".join(...)
    try:
        resp = httpx.get(url, timeout=8.0)
        ...
    except Exception:
        # haversine fallback duplicated here

# After — MLEngine delegates, knows nothing about HTTP
@staticmethod
def osrm_route(coords: list) -> dict:
    return get_default_strategy().route(coords)

# Swap globally for tests or a different provider:
set_default_strategy(MockRoutingStrategy())
```

### Tradeoff
- **Gain:** Open/Closed — new providers don't touch `MLEngine` or any caller
- **Gain:** Testable — inject `HaversineStrategy()` directly in tests, no HTTP mocking
- **Cost:** One extra import per call site; strategy must be set before first use

---

## 2. Chain of Responsibility — Violation Checking

**File:** `engines/violations.py`  
**Consumer:** `engines/ai.py` → `AIEngine.check_violations()`

### Problem
`AIEngine.check_violations()` was a 50-line method with three mixed concerns — time windows, driver hours, and capacity — all in one loop. Adding a new rule (e.g. rest-break compliance) required editing the method and risking regressions in the other checks.

### Pattern applied
Three independent handlers (`TimeWindowChecker`, `DriverHoursChecker`, `CapacityChecker`) form a linked chain. Each handles one concern and forwards to the next. `build_chain()` assembles them.

```python
# Before — all concerns mixed in one method
def check_violations(self, itinerary, constraints):
    for s in stops:
        # time window logic ...
        # driver hours logic ...
    # capacity logic ...

# After — AIEngine delegates to the chain
def check_violations(self, itinerary, constraints):
    violations = []
    build_chain().check(stops, constraints, base_dt, violations)
    return violations

# Adding a new rule: create RestBreakChecker, append to build_chain()
# Zero changes to existing checkers or AIEngine.
```

### Tradeoff
- **Gain:** Single Responsibility per checker; Open/Closed for new rules
- **Gain:** Each checker is independently unit-testable
- **Cost:** Slight indirection — violations list is mutated by reference through the chain rather than returned. Acceptable for a synchronous, single-threaded context.

---

## 3. Factory Pattern — Engine Construction

**File:** `engines/factory.py`  
**Consumer:** `app.py`

### Problem
`app.py` called `AIEngine()`, `MLEngine()` directly. Every constructor argument, dependency wiring, and future constructor change required edits in the entry point — a file that should know nothing about engine internals.

### Pattern applied
`EngineFactory` centralises all construction. `app.py` calls `EngineFactory.create_ai_engine()` — it has no import of `AIEngine` at all.

```python
# Before — app.py knows constructor details
from engines import MLEngine, AIEngine

@st.cache_resource()
def get_ai_engine():
    return AIEngine()

# After — app.py depends only on the factory
from engines.factory import EngineFactory

@st.cache_resource()
def get_ai_engine():
    return EngineFactory.create_ai_engine()

# Switching to a different LLM engine: change factory only, app.py untouched.
```

### Tradeoff
- **Gain:** Construction logic has one home; swapping implementations is a one-line change
- **Cost:** One more file to navigate for new contributors; minor overhead for a project of this size

---

## 4. Repository Pattern — Data Access

**Files:** `core/repository.py`, `data.py`  
**Consumer:** `app.py` → `DataLoader`

### Problem
`DataLoader` mixed file-existence checks, CSV reading, date coercion, and data-generation logic in one class. Each concern was tangled — adding a new data source required modifying a class that already knew about three different files and a code-generation module.

### Pattern applied
`StopsRepository`, `RoutesRepository`, `KBRepository` each own one data concern. The coupled generation step (routes depend on stops) is isolated in `_DataSeeder`. `DataLoader` becomes a thin facade that calls `make_repositories()`.

```python
# Before — DataLoader owns everything
class DataLoader:
    def load_stops(self):
        if not os.path.exists(self.STOPS_FILE):
            self._generate_and_save()   # generates ALL three files
        return pd.read_csv(...)

# After — each repository owns its file; seeder owns generation
class StopsRepository(DataRepository):
    def load(self) -> pd.DataFrame:
        if not self.exists() and self._seeder:
            self._seeder.seed_if_needed()
        return pd.read_csv(self._path, parse_dates=[...])

# DataLoader is now a one-liner facade:
class DataLoader:
    def __init__(self):
        self._stops_repo, self._routes_repo, self._kb_repo = make_repositories()
    def load_stops(self):
        return self._stops_repo.load()
```

### Tradeoff
- **Gain:** Each repository is independently mockable; adding a new data source is additive
- **Gain:** Generation coupling is explicit and isolated in `_DataSeeder`
- **Cost:** More files for what was a simple CSV loader. Justified when data sources grow.

---

## 5. Adapter Pattern — HTTP Client

**File:** `core/http_client.py`  
**Consumers:** `engines/weather.py`, `engines/geo.py`, `engines/routing.py`

### Problem
`WeatherEngine`, `GeoEngine`, and `OSRMStrategy` each held raw `httpx.get()` calls with duplicated timeout, status-check, follow_redirects, and fallback-URL logic. A change to retry behaviour required editing three different files.

### Pattern applied
`HttpAdapter` abstract interface with `HttpxAdapter` as the concrete implementation. The adapter handles multi-URL fallback (https → http), status-code validation, and JSON parsing in one place. Engines call `get_http_adapter().get(url)`.

```python
# Before — duplicated in each engine
resp = httpx.get(url, timeout=8.0, follow_redirects=True)
if resp.status_code != 200:
    continue
data = resp.json()

# After — engines call the adapter
data = get_http_adapter().get([https_url, http_url])
if data:
    # use data directly

# For tests: inject a mock adapter
set_http_adapter(MockHttpAdapter({"hourly": {...}}))
```

### Tradeoff
- **Gain:** HTTP retry/timeout policy has one home; testable without network
- **Gain:** Swapping `httpx` for `aiohttp` or `requests` is a single-class change
- **Cost:** Weather's session-state caching still lives in `WeatherEngine` (intentional — caching is domain logic, not transport logic). The adapter only handles the network call.

---

## 6. Observer Pattern — Event Bus

**File:** `core/events.py`  
**Potential consumers:** `views/planner.py`, `views/assistant.py`

### Problem
Views write results directly into `st.session_state` and other views read from it — implicit coupling with no documented contract. If a key is renamed or removed, every reader silently breaks. There is no single place to see "what events does the planner emit?"

### Pattern applied
Synchronous `EventBus` with named events as string constants (`ITINERARY_GENERATED`, `DATA_LOADED`, `CONSTRAINTS_UPDATED`). Publishers call `bus.publish(event, payload)`; subscribers register handlers via `bus.subscribe(event, handler)`.

```python
# Publishing (in planner.py after generation)
bus = get_event_bus()
bus.publish(ITINERARY_GENERATED, {
    "itinerary":   itinerary,
    "constraints": constraints,
    "source":      "dataset",
})

# Subscribing (in a panel that reacts to new itineraries)
bus.subscribe(ITINERARY_GENERATED, lambda p: update_fuel_panel(p["itinerary"]))
```

**Note:** In the current Streamlit architecture `st.session_state` is still used as the state store (required by Streamlit's execution model). The EventBus sits **on top** as a notification layer — it notifies handlers when state changes, so panels don't need to poll session state directly. Full migration to EventBus-driven state is a future step once the app moves to a non-Streamlit frontend.

### Tradeoff
- **Gain:** Documented event contracts; publishers and subscribers don't reference each other
- **Gain:** Easy to add cross-cutting concerns (logging, analytics) by subscribing without touching producers
- **Cost:** In Streamlit's synchronous re-render model, subscriptions registered in one run may not persist to the next unless tied to session state. The bus is most useful for in-run side effects (e.g. updating a cache or writing to session state in a handler).

---

## Summary Table

| Pattern | File | Problem Solved | Open for Extension |
|---|---|---|---|
| **Strategy** | `engines/routing.py` | Routing algorithm coupled to MLEngine | Add new provider without touching MLEngine |
| **Chain of Responsibility** | `engines/violations.py` | All violation types in one 50-line method | Add new checker class, append to chain |
| **Factory** | `engines/factory.py` | app.py knew engine constructors | Swap implementation in factory, app.py unchanged |
| **Repository** | `core/repository.py` | DataLoader mixed 3 data concerns + generation | Each source is independent and mockable |
| **Adapter** | `core/http_client.py` | httpx calls duplicated across 3 engines | Swap HTTP library or inject mock in one place |
| **Observer** | `core/events.py` | Views coupled via implicit session_state keys | Pub/sub with documented event contracts |

---

## What Was Deliberately Not Extracted

| Concept | Reason left as-is |
|---|---|
| Streamlit session state | Streamlit's execution model requires it; the EventBus layer is additive, not a replacement |
| `@st.cache_resource` wrappers | These are Streamlit-specific lifecycle hooks, not construction logic — they stay in `app.py` |
| `RAGEngine` internals | ChromaDB + LangChain splitter are already cohesive; extracting further would create abstraction for its own sake |
| CSS / `inject_css()` | UI-only concern already isolated in `config.py` |
| Plotly chart methods in `Dashboard` | Already a cohesive factory; no mixed concerns present |
