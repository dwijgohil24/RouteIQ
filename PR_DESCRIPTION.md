# 🚗 feat: Animated Vehicle Route Map + MIT License

## What this PR does

Adds a **live animated vehicle** that traces the optimized route on an interactive street-level map. When an itinerary is generated, a vehicle icon (matching the user's selected vehicle type) smoothly drives along the route over ~4 seconds, leaving a green trail behind it.

Also adds an MIT License to the project.

https://github.com/user-attachments/assets/placeholder — *(replace with a screen recording GIF)*

---

## Changes

### New files
| File | Purpose |
|---|---|
| `engines/map_renderer.py` | Folium/Leaflet animated map renderer with vehicle SVGs, path interpolation, bearing rotation, legend, and replay button |
| `LICENSE` | MIT License |

### Modified files
| File | What changed |
|---|---|
| `views/planner.py` | Integrated animated map in itinerary renderer + fuel panel optimized route map. Added toggle checkbox to switch between animated (Folium) and static (Plotly) views |
| `engines/__init__.py` | Exported `render_animated_map_in_streamlit` |
| `requirements.txt` | Added `folium>=0.16.0,<1.0.0` |

---

## Features

- **5 vehicle icons**: Car, Truck, Van, Motorcycle, Tempo — auto-selected based on the user's vehicle type choice
- **Smooth 80-frame animation** with path interpolation between stops (~4 seconds)
- **Vehicle rotates** to face the direction of travel
- **Green trail** builds up behind the vehicle as it drives
- **Dark CARTO tiles** matching RouteIQ's existing dark theme
- **Numbered stop markers** with click-to-open popups showing stop name, type, arrival/departure
- **Color-coded legend** overlay (Delivery, Pickup, Meeting, Warehouse, Customs, Rest)
- **Replay button** to re-watch the animation
- **Opt-in toggle** — checkbox defaults to animated; uncheck to see the original Plotly Scattergeo map
- **Auto-zoom** based on coordinate spread of the stops

---

## How to test

```bash
pip install -r requirements.txt    # installs folium
streamlit run app.py
```

1. Go to **Itinerary Planner** → select a route → generate an itinerary
2. The animated map should appear with a vehicle driving the route
3. Try changing Vehicle Type to Truck/Motorcycle — the icon changes
4. Uncheck "🚗 Animate vehicle on route" — should fall back to the original Plotly map
5. Check the fuel panel's "Optimized Route Map" — same animated option there

---

## Why Folium over Plotly?

The existing `Scattergeo` map is great for static plots, but Plotly has no smooth marker animation support. Folium wraps Leaflet.js which gives us:
- Street-level tile maps (more useful for logistics than political geography)
- Native JavaScript animation with `setTimeout` loops
- DivIcon for custom SVG vehicle markers
- Lightweight — `folium` is a thin wrapper, no heavy JS bundles

The original Plotly map is **preserved as a fallback** (toggle off the checkbox), so this is purely additive.

---

## Checklist

- [x] New feature is behind an opt-in toggle (non-breaking)
- [x] Original Plotly map preserved as fallback
- [x] All existing functionality untouched
- [x] Only one new dependency (`folium`)
- [x] Dark theme consistent with existing UI
- [x] Tested with all 5 vehicle types
- [x] Edge cases: single stop, two stops, large routes
- [x] MIT License added
