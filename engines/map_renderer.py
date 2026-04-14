"""
Animated route map renderer using Folium (Leaflet.js).

Generates interactive street-level maps with a vehicle icon that animates
along the route polyline over ~4 seconds. Vehicle type (car, truck, van,
motorcycle, tempo) is selected based on the user's transport/vehicle choice.

Drop-in replacement for the Plotly Scattergeo map used in dashboard.py.

KEY DESIGN NOTE: Folium's _repr_html_() wraps everything in an <iframe>
with HTML-escaped srcdoc. Any JS appended *after* _repr_html_() goes
OUTSIDE the iframe and cannot access the Leaflet map. We use
branca.element.Element + m.get_root().html.add_child() to inject our
animation script INSIDE the iframe's HTML before it gets escaped.
"""

import math
import json
import folium
from branca.element import Element
import streamlit.components.v1 as components


# ── Vehicle SVG Icons (top-down silhouettes, facing right) ───────────────
VEHICLE_ICONS = {
    "Car": {
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20" width="40" height="20">'
            '<rect x="2" y="6" width="36" height="10" rx="4" fill="#D4A843" stroke="#0D1117" stroke-width="1"/>'
            '<rect x="10" y="2" width="18" height="10" rx="3" fill="#E8C76A" stroke="#0D1117" stroke-width="1"/>'
            '<circle cx="10" cy="17" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<circle cx="30" cy="17" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<rect x="33" y="8" width="4" height="2" rx="1" fill="#E74C3C"/>'
            '<rect x="1" y="9" width="3" height="1.5" rx="0.5" fill="#F0F6FC"/>'
            '</svg>'
        ),
        "size": [40, 20],
        "anchor": [20, 10],
    },
    "Truck": {
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 22" width="48" height="22">'
            '<rect x="2" y="4" width="30" height="14" rx="2" fill="#2EA4A4" stroke="#0D1117" stroke-width="1"/>'
            '<rect x="32" y="6" width="14" height="12" rx="3" fill="#3FB950" stroke="#0D1117" stroke-width="1"/>'
            '<circle cx="12" cy="19" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<circle cx="24" cy="19" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<circle cx="40" cy="19" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<rect x="44" y="9" width="2" height="2" rx="0.5" fill="#E74C3C"/>'
            '<rect x="32" y="8" width="6" height="5" rx="1" fill="rgba(255,255,255,0.3)"/>'
            '</svg>'
        ),
        "size": [48, 22],
        "anchor": [24, 11],
    },
    "Van": {
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 20" width="44" height="20">'
            '<rect x="2" y="4" width="28" height="13" rx="2" fill="#8957E5" stroke="#0D1117" stroke-width="1"/>'
            '<rect x="30" y="5" width="12" height="12" rx="4" fill="#A371F7" stroke="#0D1117" stroke-width="1"/>'
            '<circle cx="10" cy="18" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<circle cx="36" cy="18" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<rect x="40" y="8" width="2" height="2" rx="0.5" fill="#E74C3C"/>'
            '<rect x="31" y="7" width="5" height="4" rx="1" fill="rgba(255,255,255,0.3)"/>'
            '</svg>'
        ),
        "size": [44, 20],
        "anchor": [22, 10],
    },
    "Motorcycle": {
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 20" width="36" height="20">'
            '<path d="M10,10 Q18,2 28,8" stroke="#D4A843" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
            '<rect x="12" y="8" width="14" height="5" rx="2" fill="#D4A843" stroke="#0D1117" stroke-width="0.8"/>'
            '<circle cx="8" cy="15" r="4" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<circle cx="30" cy="15" r="4" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<circle cx="8" cy="15" r="1.5" fill="#555"/>'
            '<circle cx="30" cy="15" r="1.5" fill="#555"/>'
            '</svg>'
        ),
        "size": [36, 20],
        "anchor": [18, 10],
    },
    "Tempo": {
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 22" width="44" height="22">'
            '<rect x="2" y="5" width="26" height="13" rx="2" fill="#E8873A" stroke="#0D1117" stroke-width="1"/>'
            '<rect x="28" y="6" width="14" height="12" rx="3" fill="#F0A860" stroke="#0D1117" stroke-width="1"/>'
            '<circle cx="10" cy="19" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<circle cx="22" cy="19" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<circle cx="38" cy="19" r="3" fill="#333" stroke="#888" stroke-width="0.8"/>'
            '<rect x="40" y="9" width="2" height="2" rx="0.5" fill="#E74C3C"/>'
            '<rect x="29" y="8" width="5" height="4" rx="1" fill="rgba(255,255,255,0.3)"/>'
            '</svg>'
        ),
        "size": [44, 22],
        "anchor": [22, 11],
    },
}

STOP_COLORS = {
    "Delivery": "#D4A843", "Pickup": "#3FB950", "Meeting": "#2EA4A4",
    "Warehouse": "#8957E5", "Customs": "#E74C3C", "Rest": "#8B949E",
}


def _bearing(lat1, lon1, lat2, lon2):
    """Compute bearing in degrees from point 1 to point 2."""
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2) -
         math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _interpolate_path(coords, num_points=80):
    """Interpolate between route coordinates for smooth animation frames."""
    if len(coords) < 2:
        return coords

    distances = [0.0]
    for i in range(1, len(coords)):
        dlat = coords[i][0] - coords[i - 1][0]
        dlon = coords[i][1] - coords[i - 1][1]
        distances.append(distances[-1] + math.sqrt(dlat ** 2 + dlon ** 2))

    total = distances[-1]
    if total == 0:
        return coords

    points = []
    for step in range(num_points + 1):
        target_dist = (step / num_points) * total
        seg = 0
        for j in range(1, len(distances)):
            if distances[j] >= target_dist:
                seg = j - 1
                break
        else:
            seg = len(distances) - 2

        seg_len = distances[seg + 1] - distances[seg]
        t = 0 if seg_len == 0 else (target_dist - distances[seg]) / seg_len
        lat = coords[seg][0] + t * (coords[seg + 1][0] - coords[seg][0])
        lon = coords[seg][1] + t * (coords[seg + 1][1] - coords[seg][1])
        points.append([lat, lon])

    return points


def render_animated_route_map(
    stops_list: list,
    itinerary: dict = None,
    vehicle_type: str = "Car",
    animation_duration_s: float = 4.0,
    map_height: int = 500,
) -> str:
    """Build a Folium map with animated vehicle tracing the route."""
    if not stops_list:
        return "<p>No stops to display.</p>"

    # ── Stop ordering ────────────────────────────────────────────────────
    ordered_stops = stops_list[:]
    if itinerary and "stops" in itinerary:
        seq_map = {s["stop_id"]: s["sequence"] for s in itinerary["stops"]}
        ordered_stops = sorted(stops_list, key=lambda s: seq_map.get(s["stop_id"], 999))

    lats = [s["lat"] for s in ordered_stops]
    lons = [s["lon"] for s in ordered_stops]
    center_lat, center_lon = sum(lats) / len(lats), sum(lons) / len(lons)

    # ── Auto-zoom ────────────────────────────────────────────────────────
    span = max(max(lats) - min(lats), max(lons) - min(lons))
    zoom = (14 if span < 0.05 else 13 if span < 0.1 else 12 if span < 0.3
            else 11 if span < 0.5 else 10 if span < 1.0 else 9 if span < 2.0
            else 8 if span < 5.0 else 7)

    # ── Build map ────────────────────────────────────────────────────────
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles="cartodbdark_matter", control_scale=True)

    # Planned route polyline (dashed gold)
    route_coords = [[s["lat"], s["lon"]] for s in ordered_stops]
    folium.PolyLine(route_coords, color="#D4A843", weight=3,
                    opacity=0.7, dash_array="8 4").add_to(m)

    # Stop markers
    for i, stop in enumerate(ordered_stops):
        color = STOP_COLORS.get(stop.get("stop_type", "Delivery"), "#D4A843")
        seq = i + 1
        arrival = departure = ""
        if itinerary and "stops" in itinerary:
            ist = next((s for s in itinerary["stops"]
                        if s["stop_id"] == stop["stop_id"]), None)
            if ist:
                arrival, departure = ist.get("arrival_time", ""), ist.get("departure_time", "")

        popup = (f'<div style="font-family:sans-serif;font-size:13px;min-width:150px">'
                 f'<b>{seq}. {stop["location_name"]}</b><br>'
                 f'<span style="color:#555;font-size:11px">Type: {stop.get("stop_type","—")}')
        if arrival:
            popup += f'<br>Arrival: {arrival}'
        if departure:
            popup += f'<br>Departure: {departure}'
        popup += '</span></div>'

        icon_html = (
            f'<div style="background:{color};color:#0D1117;width:26px;height:26px;'
            f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
            f'font-weight:700;font-size:12px;font-family:sans-serif;'
            f'border:2px solid #0D1117;box-shadow:0 2px 6px rgba(0,0,0,0.4)">'
            f'{seq}</div>')

        folium.Marker(
            [stop["lat"], stop["lon"]],
            popup=folium.Popup(popup, max_width=220),
            icon=folium.DivIcon(html=icon_html, icon_size=[26, 26], icon_anchor=[13, 13]),
        ).add_to(m)

    # ── Animation + Legend + Replay ──────────────────────────────────────
    vehicle = VEHICLE_ICONS.get(vehicle_type, VEHICLE_ICONS["Car"])
    interpolated = _interpolate_path(route_coords, num_points=80)
    interval_ms = int((animation_duration_s * 1000) / max(len(interpolated), 1))
    svg_esc = vehicle["svg"].replace("\\", "\\\\").replace("`", "\\`")

    legend_items = "".join(
        f'<div class="legend-item"><span class="legend-dot" style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{c};margin-right:6px;vertical-align:middle"></span>{t}</div>'
        for t, c in STOP_COLORS.items()
    )

    # All custom HTML/JS/CSS — injected into Folium's tree via Element
    custom = f"""
    <style>
    .vehicle-icon-wrapper {{ background:none!important; border:none!important; }}
    #routeiq-vehicle {{ filter:drop-shadow(0 1px 3px rgba(0,0,0,0.5)); }}

    /* ── Overlay container (anchored to map, not viewport) ── */
    #routeiq-overlays {{
        position: absolute;
        bottom: 0; left: 0; right: 0;
        z-index: 9999;
        pointer-events: none;
        padding: 10px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 8px;
    }}
    #routeiq-overlays > * {{ pointer-events: auto; }}

    /* ── Legend ── */
    #routeiq-legend {{
        background: rgba(13,17,23,0.92);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 8px 12px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 11px;
        color: #C9D1D9;
        line-height: 1.7;
        max-width: 160px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }}
    #routeiq-legend-title {{
        font-weight: 700;
        color: #F0F6FC;
        font-size: 11px;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: pointer;
        user-select: none;
        -webkit-user-select: none;
    }}
    #routeiq-legend-title .arrow {{
        font-size: 9px;
        transition: transform 0.2s;
    }}
    #routeiq-legend-body {{ transition: max-height 0.3s ease, opacity 0.2s; overflow: hidden; }}

    /* ── Replay button ── */
    #routeiq-replay {{
        background: rgba(13,17,23,0.92);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 6px 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 12px;
        color: #D4A843;
        cursor: pointer;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        white-space: nowrap;
        flex-shrink: 0;
    }}
    #routeiq-replay:hover {{ background: rgba(30,35,44,0.95); }}
    #routeiq-replay:active {{ transform: scale(0.97); }}

    /* ── Mobile: screens < 480px ── */
    @media (max-width: 480px) {{
        #routeiq-overlays {{ padding: 6px; gap: 6px; }}
        #routeiq-legend {{
            font-size: 10px;
            padding: 6px 10px;
            max-width: 130px;
            line-height: 1.5;
        }}
        #routeiq-legend-title {{ font-size: 10px; }}
        #routeiq-legend .legend-item {{ font-size: 10px; }}
        #routeiq-legend .legend-dot {{
            width: 8px; height: 8px; margin-right: 4px;
        }}
        /* Auto-collapse legend on mobile */
        #routeiq-legend-body {{ max-height: 0; opacity: 0; }}
        #routeiq-legend-title .arrow {{ transform: rotate(-90deg); }}

        #routeiq-replay {{
            font-size: 11px;
            padding: 5px 10px;
        }}
    }}

    /* ── Tablet: 481-768px ── */
    @media (min-width: 481px) and (max-width: 768px) {{
        #routeiq-legend {{ font-size: 10px; max-width: 145px; }}
        #routeiq-replay {{ font-size: 11px; }}
    }}
    </style>

    <div id="routeiq-overlays">
        <div id="routeiq-legend">
            <div id="routeiq-legend-title" onclick="toggleLegend()">
                Route legend <span class="arrow">&#9660;</span>
            </div>
            <div id="routeiq-legend-body">
                {legend_items}
                <div style="margin-top:3px;border-top:1px solid #30363D;padding-top:3px">
                    <span style="color:#D4A843">- - -</span> Planned route<br>
                    <span style="color:#3FB950">___</span> Vehicle trail
                </div>
            </div>
        </div>

        <div id="routeiq-replay" onclick="location.reload()">
            &#x21bb; Replay
        </div>
    </div>

    <script>
    // Legend toggle (collapsed by default on mobile, open on desktop)
    function toggleLegend() {{
        var body = document.getElementById('routeiq-legend-body');
        var arrow = document.querySelector('#routeiq-legend-title .arrow');
        if (!body) return;
        if (body.style.maxHeight && body.style.maxHeight !== '0px') {{
            body.style.maxHeight = '0px';
            body.style.opacity = '0';
            if (arrow) arrow.style.transform = 'rotate(-90deg)';
        }} else {{
            body.style.maxHeight = '200px';
            body.style.opacity = '1';
            if (arrow) arrow.style.transform = 'rotate(0deg)';
        }}
    }}

    // Auto-expand legend on desktop, keep collapsed on mobile
    (function() {{
        var mq = window.matchMedia('(min-width: 481px)');
        if (mq.matches) {{
            var body = document.getElementById('routeiq-legend-body');
            if (body) {{ body.style.maxHeight = '200px'; body.style.opacity = '1'; }}
        }}
    }})();

    // ── Vehicle animation ──
    (function() {{
        var path = {json.dumps(interpolated)};
        var vehicleSvg = `{svg_esc}`;
        var iconSize = {json.dumps(vehicle["size"])};
        var iconAnchor = {json.dumps(vehicle["anchor"])};
        var intervalMs = {interval_ms};

        function findMap() {{
            for (var key in window) {{
                if (key.indexOf('map_') === 0) {{
                    try {{
                        var obj = window[key];
                        if (obj && typeof obj.getCenter === 'function' &&
                            typeof obj.addLayer === 'function' &&
                            typeof obj.getZoom === 'function') return obj;
                    }} catch(e) {{}}
                }}
            }}
            return null;
        }}

        function run(map) {{
            var step = 0;
            var icon = L.divIcon({{
                html: '<div id="routeiq-vehicle" style="transition:transform 0.08s linear">' + vehicleSvg + '</div>',
                iconSize: iconSize, iconAnchor: iconAnchor,
                className: 'vehicle-icon-wrapper',
            }});
            var marker = L.marker(path[0], {{ icon: icon, zIndexOffset: 1000 }}).addTo(map);
            var trail = L.polyline([], {{ color: '#3FB950', weight: 4, opacity: 0.9 }}).addTo(map);

            function bearing(p1, p2) {{
                var dLon = (p2[1]-p1[1])*Math.PI/180;
                var la1 = p1[0]*Math.PI/180, la2 = p2[0]*Math.PI/180;
                var y = Math.sin(dLon)*Math.cos(la2);
                var x = Math.cos(la1)*Math.sin(la2) - Math.sin(la1)*Math.cos(la2)*Math.cos(dLon);
                return ((Math.atan2(y,x)*180/Math.PI)+360)%360;
            }}

            function tick() {{
                if (step >= path.length) {{
                    var el = document.getElementById('routeiq-vehicle');
                    if (el) el.style.filter = 'drop-shadow(0 0 8px #3FB950)';
                    return;
                }}
                var pos = path[step];
                marker.setLatLng(pos);
                trail.addLatLng(pos);
                if (step < path.length-1) {{
                    var b = bearing(pos, path[step+1]);
                    var el = document.getElementById('routeiq-vehicle');
                    if (el) el.style.transform = 'rotate('+(b-90)+'deg)';
                }}
                step++;
                setTimeout(tick, intervalMs);
            }}
            setTimeout(tick, 800);
        }}

        var attempts = 0;
        var poller = setInterval(function() {{
            attempts++;
            var map = findMap();
            if (map) {{ clearInterval(poller); run(map); }}
            else if (attempts > 50) clearInterval(poller);
        }}, 100);
    }})();
    </script>
    """

    m.get_root().html.add_child(Element(custom))
    return m._repr_html_()


def render_animated_map_in_streamlit(
    stops_list: list, itinerary: dict = None,
    vehicle_type: str = "Car", animation_duration_s: float = 4.0, height: int = 500,
):
    """Render the animated route map directly in Streamlit."""
    html = render_animated_route_map(
        stops_list, itinerary, vehicle_type, animation_duration_s, height)
    components.html(html, height=height, scrolling=False)
