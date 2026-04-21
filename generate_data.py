"""
generate_data.py — generates stops.csv, routes.csv, and logistics_kb.csv
Run once: python generate_data.py
Also auto-called if CSVs are missing.

Dataset:
  - 60 PAN-India locations across 12 geographic regions
  - 100 routes with asymmetric stop counts (5-25 stops)
  - Mix of local (city-level), regional, and inter-city PAN-India routes
"""
import pandas as pd
import random
from datetime import datetime, timedelta
import math

random.seed(42)

# ── 60 PAN-India locations across 12 regions ──────────────────────────────────
LOCATIONS = [
    # Mumbai Metro (0-10)
    ("Dadar Warehouse",           19.0178,  72.8478),
    ("Andheri Distribution Hub",  19.1196,  72.8464),
    ("BKC Corporate HQ",          19.0658,  72.8692),
    ("Thane Depot",               19.2183,  72.9781),
    ("Navi Mumbai Logistics",     19.0330,  73.0297),
    ("Bhiwandi Mega Depot",       19.2979,  73.0586),
    ("Kurla Sorting Center",      19.0728,  72.8826),
    ("Mulund Cold Storage",       19.1726,  72.9559),
    ("Panvel Port Gate",          18.9894,  73.1175),
    ("Uran Customs Terminal",     18.8921,  72.9448),
    ("Kalyan Junction",           19.2403,  73.1305),

    # Pune (11-15)
    ("Pune Warehouse Central",    18.5204,  73.8567),
    ("Hinjewadi Tech Zone",       18.5912,  73.7389),
    ("Hadapsar Export Unit",      18.5089,  73.9259),
    ("Pimpri Depot",              18.6298,  73.7997),
    ("Lonavala Transit Stop",     18.7515,  73.4061),

    # Delhi NCR (16-21)
    ("Connaught Place Hub",       28.6315,  77.2167),
    ("Gurgaon Logistics Park",    28.4595,  77.0266),
    ("Noida Distribution Center", 28.5355,  77.3910),
    ("Faridabad Warehouse",       28.4089,  77.3178),
    ("Delhi Cargo Terminal",      28.5562,  77.1000),
    ("Ghaziabad Cold Chain",      28.6692,  77.4538),

    # Bangalore (22-26)
    ("Whitefield Warehouse",      12.9698,  77.7500),
    ("Electronic City Hub",       12.8399,  77.6770),
    ("Peenya Industrial Depot",   13.0283,  77.5180),
    ("Yeshwanthpur Junction",     13.0299,  77.5520),
    ("Bommanahalli Pickup",       12.9133,  77.6398),

    # Chennai (27-30)
    ("Chennai Port Logistics",    13.0967,  80.2900),
    ("Ambattur Industrial",       13.0978,  80.1609),
    ("Sriperumbudur Hub",         12.9674,  79.9482),
    ("Madhavaram Depot",          13.1499,  80.2318),

    # Hyderabad (31-34)
    ("HITEC City Warehouse",      17.4435,  78.3772),
    ("Patancheru Depot",          17.5334,  78.2636),
    ("Uppal Distribution",        17.4062,  78.5604),
    ("Shamshabad Cargo",          17.2403,  78.4294),

    # Kolkata (35-38)
    ("Kolkata Dock Logistics",    22.5726,  88.3639),
    ("Howrah Industrial Hub",     22.6180,  88.3292),
    ("Dankuni Distribution",      22.6748,  88.2731),
    ("Salt Lake Warehouse",       22.5831,  88.4177),

    # Gujarat (39-42)
    ("Ahmedabad GIDC",            23.0225,  72.5714),
    ("Surat Textile Hub",         21.1702,  72.8311),
    ("Vadodara Depot",            22.3072,  73.1812),
    ("Rajkot Warehouse",          22.3039,  70.8022),

    # North India (43-48)
    ("Jaipur Logistics Park",     26.9124,  75.7873),
    ("Chandigarh Distribution",   30.7333,  76.7794),
    ("Ludhiana Warehouse",        30.9010,  75.8573),
    ("Lucknow Cargo Hub",         26.8467,  80.9462),
    ("Agra Distribution",         27.1767,  78.0081),
    ("Kanpur Industrial",         26.4499,  80.3319),

    # South India (49-52)
    ("Coimbatore Industrial",     11.0168,  76.9558),
    ("Kochi Port Terminal",        9.9312,  76.2673),
    ("Madurai Warehouse",          9.9252,  78.1198),
    ("Visakhapatnam Port",        17.6868,  83.2185),

    # East India (53-56)
    ("Bhubaneswar Depot",         20.2961,  85.8245),
    ("Patna Cargo Center",        25.5941,  85.1376),
    ("Guwahati North Hub",        26.1445,  91.7362),
    ("Ranchi Distribution",       23.3441,  85.3096),

    # Central India (57-62)
    ("Nagpur Cold Chain",         21.1458,  79.0882),
    ("Indore Logistics Hub",      22.7196,  75.8577),
    ("Bhopal Central Depot",      23.2599,  77.4126),
    ("Raipur Warehouse",          21.2514,  81.6296),
    ("Aurangabad Hub",            19.8762,  75.3433),
    ("Nashik Depot",              19.9975,  73.7898),
]

# Region index ranges (for geographically coherent local/regional routes)
_REGIONS = [
    list(range(0,  11)),   # Mumbai
    list(range(11, 16)),   # Pune
    list(range(16, 22)),   # Delhi NCR
    list(range(22, 27)),   # Bangalore
    list(range(27, 31)),   # Chennai
    list(range(31, 35)),   # Hyderabad
    list(range(35, 39)),   # Kolkata
    list(range(39, 43)),   # Gujarat
    list(range(43, 49)),   # North India
    list(range(49, 53)),   # South India
    list(range(53, 57)),   # East India
    list(range(57, 63)),   # Central India
]

STOP_TYPES    = ["Delivery", "Pickup", "Meeting", "Warehouse", "Customs", "Rest"]
PRIORITIES    = ["High", "Medium", "Low"]
STATUSES      = ["On Time", "Delayed", "Scheduled", "Cancelled"]
DELAY_REASONS = [
    "Traffic congestion", "Vehicle breakdown", "Weather delay",
    "Customer not available", "Document missing", "Customs hold",
    "Driver illness", "", "", "", "", "",
]
TRANSPORT_MODES = ["Road", "Rail", "Air", "Sea"]
VEHICLE_TYPES   = ["Truck", "Van", "Tempo", "Motorcycle", "Refrigerated Truck"]
DRIVERS = [
    "Ramesh Kumar",  "Suresh Patil",  "Rajesh Singh",   "Anil Sharma",
    "Vinod Yadav",   "Priya Nair",    "Deepak Mehta",   "Kavita Joshi",
    "Manish Gupta",  "Santosh Reddy", "Arjun Verma",    "Neha Desai",
    "Ravi Krishnan", "Sunita Rao",    "Amit Tiwari",
]

# Asymmetric stop-count profile: 100 routes
_STOP_COUNTS = (
    [random.randint(5,  8)  for _ in range(30)] +   # short  — local delivery
    [random.randint(10, 15) for _ in range(45)] +   # medium — regional
    [random.randint(16, 25) for _ in range(25)]     # long   — inter-city / PAN India
)
random.shuffle(_STOP_COUNTS)


def _pick_route_locations(n_stops: int, route_idx: int) -> list:
    """Return n_stops location tuples.

    Short/medium routes stay within one or two regions.
    Long routes (>= 15 stops) span multiple regions for PAN India coverage.
    """
    if n_stops >= 15:
        # PAN India: draw from entire location pool
        pool = LOCATIONS[:]
    else:
        # Local/regional: pick 1-2 adjacent regions as primary pool
        primary   = _REGIONS[route_idx % len(_REGIONS)]
        secondary = _REGIONS[(route_idx + 1) % len(_REGIONS)]
        pool_idx  = list(set(primary + secondary))
        pool      = [LOCATIONS[i] for i in pool_idx if i < len(LOCATIONS)]
        # Pad from full list if the regional pool is too small
        if len(pool) < n_stops:
            extras = [l for l in LOCATIONS if l not in pool]
            pool  += random.sample(extras, min(n_stops - len(pool), len(extras)))

    return random.sample(pool, min(n_stops, len(pool)))


def generate_stops(n_routes: int = 100) -> pd.DataFrame:
    rows      = []
    base_date = datetime(2024, 1, 1)
    stop_idx  = 0

    for route_idx in range(n_routes):
        rid     = f"RT-{100 + route_idx}"
        n_stops = _STOP_COUNTS[route_idx]
        locs    = _pick_route_locations(n_stops, route_idx)

        for loc in locs:
            stop_type    = random.choices(STOP_TYPES, weights=[35, 20, 15, 15, 10, 5])[0]
            priority     = random.choices(PRIORITIES, weights=[20, 50, 30])[0]
            tw_start_h   = random.randint(7, 14)
            tw_end_h     = tw_start_h + random.randint(2, 4)
            date         = base_date + timedelta(days=random.randint(0, 89))
            tw_start     = datetime(date.year, date.month, date.day,
                                    tw_start_h, random.choice([0, 30]))
            tw_end       = datetime(date.year, date.month, date.day,
                                    min(tw_end_h, 20), random.choice([0, 30]))
            status       = random.choices(STATUSES, weights=[55, 20, 20, 5])[0]
            delay_reason = (random.choice([r for r in DELAY_REASONS if r])
                            if status == "Delayed" else "")
            stop_idx    += 1

            rows.append({
                "stop_id":              f"STP-{stop_idx:06d}",
                "route_id":             rid,
                "location_name":        loc[0],
                "lat":                  loc[1],
                "lon":                  loc[2],
                "stop_type":            stop_type,
                "priority":             priority,
                "status":               status,
                "delay_reason":         delay_reason,
                "scheduled_date":       date.strftime("%Y-%m-%d"),
                "time_window_start":    tw_start.strftime("%Y-%m-%d %H:%M"),
                "time_window_end":      tw_end.strftime("%Y-%m-%d %H:%M"),
                "service_duration_min": {
                    "Delivery": 15, "Pickup": 10, "Meeting": 45,
                    "Warehouse": 30, "Customs": 60, "Rest": 20,
                }[stop_type],
                "cargo_weight_kg":  (random.randint(10, 2000)
                                     if stop_type in ("Delivery", "Pickup", "Warehouse")
                                     else 0),
                "cargo_type":       random.choice([
                    "Electronics", "FMCG", "Pharmaceuticals", "Auto Parts",
                    "Textiles", "Perishables", "Documents",
                ]),
                "contact_person":   random.choice(DRIVERS),
                "notes":            random.choice([
                    "Call 30 min before arrival", "Gate closes at 17:00",
                    "Requires forklift", "Temperature controlled",
                    "High security zone", "Parking available", "", "",
                ]),
            })

    return pd.DataFrame(rows)


def _haversine(lat1, lon1, lat2, lon2):
    R    = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def generate_routes(stops_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid in stops_df["route_id"].unique():
        rdf    = stops_df[stops_df["route_id"] == rid]
        coords = list(zip(rdf["lat"], rdf["lon"]))
        dist   = sum(
            _haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
            for i in range(len(coords) - 1)
        )
        mode        = random.choice(TRANSPORT_MODES)
        speed       = {"Road": 35, "Rail": 60, "Air": 800, "Sea": 25}[mode]
        duration_m  = int((dist / speed) * 60 + len(rdf) * 20)
        fuel_cost   = round(dist * {"Road": 8, "Rail": 4, "Air": 50, "Sea": 12}[mode], 0)
        on_time_pct = round(100 * (rdf["status"] == "On Time").mean(), 1)
        date        = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 89))

        rows.append({
            "route_id":                rid,
            "driver":                  random.choice(DRIVERS),
            "vehicle_type":            random.choice(VEHICLE_TYPES),
            "transport_mode":          mode,
            "n_stops":                 len(rdf),
            "distance_km":             round(dist, 2),
            "total_distance_km":       round(dist, 2),
            "duration_min":            duration_m,
            "estimated_fuel_cost_inr": fuel_cost,
            "on_time_pct":             on_time_pct,
            "route_date":              date.strftime("%Y-%m-%d"),
            "start_location":          rdf.iloc[0]["location_name"],
            "end_location":            rdf.iloc[-1]["location_name"],
            "status":                  random.choices(
                ["Completed", "In Progress", "Planned"], weights=[60, 20, 20])[0],
            "total_cargo_kg":          int(rdf["cargo_weight_kg"].sum()),
        })
    return pd.DataFrame(rows)


KB_DATA = [
    {"category": "Routing", "topic": "Route optimization basics",
     "content": "Route optimization aims to find the shortest or fastest sequence of stops. Key algorithms include Nearest Neighbor (greedy), Clarke-Wright Savings, and Genetic Algorithms. For urban logistics with time windows, VRPTW (Vehicle Routing Problem with Time Windows) solvers are industry standard."},
    {"category": "Routing", "topic": "Time window compliance",
     "content": "Time windows are customer-specified delivery slots. Hard windows (must arrive within slot) vs soft windows (penalty for violation). Best practice: plan buffer time of 10-15 min per stop. Sort stops by earliest deadline first when feasible."},
    {"category": "Routing", "topic": "Last-mile delivery challenges",
     "content": "Last-mile delivery accounts for 53% of total logistics cost. Urban challenges include parking restrictions, narrow lanes, and traffic signals. Solutions: micro-fulfillment centers, electric cargo bikes for inner city, slot-based delivery appointments."},
    {"category": "Routing", "topic": "Multi-stop route planning",
     "content": "For routes with 10+ stops, consider splitting by zone/cluster first, then optimize within each zone. Use geographic clustering (k-means or density-based) to group nearby stops before applying TSP heuristics."},
    {"category": "Routing", "topic": "PAN India logistics corridors",
     "content": "Key Indian logistics corridors: Delhi-Mumbai Industrial Corridor (DMIC), Chennai-Bangalore Industrial Corridor, Eastern Dedicated Freight Corridor (Ludhiana to Dankuni), Western Dedicated Freight Corridor (Dadri to JNPT). GST implementation unified inter-state movement with E-way bills."},
    {"category": "Fleet", "topic": "Vehicle capacity planning",
     "content": "Match vehicle capacity to cargo volume + weight. Rule: load vehicles to 80-85% capacity to allow for returns and ad-hoc pickups. Refrigerated trucks add 15-20% fuel cost. Always check axle weight limits for heavy cargo."},
    {"category": "Fleet", "topic": "Driver hours regulations",
     "content": "In India, Motor Transport Workers Act limits driving to 9 hours/day with mandatory 30-min break after 5 hours. Maximum 48 hours/week. Maintain driver logs for compliance. Fatigue-related accidents peak between 2-6 AM and 2-4 PM."},
    {"category": "Fleet", "topic": "Fuel cost estimation",
     "content": "Average fuel consumption: trucks 4-6 km/L, vans 10-14 km/L. Fuel cost per km: truck ₹18-25, van ₹8-12. Include idle time (1L/hour idling). GPS route optimization typically reduces fuel costs by 15-20%."},
    {"category": "Documentation", "topic": "E-way bill requirements",
     "content": "E-way bill required for interstate goods movement >₹50,000. Generate on GST portal before dispatch. Valid for 1 day per 100 km distance. Carry bill of lading, delivery challan, and tax invoice. Customs stop? Bring advance authorization."},
    {"category": "Documentation", "topic": "Customs clearance process",
     "content": "Import customs: file Bill of Entry within 30 days of vessel arrival. Required: Invoice, packing list, Bill of Lading, Import Export Code. Customs examines 15-20% of shipments. Pre-clearance (prior to arrival) reduces dwell time by 60%."},
    {"category": "Documentation", "topic": "Proof of delivery (POD)",
     "content": "POD must include: recipient name, signature, date/time, item count confirmation. Digital POD via mobile app is legally valid. Capture geo-tagged photo at delivery. POD disputes: retain for 3 years. Undelivered items: return-to-origin protocol."},
    {"category": "Scheduling", "topic": "Delivery slot optimization",
     "content": "Cluster time-sensitive deliveries (High priority) in morning slots (8-11 AM) when traffic is lighter. Schedule Customs and Warehouse stops mid-morning. Meetings should be confirmed 24 hours in advance. Allow 20% buffer in schedule for delays."},
    {"category": "Scheduling", "topic": "Dynamic re-routing",
     "content": "When delays occur: (1) identify impacted downstream stops, (2) notify customers proactively, (3) check if stops can be swapped or rescheduled, (4) escalate critical stops. Live traffic APIs (HERE, Google Maps) can trigger automatic rerouting."},
    {"category": "Compliance", "topic": "Hazardous materials transport",
     "content": "Hazmats require ADG/IMDG compliant packaging, MSDS documentation, trained driver with HAZMAT certification. Separate from food/pharma cargo. Report spills immediately to SPCB. Insurance cover must explicitly include hazardous cargo."},
    {"category": "Compliance", "topic": "Cold chain management",
     "content": "Pharma/perishable cold chain: maintain 2-8°C (pharma) or 0-4°C (food). IoT temperature loggers must record every 15 min. Break in cold chain must be documented and reported. Pre-cool truck 2 hours before loading."},
    {"category": "Performance", "topic": "KPI benchmarks for logistics",
     "content": "Industry benchmarks: On-time delivery >92%, Order accuracy >99.5%, Cost per delivery <₹150 (urban), Vehicle utilization >80%, First attempt delivery success >85%. Track daily. Weekly review with routing team improves KPIs by 10-15%."},
    {"category": "Performance", "topic": "Reducing delivery failures",
     "content": "Top failure causes: wrong address (22%), customer unavailable (38%), business closed (15%), access issues (12%). Mitigations: address validation at order entry, pre-delivery SMS/call, flexible re-delivery slots, OTP-based confirmation."},
    {"category": "Technology", "topic": "GPS and telematics",
     "content": "GPS tracking enables real-time vehicle location, speed monitoring, and geofencing alerts. Telematics data: harsh braking, acceleration, idling reduce maintenance cost by 12%. Integration with TMS enables automatic ETA updates to customers."},
    {"category": "Emergency", "topic": "Breakdown protocol",
     "content": "Vehicle breakdown: (1) Safe stop with hazard lights, (2) Notify dispatcher immediately, (3) Arrange replacement vehicle or tow, (4) Transfer priority cargo, (5) Update all affected customers, (6) File incident report within 4 hours."},
    {"category": "Emergency", "topic": "Cargo damage claims",
     "content": "Cargo damage: document with photos at delivery, note on POD, notify insurer within 24 hours. File claim with carrier within 3 days. Required: photos, POD with damage noted, original invoice, packing list."},
]


def generate_kb():
    return pd.DataFrame(KB_DATA)


if __name__ == "__main__":
    stops_df  = generate_stops(100)
    routes_df = generate_routes(stops_df)
    kb_df     = generate_kb()
    stops_df.to_csv("stops.csv",        index=False)
    routes_df.to_csv("routes.csv",       index=False)
    kb_df.to_csv("logistics_kb.csv",    index=False)
    print(f"Generated stops.csv        ({len(stops_df)} rows, "
          f"{stops_df['route_id'].nunique()} routes)")
    print(f"Generated routes.csv       ({len(routes_df)} rows)")
    print(f"Generated logistics_kb.csv ({len(kb_df)} rows)")
    print(f"Stops per route: min={stops_df.groupby('route_id').size().min()}, "
          f"max={stops_df.groupby('route_id').size().max()}, "
          f"avg={stops_df.groupby('route_id').size().mean():.1f}")
