"""
generate_data.py — generates stops.csv, routes.csv, and logistics_kb.csv
Run once: python generate_data.py
Also auto-called if CSVs are missing.
"""
import pandas as pd
import random
from datetime import datetime, timedelta
import math

random.seed(42)

# ── Mumbai / Pune metro area stop locations ──
LOCATIONS = [
    ("Dadar Warehouse",         19.0178, 72.8478),
    ("Andheri Distribution Hub",19.1196, 72.8464),
    ("Bandra Client Office",    19.0596, 72.8295),
    ("Thane Depot",             19.2183, 72.9781),
    ("Navi Mumbai Logistics",   19.0330, 73.0297),
    ("BKC Corporate HQ",        19.0658, 72.8692),
    ("Kurla Sorting Center",    19.0728, 72.8826),
    ("Mulund Cold Storage",     19.1726, 72.9559),
    ("Powai Tech Park",         19.1176, 72.9060),
    ("Vikhroli Warehouse",      19.1076, 72.9265),
    ("Ghatkopar Hub",           19.0863, 72.9081),
    ("Chembur Office",          19.0522, 72.8996),
    ("Vashi Distribution",      19.0771, 73.0075),
    ("Airoli Customs",          19.1607, 72.9991),
    ("Bhiwandi Mega Depot",     19.2979, 73.0586),
    ("Pune Warehouse Central",  18.5204, 73.8567),
    ("Hinjewadi Tech Zone",     18.5912, 73.7389),
    ("Hadapsar Export Unit",    18.5089, 73.9259),
    ("Kharadi Hub",             18.5515, 73.9437),
    ("Pimpri Depot",            18.6298, 73.7997),
    ("Kalyan Junction",         19.2403, 73.1305),
    ("Dombivli Pickup Point",   19.2152, 73.0883),
    ("Badlapur Storage",        19.1538, 73.2574),
    ("Panvel Port Gate",        18.9894, 73.1175),
    ("Uran Customs Terminal",   18.8921, 72.9448),
    ("Lonavala Transit Stop",   18.7515, 73.4061),
    ("Talegaon Distribution",   18.7330, 73.6756),
    ("Nashik Depot",            19.9975, 73.7898),
    ("Aurangabad Hub",          19.8762, 75.3433),
    ("Nagpur Cold Chain",       21.1458, 79.0882),
]

STOP_TYPES  = ["Delivery","Pickup","Meeting","Warehouse","Customs","Rest"]
PRIORITIES  = ["High","Medium","Low"]
STATUSES    = ["On Time","Delayed","Scheduled","Cancelled"]
DELAY_REASONS = [
    "Traffic congestion","Vehicle breakdown","Weather delay","Customer not available",
    "Document missing","Customs hold","Driver illness","","","","","",  # weighted blanks
]
TRANSPORT_MODES = ["Road","Rail","Air","Sea"]
VEHICLE_TYPES   = ["Truck","Van","Tempo","Motorcycle","Refrigerated Truck"]
DRIVERS = [
    "Ramesh Kumar","Suresh Patil","Rajesh Singh","Anil Sharma","Vinod Yadav",
    "Priya Nair","Deepak Mehta","Kavita Joshi","Manish Gupta","Santosh Reddy",
]


def generate_stops(n=60):
    rows = []
    route_ids = [f"RT-{100 + i}" for i in range(n // 5 + 1)]
    base_date = datetime(2024, 3, 1)

    for i in range(n):
        loc = random.choice(LOCATIONS)
        stop_type = random.choices(STOP_TYPES, weights=[35, 20, 15, 15, 10, 5])[0]
        priority  = random.choices(PRIORITIES, weights=[20, 50, 30])[0]

        # Time windows
        tw_start_h = random.randint(7, 14)
        tw_end_h   = tw_start_h + random.randint(2, 4)
        date       = base_date + timedelta(days=random.randint(0, 59))

        tw_start = datetime(date.year, date.month, date.day, tw_start_h, random.choice([0, 30]))
        tw_end   = datetime(date.year, date.month, date.day, min(tw_end_h, 20), random.choice([0, 30]))

        status = random.choices(STATUSES, weights=[55, 20, 20, 5])[0]
        delay_reason = ""
        if status == "Delayed":
            delay_reason = random.choice([r for r in DELAY_REASONS if r])

        # Slight coordinate jitter for realism
        lat = loc[1] + random.uniform(-0.05, 0.05)
        lon = loc[2] + random.uniform(-0.05, 0.05)

        rows.append({
            "stop_id":           f"STP-{2024000 + i + 1}",
            "route_id":          route_ids[i // 5],
            "location_name":     loc[0],
            "lat":               round(lat, 6),
            "lon":               round(lon, 6),
            "stop_type":         stop_type,
            "priority":          priority,
            "status":            status,
            "delay_reason":      delay_reason,
            "scheduled_date":    date.strftime("%Y-%m-%d"),
            "time_window_start": tw_start.strftime("%Y-%m-%d %H:%M"),
            "time_window_end":   tw_end.strftime("%Y-%m-%d %H:%M"),
            "service_duration_min": {
                "Delivery":15, "Pickup":10, "Meeting":45,
                "Warehouse":30, "Customs":60, "Rest":20
            }[stop_type],
            "cargo_weight_kg":   random.randint(10, 2000) if stop_type in ["Delivery","Pickup","Warehouse"] else 0,
            "cargo_type":        random.choice(["Electronics","FMCG","Pharmaceuticals","Auto Parts","Textiles","Perishables","Documents"]),
            "contact_person":    random.choice(DRIVERS),
            "notes":             random.choice([
                "Call 30 min before arrival",
                "Gate closes at 17:00",
                "Requires forklift",
                "Temperature controlled",
                "High security zone",
                "Parking available",
                "",""
            ]),
        })
    return pd.DataFrame(rows)


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def generate_routes(stops_df):
    rows = []
    route_ids = stops_df["route_id"].unique()
    base_date = datetime(2024, 3, 1)

    for rid in route_ids:
        route_stops = stops_df[stops_df["route_id"] == rid]
        n_stops = len(route_stops)

        # Compute approx distance
        coords = list(zip(route_stops["lat"], route_stops["lon"]))
        dist = 0
        for i in range(len(coords)-1):
            dist += _haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])

        mode = random.choice(TRANSPORT_MODES)
        speed = {"Road": 35, "Rail": 60, "Air": 800, "Sea": 25}[mode]
        duration_min = int((dist / speed) * 60 + n_stops * 20)
        fuel_cost = round(dist * {"Road":8, "Rail":4, "Air":50, "Sea":12}[mode], 0)
        on_time_pct = round(100 * (route_stops["status"] == "On Time").mean(), 1)
        date = base_date + timedelta(days=random.randint(0, 59))

        rows.append({
            "route_id":                   rid,
            "driver":                     random.choice(DRIVERS),
            "vehicle_type":               random.choice(VEHICLE_TYPES),
            "transport_mode":             mode,
            "n_stops":                    n_stops,
            "distance_km":                round(dist, 2),
            "total_distance_km":          round(dist, 2),  # alias
            "duration_min":               duration_min,
            "estimated_fuel_cost_inr":    fuel_cost,
            "on_time_pct":                on_time_pct,
            "route_date":                 date.strftime("%Y-%m-%d"),
            "start_location":             route_stops.iloc[0]["location_name"] if not route_stops.empty else "",
            "end_location":               route_stops.iloc[-1]["location_name"] if not route_stops.empty else "",
            "status":                     random.choices(["Completed","In Progress","Planned"], weights=[60,20,20])[0],
            "total_cargo_kg":             int(route_stops["cargo_weight_kg"].sum()),
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

    {"category": "Technology", "topic": "TMS integration",
     "content": "Transport Management Systems (TMS) automate: rate shopping, load optimization, carrier booking, document generation, and invoice reconciliation. ROI typically realized in 6-12 months. API integration with WMS and ERP is essential for end-to-end visibility."},

    {"category": "Emergency", "topic": "Breakdown protocol",
     "content": "Vehicle breakdown: (1) Safe stop with hazard lights, (2) Notify dispatcher immediately, (3) Arrange replacement vehicle or tow, (4) Transfer priority cargo, (5) Update all affected customers, (6) File incident report within 4 hours. Keep emergency kit: reflectors, first aid, fire extinguisher."},

    {"category": "Emergency", "topic": "Cargo damage claims",
     "content": "Cargo damage: document with photos at delivery, note on POD, notify insurer within 24 hours. File claim with carrier within 3 days. Required: photos, POD with damage noted, original invoice, packing list. Settlement: replacement value or repair cost, whichever lower."},
]


def generate_kb():
    return pd.DataFrame(KB_DATA)


if __name__ == "__main__":
    stops_df  = generate_stops(60)
    routes_df = generate_routes(stops_df)
    kb_df     = generate_kb()
    stops_df.to_csv("stops.csv", index=False)
    routes_df.to_csv("routes.csv", index=False)
    kb_df.to_csv("logistics_kb.csv", index=False)
    print(f"Generated stops.csv       ({len(stops_df)} rows)")
    print(f"Generated routes.csv      ({len(routes_df)} rows)")
    print(f"Generated logistics_kb.csv ({len(kb_df)} rows)")
