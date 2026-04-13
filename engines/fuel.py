from .ml import MLEngine


class FuelEngine:
    CITY_PRICES = {
        "Mumbai":        {"petrol": 103.44, "diesel": 89.97},
        "Pune":          {"petrol": 103.57, "diesel": 90.10},
        "Nashik":        {"petrol": 103.20, "diesel": 89.75},
        "Nagpur":        {"petrol": 103.80, "diesel": 90.15},
        "Aurangabad":    {"petrol": 103.35, "diesel": 89.85},
        "Delhi":         {"petrol": 94.72,  "diesel": 87.62},
        "Gurgaon":       {"petrol": 95.10,  "diesel": 88.05},
        "Noida":         {"petrol": 94.85,  "diesel": 87.80},
        "Bengaluru":     {"petrol": 102.86, "diesel": 88.94},
        "Mysuru":        {"petrol": 102.60, "diesel": 88.70},
        "Chennai":       {"petrol": 100.75, "diesel": 92.34},
        "Coimbatore":    {"petrol": 100.50, "diesel": 92.10},
        "Hyderabad":     {"petrol": 107.41, "diesel": 95.65},
        "Visakhapatnam": {"petrol": 106.80, "diesel": 95.20},
        "Ahmedabad":     {"petrol": 96.63,  "diesel": 92.38},
        "Surat":         {"petrol": 96.50,  "diesel": 92.25},
        "Jaipur":        {"petrol": 104.88, "diesel": 90.36},
        "Lucknow":       {"petrol": 94.65,  "diesel": 87.76},
        "Kanpur":        {"petrol": 94.55,  "diesel": 87.65},
        "Kolkata":       {"petrol": 103.94, "diesel": 90.76},
        "default":       {"petrol": 101.50, "diesel": 90.00},
    }

    VEHICLE_EFFICIENCY = {
        "Truck": 5.5, "Van": 12.0, "Tempo": 9.0,
        "Car": 15.0, "Motorcycle": 40.0, "default": 8.0,
    }

    VEHICLE_FUEL_TYPE = {
        "Truck": "diesel", "Tempo": "diesel",
        "Van": "diesel", "Car": "petrol", "Motorcycle": "petrol",
    }

    @classmethod
    def get_price(cls, city: str, fuel_type: str = "diesel") -> float:
        city_lower = city.lower()
        for name, prices in cls.CITY_PRICES.items():
            if name.lower() in city_lower or city_lower in name.lower():
                return prices.get(fuel_type, prices["diesel"])
        return cls.CITY_PRICES["default"].get(fuel_type, 90.0)

    @classmethod
    def compute_fuel_cost(
        cls,
        distance_km: float,
        vehicle_type: str,
        city: str = "Mumbai",
        override_price: float = None,
        override_efficiency: float = None,
    ) -> dict:
        fuel_type  = cls.VEHICLE_FUEL_TYPE.get(vehicle_type, "diesel")
        efficiency = override_efficiency or cls.VEHICLE_EFFICIENCY.get(vehicle_type, cls.VEHICLE_EFFICIENCY["default"])
        price      = override_price      or cls.get_price(city, fuel_type)
        litres     = distance_km / efficiency if efficiency > 0 else 0
        cost       = round(litres * price, 2)
        return {
            "litres_consumed": round(litres, 2),
            "price_per_litre": price,
            "total_cost_inr":  cost,
            "efficiency_kmpl": efficiency,
            "fuel_type":       fuel_type,
            "city":            city,
        }

    @classmethod
    def savings_analysis(
        cls,
        stops_list: list,
        vehicle_type: str,
        city: str = "Mumbai",
        override_price: float = None,
        override_efficiency: float = None,
    ) -> dict:
        if len(stops_list) < 2:
            return {}

        coords = [(s["lat"], s["lon"]) for s in stops_list]

        osrm_orig = MLEngine.osrm_route(coords)
        orig_km   = osrm_orig["total_distance_km"]

        nn_order  = MLEngine().nearest_neighbor_route(coords)
        coords_nn = [coords[i] for i in nn_order]
        osrm_opt  = MLEngine.osrm_route(coords_nn)
        opt_km    = osrm_opt["total_distance_km"]

        if opt_km >= orig_km:
            opt_km   = orig_km
            osrm_opt = osrm_orig
            nn_order = list(range(len(stops_list)))

        saved_km   = round(orig_km - opt_km, 2)
        saving_pct = round(saved_km / orig_km * 100, 1) if orig_km > 0 else 0

        fuel_orig  = cls.compute_fuel_cost(orig_km, vehicle_type, city, override_price, override_efficiency)
        fuel_opt   = cls.compute_fuel_cost(opt_km,  vehicle_type, city, override_price, override_efficiency)
        saved_cost = round(fuel_orig["total_cost_inr"] - fuel_opt["total_cost_inr"], 2)

        per_stop  = []
        orig_legs = osrm_orig.get("legs", [])
        opt_legs  = osrm_opt.get("legs", [])
        for i, s in enumerate(stops_list[1:]):
            orig_leg = orig_legs[i]["distance_km"] if i < len(orig_legs) else 0
            opt_s    = stops_list[nn_order[i+1]] if (i+1) < len(nn_order) else s
            opt_leg  = opt_legs[i]["distance_km"] if i < len(opt_legs) else 0
            per_stop.append({
                "leg":            i + 1,
                "original_stop":  s["location_name"],
                "optimized_stop": opt_s["location_name"],
                "original_km":    orig_leg,
                "optimized_km":   opt_leg,
                "saved_km":       round(orig_leg - opt_leg, 2),
            })

        return {
            "original_km":           orig_km,
            "optimized_km":          opt_km,
            "saved_km":              saved_km,
            "saving_pct":            saving_pct,
            "original_cost_inr":     fuel_orig["total_cost_inr"],
            "optimized_cost_inr":    fuel_opt["total_cost_inr"],
            "saved_cost_inr":        saved_cost,
            "fuel_detail_original":  fuel_orig,
            "fuel_detail_optimized": fuel_opt,
            "original_order":        [s["location_name"] for s in stops_list],
            "optimized_order":       [stops_list[i]["location_name"] for i in nn_order],
            "per_stop_savings":      per_stop,
            "routing_source":        osrm_orig["source"],
        }
