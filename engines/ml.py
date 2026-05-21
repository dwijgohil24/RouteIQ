import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .routing import get_default_strategy


class MLEngine:
    def __init__(self):
        self.vectorizer     = None
        self.kmeans         = None
        self.feature_matrix = None

    def cluster_stops(self, stops_df, n_clusters=4):
        texts = (
            stops_df["stop_type"].fillna("") + " " +
            stops_df["location_name"].fillna("") + " " +
            stops_df["notes"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        X = self.vectorizer.fit_transform(texts)
        self.feature_matrix = X
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        pca    = PCA(n_components=2, random_state=42)
        X_2d   = pca.fit_transform(X.toarray())
        return labels, X_2d

    def get_cluster_keywords(self, top_n=5):
        if self.vectorizer is None or self.kmeans is None:
            return {}
        terms    = self.vectorizer.get_feature_names_out()
        keywords = {}
        for i, center in enumerate(self.kmeans.cluster_centers_):
            top_idx     = center.argsort()[-top_n:][::-1]
            keywords[i] = [terms[j] for j in top_idx]
        return keywords

    def nearest_neighbor_route(self, coords):
        if len(coords) <= 1:
            return list(range(len(coords)))
        unvisited = list(range(1, len(coords)))
        route     = [0]
        while unvisited:
            curr    = route[-1]
            nearest = min(unvisited, key=lambda j: self._haversine(coords[curr], coords[j]))
            route.append(nearest)
            unvisited.remove(nearest)
        return route

    @staticmethod
    def _haversine(c1, c2):
        lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
        lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def compute_distance_km(lat1, lon1, lat2, lon2):
        return MLEngine._haversine((lat1, lon1), (lat2, lon2))

    @staticmethod
    def osrm_route(coords: list) -> dict:
        """Delegates to the module-level routing strategy (Strategy Pattern).
        Swap the strategy via engines.routing.set_default_strategy() without
        touching this method or any of its callers."""
        if len(coords) < 2:
            return {"legs": [], "total_distance_km": 0.0,
                    "total_duration_min": 0.0, "source": "osrm",
                    "route_geometry": None}
        return get_default_strategy().route(coords)
