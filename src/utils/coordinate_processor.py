# utils/coordinate_processor.py
import math
from typing import Tuple, List, Dict

# -----------------------------
# Haversine Distance
# -----------------------------
def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance in km between two lat/lng points"""
    R = 6371  # km
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat, delta_lng = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# -----------------------------
# Coordinate Utilities
# -----------------------------
def validate_coordinates(lat: float, lng: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lng <= 180

def average_coordinates(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not coords:
        return (0.0, 0.0)
    avg_lat = sum(lat for lat, _ in coords) / len(coords)
    avg_lng = sum(lng for _, lng in coords) / len(coords)
    return avg_lat, avg_lng

def coordinates_dict(lat: float, lng: float) -> Dict[str, float]:
    return {"latitude": lat, "longitude": lng}
