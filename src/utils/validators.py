# src/utils/validators.py

from typing import Any, List, Dict, Union, Optional

# -----------------------------
# Basic Validators
# -----------------------------
def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())

def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0

def is_list_of_dicts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(i, dict) for i in value)

def is_dict_with_keys(value: Any, keys: List[str]) -> bool:
    return isinstance(value, dict) and all(k in value for k in keys)

def is_valid_coordinate(coord: Dict[str, Any]) -> bool:
    try:
        lat, lng = float(coord.get("latitude", 0)), float(coord.get("longitude", 0))
        return -90 <= lat <= 90 and -180 <= lng <= 180
    except (ValueError, TypeError):
        return False

# -----------------------------
# Coordinate Validators
# -----------------------------
def validate_coordinate(lat: float, lng: float) -> bool:
    """Validate single coordinate pair"""
    return -90 <= lat <= 90 and -180 <= lng <= 180

def validate_coordinates(start_lat: float, start_lng: float, end_lat: float, end_lng: float, waypoint_lats: Optional[List[float]] = None, waypoint_lngs: Optional[List[float]] = None) -> bool:
    """Validate coordinates including waypoints"""
    # Validate start and end coordinates (existing logic)
    if not (-90 <= start_lat <= 90) or not (-180 <= start_lng <= 180):
        return False
    if not (-90 <= end_lat <= 90) or not (-180 <= end_lng <= 180):
        return False
    
    # Validate waypoints if provided
    if waypoint_lats and waypoint_lngs:
        if len(waypoint_lats) != len(waypoint_lngs):
            return False
        
        for lat, lng in zip(waypoint_lats, waypoint_lngs):
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return False
    
    return True

# -----------------------------
# Combined Checks
# -----------------------------
def validate_chunks(chunks: Union[List[Dict], None]) -> bool:
    if not chunks or not isinstance(chunks, list):
        return False
    for c in chunks:
        if not is_dict_with_keys(c, ["text"]) or not is_non_empty_string(c["text"]):
            return False
    return True