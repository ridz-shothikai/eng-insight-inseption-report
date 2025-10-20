# src/utils/validators.py

from typing import Any, List, Dict, Union

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

def validate_coordinates(start_lat: float, start_lng: float, 
                        end_lat: float, end_lng: float) -> bool:
    """
    Validate start and end coordinates.
    
    Args:
        start_lat, start_lng: Start point coordinates
        end_lat, end_lng: End point coordinates
    
    Returns:
        True if all coordinates are valid, False otherwise
    """
    return (validate_coordinate(start_lat, start_lng) and 
            validate_coordinate(end_lat, end_lng))

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