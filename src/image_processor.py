# src/image_processor.py

import os
import math
import logging
from typing import List, Dict, Optional, Callable
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests
import polyline
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY")


# ---------------- Distance Calculation ---------------- #

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate Haversine distance in kilometers"""
    R = 6371  # Earth radius in km
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat, delta_lng = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lng / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


# ---------------- Google Routes API ---------------- #

def get_route_from_google(
    start_lat: float, 
    start_lng: float, 
    end_lat: float, 
    end_lng: float
) -> tuple:
    """
    Get route coordinates using Google Routes API
    
    Returns:
        Tuple of (coordinates, distance_km, duration_minutes)
    """
    if not API_KEY:
        logger.warning("Google API key not found, using fallback straight line")
        return (
            [(start_lat, start_lng), (end_lat, end_lng)],
            calculate_distance(start_lat, start_lng, end_lat, end_lng),
            0
        )
    
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline"
    }
    body = {
        "origin": {
            "location": {
                "latLng": {"latitude": start_lat, "longitude": start_lng}
            }
        },
        "destination": {
            "location": {
                "latLng": {"latitude": end_lat, "longitude": end_lng}
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "polylineQuality": "HIGH_QUALITY"
    }
    
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        if "routes" in data and len(data["routes"]) > 0:
            route = data["routes"][0]
            coords = polyline.decode(route["polyline"]["encodedPolyline"])
            distance = route["distanceMeters"] / 1000  # Convert to km
            duration_sec = int(route["duration"].rstrip("s"))
            duration_min = duration_sec / 60
            
            logger.info(f"Route fetched: {distance:.2f} km, {duration_min:.1f} min")
            return coords, distance, duration_min
    
    except Exception as e:
        logger.warning(f"Google Routes API failed: {e}, using fallback")
    
    # Fallback to straight line
    fallback_coords = [(start_lat, start_lng), (end_lat, end_lng)]
    fallback_distance = calculate_distance(start_lat, start_lng, end_lat, end_lng)
    
    return fallback_coords, fallback_distance, 0


def sample_route_coords(route_coords: List[tuple], max_points: int = 80) -> List[tuple]:
    """
    Reduce route points to avoid URL length limits for Static Maps API
    
    Args:
        route_coords: List of (lat, lng) tuples
        max_points: Maximum number of points to keep
        
    Returns:
        Sampled list of coordinates
    """
    if len(route_coords) <= max_points:
        return route_coords
    
    step = len(route_coords) // (max_points - 2)
    sampled = (
        [route_coords[0]] + 
        [route_coords[i] for i in range(step, len(route_coords) - 1, step)] + 
        [route_coords[-1]]
    )
    
    return sampled


def get_route_image(
    start_lat: float, 
    start_lng: float, 
    end_lat: float, 
    end_lng: float, 
    size: str = "800x600"
) -> Optional[Image.Image]:
    """
    Generate route map image using Google Static Maps API
    
    Args:
        start_lat, start_lng: Starting coordinates
        end_lat, end_lng: Ending coordinates
        size: Image size in format "widthxheight"
        
    Returns:
        PIL Image or None if failed
    """
    if not API_KEY:
        logger.error("Google API key not configured")
        return None
    
    try:
        # Get route coordinates
        route_coords, distance, duration = get_route_from_google(
            start_lat, start_lng, end_lat, end_lng
        )
        
        # Sample coordinates to avoid URL length issues
        sampled_coords = sample_route_coords(route_coords)
        
        # Build path string
        path = "|".join([f"{lat},{lng}" for lat, lng in sampled_coords])
        
        # Define markers
        markers_start = f"color:green|label:A|{start_lat},{start_lng}"
        markers_end = f"color:red|label:B|{end_lat},{end_lng}"
        
        # Build Static Maps API request
        params = {
            "size": size,
            "markers": [markers_start, markers_end],
            "path": f"color:0x0000ff|weight:5|{path}",
            "key": API_KEY
        }
        
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/staticmap",
            params=params,
            timeout=30
        )
        
        if resp.status_code == 200:
            logger.info(f"Route map image generated successfully")
            return Image.open(BytesIO(resp.content))
        else:
            logger.warning(f"Failed to fetch route image: {resp.status_code}")
            return None
    
    except Exception as e:
        logger.error(f"Error generating route image: {e}")
        return None


# ---------------- PDF Generation ---------------- #

def create_pdf_from_images(images: List[Image.Image], output_file: str):
    """
    Combine PIL images into a single PDF
    
    Args:
        images: List of PIL Image objects
        output_file: Path to save the PDF
    """
    if not images:
        raise ValueError("No images to combine into PDF")
    
    # Convert all images to RGB mode (required for PDF)
    rgb_images = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        rgb_images.append(img)
    
    # Save as PDF
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rgb_images[0].save(
        str(output_path),
        save_all=True,
        append_images=rgb_images[1:]
    )
    
    logger.info(f"PDF created with {len(rgb_images)} images: {output_file}")


# ---------------- Main Processing Function ---------------- #

def process_images_to_pdf(
    image_paths: List[str],
    coordinate_data: Dict,
    output_pdf_path: str,
    session_id: Optional[str] = None,
    progress_store: Optional[Dict[str, Dict[str, float]]] = None
) -> bool:
    """
    Process user-uploaded images and route coordinates into a PDF
    
    Args:
        image_paths: List of paths to uploaded images
        coordinate_data: Dict with 'start' and 'end' coordinates
        output_pdf_path: Path to save output PDF
        session_id: Session identifier for progress tracking
        progress_store: Optional progress dictionary to update
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Progress callback
        def update_progress(value: float):
            if progress_store and session_id and session_id in progress_store:
                progress_store[session_id]["images"] = value
        
        update_progress(10.0)
        
        pil_images = []

        # Add route image if coordinates provided
        if coordinate_data:
            start = coordinate_data.get("start")
            end = coordinate_data.get("end")
            
            if start and end:
                logger.info(f"Generating route map from ({start['latitude']}, {start['longitude']}) "
                          f"to ({end['latitude']}, {end['longitude']})")
                
                route_img = get_route_image(
                    start["latitude"],
                    start["longitude"],
                    end["latitude"],
                    end["longitude"]
                )
                
                if route_img:
                    pil_images.append(route_img)
                    logger.info("Route image added successfully")
                else:
                    logger.warning("Failed to generate route image")
        
        update_progress(40.0)

        # Add user-uploaded images
        logger.info(f"Processing {len(image_paths)} user images")
        
        for idx, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path)
                pil_images.append(img)
                logger.info(f"Image loaded: {Path(img_path).name}")
                
                # Update progress
                progress = 40.0 + (idx + 1) / len(image_paths) * 50.0
                update_progress(progress)
                
            except Exception as e:
                logger.warning(f"Failed to open image {img_path}: {e}")

        # Generate PDF
        if pil_images:
            update_progress(95.0)
            create_pdf_from_images(pil_images, output_pdf_path)
            update_progress(100.0)
            
            logger.info(f"Image processing complete: {output_pdf_path}")
            return True
        else:
            logger.error("No images available to create PDF")
            return False

    except Exception as e:
        logger.error(f"Image processing failed: {e}", exc_info=True)
        return False