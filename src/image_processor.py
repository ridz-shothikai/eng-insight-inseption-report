# src/image_processor.py

import os
import json
import logging
import time
import shutil
import math
import requests
import polyline
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------- Image Categories ---------------- #

IMAGE_CATEGORIES = {
    "executive_summary": "High-level project overview, objectives, key highlights, summary graphics",
    "introduction": "Project background, context, stakeholders, introductory visuals",
    "site_appreciation": "Site location, geography, terrain, conditions, access, site photos",
    "methodology": "Construction approach, techniques, procedures, technical processes, workflow diagrams",
    "task_assignment": "Team structure, roles, responsibilities, manning, organizational charts",
    "cross_sections": "Engineering designs, drawings, technical specifications, cross-sectional views",
    "design_standards": "Standards, codes, compliance requirements, specifications, technical standards",
    "work_programme": "Timeline, schedule, phases, milestones, activities, Gantt charts",
    "development": "Implementation strategy, staging, phasing, development plans",
    "quality_assurance": "QA/QC plans, testing, inspections, procedures, quality control",
    "checklists": "Operational checklists, verification lists, compliance checks",
    "summary_conclusion": "Summary, recommendations, conclusions, final overview",
    "compliances": "Regulatory compliance, legal requirements, certifications, permits"
}


# ---------------- Distance & Route Helpers ---------------- #

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate Haversine distance in kilometers"""
    R = 6371
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat, delta_lng = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lng / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_route_from_google(
    start_lat: float, 
    start_lng: float, 
    end_lat: float, 
    end_lng: float
) -> tuple:
    """Get route coordinates using Google Routes API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("Google API key not found, using fallback straight line")
        return (
            [(start_lat, start_lng), (end_lat, end_lng)],
            calculate_distance(start_lat, start_lng, end_lat, end_lng),
            0
        )
    
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
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
            distance = route["distanceMeters"] / 1000
            duration_sec = int(route["duration"].rstrip("s"))
            duration_min = duration_sec / 60
            
            logger.info(f"Route fetched: {distance:.2f} km, {duration_min:.1f} min")
            return coords, distance, duration_min
    
    except Exception as e:
        logger.warning(f"Google Routes API failed: {e}, using fallback")
    
    fallback_coords = [(start_lat, start_lng), (end_lat, end_lng)]
    fallback_distance = calculate_distance(start_lat, start_lng, end_lat, end_lng)
    
    return fallback_coords, fallback_distance, 0


def sample_route_coords(route_coords: List[tuple], max_points: int = 80) -> List[tuple]:
    """Reduce route points to avoid URL length limits"""
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
    size: str = "1920x1080"
) -> Optional[Image.Image]:
    """Generate route map image using Google Static Maps API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Google API key not configured")
        return None
    
    try:
        route_coords, distance, duration = get_route_from_google(
            start_lat, start_lng, end_lat, end_lng
        )
        
        sampled_coords = sample_route_coords(route_coords)
        path = "|".join([f"{lat},{lng}" for lat, lng in sampled_coords])
        
        markers_start = f"color:green|label:A|{start_lat},{start_lng}"
        markers_end = f"color:red|label:B|{end_lat},{end_lng}"
        
        params = {
            "size": size,
            "markers": [markers_start, markers_end],
            "path": f"color:0x0000ff|weight:5|{path}",
            "key": api_key
        }
        
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/staticmap",
            params=params,
            timeout=30
        )
        
        if resp.status_code == 200:
            logger.info(f"Route map image generated successfully")
            img = Image.open(BytesIO(resp.content))
            return img
        else:
            logger.warning(f"Failed to fetch route image: {resp.status_code}")
            return None
    
    except Exception as e:
        logger.error(f"Error generating route image: {e}")
        return None


# ---------------- Image Classifier ---------------- #

class ImageClassifier:
    """Classify images using Google Gemini Vision API"""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        logger.info("ImageClassifier initialized with Gemini model")
    
    def _load_image_for_classification(self, image_path: str) -> Optional[Image.Image]:
        """Load and prepare image for API"""
        try:
            img = Image.open(image_path)
            
            # Resize if too large (to save API quota)
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _build_classification_prompt(self) -> str:
        """Build prompt for image classification"""
        categories_text = "\n".join([
            f"{idx + 1}. {cat}: {desc}"
            for idx, (cat, desc) in enumerate(IMAGE_CATEGORIES.items())
        ])
        
        prompt = f"""You are an expert in analyzing construction and engineering project documentation images.

Analyze this image and classify it into ONE of the following categories based on its content:

{categories_text}

**Instructions:**
- Look at the image content carefully
- Identify key visual elements (site photos, diagrams, charts, text, technical drawings, etc.)
- Determine which category best fits the image
- Respond with ONLY the category name (e.g., "site_appreciation")
- Do NOT include any explanation or additional text
- If the image doesn't clearly fit any category, use the most relevant one

Category:"""
        
        return prompt
    
    def _build_caption_prompt(self) -> str:
        """Build prompt for generating image caption"""
        prompt = """You are an expert in analyzing construction and engineering project documentation images.

Analyze this image and provide a brief, descriptive caption (maximum 10-12 words) that describes what is shown in the image.

**Instructions:**
- Be concise and specific
- Focus on the main subject or feature visible in the image
- Use professional engineering/construction terminology
- Keep it under 12 words
- Do NOT include quotation marks in your response
- Respond with ONLY the caption text, no prefix like "Caption:"

Caption:"""
        
        return prompt
    
    def classify_image(self, image_path: str, max_retries: int = 3) -> str:
        """
        Classify a single image
        
        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Category name as string
        """
        img = self._load_image_for_classification(image_path)
        if not img:
            logger.warning(f"Could not load image, defaulting to 'site_appreciation'")
            return "site_appreciation"
        
        prompt = self._build_classification_prompt()
        
        for attempt in range(max_retries):
            try:
                # Add delay to respect rate limits
                time.sleep(1)
                
                # Generate classification
                response = self.model.generate_content([prompt, img])
                category = response.text.strip().lower()
                
                # Clean up response
                category = category.replace('"', '').replace("'", "").strip()
                
                # Validate category
                if category in IMAGE_CATEGORIES:
                    logger.info(f"✓ Classified: {Path(image_path).name} → {category}")
                    return category
                else:
                    logger.warning(f"Invalid category '{category}', trying again...")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        logger.warning(f"Max retries reached, defaulting to 'site_appreciation'")
                        return "site_appreciation"
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s... (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Classification error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        return "site_appreciation"
        
        return "site_appreciation"
    
    def generate_caption(self, image_path: str, max_retries: int = 3) -> str:
        """
        Generate a caption for a single image
        
        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Caption text as string
        """
        img = self._load_image_for_classification(image_path)
        if not img:
            logger.warning(f"Could not load image for caption generation")
            return "Image description unavailable"
        
        prompt = self._build_caption_prompt()
        
        for attempt in range(max_retries):
            try:
                # Add delay to respect rate limits
                time.sleep(1)
                
                # Generate caption
                response = self.model.generate_content([prompt, img])
                caption = response.text.strip()
                
                # Clean up response - remove quotes and common prefixes
                caption = caption.replace('"', '').replace("'", "").strip()
                caption = caption.replace("Caption:", "").strip()
                
                logger.info(f"✓ Caption generated: {Path(image_path).name} → {caption}")
                return caption
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s... (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Caption generation error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        return "Image description unavailable"
        
        return "Image description unavailable"


# ---------------- Image Processing Functions ---------------- #

def add_metadata_to_image(
    image_path: str,
    output_path: str,
    coordinate_data: Dict,
    metadata_text: Optional[str] = None
) -> str:
    """
    Add metadata overlay to an image
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        coordinate_data: Dictionary with start and end coordinates
        metadata_text: Optional custom text to add
        
    Returns:
        Path to processed image
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Simply save the image without any overlay
        img.save(output_path, quality=95)
        logger.info(f"Processed image saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        # If processing fails, just copy the original
        shutil.copy2(image_path, output_path)
        return output_path


def process_images(
    image_paths: List[str],
    coordinate_data: Dict,
    output_dir: str,
    classified_images_json_path: str,
    session_id: str,
    progress_store: Dict
):
    """
    Process images: add metadata, classify them, generate captions, and save as individual image files.
    Also generates and processes a route map image if coordinates are provided.
    """
    try:
        logger.info(f"Session {session_id}: Starting image processing")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize classifier
        classifier = ImageClassifier()
        
        # --- Generate route map if coordinates provided ---
        all_image_paths = list(image_paths)  # Start with user images
        
        if coordinate_data and "start" in coordinate_data and "end" in coordinate_data:
            start = coordinate_data["start"]
            end = coordinate_data["end"]
            route_img = get_route_image(
                start["latitude"], start["longitude"],
                end["latitude"], end["longitude"]
            )
            
            if route_img:
                # Standardize route image size (pad to 1920x1080)
                if route_img.mode not in ("RGB", "RGBA"):
                    route_img = route_img.convert("RGB")
                
                route_img = ImageOps.pad(
                    route_img,
                    (1920, 1080),
                    method=Image.Resampling.LANCZOS,
                    color=(255, 255, 255)
                )
                
                # Save route image
                route_image_path = output_path / "route_map_processed_000.jpg"
                route_img.save(route_image_path, quality=95)
                all_image_paths.insert(0, str(route_image_path))  # Process first
                logger.info(f"Route map saved: {route_image_path}")
            else:
                logger.warning("Failed to generate route map")
        
        total_to_process = len(all_image_paths)
        processed_results = []
        
        # Process all images (including route map)
        for idx, image_path in enumerate(all_image_paths):
            logger.info(f"Session {session_id}: Processing image {idx + 1}/{total_to_process}")
            
            # Determine output filename
            original_ext = Path(image_path).suffix.lower()
            if original_ext not in ['.jpg', '.jpeg', '.png']:
                original_ext = '.jpg'
            
            if "route_map" in Path(image_path).name:
                output_filename = "route_map_processed_000.jpg"
            else:
                original_filename = Path(image_path).stem
                if '_' in original_filename:
                    parts = original_filename.split('_', 1)
                    if parts[0].isdigit():
                        original_filename = parts[1]
                output_filename = f"{original_filename}_processed_{idx+1:03d}{original_ext}"
            
            output_image_path = output_path / output_filename
            
            # Add metadata
            metadata_text = f"Image {idx + 1}/{total_to_process}"
            if "route_map" in Path(image_path).name:
                metadata_text = "Project Route Map"
            
            processed_path = add_metadata_to_image(
                image_path=image_path,
                output_path=str(output_image_path),
                coordinate_data=coordinate_data,
                metadata_text=metadata_text
            )
            
            # Classify and caption
            category = classifier.classify_image(processed_path)
            caption = classifier.generate_caption(processed_path)
            
            # Store result with relative path
            relative_img_path = f"processed_images/{output_filename}"
            processed_results.append({
                "category": category,
                "img_path": relative_img_path,
                "caption": caption
            })
            
            # Update progress
            progress = ((idx + 1) / total_to_process) * 100
            progress_store[session_id]["images"] = progress
            
            logger.info(f"Session {session_id}: Image {idx + 1}/{total_to_process} processed")
            logger.info(f"  Category: {category}")
            logger.info(f"  Caption: {caption}")
        
        # Save classification results
        with open(classified_images_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session {session_id}: All images processed. Results saved to {classified_images_json_path}")
        logger.info(f"Session {session_id}: Processed images saved to {output_dir}")
        
        # Print summary
        from collections import Counter
        categories = [item["category"] for item in processed_results]
        category_counts = Counter(categories)
        
        logger.info(f"\nSession {session_id}: Classification Summary:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} images")
        
        progress_store[session_id]["images"] = 100.0
        
    except Exception as e:
        logger.error(f"Session {session_id}: Error in image processing - {str(e)}", exc_info=True)
        raise


# ---------------- Standalone Usage ---------------- #

# if __name__ == "__main__":
#     # Example usage
#     test_images = ["test_image1.jpg", "test_image2.jpg"]
#     test_coords = {
#         "start": {"latitude": 23.8103, "longitude": 90.4125},
#         "end": {"latitude": 23.8203, "longitude": 90.4225}
#     }
#     
#     progress = {"test": {"images": 0.0}}
#     
#     process_images(
#         image_paths=test_images,
#         coordinate_data=test_coords,
#         output_dir="processed_images",
#         classified_images_json_path="classified_images.json",
#         session_id="test",
#         progress_store=progress
#     )