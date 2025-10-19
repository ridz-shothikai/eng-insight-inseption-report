# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
import logging
from datetime import datetime
import asyncio
import json

# Import your processing modules
from src.ocr import process_ocr
from src.chunking import chunk_text
from src.classifier import classify_chunks
from src.report_generator import generate_inception_report
from src.image_processor import process_images
from src.utils.file_handler import cleanup_old_files, ensure_directories
from src.utils.validators import validate_coordinates
from src.utils.log_streamer import log_streamer, SessionLogHandler

#Import Mongodb packages
from src.database.mongodb import mongodb, get_database
from src.database.crud import session_crud, markdown_crud, file_crud
from bson import ObjectId
import json

# ---------------------------
# Logging setup
# ---------------------------
# Ensure logs directory exists
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add session log handler for streaming
session_log_handler = SessionLogHandler(log_streamer)
session_log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(session_log_handler)

# ---------------------------
# Session logging helper
# ---------------------------
def log_with_session(message: str, session_id: str, level=logging.INFO):
    """Helper to log with session context"""
    logger.log(level, message, extra={'session_id': session_id})

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="RFP Processing API",
    description="API for processing RFP documents and generating reports",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Directories
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
RFP_UPLOAD_DIR = UPLOAD_DIR / "rfp_documents"
IMAGE_UPLOAD_DIR = UPLOAD_DIR / "user_images"

ensure_directories([
    UPLOAD_DIR,
    OUTPUT_DIR,
    RFP_UPLOAD_DIR,
    IMAGE_UPLOAD_DIR,
    BASE_DIR / "logs"
])

# ---------------------------
# In-memory progress store
# ---------------------------
progress_store: Dict[str, Dict[str, float]] = {}
# Structure: progress_store[session_id] = {
#     "ocr": 0.0, "images": 0.0, "chunking": 0.0, "classification": 0.0,
#     "report": 0.0, "completed": 0.0
# }

# ---------------------------
# In-memory markdown store
# ---------------------------
markdown_store: Dict[str, Dict[str, str]] = {}
# Structure: markdown_store[session_id] = {
#     "executive_summary": "markdown content...",
#     "introduction": "markdown content...",
#     ...
# }

# ---------------------------
# Custom Functions
# ---------------------------

class SessionUpdateRequest(BaseModel):
    status: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    coordinate_data: Optional[Dict[str, Any]] = None
    original_files: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        extra = "forbid"  # Reject any fields not explicitly defined

def create_markdown_stream_handler(session_id: str):
    """Create a callback function to handle markdown streaming AND accumulation for a session"""
    async def stream_handler(section_id: str, chunk: str):
        # Accumulate markdown in memory store (for real-time access)
        if session_id not in markdown_store:
            markdown_store[session_id] = {}
        if section_id not in markdown_store[session_id]:
            markdown_store[session_id][section_id] = ""
        markdown_store[session_id][section_id] += chunk
        
        # Save to MongoDB (for persistence)
        try:
            await markdown_crud.save_markdown_section(
                session_id, section_id, markdown_store[session_id][section_id]
            )
        except Exception as e:
            logger.error(f"Error saving markdown to MongoDB: {e}")
        
        # Also stream it via log_streamer for real-time display
        log_streamer.broadcast(
            session_id, 
            f"MARKDOWN||{section_id}||{chunk}"
        )
    return stream_handler

async def create_indexes():
    """Create MongoDB indexes"""
    db = get_database()
    
    # Session indexes
    await db.sessions.create_index("session_id", unique=True)
    await db.sessions.create_index("created_at")
    await db.sessions.create_index("status")
    
    # Markdown indexes
    await db.markdown_sections.create_index([("session_id", 1), ("section_id", 1)], unique=True)
    await db.markdown_sections.create_index("session_id")
    
    # File indexes
    await db.processed_files.create_index([("session_id", 1), ("file_type", 1)])
    await db.processed_files.create_index("session_id")
    
    logger.info("✅ MongoDB indexes created")


# ---------------------------
# Startup & Shutdown events
# ---------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting RFP Processing API...")

    # Connect to MongoDB
    await mongodb.connect()

    # Create indexes
    await create_indexes()

    cleanup_old_files(OUTPUT_DIR, days=7)
    cleanup_old_files(UPLOAD_DIR, days=7)
    
    # Start background tasks
    asyncio.create_task(cleanup_old_progress())
    asyncio.create_task(cleanup_old_history()) 

async def cleanup_old_progress():
    """Cleanup completed sessions from progress_store after 1 hour"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        current_time = datetime.now()
        to_remove = []
        
        for session_id, progress in progress_store.items():
            if progress.get("completed", 0) >= 100:
                # Extract timestamp from session_id
                try:
                    session_time = datetime.strptime(session_id[:15], "%Y%m%d_%H%M%S")
                    if (current_time - session_time).total_seconds() > 3600:  # 1 hour old
                        to_remove.append(session_id)
                except:
                    pass
        
        for session_id in to_remove:
            progress_store.pop(session_id, None)
            logger.info(f"Cleaned up old progress for session: {session_id}")

async def cleanup_old_history():
    """Cleanup old log history periodically"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        try:
            removed = log_streamer.cleanup_old_history()
            if removed > 0:
                logger.info(f"Cleaned up history for {removed} old sessions")
        except Exception as e:
            logger.error(f"Error cleaning up history: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down RFP Processing API...")
    await mongodb.close()

# ---------------------------
# Root & Health
# ---------------------------
@app.get("/")
async def root():
    return {"message": "RFP Processing API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ---------------------------
# Main RFP processing endpoint
# ---------------------------
async def log_progress(session_id: str):
    while session_id in progress_store and progress_store[session_id]["completed"] < 100.0:
        log_with_session(f"Progress: {progress_store[session_id]}", session_id)
        await asyncio.sleep(5)  # log every 5 seconds

@app.post("/process-rfp")
async def process_rfp(
    rfp_document: UploadFile = File(..., description="RFP document (PDF/Image)"),
    start_latitude: float = Form(...),
    start_longitude: float = Form(...),
    end_latitude: float = Form(...),
    end_longitude: float = Form(...),
    images: List[UploadFile] = File(...),
):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_with_session("🚀 Starting RFP processing", session_id)
    
    try:
        # ---------------------------
        # Validation
        # ---------------------------
        if not validate_coordinates(start_latitude, start_longitude, end_latitude, end_longitude):
            raise HTTPException(status_code=400, detail="Invalid coordinates provided")

        allowed_rfp_types = ["application/pdf", "image/jpeg", "image/png", "image/tiff"]
        if rfp_document.content_type not in allowed_rfp_types:
            raise HTTPException(status_code=400, detail="Invalid RFP document type")

        allowed_image_types = ["image/jpeg", "image/png", "image/jpg"]
        for img in images:
            if img.content_type not in allowed_image_types:
                raise HTTPException(status_code=400, detail=f"Invalid image type for {img.filename}")

        log_with_session(f"✓ Validation passed: {len(images)} images, coordinates validated", session_id)

        # ---------------------------
        # Directories
        # ---------------------------
        session_upload_dir = RFP_UPLOAD_DIR / session_id
        session_image_dir = IMAGE_UPLOAD_DIR / session_id
        session_output_dir = OUTPUT_DIR / session_id
        processed_images_dir = session_output_dir / "processed_images"
        ensure_directories([session_upload_dir, session_image_dir, session_output_dir, processed_images_dir])

        # Save RFP
        rfp_path = session_upload_dir / rfp_document.filename
        with open(rfp_path, "wb") as f:
            shutil.copyfileobj(rfp_document.file, f)
        log_with_session(f"📄 Saved RFP document: {rfp_document.filename}", session_id)

        # Save images
        image_paths = []
        for idx, image in enumerate(images):
            img_path = session_image_dir / f"{idx}_{image.filename}"
            with open(img_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            image_paths.append(img_path)
        log_with_session(f"📸 Saved {len(images)} images", session_id)

        # ---------------------------
        # Initialize progress
        # ---------------------------
        progress_store[session_id] = {
            "ocr": 0.0,
            "images": 0.0,
            "chunking": 0.0,
            "classification": 0.0,
            "report": 0.0,
            "completed": 0.0
        }

        markdown_store[session_id] = {} 

        coordinate_data = {
            "start": {"latitude": start_latitude, "longitude": start_longitude},
            "end": {"latitude": end_latitude, "longitude": end_longitude}
        }

        # ---------------------------
        # CREATE SESSION IN MONGODB FIRST (BEFORE BACKGROUND PROCESSING)
        # ---------------------------
        session_data = {
            "session_id": session_id,
            "status": "processing",
            "progress": {
                "ocr": 0.0, "images": 0.0, "chunking": 0.0, 
                "classification": 0.0, "report": 0.0, "completed": 0.0
            },
            "coordinate_data": {
                "start": {"latitude": start_latitude, "longitude": start_longitude},
                "end": {"latitude": end_latitude, "longitude": end_longitude}
            },
            "original_files": {
                "rfp": [rfp_document.filename],
                "images": [img.filename for img in images]
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Save session to MongoDB immediately
        session_db_id = await session_crud.create_session(session_data)
        logger.info(f"✅ Session created in MongoDB with ID: {session_db_id}")

        # ---------------------------
        # Start processing in background
        # ---------------------------
        asyncio.create_task(process_rfp_background(
            session_id=session_id,
            rfp_path=rfp_path,
            image_paths=image_paths,
            coordinate_data=coordinate_data,
            ocr_output_path=session_output_dir / "ocr_output.txt",
            classified_images_json_path=session_output_dir / "classified_images.json",
            chunked_output_path=session_output_dir / "chunked_output.json",
            classified_output_path=session_output_dir / "classified_output.json",
            inception_pdf_path=session_output_dir / "inception.pdf",
            processed_images_dir=processed_images_dir
        ))

        # ---------------------------
        # Return session ID immediately for redirection
        # ---------------------------
        return JSONResponse(
            status_code=202,
            content={
                "message": "RFP processing started successfully",
                "session_id": session_id,
                "progress_url": f"/progress/{session_id}",
                "status": "processing"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        log_with_session(f"❌ Error: {str(e)}", session_id, logging.ERROR)
        # Even if there's an error, try to save the failed session to MongoDB
        try:
            error_session_data = {
                "session_id": session_id,
                "status": "failed",
                "progress": {"completed": -1},
                "coordinate_data": {
                    "start": {"latitude": start_latitude, "longitude": start_longitude},
                    "end": {"latitude": end_latitude, "longitude": end_longitude}
                },
                "original_files": {
                    "rfp": [rfp_document.filename if 'rfp_document' in locals() else "unknown"],
                    "images": [img.filename for img in images] if 'images' in locals() else []
                },
                "error": str(e),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            await session_crud.create_session(error_session_data)
        except:
            pass  # If MongoDB save fails, at least we tried
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# ---------------------------
# Background processing task
# ---------------------------
async def process_rfp_background(
    session_id: str,
    rfp_path: Path,
    image_paths: List[Path],
    coordinate_data: Dict,
    ocr_output_path: Path,
    classified_images_json_path: Path,
    chunked_output_path: Path,
    classified_output_path: Path,
    inception_pdf_path: Path,
    processed_images_dir: Path
):
    """Background task to process RFP without blocking the main request"""
    try:
        # Start progress logging
        asyncio.create_task(log_progress(session_id))
        
        log_with_session("⚡ Running OCR and image processing in parallel", session_id)

        # Update progress in MongoDB
        await session_crud.update_session_progress(session_id, {
            "ocr": 0.0, "images": 0.0, "chunking": 0.0, 
            "classification": 0.0, "report": 0.0, "completed": 0.0
        })
        
        # Parallel tasks: OCR + Images
        await asyncio.gather(
            asyncio.to_thread(process_ocr, str(rfp_path), str(ocr_output_path), session_id, progress_store),
            asyncio.to_thread(
                process_images,
                [str(p) for p in image_paths],
                coordinate_data,
                str(processed_images_dir),
                str(classified_images_json_path),
                session_id,
                progress_store
            )
        )

        # Update progress
        progress_store[session_id]["images"] = 100.0
        await session_crud.update_session_progress(session_id, progress_store[session_id])

        if not ocr_output_path.exists():
            raise Exception("OCR processing failed")
        if not classified_images_json_path.exists():
            raise Exception("Image classification failed")
        if not processed_images_dir.exists() or not any(processed_images_dir.iterdir()):
            raise Exception("Image processing failed - no images generated")

        # Save file metadata to MongoDB
        try:
            if ocr_output_path.exists():
                await file_crud.save_file_metadata(
                    session_id, "ocr", str(ocr_output_path), ocr_output_path.stat().st_size
                )
            if classified_images_json_path.exists():
                await file_crud.save_file_metadata(
                    session_id, "classified_images", str(classified_images_json_path), 
                    classified_images_json_path.stat().st_size
                )
        except Exception as e:
            logger.error(f"Error saving file metadata to MongoDB: {e}")

        log_with_session("✓ Parallel processing completed", session_id)

        # Sequential dependent steps
        log_with_session("📝 Starting text chunking", session_id)
        progress_store[session_id]["chunking"] = 0.0
        await asyncio.to_thread(
            chunk_text,
            input_path=str(ocr_output_path),
            output_path=str(chunked_output_path),
            chunk_size=512,
            overlap=50,
            session_id=session_id,
            progress_store=progress_store
        )
        progress_store[session_id]["chunking"] = 100.0
        
        # Save chunked file metadata
        if chunked_output_path.exists():
            await file_crud.save_file_metadata(
                session_id, "chunked", str(chunked_output_path), chunked_output_path.stat().st_size
            )
        
        log_with_session("✓ Text chunking completed", session_id)

        log_with_session("🏷️ Starting chunk classification", session_id)
        progress_store[session_id]["classification"] = 0.0
        await asyncio.to_thread(classify_chunks, str(chunked_output_path), str(classified_output_path), 5, session_id, progress_store)
        progress_store[session_id]["classification"] = 100.0
        
        # Save classified file metadata
        if classified_output_path.exists():
            await file_crud.save_file_metadata(
                session_id, "classified", str(classified_output_path), classified_output_path.stat().st_size
            )
        
        log_with_session("✓ Chunk classification completed", session_id)

        log_with_session("📊 Generating inception report", session_id)
        progress_store[session_id]["report"] = 0.0

        # Create markdown stream handler
        markdown_stream_handler = create_markdown_stream_handler(session_id)

        await asyncio.to_thread(
            generate_inception_report,
            str(classified_output_path),
            str(inception_pdf_path),
            str(ocr_output_path),
            session_id,
            progress_store,
            stream_callback=markdown_stream_handler
        )
        progress_store[session_id]["report"] = 100.0
        progress_store[session_id]["completed"] = 100.0
        
        # Save final report metadata
        if inception_pdf_path.exists():
            await file_crud.save_file_metadata(
                session_id, "inception", str(inception_pdf_path), inception_pdf_path.stat().st_size
            )
        
        # Update final status in MongoDB
        await session_crud.update_session_status(session_id, "completed")
        await session_crud.update_session_progress(session_id, progress_store[session_id])
        
        log_with_session("✅ Processing complete! Report generated", session_id)

    except Exception as e:
        log_with_session(f"❌ Background processing error: {str(e)}", session_id, logging.ERROR)
        # Update MongoDB with error status
        await session_crud.update_session_status(session_id, "failed", str(e))
        # Optionally update progress to indicate failure
        progress_store[session_id]["error"] = str(e)
        progress_store[session_id]["completed"] = -1  # Indicate failure


############################
#add image url endpoint
############################

@app.get("/route-url/{session_id}")
async def get_route_url(session_id: str):
    """Get the Google Maps route image URL for a session"""
    try:
        # Check if session exists
        if session_id not in progress_store:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Path to the classified images JSON
        classified_images_path = OUTPUT_DIR / session_id / "classified_images.json"
        
        if not classified_images_path.exists():
            raise HTTPException(status_code=404, detail="Route data not found for this session")
        
        # Load the JSON data
        with open(classified_images_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract route URL
        route_url = data.get("route_image_url")
        
        if not route_url:
            raise HTTPException(status_code=404, detail="Route URL not available")
        
        return {
            "session_id": session_id,
            "route_image_url": route_url,
            "message": "Route URL retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting route URL for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving route URL: {str(e)}")

# ---------------------------
# Progress endpoint
# ---------------------------
@app.get("/progress/{session_id}")
async def get_progress(session_id: str):
    if session_id not in progress_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return progress_store[session_id]

# ---------------------------
# Download intermediate files
# ---------------------------
@app.get("/download/{session_id}/{file_type}")
async def download_intermediate_file(session_id: str, file_type: str):
    file_mapping = {
        "ocr": "ocr_output.txt",
        "chunked": "chunked_output.json",
        "classified": "classified_output.json",
        "inception": "inception.pdf",
        "classified_images": "classified_images.json",
        "processed_images": "processed_images"
    }
    
    # Add media type mapping for proper content type headers
    media_type_mapping = {
        "ocr": "text/plain",
        "chunked": "application/json",
        "classified": "application/json",
        "inception": "application/pdf",
        "classified_images": "application/json"
    }
    
    if file_type not in file_mapping:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # SPECIAL CASE: For "inception" file type - try session-specific first, then static fallback
    if file_type == "inception":
        # First try session-specific file
        session_file_path = OUTPUT_DIR / session_id / "inception.pdf"
        if session_file_path.exists():
            file_path = session_file_path
        else:
            # Fallback to static file
            file_path = BASE_DIR / "inception" / "inception.pdf"
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="Inception PDF not found")
    
    # For all other file types, use the session-specific files
    else:
        file_path = OUTPUT_DIR / session_id / file_mapping[file_type]
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Handle directory download (processed images)
    if file_type == "processed_images" and file_path.is_dir():
        image_files = [f for f in file_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']]
        
        if not image_files:
            raise HTTPException(status_code=404, detail="No processed images found")
        
        zip_buffer = BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for image_file in image_files:
                    zip_file.write(image_file, arcname=image_file.name)
            
            zip_buffer.seek(0)
            
            return StreamingResponse(
                content=zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename=processed_images_{session_id}.zip"
                }
            )
        except Exception as e:
            logger.error(f"Error creating zip for session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error creating download package")
    
    # Handle single file download with proper media type
    return FileResponse(
        path=str(file_path), 
        filename=file_path.name,  # Use actual file name to handle both session and static files
        media_type=media_type_mapping.get(file_type, "application/octet-stream"),
        headers={
            "Content-Disposition": f"attachment; filename={file_path.name}"
        }
    )

# ---------------------------
# Get all sessions endpoint
# ---------------------------
@app.get("/sessions")
async def get_all_sessions():
    """Get all active session IDs and their progress"""
    try:
        # Get sessions from MongoDB
        mongo_sessions = await session_crud.get_all_sessions(limit=100)
        
        # Convert ObjectId to string for JSON serialization
        serialized_sessions = []
        for session in mongo_sessions:
            # Convert ObjectId to string
            if '_id' in session:
                session['_id'] = str(session['_id'])
            serialized_sessions.append(session)
        
        # Also include in-memory sessions for backward compatibility
        memory_sessions = []
        for session_id, progress in progress_store.items():
            memory_sessions.append({
                "session_id": session_id,
                "progress": progress,
                "is_complete": progress.get("completed", 0) >= 100.0,
                "source": "memory"
            })
        
        return {
            "mongo_sessions": serialized_sessions,
            "memory_sessions": memory_sessions,
            "mongo_count": len(serialized_sessions),
            "memory_count": len(memory_sessions)
        }
        
    except Exception as e:
        logger.error(f"Error fetching sessions from MongoDB: {e}")
        # Fallback to memory store only
        if not progress_store:
            return {"sessions": [], "count": 0}
        
        sessions = []
        for session_id, progress in progress_store.items():
            sessions.append({
                "session_id": session_id,
                "progress": progress,
                "is_complete": progress.get("completed", 0) >= 100.0
            })
        
        return {
            "sessions": sessions,
            "count": len(sessions),
            "note": "Using memory store due to MongoDB error"
        }

# ---------------------------
# Get specific session details
# ---------------------------
@app.get("/sessions/{session_id}")
async def get_mongo_session(session_id: str):
    """Get a specific session from MongoDB"""
    try:
        session = await session_crud.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found in MongoDB")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session from MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching session: {str(e)}")
    
# ---------------------------
# Session CRUD operations
# ---------------------------

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its associated data"""
    try:
        # Check if session exists
        session = await session_crud.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete session from MongoDB
        await session_crud.delete_session(session_id)
        
        # Delete associated markdown sections
        await markdown_crud.delete_session_markdown(session_id)
        
        # Delete associated file metadata
        await file_crud.delete_session_files(session_id)
        
        # Clean up local files
        session_dirs = [
            RFP_UPLOAD_DIR / session_id,
            IMAGE_UPLOAD_DIR / session_id,
            OUTPUT_DIR / session_id
        ]
        for dir_path in session_dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)
        
        # Clean up in-memory stores
        progress_store.pop(session_id, None)
        markdown_store.pop(session_id, None)
        log_streamer.clear_history(session_id)
        
        logger.info(f"🗑️ Session deleted: {session_id}")
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.put("/sessions/{session_id}")
async def update_session(
    session_id: str,
    updates: SessionUpdateRequest
):
    """Update session metadata"""
    try:
        session = await session_crud.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Convert to dict, keeping only provided fields
        update_dict = updates.dict(exclude_unset=True)
        
        allowed_updates = {
            "status", "progress", "coordinate_data", 
            "original_files", "error"
        }
        
        filtered_updates = {
            k: v for k, v in update_dict.items() 
            if k in allowed_updates and v is not None
        }
        
        if not filtered_updates:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        
        filtered_updates["updated_at"] = datetime.now()
        
        # Update MongoDB
        await session_crud.sessions.update_one(
            {"session_id": session_id},
            {"$set": filtered_updates}
        )
        
        # Update in-memory progress if needed
        if "progress" in filtered_updates and session_id in progress_store:
            progress_store[session_id].update(filtered_updates["progress"])
        
        logger.info(f"📝 Session updated: {session_id}")
        return {"message": f"Session {session_id} updated successfully", "updates": filtered_updates}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

@app.post("/sessions/{session_id}/clone")
async def clone_session(session_id: str, new_session_id: str = None):
    """Clone an existing session with a new session ID"""
    try:
        # Get original session
        original_session = await session_crud.get_session(session_id)
        if not original_session:
            raise HTTPException(status_code=404, detail="Original session not found")
        
        # Generate new session ID if not provided
        if not new_session_id:
            new_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if new session ID already exists
        existing_session = await session_crud.get_session(new_session_id)
        if existing_session:
            raise HTTPException(status_code=400, detail="New session ID already exists")
        
        # Create cloned session data
        cloned_session = original_session.copy()
        cloned_session["session_id"] = new_session_id
        cloned_session["status"] = "cloned"
        cloned_session["progress"] = {"completed": 0.0}  # Reset progress
        cloned_session["created_at"] = datetime.now()
        cloned_session["updated_at"] = datetime.now()
        
        # Remove MongoDB _id to create new document
        if "_id" in cloned_session:
            del cloned_session["_id"]
        
        # Save cloned session to MongoDB
        await session_crud.create_session(cloned_session)
        
        # Clone markdown sections if they exist
        try:
            original_markdown = await markdown_crud.get_session_markdown(session_id)
            for section_id, content in original_markdown.items():
                await markdown_crud.save_markdown_section(new_session_id, section_id, content)
        except Exception as e:
            logger.warning(f"Could not clone markdown sections: {e}")
        
        # Clone file metadata
        try:
            original_files = await file_crud.get_session_files(session_id)
            for file_meta in original_files:
                # Create new file metadata with new session ID
                new_file_meta = file_meta.copy()
                new_file_meta["session_id"] = new_session_id
                new_file_meta["created_at"] = datetime.now()
                if "_id" in new_file_meta:
                    del new_file_meta["_id"]
                
                await file_crud.processed_files.insert_one(new_file_meta)
        except Exception as e:
            logger.warning(f"Could not clone file metadata: {e}")
        
        # Initialize in-memory stores for new session
        progress_store[new_session_id] = {"completed": 0.0}
        markdown_store[new_session_id] = {}
        
        logger.info(f"🌀 Session cloned: {session_id} -> {new_session_id}")
        return {
            "message": f"Session cloned successfully",
            "original_session_id": session_id,
            "new_session_id": new_session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cloning session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cloning session: {str(e)}")

@app.post("/sessions/merge")
async def merge_sessions(session_ids: List[str], new_session_id: str = None):
    """Merge multiple sessions into a new session"""
    try:
        if len(session_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 sessions required for merging")
        
        # Generate new session ID if not provided
        if not new_session_id:
            new_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if new session ID already exists
        existing_session = await session_crud.get_session(new_session_id)
        if existing_session:
            raise HTTPException(status_code=400, detail="New session ID already exists")
        
        # Get all source sessions
        source_sessions = []
        for session_id in session_ids:
            session = await session_crud.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            source_sessions.append(session)
        
        # Create merged session data
        merged_session = {
            "session_id": new_session_id,
            "status": "merged",
            "progress": {"completed": 100.0},  # Mark as completed since we're merging existing data
            "coordinate_data": source_sessions[0].get("coordinate_data", {}),  # Use first session's coordinates
            "original_files": {
                "rfp": [],
                "images": []
            },
            "source_sessions": session_ids,  # Track which sessions were merged
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Aggregate files from all source sessions
        for session in source_sessions:
            original_files = session.get("original_files", {})
            merged_session["original_files"]["rfp"].extend(original_files.get("rfp", []))
            merged_session["original_files"]["images"].extend(original_files.get("images", []))
        
        # Save merged session to MongoDB
        await session_crud.create_session(merged_session)
        
        # Merge markdown sections from all source sessions
        all_markdown = {}
        for session_id in session_ids:
            try:
                session_markdown = await markdown_crud.get_session_markdown(session_id)
                for section_id, content in session_markdown.items():
                    # Append content if section already exists, otherwise create new
                    if section_id in all_markdown:
                        all_markdown[section_id] += f"\n\n--- Merged from {session_id} ---\n\n{content}"
                    else:
                        all_markdown[section_id] = content
            except Exception as e:
                logger.warning(f"Could not merge markdown from {session_id}: {e}")
        
        # Save merged markdown
        for section_id, content in all_markdown.items():
            await markdown_crud.save_markdown_section(new_session_id, section_id, content)
        
        # Initialize in-memory stores
        progress_store[new_session_id] = {"completed": 100.0}
        markdown_store[new_session_id] = all_markdown
        
        logger.info(f"🔗 Sessions merged into: {new_session_id}")
        return {
            "message": f"Sessions merged successfully",
            "merged_session_id": new_session_id,
            "source_sessions": session_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error merging sessions: {str(e)}")

@app.patch("/sessions/{session_id}/metadata")
async def update_session_metadata(
    session_id: str,
    coordinate_data: dict = None,
    original_files: dict = None
):
    """Update specific session metadata (coordinates or files)"""
    try:
        session = await session_crud.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        update_data = {"updated_at": datetime.now()}
        
        if coordinate_data:
            update_data["coordinate_data"] = coordinate_data
        
        if original_files:
            update_data["original_files"] = original_files
        
        if len(update_data) == 1:  # Only updated_at was set
            raise HTTPException(status_code=400, detail="No valid metadata provided for update")
        
        # Update session in MongoDB
        await session_crud.sessions.update_one(
            {"session_id": session_id},
            {"$set": update_data}
        )
        
        logger.info(f"📋 Session metadata updated: {session_id}")
        return {"message": f"Session {session_id} metadata updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session metadata {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating session metadata: {str(e)}")

# ---------------------------
# Log streaming endpoint (SSE)
# ---------------------------
@app.get("/fetch_markdown/{session_id}")
async def fetch_markdown(session_id: str):
    """Stream logs, progress, and markdown for a specific session using Server-Sent Events"""
    
    async def event_generator():
        queue = log_streamer.add_client(session_id)
        last_progress = {}
        
        try:
            # Send initial connection message
            yield {
                "event": "connected",
                "data": json.dumps({"message": f"Connected to log stream for session {session_id}"})
            }
            
            while True:
                try:
                    # Wait for logs with timeout
                    log_message = await asyncio.wait_for(queue.get(), timeout=0.5)
                    
                    # Check if this is a markdown stream message
                    if log_message.startswith("MARKDOWN||"):
                        # Parse markdown stream: "MARKDOWN||section_id||chunk"
                        parts = log_message.split("||", 2)
                        if len(parts) == 3:
                            yield {
                                "event": "markdown",
                                "data": json.dumps({
                                    "section_id": parts[1],
                                    "chunk": parts[2]
                                })
                            }
                            continue
                    
                    # Regular log message
                    yield {
                        "event": "log",
                        "data": json.dumps({"message": log_message})
                    }
                    
                except asyncio.TimeoutError:
                    # No log received, send progress update instead
                    if session_id in progress_store:
                        current_progress = progress_store[session_id]
                        if current_progress != last_progress:
                            yield {
                                "event": "progress", 
                                "data": json.dumps(current_progress)
                            }
                            last_progress = current_progress.copy()
                        
                        # Check if completed
                        if current_progress.get("completed", 0) >= 100:
                            yield {
                                "event": "complete",
                                "data": json.dumps({"message": "Processing complete"})
                            }
                            break
                    
        except asyncio.CancelledError:
            log_streamer.remove_client(session_id, queue)
            raise
        finally:
            log_streamer.remove_client(session_id, queue)
    
    return EventSourceResponse(event_generator())


# ---------------------------
# Get complete markdown endpoint (non-streaming)
# ---------------------------
@app.get("/get_markdown/{session_id}")
async def get_markdown(session_id: str):
    """Fetch complete accumulated markdown for a session"""
    
    # Try to get from MongoDB first, then fallback to memory store
    markdown_data = {}
    
    try:
        # Try MongoDB first
        db_markdown = await markdown_crud.get_session_markdown(session_id)
        if db_markdown:
            markdown_data = db_markdown
        elif session_id in markdown_store:
            markdown_data = markdown_store[session_id]
        else:
            raise HTTPException(status_code=404, detail="Session not found or no markdown generated yet")
    except Exception as e:
        logger.error(f"Error fetching markdown from MongoDB: {e}")
        # Fallback to memory store
        if session_id in markdown_store:
            markdown_data = markdown_store[session_id]
        else:
            raise HTTPException(status_code=404, detail="Session not found or no markdown generated yet")
    
    if not markdown_data:
        raise HTTPException(status_code=404, detail="No markdown data available yet")
    
    section_order = [
        "executive_summary", "introduction", "site_appreciation", "methodology",
        "task_assignment", "cross_sections", "design_standards", "work_programme",
        "development", "quality_assurance", "checklists", "summary_conclusion", "compliances",
        "appendix_irc_codes", "appendix_monsoon", "appendix_equipment",
        "appendix_testing", "appendix_compliance_matrix"
    ]
    
    # Build full markdown
    full_markdown = ""
    completed_sections = 0
    
    for section_id in section_order:
        if section_id in markdown_data and markdown_data[section_id].strip():
            full_markdown += markdown_data[section_id] + "\n\n"
            completed_sections += 1
    
    # Get progress from MongoDB if available
    report_progress = 0
    is_complete = False
    try:
        session_data = await session_crud.get_session(session_id)
        if session_data:
            report_progress = session_data.get("progress", {}).get("report", 0)
            is_complete = session_data.get("status") == "completed"
    except Exception as e:
        logger.error(f"Error fetching session progress from MongoDB: {e}")
        # Fallback to memory store
        if session_id in progress_store:
            report_progress = progress_store[session_id].get("report", 0)
            is_complete = progress_store[session_id].get("completed", 0) >= 100
    
    return {
        "session_id": session_id,
        "markdown": full_markdown.strip(),
        "sections": markdown_data,
        "progress": {
            "completed_sections": completed_sections,
            "total_sections": len(section_order),
            "percentage": (completed_sections / len(section_order)) * 100,
            "report_progress": report_progress,
            "is_complete": is_complete
        }
    }
# ---------------------------
# Cleanup session files
# ---------------------------
@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    try:
        session_dirs = [
            RFP_UPLOAD_DIR / session_id,
            IMAGE_UPLOAD_DIR / session_id,
            OUTPUT_DIR / session_id
        ]
        for dir_path in session_dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)
        
        # Clean up in-memory stores
        progress_store.pop(session_id, None)
        markdown_store.pop(session_id, None)
        log_streamer.clear_history(session_id)
        
        logger.info(f"Cleaned up session: {session_id}")
        return {"message": f"Session {session_id} cleaned up successfully"}
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )