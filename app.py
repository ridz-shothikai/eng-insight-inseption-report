# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import List, Dict
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
# Startup & Shutdown events
# ---------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting RFP Processing API...")
    cleanup_old_files(OUTPUT_DIR, days=7)
    cleanup_old_files(UPLOAD_DIR, days=7)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down RFP Processing API...")

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
        log_with_session(f"Progress: {progress_store[session_id]}", session_id)  # CHANGED
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
    log_with_session("🚀 Starting RFP processing", session_id)  # CHANGED
    asyncio.create_task(log_progress(session_id))

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

        log_with_session(f"✓ Validation passed: {len(images)} images, coordinates validated", session_id)  # ADDED

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
        log_with_session(f"📄 Saved RFP document: {rfp_document.filename}", session_id)  # ADDED

        # Save images
        image_paths = []
        for idx, image in enumerate(images):
            img_path = session_image_dir / f"{idx}_{image.filename}"
            with open(img_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            image_paths.append(img_path)
        log_with_session(f"📸 Saved {len(images)} images", session_id)  # ADDED

        # ---------------------------
        # Prepare output paths
        # ---------------------------
        ocr_output_path = session_output_dir / "ocr_output.txt"
        classified_images_json_path = session_output_dir / "classified_images.json"
        chunked_output_path = session_output_dir / "chunked_output.json"
        classified_output_path = session_output_dir / "classified_output.json"
        inception_pdf_path = session_output_dir / "inception.pdf"

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

        coordinate_data = {
            "start": {"latitude": start_latitude, "longitude": start_longitude},
            "end": {"latitude": end_latitude, "longitude": end_longitude}
        }

        # ---------------------------
        # Parallel tasks: OCR + Images
        # ---------------------------
        log_with_session("⚡ Running OCR and image processing in parallel", session_id)  # CHANGED
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
        progress_store[session_id]["images"] = 100.0

        if not ocr_output_path.exists():
            raise HTTPException(status_code=500, detail="OCR processing failed")
        if not classified_images_json_path.exists():
            raise HTTPException(status_code=500, detail="Image classification failed")
        if not processed_images_dir.exists() or not any(processed_images_dir.iterdir()):
            raise HTTPException(status_code=500, detail="Image processing failed - no images generated")

        log_with_session("✓ Parallel processing completed", session_id)  # ADDED

        # ---------------------------
        # Sequential dependent steps
        # ---------------------------
        log_with_session("📝 Starting text chunking", session_id)  # ADDED
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
        log_with_session("✓ Text chunking completed", session_id)  # ADDED

        log_with_session("🏷️ Starting chunk classification", session_id)  # ADDED
        progress_store[session_id]["classification"] = 0.0
        await asyncio.to_thread(classify_chunks, str(chunked_output_path), str(classified_output_path), 5, session_id, progress_store)
        progress_store[session_id]["classification"] = 100.0
        log_with_session("✓ Chunk classification completed", session_id)  # ADDED

        log_with_session("📊 Generating inception report", session_id)  # ADDED
        progress_store[session_id]["report"] = 0.0
        await asyncio.to_thread(
            generate_inception_report,
            str(classified_output_path),
            str(inception_pdf_path),
            str(ocr_output_path),
            session_id,
            progress_store
        )
        progress_store[session_id]["report"] = 100.0
        progress_store[session_id]["completed"] = 100.0
        log_with_session("✅ Processing complete! Report generated", session_id)  # ADDED

        # ---------------------------
        # Return inception PDF as final output
        # ---------------------------
        return FileResponse(
            path=str(inception_pdf_path),
            media_type="application/pdf",
            filename=f"rfp_report_{session_id}.pdf",
            headers={"Content-Disposition": f"attachment; filename=rfp_report_{session_id}.pdf"}
        )

    except HTTPException:
        raise
    except Exception as e:
        log_with_session(f"❌ Error: {str(e)}", session_id, logging.ERROR)  # CHANGED
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

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
        "classified_images": "classified_images.json"
    }
    if file_type not in file_mapping:
        raise HTTPException(status_code=400, detail="Invalid file type")
    file_path = OUTPUT_DIR / session_id / file_mapping[file_type]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=file_mapping[file_type])

# ---------------------------
# Get all sessions endpoint
# ---------------------------
@app.get("/sessions")
async def get_all_sessions():
    """Get all active session IDs and their progress"""
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
        "count": len(sessions)
    }

# ---------------------------
# Get specific session details
# ---------------------------
@app.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get details for a specific session"""
    if session_id not in progress_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_dir = OUTPUT_DIR / session_id
    files_available = []
    
    if session_dir.exists():
        file_checks = {
            "ocr": session_dir / "ocr_output.txt",
            "chunked": session_dir / "chunked_output.json",
            "classified": session_dir / "classified_output.json",
            "inception": session_dir / "inception.pdf",
            "classified_images": session_dir / "classified_images.json",
            "processed_images": session_dir / "processed_images"
        }
        
        for file_type, path in file_checks.items():
            if path.exists():
                if path.is_dir():
                    files_available.append({
                        "type": file_type,
                        "count": len(list(path.iterdir()))
                    })
                else:
                    files_available.append({
                        "type": file_type,
                        "size": path.stat().st_size
                    })
    
    return {
        "session_id": session_id,
        "progress": progress_store[session_id],
        "files_available": files_available
    }

# ---------------------------
# Log streaming endpoint (SSE)
# ---------------------------
@app.get("/stream-logs/{session_id}")
async def stream_logs(session_id: str):
    """Stream logs and progress for a specific session using Server-Sent Events"""
    
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
        progress_store.pop(session_id, None)
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