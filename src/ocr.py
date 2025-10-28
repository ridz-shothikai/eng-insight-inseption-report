# src/ocr.py

import os
import requests
import base64
import time
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Tuple, Callable 
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path
from PIL import Image
from src.utils.log_streamer import log_streamer, SessionLogHandler

import threading, time, google.generativeai as genai

load_dotenv()

# Setup logger with session handler
logger = logging.getLogger(__name__)

# ---------- global free-tier gate ----------
_RPM_LOCK = threading.Lock()
_LAST_CALL = 0
_MIN_INTERVAL = 0.05  # 600 RPM paid tier = 10 requests/second

def _wait_for_free_slot():
    global _LAST_CALL
    with _RPM_LOCK:
        elapsed = time.time() - _LAST_CALL
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _LAST_CALL = time.time()

# Add session log handler if not already added
if not any(isinstance(h, SessionLogHandler) for h in logger.handlers):
    session_log_handler = SessionLogHandler(log_streamer)
    session_log_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(session_log_handler)

# Helper function for session-aware logging
def log_with_session(message: str, session_id: str = None, level=logging.INFO):
    """Helper to log with session context for streaming"""
    if session_id:
        logger.log(level, message, extra={'session_id': session_id})
    else:
        logger.log(level, message)

def process_single_page_worker(
    page_data: Tuple[int, Image.Image],
    api_key: str,
    session_id: str = None
) -> Tuple[int, str]:
    page_num, page_image = page_data

    genai.configure(api_key=api_key)
    model_name = os.getenv("llm_primary_model", "gemini-2.5-flash-lite")
    model = genai.GenerativeModel(model_name)

    buffered = BytesIO()
    page_image.save(buffered, format="JPEG", quality=70, optimize=True)
    image_bytes = buffered.getvalue()

    prompt = "Return only the text visible in this document page. Do not add comments."

    for attempt in range(3):
        try:
            _wait_for_free_slot()          # ------ rate-limit gate ------
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                request_options={"timeout": 20}
            )
            return page_num, (response.text or "").strip()

        except Exception as e:
            if "429" in str(e):
                # honour server-side delay once we are above the 15 rpm
                delay = 10
                log_with_session(
                    f"⏱️  429 from Gemini on page {page_num} – waiting {delay}s …",
                    session_id, logging.WARNING
                )
                time.sleep(delay)
                continue
            # any other error
            if attempt == 2:
                log_with_session(f"❌ Gemini failed on page {page_num}: {e}", session_id, logging.ERROR)
                return page_num, ""
            time.sleep(1 ** attempt)

    return page_num, ""


class GeminiVisionOCR:
    """Gemini Vision OCR processor optimized for PDFs and images"""

    def __init__(
        self, 
        api_key: Optional[str] = None,
        max_workers: Optional[int] = None,
        dpi: int = 150,
        chunk_size: int = 100,
        session_id: str = None
    ):
        """
        Initialize Gemini Vision OCR processor
        
        Args:
            api_key: Gemini Vision API key (defaults to env var)
            max_workers: Number of parallel workers (defaults to 10)
            dpi: DPI for PDF to image conversion
            chunk_size: Number of pages to process in each chunk
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini Vision API key not found. Set GOOGLE_API_KEY environment variable.")
        
        self.max_workers = max_workers or min(8, (os.cpu_count() or 4) * 2) 
        self.dpi = dpi
        self.chunk_size = chunk_size
        self.session_id = session_id 
        self.session = requests.Session()
        
        log_with_session(
            f"Gemini Vision OCR initialized: "
            f"{self.max_workers} workers, DPI={dpi}, chunk_size={chunk_size}",
            session_id
        )
    
    def process_pdf(self, pdf_path: str, progress_callback=None) -> str:
        """
        Process PDF with Gemini Vision API in parallel chunks
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates (0-100)
            
        Returns:
            Full extracted text from all pages
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            log_with_session(f"❌ PDF file not found: {pdf_path}", self.session_id, logging.ERROR)
            return ""
        
        log_with_session(f"Processing PDF: {pdf_path}", self.session_id)
        start_time = time.time()
        
        # Get total page count
        try:
            info = pdfinfo_from_path(str(pdf_path))
            total_pages = info["Pages"]
            log_with_session(f"PDF has {total_pages} pages", self.session_id)
        except Exception as e:
            log_with_session(f"❌ Could not read PDF info: {e}", self.session_id, logging.ERROR)
            return ""
        
        results = {}
        pages_processed = 0
        
        # Process in chunks to manage memory
        for start_page in range(1, total_pages + 1, self.chunk_size):
            end_page = min(start_page + self.chunk_size - 1, total_pages)
            chunk_pages = end_page - start_page + 1
            
            log_with_session(f"Converting pages {start_page}-{end_page} to images...", self.session_id)
            chunk_start = time.time()
            
            try:
                # Convert chunk to images
                pages = convert_from_path(
                    str(pdf_path),
                    dpi=self.dpi,
                    first_page=start_page,
                    last_page=end_page,
                    thread_count=8
                )
                
                conversion_time = time.time() - chunk_start
                log_with_session(f"Chunk conversion took {conversion_time:.2f}s", self.session_id)
                
                # Prepare page data
                page_data = [(start_page + i, page) for i, page in enumerate(pages)]
                
                # Process chunk in parallel
                log_with_session(f"Processing {len(pages)} pages with {self.max_workers} workers...", self.session_id)
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(process_single_page_worker, data, self.api_key, self.session_id): data[0]  
                        for data in page_data
                    }
                    
                    for future in as_completed(futures):
                        page_num = futures[future]
                        try:
                            result_page_num, text = future.result()
                            results[result_page_num] = text
                            pages_processed += 1
                            
                            log_with_session(f"✓ Completed page {result_page_num}/{total_pages}...", self.session_id)
                            
                            # Update progress
                            if progress_callback:
                                progress_value = (pages_processed / total_pages) * 100.0
                                progress_callback(progress_value)
                                
                        except Exception as e:
                            log_with_session(f"❌ Page {page_num} generated exception: {e}", self.session_id, logging.ERROR)
                            results[page_num] = ""
                            pages_processed += 1
                
                # Clear memory
                del pages
                
                chunk_time = time.time() - chunk_start
                log_with_session(f"Chunk {start_page}-{end_page} completed in {chunk_time:.2f}s", self.session_id)
                
            except Exception as e:
                log_with_session(f"❌ Failed to process chunk {start_page}-{end_page}: {e}", self.session_id, logging.ERROR)
                # Mark failed pages as empty
                for page_num in range(start_page, end_page + 1):
                    if page_num not in results:
                        results[page_num] = ""
                        pages_processed += 1
        
        # Assemble results in order
        full_text = ""
        for i in range(1, total_pages + 1):
            page_text = results.get(i, "")
            full_text += f"\n--- Page {i} ---\n{page_text}\n"
        
        total_time = time.time() - start_time
        log_with_session(f"✅ OCR complete! Total time: {total_time:.2f}s...", self.session_id)
        
        # Mark as complete
        if progress_callback:
            progress_callback(100.0)
        
        return full_text
    
    def process_image_file(self, image_path: str, progress_callback=None) -> str:
        """
        Process a single image file with Gemini Vision API
        
        Args:
            image_path: Path to image file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Extracted text from image
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            log_with_session(f"❌ Image file not found: {image_path}", self.session_id, logging.ERROR)
            return ""
        
        try:
            if progress_callback:
                progress_callback(50.0)
            
            log_with_session(f"🖼️ Processing image: {image_path}", self.session_id)
            
            # Load and process image
            image = Image.open(image_path)
            _, text = process_single_page_worker((1, image), self.api_key, self.session_id)
            
            if progress_callback:
                progress_callback(100.0)
            
            return f"\n--- Image: {image_path.name} ---\n{text}\n"
            
        except Exception as e:
            log_with_session(f"❌ Error processing image {image_path}: {e}", self.session_id, logging.ERROR)
            return ""
    
    def process_document(self, input_path: str, output_path: str, progress_callback=None) -> bool:
        """
        Main entry: process PDF or image and save OCR text
        
        Args:
            input_path: Path to input file (PDF or image)
            output_path: Path to save extracted text
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            if not input_path.exists():
                log_with_session(f"❌ Input file not found: {input_path}", self.session_id, logging.ERROR)
                return False
            
            file_ext = input_path.suffix.lower()
            log_with_session(f"📄 Starting OCR for {input_path.name} (type: {file_ext})", self.session_id)

            
            # Route to appropriate processor
            if file_ext == ".pdf":
                text = self.process_pdf(str(input_path), progress_callback)
            elif file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
                text = self.process_image_file(str(input_path), progress_callback)
            else:
                log_with_session(f"❌ Unsupported file type: {file_ext}", self.session_id, logging.ERROR)
                return False
            
            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            log_with_session(f"✅ OCR complete for {input_path.name}, saved to {output_path}", self.session_id)
            return True
            
        except Exception as e:
            log_with_session(f"❌ OCR failed: {e}", self.session_id, logging.ERROR)
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        self.session.close()

    


# Integration function for backend (FastAPI compatible)
def process_ocr(
    input_path: str,
    output_path: str,
    session_id: str,
    progress_store: Optional[Dict[str, Dict[str, float]]] = None,
    api_key: Optional[str] = None
) -> bool:
    """
    Wrapper to be called from FastAPI endpoint
    
    Args:
        input_path: Path to input file
        output_path: Path to save output
        session_id: Session identifier for progress tracking
        progress_store: Optional dictionary to store progress
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        True if successful, False otherwise
    """
    def progress_callback(value: float):
        """Callback to update progress store"""
        if progress_store and session_id in progress_store:
            progress_store[session_id]["ocr"] = value
    
    try:
        log_with_session(f"🔍 Starting OCR processing", session_id)
        ocr = GeminiVisionOCR(api_key=api_key, session_id=session_id)
        success = ocr.process_document(input_path, output_path, progress_callback)
        ocr.cleanup()
        log_with_session(f"✅ OCR processing complete", session_id)
        return success
    except Exception as e:
        log_with_session(f"❌ OCR process failed: {e}", session_id, logging.ERROR)
        return False


# # CLI entry point
# if __name__ == "__main__":
#     import sys
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     # Default values
#     DEFAULT_INPUT = "rfp.pdf"
#     DEFAULT_OUTPUT = "ocr_output.txt"
    
#     # Parse arguments
#     if len(sys.argv) == 1:
#         # No arguments: use defaults
#         input_file = DEFAULT_INPUT
#         output_file = DEFAULT_OUTPUT
#     elif len(sys.argv) == 2:
#         # One argument: treat as input file, use default output
#         input_file = sys.argv[1]
#         output_file = DEFAULT_OUTPUT
#     elif len(sys.argv) == 3:
#         # Two arguments: custom input and output
#         input_file = sys.argv[1]
#         output_file = sys.argv[2]
#     else:
#         print("Usage:")
#         print(f"  python gcv_ocr.py                    # Uses {DEFAULT_INPUT} -> {DEFAULT_OUTPUT}")
#         print(f"  python gcv_ocr.py <input_file>       # Uses <input_file> -> {DEFAULT_OUTPUT}")
#         print(f"  python gcv_ocr.py <input> <output>   # Custom input and output")
#         sys.exit(1)
    
#     # Check if input exists
#     if not Path(input_file).exists():
#         print(f"❌ Error: Input file '{input_file}' not found")
#         sys.exit(1)
    
#     print(f"📄 Input:  {input_file}")
#     print(f"📝 Output: {output_file}")
#     print()
    
#     start_time = time.time()
#     ocr = GoogleCloudVisionOCR()
#     success = ocr.process_document(input_file, output_file)
#     ocr.cleanup()
#     elapsed = time.time() - start_time
    
#     if success:
#         print(f"\n✅ OCR completed in {elapsed:.1f}s")
#         print(f"   Output saved to: {output_file}")
#     else:
#         print("\n❌ OCR failed")
#         sys.exit(1)