# src/pdf_merger.py

import logging
from pathlib import Path
from typing import Optional, Dict
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def merge_pdfs(
    inception_pdf_path: str,
    images_pdf_path: str,
    final_pdf_path: str,
    session_id: Optional[str] = None,
    progress_store: Optional[Dict[str, Dict[str, float]]] = None
) -> bool:
    """
    Merge PDFs in the following order:
    1. First page of inception.pdf (cover page)
    2. All pages from images_combined.pdf (route map + user images)
    3. Remaining pages from inception.pdf (report content)
    
    Args:
        inception_pdf_path: Path to inception report PDF
        images_pdf_path: Path to images combined PDF
        final_pdf_path: Path to save final merged PDF
        session_id: Session identifier for progress tracking
        progress_store: Optional progress dictionary to update
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Progress callback
        def update_progress(value: float):
            if progress_store and session_id and session_id in progress_store:
                progress_store[session_id]["merging"] = value
        
        update_progress(10.0)
        
        # Validate input paths
        inception_pdf = Path(inception_pdf_path)
        images_pdf = Path(images_pdf_path)
        final_pdf = Path(final_pdf_path)
        
        if not inception_pdf.exists():
            logger.error(f"Inception PDF not found: {inception_pdf}")
            raise FileNotFoundError(f"Inception PDF not found: {inception_pdf}")
        
        if not images_pdf.exists():
            logger.error(f"Images PDF not found: {images_pdf}")
            raise FileNotFoundError(f"Images PDF not found: {images_pdf}")
        
        update_progress(20.0)
        
        # Initialize PDF writer
        writer = PdfWriter()
        
        # Read PDFs
        logger.info(f"Reading inception PDF: {inception_pdf}")
        inception_reader = PdfReader(str(inception_pdf))
        
        logger.info(f"Reading images PDF: {images_pdf}")
        images_reader = PdfReader(str(images_pdf))
        
        update_progress(30.0)
        
        # Add first page of inception.pdf (cover page)
        if len(inception_reader.pages) > 0:
            writer.add_page(inception_reader.pages[0])
            logger.info("Added cover page from inception.pdf")
        else:
            logger.warning("Inception PDF has no pages")
        
        update_progress(50.0)
        
        # Add all pages from images_combined.pdf
        for idx, page in enumerate(images_reader.pages):
            writer.add_page(page)
        
        logger.info(f"Added {len(images_reader.pages)} pages from images_combined.pdf")
        update_progress(70.0)
        
        # Add remaining pages from inception.pdf (skip first page)
        remaining_pages = inception_reader.pages[1:]
        for idx, page in enumerate(remaining_pages):
            writer.add_page(page)
        
        logger.info(f"Added {len(remaining_pages)} remaining pages from inception.pdf")
        update_progress(90.0)
        
        # Write final merged PDF
        final_pdf.parent.mkdir(parents=True, exist_ok=True)
        
        with open(final_pdf, "wb") as f:
            writer.write(f)
        
        logger.info(f"Final merged PDF created: {final_pdf}")
        logger.info(f"Total pages in final PDF: {len(writer.pages)}")
        
        update_progress(100.0)
        
        return True
    
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"PDF merging failed: {e}", exc_info=True)
        return False