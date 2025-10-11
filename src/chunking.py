import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Callable

logger = logging.getLogger(__name__)


class TextChunk:
    """Represents a chunk of text with metadata"""

    def __init__(self, chunk_id: int, text: str, start_pos: int, end_pos: int, token_count: int):
        self.chunk_id = chunk_id
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.token_count = token_count

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "token_count": self.token_count,
        }


class ChunkingEngine:
    """
    Splits extracted text into manageable chunks.
    Uses token-based chunking with overlap for context preservation.
    """

    def __init__(
        self, 
        chunk_size: int = 512, 
        overlap: int = 50,
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """
        Initialize chunking engine
        
        Args:
            chunk_size: Maximum tokens per chunk (default: 512)
            overlap: Number of overlapping tokens between chunks (default: 50)
            progress_callback: Optional callback for progress updates (0-100)
        """
        # Ensure integer types to prevent type comparison errors
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.progress_callback = progress_callback
        
        logger.info(f"ChunkingEngine initialized: chunk_size={self.chunk_size}, overlap={self.overlap}")

    def update_progress(self, value: float):
        """Update progress via callback if available"""
        if self.progress_callback:
            try:
                self.progress_callback(min(100.0, max(0.0, value)))
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (~4 chars per token)"""
        return max(1, len(text) // 4)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences; supports English + Hindi punctuation"""
        sentences = re.split(r'[.!?।]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> List[TextChunk]:
        """Create overlapping chunks from text"""
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        self.update_progress(10.0)
        
        sentences = self._split_into_sentences(text)
        total_sentences = len(sentences)
        
        logger.info(f"Splitting {len(text)} characters into chunks ({total_sentences} sentences)")
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        position = 0
        chunk_id = 0

        for idx, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start_pos=position - len(chunk_text),
                        end_pos=position,
                        token_count=current_tokens,
                    )
                )
                chunk_id += 1

                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            position += len(sentence) + 1
            
            if idx % 100 == 0 and total_sentences > 0:
                progress = 10.0 + (idx / total_sentences) * 80.0
                self.update_progress(progress)

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_pos=position - len(chunk_text),
                    end_pos=position,
                    token_count=current_tokens,
                )
            )

        self.update_progress(100.0)
        logger.info(f"Created {len(chunks)} chunks from text")
        
        return chunks


def chunk_text(
    input_path: str,
    output_path: str,
    chunk_size: int = 512,
    overlap: int = 50,
    session_id: Optional[str] = None,
    progress_store: Optional[Dict[str, Dict[str, float]]] = None
) -> bool:
    """
    Read OCR output and generate chunked JSON output
    
    Args:
        input_path: Path to OCR text file
        output_path: Path to save chunked JSON output
        chunk_size: Maximum tokens per chunk (default: 512)
        overlap: Overlapping tokens between chunks (default: 50)
        session_id: Session identifier for progress tracking
        progress_store: Optional progress dictionary to update
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert parameters to int to handle string inputs from CLI or config files
        chunk_size = int(chunk_size)
        overlap = int(overlap)
        
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return False

        logger.info(f"Reading OCR output from {input_path}")
        text = input_path.read_text(encoding="utf-8").strip()
        
        if not text:
            logger.warning("OCR output is empty. Creating empty chunk file.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return True

        def progress_callback(value: float):
            if progress_store and session_id and session_id in progress_store:
                progress_store[session_id]["chunking"] = value

        engine = ChunkingEngine(
            chunk_size=chunk_size,
            overlap=overlap,
            progress_callback=progress_callback if progress_store else None
        )
        chunks = engine.create_chunks(text)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() for c in chunks],
                f,
                ensure_ascii=False,
                indent=2
            )

        logger.info(f"Chunking complete: {len(chunks)} chunks saved to {output_path}")
        
        if progress_store and session_id and session_id in progress_store:
            progress_store[session_id]["chunking"] = 100.0
        
        return True

    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        return False