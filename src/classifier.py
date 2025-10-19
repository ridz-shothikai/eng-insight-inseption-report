# src/classifier.py

import json
import os
import asyncio
import logging
from typing import List, Dict, Set, Optional, Callable
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

import google.generativeai as genai

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

logger = logging.getLogger(__name__)


class TextChunk:
    def __init__(self, chunk_id: int, text: str, start_pos: int, end_pos: int, token_count: int):
        self.chunk_id = chunk_id
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.token_count = token_count

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            chunk_id=data['chunk_id'],
            text=data['text'],
            start_pos=data['start_pos'],
            end_pos=data['end_pos'],
            token_count=data['token_count']
        )

    @property
    def __dict__(self):
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'token_count': self.token_count
        }


class ContentClassifier:
    """Multi-label classifier for RFP text chunks using Google Gemini or keyword fallback"""

    CATEGORIES = [
        "executive_summary",
        "introduction",
        "site_appreciation",
        "methodology",
        "task_assignment",
        "cross_sections",
        "design_standards",
        "work_programme",
        "development",
        "quality_assurance",
        "checklists",
        "summary_conclusion",
        "compliances"
    ]

    KEYWORDS = {
        "executive_summary": ["executive", "summary", "overview", "highlights", "सारांश", "प्रमुख"],
        "introduction": ["introduction", "background", "परिचय", "पृष्ठभूमि", "उद्देश्य"],
        "site_appreciation": ["site", "location", "geography", "terrain", "स्थल", "स्थान", "भूगोल"],
        "methodology": ["methodology", "approach", "construction", "technique", "पद्धति", "निर्माण", "प्रक्रिया"],
        "task_assignment": ["task", "assignment", "team", "responsibility", "कार्य", "टीम", "दायित्व"],
        "cross_sections": ["cross section", "design", "drawing", "डिजाइन", "खंड", "रचना"],
        "design_standards": ["standard", "code", "specification", "मानक", "कोड", "विशिष्टता"],
        "work_programme": ["schedule", "timeline", "programme", "phase", "समय-सारणी", "कार्यक्रम", "चरण"],
        "development": ["development", "implementation", "विकास", "कार्यान्वयन", "निर्माण"],
        "quality_assurance": ["quality", "qa", "qc", "testing", "inspection", "गुणवत्ता", "परीक्षण", "निरीक्षण"],
        "checklists": ["checklist", "verification", "जांच-सूची", "सत्यापन", "चेकलिस्ट"],
        "summary_conclusion": ["conclusion", "summary", "recommendation", "निष्कर्ष", "सारांश", "सिफारिश"],
        "compliances": ["compliance", "regulatory", "legal", "अनुपालन", "विनियामक", "कानूनी"]
    }

    def __init__(self, progress_callback: Optional[Callable[[float], None]] = None):
        """
        Initialize classifier with Gemini AI
        
        Args:
            progress_callback: Optional callback for progress updates (0-100)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            raise ValueError("GOOGLE_API_KEY not found in .env")
        
        genai.configure(api_key=api_key)
        
        # Primary and fallback models
        self.primary_model_name = "gemini-2.5-flash-lite"
        self.fallback_model_name = "gemini-2.5-flash-lite"
        
        self.model = genai.GenerativeModel(self.primary_model_name)
        self.fallback_model = genai.GenerativeModel(self.fallback_model_name)
        self.progress_callback = progress_callback
        
        logger.info(f"ContentClassifier initialized with {self.primary_model_name} (fallback: {self.fallback_model_name})")

    def update_progress(self, value: float):
        """Update progress via callback if available"""
        if self.progress_callback:
            try:
                self.progress_callback(min(100.0, max(0.0, value)))
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _fallback_classification(self, text: str) -> Set[str]:
        """Keyword-based multi-label classification fallback"""
        text_lower = text.lower()
        categories = {
            cat for cat, words in self.KEYWORDS.items() 
            if any(w in text_lower for w in words)
        }
        return categories if categories else {"introduction"}

    def _build_prompt(self, text: str) -> str:
        """Build classification prompt for Gemini"""
        return f"""You are analyzing a construction/engineering RFP document in Hindi and English.
Analyze this text chunk and identify ALL relevant categories from the list below. A single chunk can belong to multiple categories.

CATEGORIES:
{', '.join(self.CATEGORIES)}

Text to classify:
{text[:800]}...

Respond with a JSON array of category names. Return ONLY the JSON array, no explanations."""

    async def _classify_chunk(self, chunk: TextChunk, max_retries: int = 2) -> Set[str]:
        """Classify a single chunk using Gemini with fallback; ultimate fallback to keywords"""
        
        prompt = self._build_prompt(chunk.text)
        
        # Try primary model first
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                resp_text = response.text.strip()

                # Try to parse JSON array
                import re
                try:
                    if resp_text.startswith('[') and resp_text.endswith(']'):
                        categories = json.loads(resp_text)
                    else:
                        match = re.search(r'\[.*\]', resp_text, re.DOTALL)
                        categories = json.loads(match.group()) if match else []
                except Exception:
                    categories = []

                valid = {c for c in categories if c in self.CATEGORIES}
                if valid:
                    return valid
                
                # If no valid categories, try again
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                    
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "rateLimit" in str(e):
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limited on {self.primary_model_name} for chunk {chunk.chunk_id}. Waiting {wait_time}s... (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"Primary model error for chunk {chunk.chunk_id}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    else:
                        break
        
        # Fallback to weaker model
        logger.warning(f"⚠️ Primary model failed for chunk {chunk.chunk_id}, trying fallback: {self.fallback_model_name}")
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(self.fallback_model.generate_content, prompt)
                resp_text = response.text.strip()

                # Try to parse JSON array
                import re
                try:
                    if resp_text.startswith('[') and resp_text.endswith(']'):
                        categories = json.loads(resp_text)
                    else:
                        match = re.search(r'\[.*\]', resp_text, re.DOTALL)
                        categories = json.loads(match.group()) if match else []
                except Exception:
                    categories = []

                valid = {c for c in categories if c in self.CATEGORIES}
                if valid:
                    logger.info(f"✓ Classified chunk {chunk.chunk_id} with fallback model")
                    return valid
                
                # If no valid categories, try again
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                    
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "rateLimit" in str(e):
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limited on {self.fallback_model_name} for chunk {chunk.chunk_id}. Waiting {wait_time}s... (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"Fallback model error for chunk {chunk.chunk_id}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
        
        # Ultimate fallback to keyword-based classification
        logger.warning(f"All models failed for chunk {chunk.chunk_id}, using keyword fallback")
        return self._fallback_classification(chunk.text)

    async def classify_chunks(self, chunks: List[TextChunk], concurrency: int = 5) -> Dict[str, List[TextChunk]]:
        """
        Classify all chunks into categories
        
        Args:
            chunks: List of text chunks to classify
            concurrency: Number of concurrent API calls
            
        Returns:
            Dictionary mapping categories to chunks
        """
        if not chunks:
            logger.warning("No chunks to classify")
            return {cat: [] for cat in self.CATEGORIES}
        
        logger.info(f"Classifying {len(chunks)} chunks with concurrency={concurrency}")
        self.update_progress(10.0)
        
        sem = asyncio.Semaphore(concurrency)

        async def classify(c: TextChunk, idx: int):
            async with sem:
                result = await self._classify_chunk(c)
                # Update progress
                progress = 10.0 + (idx + 1) / len(chunks) * 80.0
                self.update_progress(progress)
                return result

        tasks = [classify(c, i) for i, c in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        categorized = defaultdict(list)
        for chunk, cats in zip(chunks, results):
            for cat in cats:
                categorized[cat].append(chunk)

        self.update_progress(100.0)
        logger.info(f"Classification complete. Chunks distributed across {len(categorized)} categories")
        
        return {cat: categorized.get(cat, []) for cat in self.CATEGORIES}


def load_chunks(file_path: str) -> List[TextChunk]:
    """Load chunks from JSON file"""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Chunk file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = [TextChunk.from_dict(c) for c in data]
        logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        return []


def save_classification(categorized: Dict[str, List[TextChunk]], output_file: str):
    """Save classification results to JSON"""
    try:
        # Build chunk mapping
        chunk_mapping = {}
        for cat, chunks in categorized.items():
            for c in chunks:
                if c.chunk_id not in chunk_mapping:
                    chunk_mapping[c.chunk_id] = []
                chunk_mapping[c.chunk_id].append(cat)
        
        output = {
            "categories": {
                cat: [c.__dict__ for c in chunks] 
                for cat, chunks in categorized.items() if chunks
            },
            "chunk_mapping": chunk_mapping
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Classification saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save classification: {e}")
        raise


def classify_chunks(
    input_path: str,
    output_path: str,
    concurrency: int = 5,
    session_id: Optional[str] = None,
    progress_store: Optional[Dict[str, Dict[str, float]]] = None
) -> bool:
    """
    Main function to classify chunks
    
    Args:
        input_path: Path to chunked JSON file
        output_path: Path to save classified output
        concurrency: Number of concurrent Gemini API calls
        session_id: Session identifier for progress tracking
        progress_store: Optional progress dictionary to update
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load chunks
        chunks = load_chunks(input_path)
        if not chunks:
            logger.warning("No chunks to classify, creating empty output")
            save_classification({cat: [] for cat in ContentClassifier.CATEGORIES}, output_path)
            return True

        # Progress callback
        def progress_callback(value: float):
            if progress_store and session_id and session_id in progress_store:
                progress_store[session_id]["classification"] = value

        # Run classification
        classifier = ContentClassifier(progress_callback=progress_callback if progress_store else None)
        categorized = asyncio.run(classifier.classify_chunks(chunks, concurrency=concurrency))
        
        # Save results
        save_classification(categorized, output_path)
        
        # Mark complete
        if progress_store and session_id and session_id in progress_store:
            progress_store[session_id]["classification"] = 100.0
        
        logger.info(f"Classification complete: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        return False