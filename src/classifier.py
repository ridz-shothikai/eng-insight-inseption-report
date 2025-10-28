import json
import os
import asyncio
import logging
from typing import List, Dict, Set, Optional, Callable
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
import warnings
import hashlib
import random

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
    """Optimized multi-label classifier using batch processing and aggressive concurrency"""

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

    def __init__(self, 
                 progress_callback: Optional[Callable[[float], None]] = None,
                 use_cache: bool = True,
                 cache_size: int = 1000):
        """
        Initialize classifier with optimized settings for paid API
        
        Args:
            progress_callback: Optional callback for progress updates (0-100)
            use_cache: Enable response caching for similar chunks
            cache_size: Maximum number of cached responses
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            raise ValueError("GOOGLE_API_KEY not found in .env")
        
        genai.configure(api_key=api_key)
        
        # Use environment variables with optimized defaults
        self.primary_model_name = os.getenv("llm_primary_model", "gemini-2.0-flash-light")
        self.fallback_model_name = os.getenv("llm_secondary_model", "gemini-2.0-flash-light")
        
        self.model = genai.GenerativeModel(self.primary_model_name)
        self.fallback_model = genai.GenerativeModel(self.fallback_model_name)
        self.progress_callback = progress_callback
        
        # Caching
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.cache_size = cache_size
        
        logger.info(f"ContentClassifier initialized with {self.primary_model_name} (cache: {use_cache})")

    def update_progress(self, value: float):
        """Update progress via callback if available"""
        if self.progress_callback:
            try:
                self.progress_callback(min(100.0, max(0.0, value)))
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text content"""
        # Use first 1000 chars for cache key to handle similar chunks
        return hashlib.md5(text[:1000].encode('utf-8')).hexdigest()

    def _fallback_classification(self, text: str) -> Set[str]:
        """Keyword-based multi-label classification fallback"""
        text_lower = text.lower()
        categories = {
            cat for cat, words in self.KEYWORDS.items() 
            if any(w in text_lower for w in words)
        }
        return categories if categories else {"introduction"}

    def _build_batch_prompt(self, chunks: List[TextChunk]) -> str:
        """Build optimized batch classification prompt"""
        chunks_text = ""
        for chunk in chunks:
            # Truncate each chunk to 500 chars to fit more in one request
            chunks_text += f"\n[ID:{chunk.chunk_id}] {chunk.text[:500]}...\n"
        
        return f"""Classify RFP chunks into categories. Return JSON object mapping chunk ID to category array.

Categories: {', '.join(self.CATEGORIES)}

Chunks:{chunks_text}

Return JSON like: {{"0": ["category1"], "1": ["category1", "category2"]}}
JSON only, no markdown:"""

    def _parse_batch_response(self, response_text: str, chunks: List[TextChunk]) -> List[Set[str]]:
        """Parse batch classification response"""
        import re
        
        try:
            # Try direct JSON parsing
            if response_text.startswith('{') and response_text.endswith('}'):
                result = json.loads(response_text)
            else:
                # Extract JSON from markdown or mixed content
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    return None
            
            # Parse results for each chunk
            parsed_results = []
            for chunk in chunks:
                cats = result.get(str(chunk.chunk_id), [])
                valid = {c for c in cats if c in self.CATEGORIES}
                parsed_results.append(valid if valid else None)
            
            return parsed_results
            
        except Exception as e:
            logger.warning(f"Failed to parse batch response: {e}")
            return None

    async def _classify_batch(self, chunks: List[TextChunk], model, model_name: str) -> Optional[List[Set[str]]]:
        """Classify a batch of chunks in a single API call"""
        if not chunks:
            return []
        
        prompt = self._build_batch_prompt(chunks)
        
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            results = self._parse_batch_response(response.text.strip(), chunks)
            
            if results and any(r for r in results):
                logger.debug(f"✓ Batch classified {len(chunks)} chunks with {model_name}")
                return results
            
            return None
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning(f"Rate limit on {model_name} for batch")
            else:
                logger.debug(f"Batch classification error on {model_name}: {e}")
            return None

    async def _classify_single(self, chunk: TextChunk, model, model_name: str) -> Optional[Set[str]]:
        """Classify a single chunk (fallback from batch)"""
        # Check cache first
        if self.use_cache:
            cache_key = self._get_cache_key(chunk.text)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for chunk {chunk.chunk_id}")
                return self.cache[cache_key]
        
        prompt = f"""Classify this RFP text. Return JSON array of categories only.

Categories: {', '.join(self.CATEGORIES)}

Text: {chunk.text[:600]}...

JSON array:"""
        
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            resp_text = response.text.strip()
            
            # Parse response
            import re
            if resp_text.startswith('[') and resp_text.endswith(']'):
                categories = json.loads(resp_text)
            else:
                match = re.search(r'\[.*?\]', resp_text, re.DOTALL)
                categories = json.loads(match.group()) if match else []
            
            valid = {c for c in categories if c in self.CATEGORIES}
            
            # Update cache
            if self.use_cache and valid:
                if len(self.cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    self.cache.pop(next(iter(self.cache)))
                cache_key = self._get_cache_key(chunk.text)
                self.cache[cache_key] = valid
            
            return valid if valid else None
            
        except Exception as e:
            logger.debug(f"Single classification error on {model_name}: {e}")
            return None

    async def _classify_chunk_with_fallback(self, chunk: TextChunk, idx: int, total: int) -> Set[str]:
        """
        Classify a single chunk with automatic fallback strategy
        Tries: primary model -> fallback model -> keywords
        """
        # Try primary model
        result = await self._classify_single(chunk, self.model, self.primary_model_name)
        if result:
            return result
        
        # Try fallback model with small delay
        await asyncio.sleep(0.1)
        result = await self._classify_single(chunk, self.fallback_model, self.fallback_model_name)
        if result:
            logger.info(f"Used fallback model for chunk {chunk.chunk_id}")
            return result
        
        # Ultimate fallback to keywords
        logger.warning(f"Using keyword fallback for chunk {chunk.chunk_id}")
        return self._fallback_classification(chunk.text)

    async def _process_batch_with_fallback(self, chunks: List[TextChunk], batch_idx: int, total_batches: int) -> List[Set[str]]:
        """
        Process a batch with automatic fallback to individual classification
        """
        if not chunks:
            return []
        
        # Try batch classification with primary model
        batch_results = await self._classify_batch(chunks, self.model, self.primary_model_name)
        
        if batch_results and all(r for r in batch_results):
            # All chunks successfully classified
            return batch_results
        
        # If batch failed or partial results, process individually
        if batch_results:
            # Some succeeded, only retry failed ones
            final_results = []
            for i, (chunk, result) in enumerate(zip(chunks, batch_results)):
                if result:
                    final_results.append(result)
                else:
                    final_results.append(await self._classify_chunk_with_fallback(
                        chunk, i, len(chunks)
                    ))
            return final_results
        else:
            # Entire batch failed, process all individually
            logger.info(f"Batch {batch_idx}/{total_batches} failed, falling back to individual classification")
            tasks = [
                self._classify_chunk_with_fallback(chunk, i, len(chunks))
                for i, chunk in enumerate(chunks)
            ]
            return await asyncio.gather(*tasks)

    async def classify_chunks(self, 
                             chunks: List[TextChunk], 
                             concurrency: int = 100,
                             batch_size: int = 8) -> Dict[str, List[TextChunk]]:
        """
        Classify all chunks with optimized batch processing and high concurrency
        
        Args:
            chunks: List of text chunks to classify
            concurrency: Number of concurrent API calls (default 100 for paid API)
            batch_size: Number of chunks per batch request (default 8)
            
        Returns:
            Dictionary mapping categories to chunks
        """
        if not chunks:
            logger.warning("No chunks to classify")
            return {cat: [] for cat in self.CATEGORIES}
        
        logger.info(f"Classifying {len(chunks)} chunks (concurrency={concurrency}, batch_size={batch_size})")
        self.update_progress(10.0)
        
        # Split chunks into batches
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        total_batches = len(batches)
        logger.info(f"Processing {total_batches} batches")
        
        # Semaphore for concurrency control
        sem = asyncio.Semaphore(concurrency)
        
        async def process_batch_with_sem(batch: List[TextChunk], batch_idx: int):
            async with sem:
                results = await self._process_batch_with_fallback(batch, batch_idx, total_batches)
                # Update progress
                progress = 10.0 + ((batch_idx + 1) / total_batches) * 85.0
                self.update_progress(progress)
                return list(zip(batch, results))
        
        # Process all batches concurrently
        tasks = [process_batch_with_sem(batch, idx) for idx, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten and categorize results
        categorized = defaultdict(list)
        for batch_result in batch_results:
            for chunk, cats in batch_result:
                for cat in cats:
                    categorized[cat].append(chunk)
        
        self.update_progress(100.0)
        
        # Log statistics
        total_chunks = len(chunks)
        chunks_per_category = {cat: len(chunks) for cat, chunks in categorized.items()}
        logger.info(f"Classification complete: {total_chunks} chunks across {len(categorized)} categories")
        logger.info(f"Distribution: {chunks_per_category}")
        
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
    concurrency: int = 100,
    batch_size: int = 8,
    session_id: Optional[str] = None,
    progress_store: Optional[Dict[str, Dict[str, float]]] = None,
    use_cache: bool = True
) -> bool:
    """
    Main function to classify chunks with optimized settings
    
    Args:
        input_path: Path to chunked JSON file
        output_path: Path to save classified output
        concurrency: Number of concurrent API calls (default 100 for paid API)
        batch_size: Chunks per batch request (default 8)
        session_id: Session identifier for progress tracking
        progress_store: Optional progress dictionary to update
        use_cache: Enable response caching
        
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

        # Run classification with optimized settings
        classifier = ContentClassifier(
            progress_callback=progress_callback if progress_store else None,
            use_cache=use_cache
        )
        
        categorized = asyncio.run(
            classifier.classify_chunks(
                chunks, 
                concurrency=concurrency,
                batch_size=batch_size
            )
        )
        
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