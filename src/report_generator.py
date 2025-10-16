# src/report_generator.py

import os
import json
import time
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
import requests
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.platypus import Image as RLImage

from reportlab.platypus import Frame
from reportlab.platypus import PageTemplate
from reportlab.pdfgen import canvas

# Import log streamer
from src.utils.log_streamer import log_streamer, SessionLogHandler

# Setup logger with session handler
logger = logging.getLogger(__name__)

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


# -----------------------------
# DATA STRUCTURES
# -----------------------------
@dataclass
class TextChunk:
    text: str
    meta: Optional[Dict] = None


@dataclass
class ReportSection:
    section_id: str
    title: str
    content: str
    word_count: int
    generated_at: datetime


# -----------------------------
# GOOGLE SEARCH TOOL
# -----------------------------
class GoogleSearchTool:
    def __init__(self, session_id: str = None):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.session_id = session_id

    def search(self, query: str, max_retries: int = 3) -> Dict:
        if not self.api_key or not self.cse_id:
            log_with_session("Google Search API credentials not found, skipping search", self.session_id, logging.WARNING)
            return {"results": []}

        params = {"key": self.api_key, "cx": self.cse_id, "q": query, "num": 3}

        for attempt in range(max_retries + 1):
            try:
                time.sleep(0.5)
                response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)

                if response.status_code == 429:
                    log_with_session(f"Rate limited. Retrying in 30s... (attempt {attempt+1})", self.session_id, logging.WARNING)
                    time.sleep(30)
                    continue

                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    log_with_session(f"Search API error: {data.get('error')}", self.session_id, logging.ERROR)
                    return {"results": []}

                results = [
                    {"title": item.get("title", ""), "snippet": item.get("snippet", "")}
                    for item in data.get("items", []) if item.get("snippet")
                ]
                return {"results": results}

            except Exception as e:
                log_with_session(f"Search error: {e}", self.session_id, logging.ERROR)
                if attempt < max_retries:
                    time.sleep(30)

        return {"results": []}


# -----------------------------
# CONTENT GENERATOR
# -----------------------------
class ContentGenerator:
    SECTION_TITLES = {
        "executive_summary": "1.0 Executive Summary",
        "introduction": "2.0 Introduction",
        "site_appreciation": "3.0 Site Appreciation",
        "methodology": "4.0 Approach and Methodology",
        "task_assignment": "5.0 Task Assignment and Manning Schedule",
        "cross_sections": "6.0 Proposed Cross Sections",
        "design_standards": "7.0 Draft Design Standards",
        "work_programme": "8.0 Work Programme",
        "development": "9.0 Development",
        "quality_assurance": "10.0 Quality Assurance Plan",
        "checklists": "11.0 Checklists",
        "summary_conclusion": "12.0 Summary and Conclusion",
        "compliances": "13.0 Compliances",
        "appendix_irc_codes": "Appendix A: IRC Codes Reference",
        "appendix_monsoon": "Appendix B: Monsoon Calendar",
        "appendix_equipment": "Appendix C: Equipment Catalog",
        "appendix_testing": "Appendix D: Testing Protocols",
        "appendix_compliance_matrix": "Appendix E: Compliance Matrix"
    }

    SEARCH_QUERIES = {
        "site_appreciation": [
            "site details road project {location}",
            "monsoon climate construction {location}",
            "construction material suppliers {location}"
        ],
        "methodology": [
            "IRC codes Indian Roads Congress",
            "MoRTH specifications Ministry Road Transport Highways",
            "NHAI guidelines National Highways Authority"
        ],
        "design_standards": [
            "IRC codes standards",
            "MoRTH specifications",
            "BIS standards road construction"
        ],
        "compliances": [
            "Indian construction labour laws Building Workers Act",
            "construction site safety regulations India",
            "environmental clearance highway {location}"
        ],
        "quality_assurance": [
            "BIS road construction quality testing",
            "MoRTH quality specifications"
        ]
    }

    def __init__(self, location: str, progress_callback: Optional[Callable[[float], None]] = None, session_id: str = None, stream_callback: Optional[Callable[[str, str], None]] = None):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
            # Primary and fallback models
        self.primary_model_name = "gemini-2.5-flash"
        self.fallback_model_name = "gemini-2.5-flash-lite"  # Weaker/cheaper model
        
        self.model = genai.GenerativeModel(self.primary_model_name)
        self.fallback_model = genai.GenerativeModel(self.fallback_model_name)

        self.location = location
        self.session_id = session_id
        self.search_tool = GoogleSearchTool(session_id=session_id)
        self.progress_callback = progress_callback
        self.stream_callback = stream_callback

        # self.safety_settings = [
        #     {
        #         "category": "HARM_CATEGORY_HARASSMENT",
        #         "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_HATE_SPEECH", 
        #         "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        #         "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        #         "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        #     },
        # ]
        
        log_with_session(f"ContentGenerator initialized for location: {location}", session_id)
        log_with_session(f"Primary model: {self.primary_model_name}, Fallback: {self.fallback_model_name}", session_id)

    def update_progress(self, value: float):
        if self.progress_callback:
            try:
                self.progress_callback(min(100.0, max(0.0, value)))
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _gather_context(self, category: str) -> str:
        queries = self.SEARCH_QUERIES.get(category, [])
        context = f"\n=== EXTERNAL RESEARCH CONTEXT FOR {self.location.upper()} ===\n"
        
        for query_template in queries:
            query = query_template.format(location=self.location)
            results = self.search_tool.search(query)
            
            if results.get("results"):
                context += f"\nQuery: {query}\n"
                for i, r in enumerate(results["results"][:2], 1):
                    context += f"  {i}. {r.get('snippet', '')}\n"
            else:
                context += f"\nQuery: {query} → No results found.\n"
        
        context += "\n=== END EXTERNAL CONTEXT ===\n"
        return context

    def _get_section_prompt(self, category: str, chunks: List[TextChunk]) -> str:
        combined_chunks = "\n".join([c.text.strip() for c in chunks if c.text.strip()]) or "No specific project details were provided."
        
        base_guidelines = f"""
**CRITICAL REQUIREMENTS:**
- Write ONLY in professional English. NO regional languages.
- Base ALL content on {self.location}, India.
- Include technical specifications and tables where relevant.
- Reference IRC, MoRTH, BIS, NHAI standards.
- Do NOT invent project-specific facts unless provided.
- Format content professionally with clear headings and structure.
"""
        
        prompt = f"""Create detailed professional content for the '{category}' section of an RFP inception report.

CLIENT INPUT:
{combined_chunks}

{base_guidelines}

Generate comprehensive, technical content suitable for a professional engineering report."""
        
        return prompt

    def _generate_with_streaming(self, prompt: str, section_id: str, max_retries: int = 3) -> str:
        # Try primary model first
        for attempt in range(max_retries + 1):
            try:
                time.sleep(1.0)
                full_response = ""
                
                response = self.model.generate_content(prompt, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        if self.stream_callback:
                            self.stream_callback(section_id, chunk.text)
                
                return full_response.strip()
                
            except Exception as e:
                if "429" in str(e) or "rateLimit" in str(e):
                    if attempt < max_retries:
                        log_with_session(f"⏱️ Rate limited on {self.primary_model_name}. Retrying in 30s... (attempt {attempt+1})", self.session_id, logging.WARNING)
                        time.sleep(30)
                    else:
                        log_with_session(f"⚠️ Max retries exceeded on {self.primary_model_name}, falling back to {self.fallback_model_name}", self.session_id, logging.WARNING)
                        break  # Exit to try fallback model
                else:
                    log_with_session(f"❌ Content generation error on {self.primary_model_name}: {e}", self.session_id, logging.ERROR)
                    break  # Exit to try fallback model
        
        # Fallback to weaker model
        log_with_session(f"🔄 Attempting with fallback model: {self.fallback_model_name}", self.session_id, logging.INFO)
        
        for attempt in range(max_retries + 1):
            try:
                time.sleep(1.0)
                full_response = ""
                
                response = self.fallback_model.generate_content(prompt, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        if self.stream_callback:
                            self.stream_callback(section_id, chunk.text)
                
                log_with_session(f"✅ Successfully generated with fallback model: {self.fallback_model_name}", self.session_id)
                return full_response.strip()
                
            except Exception as e:
                if "429" in str(e) or "rateLimit" in str(e):
                    if attempt < max_retries:
                        log_with_session(f"⏱️ Rate limited on {self.fallback_model_name}. Retrying in 30s... (attempt {attempt+1})", self.session_id, logging.WARNING)
                        time.sleep(30)
                    else:
                        log_with_session(f"❌ Max retries exceeded on fallback model {self.fallback_model_name}", self.session_id, logging.ERROR)
                        raise Exception(f"Failed to generate content with both primary and fallback models")
                else:
                    log_with_session(f"❌ Content generation error on {self.fallback_model_name}: {e}", self.session_id, logging.ERROR)
                    raise e

    
    def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Non-streaming fallback"""
        # Try primary model
        for attempt in range(max_retries + 1):
            try:
                time.sleep(1.0)
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) or "rateLimit" in str(e):
                    if attempt < max_retries:
                        log_with_session(f"⏱️ Rate limited on {self.primary_model_name}. Retrying in 30s... (attempt {attempt+1})", self.session_id, logging.WARNING)
                        time.sleep(30)
                    else:
                        log_with_session(f"⚠️ Max retries exceeded on {self.primary_model_name}, falling back to {self.fallback_model_name}", self.session_id, logging.WARNING)
                        break
                else:
                    log_with_session(f"❌ Content generation error on {self.primary_model_name}: {e}", self.session_id, logging.ERROR)
                    break
        
        # Fallback to weaker model
        log_with_session(f"🔄 Attempting with fallback model: {self.fallback_model_name}", self.session_id, logging.INFO)
        
        for attempt in range(max_retries + 1):
            try:
                time.sleep(1.0)
                response = self.fallback_model.generate_content(prompt)
                log_with_session(f"✅ Successfully generated with fallback model: {self.fallback_model_name}", self.session_id)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) or "rateLimit" in str(e):
                    if attempt < max_retries:
                        log_with_session(f"⏱️ Rate limited on {self.fallback_model_name}. Retrying in 30s... (attempt {attempt+1})", self.session_id, logging.WARNING)
                        time.sleep(30)
                    else:
                        log_with_session(f"❌ Max retries exceeded on fallback model {self.fallback_model_name}", self.session_id, logging.ERROR)
                        raise Exception(f"Failed to generate content with both primary and fallback models")
                else:
                    log_with_session(f"❌ Content generation error on {self.fallback_model_name}: {e}", self.session_id, logging.ERROR)
                    raise e

    def generate_section(self, category: str, chunks: List[TextChunk]) -> ReportSection:
        if not chunks:
            chunks = [TextChunk(text="No input provided.")]
        
        log_with_session(f"📝 Generating section: {category}", self.session_id)
        
        # Notify streaming start
        if self.stream_callback:
            self.stream_callback(category, f"## {self.SECTION_TITLES.get(category, category)}\n\n")
        
        external_context = self._gather_context(category)
        base_prompt = self._get_section_prompt(category, chunks)
        full_prompt = base_prompt + external_context
        
        # Use streaming generation if callback is provided, otherwise fallback
        if self.stream_callback:
            content = self._generate_with_streaming(full_prompt, category)
        else:
            content = self._generate_with_retry(full_prompt)

        word_count = len(content.split())
        log_with_session(f"✓ Generated '{category}' section: {word_count} words", self.session_id)
        
        return ReportSection(
            section_id=category,
            title=self.SECTION_TITLES.get(category, category.replace('_', ' ').title()),
            content=content,
            word_count=word_count,
            generated_at=datetime.now()
        )

    def generate_appendix(self, appendix_type: str) -> ReportSection:
        appendix_prompts = {
            "appendix_irc_codes": f"List and summarize relevant IRC codes for road/highway projects in {self.location}, India. Include IRC:SP:84 for urban roads, IRC:37 for rural roads, IRC:5 for traffic standards.",
            "appendix_monsoon": f"Create a detailed monsoon calendar for {self.location}, India. Include typical rainfall patterns, construction-suitable months, and weather considerations.",
            "appendix_equipment": f"List comprehensive equipment catalog for road construction projects in {self.location}, India. Include machinery specifications, capacity, and usage.",
            "appendix_testing": f"Describe material testing protocols following MoRTH specifications for road projects in {self.location}, India. Include soil testing, aggregate testing, and quality control measures.",
            "appendix_compliance_matrix": f"Create compliance matrix showing adherence to IRC, MoRTH, BIS, and NHAI standards for projects in {self.location}, India."
        }
        
        prompt = appendix_prompts.get(appendix_type, f"Create {appendix_type} for {self.location}")
        log_with_session(f"📑 Generating appendix: {appendix_type}", self.session_id)
        
        # Notify streaming start for appendices too
        if self.stream_callback:
            self.stream_callback(appendix_type, f"## {self.SECTION_TITLES.get(appendix_type, appendix_type.replace('_', ' ').title())}\n\n")
        
        # Use streaming generation if callback is provided
        if self.stream_callback:
            content = self._generate_with_streaming(prompt, appendix_type)
        else:
            content = self._generate_with_retry(prompt)

        word_count = len(content.split())
        log_with_session(f"✓ Generated '{appendix_type}' appendix: {word_count} words", self.session_id)
        
        return ReportSection(
            section_id=appendix_type,
            title=self.SECTION_TITLES.get(appendix_type, appendix_type.replace('_', ' ').title()),
            content=content,
            word_count=len(content.split()),
            generated_at=datetime.now()
        )


# -----------------------------
# PDF UTILITIES
# -----------------------------
def sanitize_for_reportlab(text: str) -> str:
    """Sanitize text for ReportLab PDF generation"""
    # Convert <br> tags to newlines BEFORE processing
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Convert markdown bold to HTML bold
    text = re.sub(r'\*\*([^*]+?)\*\*', r'<b>\1</b>', text)
    
    # Convert markdown italic to HTML italic (single asterisk)
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    
    # Escape ALL special characters first
    text = text.replace('&', '&amp;')
    text = text.replace('<', '<')
    text = text.replace('>', '>')
    
    # Now restore ONLY the safe tags that ReportLab supports
    text = text.replace('<b>', '<b>').replace('</b>', '</b>')
    text = text.replace('<i>', '<i>').replace('</i>', '</i>')
    text = text.replace('<u>', '<u>').replace('</u>', '</u>')
    text = text.replace('<strong>', '<b>').replace('</strong>', '</b>')
    text = text.replace('<em>', '<i>').replace('</em>', '</i>')
    
    return text.strip()


def parse_markdown_table(text: str) -> tuple:
    """
    Parse markdown table from text with validation
    Returns: (is_table, table_data) where table_data is list of lists
    """
    lines = text.strip().split('\n')
    
    # Check if it looks like a markdown table
    if not lines or not lines[0].startswith('|'):
        return False, None
    
    table_data = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            continue
        
        # Skip separator line (contains dashes and colons)
        if re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        
        # Parse cells
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        
        # Skip completely empty rows
        if cells and any(cell.strip() for cell in cells):
            # Clean markdown from cells
            cleaned_cells = []
            for cell in cells:
                # Remove markdown formatting
                cell = re.sub(r'\*\*(.+?)\*\*', r'\1', cell)  # Bold
                cell = re.sub(r'\*(.+?)\*', r'\1', cell)      # Italic
                cell = re.sub(r'`(.+?)`', r'\1', cell)        # Code
                cell = re.sub(r'<[^>]+>', '', cell)           # HTML tags
                cell = re.sub(r'^#{1,6}\s+', '', cell)        # Headers
                
                # Truncate very long cells
                if len(cell) > 150:
                    cell = cell[:147] + '...'
                
                cleaned_cells.append(cell)
            
            table_data.append(cleaned_cells)
    
    # Validate: must have at least header + 1 data row
    if len(table_data) < 2:
        return False, None
    
    # Normalize: ensure all rows have same column count
    if table_data:
        col_count = len(table_data[0])
        normalized_data = []
        for row in table_data:
            if len(row) < col_count:
                # Pad with empty cells
                row.extend([''] * (col_count - len(row)))
            elif len(row) > col_count:
                # Trim extra cells
                row = row[:col_count]
            normalized_data.append(row)
        table_data = normalized_data
    
    return len(table_data) > 0, table_data


def extract_subsections(content: str) -> List[str]:
    """Extract subsection titles from content"""
    subsections = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Pattern 1: Numbered subsections (3.1, 3.1.1, etc.)
        match = re.match(r'^(\d+\.)+\d+\s+(.+)', line)
        if match:
            subsections.append(line)
            continue
        
        # Pattern 2: Bold text that looks like a header
        if re.match(r'^\*\*[^*]+\*\*$', line) or re.match(r'^<b>[^<]+</b>$', line):
            clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            clean = re.sub(r'</?b>', '', clean)
            if len(clean.split()) <= 10:
                subsections.append(clean)
            continue
        
        # Pattern 3: Markdown headers (##, ###)
        if re.match(r'^#{2,6}\s+', line):
            clean = re.sub(r'^#{2,6}\s+', '', line)
            subsections.append(clean)
    
    return subsections


def strip_leading_numbering(text: str) -> str:
    """
    Remove leading numbering patterns, even when wrapped in markdown formatting.
    """
    # Pattern for markdown bold followed by numbering
    patterns = [
        # Roman numerals: require at least a dot, parenthesis, or multiple letters
        r'^\s*(\*\*)?\s*(?:\d+(?:\.\d+)*\.?|\(\d+\)|\d+\)|(?:[IVXLCDM]{2,}|[IVXLCDM])\.|\([IVXLCDM]+\)|[IVXLCDM]+\)|[A-Za-z]\.|\([A-Za-z]\)|[A-Za-z]\)|\*\s*|\•\s*)\s*(\*\*)?',
        r'^\s*\*\*\s*(?:\d+(?:\.\d+)*\.?|\(\d+\)|\d+\)|(?:[IVXLCDM]{2,}|[IVXLCDM])\.|\([IVXLCDM]+\)|[IVXLCDM]+\)|[A-Za-z]\.|\([A-Za-z]\)|[A-Za-z]\)|\*\s*|\•\s*)\s*\*\*'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # Basic pattern - more restrictive for Roman numerals
    basic_pattern = r'^\s*(?:\d+(?:\.\d+)*\.?|\(\d+\)|\d+\)|(?:[IVXLCDM]{2,}|[IVXLCDM])\.|\([IVXLCDM]+\)|[IVXLCDM]+\)|[A-Za-z]\.|\([A-Za-z]\)|[A-Za-z]\)|\*\s*|\•\s*)\s*'
    text = re.sub(basic_pattern, '', text)
    
    # Remove trailing colons and asterisks
    text = re.sub(r'[:*]+$', '', text.strip())
    
    return text.strip()

def add_page_decorations(canvas_obj, doc):
    """
    Draws header and footer on EVERY page.
    Uses absolute canvas coordinates (0,0 is bottom-left of entire page).
    """
    logger.info(f"Drawing decorations on page {doc.page}")  # Add this line
    canvas_obj.saveState()
    
    logo_dir = Path(__file__).parent.parent / "logo"
    top_img = logo_dir / "top.jpeg"
    bottom_img = logo_dir / "bottom.jpeg"
    page_width, page_height = A4

    # Header: top of page
    if top_img.exists():
        try:
            canvas_obj.drawImage(
                str(top_img),
                x=0,
                y=page_height - 0.75 * inch,
                width=page_width,
                height=0.75 * inch,
                preserveAspectRatio=False,
                mask='auto'
            )
        except Exception as e:
            logger.error(f"Failed to draw top image: {e}")

    # Footer: bottom of page
    if bottom_img.exists():
        try:
            canvas_obj.drawImage(
                str(bottom_img),
                x=0,
                y=0,
                width=page_width,
                height=0.75 * inch,
                preserveAspectRatio=False,
                mask='auto'
            )
        except Exception as e:
            logger.error(f"Failed to draw bottom image: {e}")
    
    canvas_obj.restoreState()  # Restore the state


def create_pdf_report(
    sections: List[ReportSection], 
    location: str, 
    output_path: str, 
    image_data: List[Dict] = None, 
    image_base_dir: str = None,
    session_id: str = None
):
    """Create PDF report with embedded images from classified_images.json"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Page dimensions
    page_width, page_height = A4
    
    # Margins for content (leave space for header/footer)
    left_margin = right_margin = 72
    top_margin = 72 + 0.75 * inch  # Extra space for header
    bottom_margin = 72 + 0.75 * inch  # Extra space for footer

    # Initialize document first
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin
    )

    # Create frames for body content (avoiding header/footer space)
    frame = Frame(
        left_margin,
        bottom_margin,
        page_width - left_margin - right_margin,
        page_height - top_margin - bottom_margin,
        leftPadding=6,
        bottomPadding=6,
        rightPadding=6,
        topPadding=6,
        id='normal'
    )

    # Create custom page template with the decoration function
    page_template = PageTemplate(
        id='with_decorations',
        frames=[frame],
        onPage=add_page_decorations,
        pagesize=A4
    )

    # Add the template to the document
    doc.addPageTemplates([page_template])
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title', 
        parent=styles['Heading1'], 
        fontSize=18, 
        spaceAfter=30, 
        alignment=1,
        textColor=colors.HexColor('#1a1a1a')
    )
    heading_style = ParagraphStyle(
        'Heading', 
        parent=styles['Heading2'], 
        fontSize=14, 
        spaceAfter=12, 
        spaceBefore=12, 
        keepWithNext=True,
        textColor=colors.HexColor('#2c3e50'),
        fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'Body', 
        parent=styles['BodyText'],
        fontName='Helvetica', 
        fontSize=11, 
        leading=16, 
        spaceAfter=8,
        textColor=colors.HexColor('#2c3e50'),
        alignment=0
    )
    
    # Style for table cells with word wrapping
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['BodyText'],
        fontSize=9,
        leading=12,
        alignment=0,
        wordWrap='CJK'
    )

    toc_style = ParagraphStyle(
    'TOC',
    parent=styles['BodyText'],
    fontSize=11,
    leading=18,
    textColor=colors.HexColor('#2980b9'),  # Blue for clickable look
    leftIndent=20
)
    
    story = []

    # Group images by category
    images_by_category = {}
    if image_data:
        for item in image_data:
            cat = item.get("category")
            if cat not in images_by_category:
                images_by_category[cat] = []
            images_by_category[cat].append(item)
        log_with_session(f"📸 Loaded {len(image_data)} images across {len(images_by_category)} categories",session_id)

    # Cover page
    story.append(Paragraph("INCEPTION REPORT", title_style))
    story.append(Spacer(1, 100))
    story.append(Paragraph(f"Road/Highway/Bridge Project", heading_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>Location: {location}, India</b>", body_style))
    story.append(Spacer(1, 50))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y')}", body_style))
    story.append(PageBreak())

    # Table of Contents
    story.append(Paragraph("Table of Contents", heading_style))
    story.append(Spacer(1, 20))
    
    # TOC subsection style
    toc_sub_style = ParagraphStyle(
        'TOCSub',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        spaceBefore=2,
        spaceAfter=2,
        textColor=colors.HexColor('#555555')
    )
    
    for i, sec in enumerate(sections, 1):
        section_id = f"section_{i}"
        clean_title = strip_leading_numbering(sec.title)
        # Create clickable link
        toc_entry = f'<link href="#{section_id}" color="blue">{i}. {clean_title}</link>'
        story.append(Paragraph(toc_entry, toc_style))
        
        subsections = extract_subsections(sec.content)
        for subsection in subsections[1:]:
            clean_sub = strip_leading_numbering(subsection.strip())
            story.append(Paragraph(f"  ◦ {clean_sub}", toc_sub_style))
        
        story.append(Spacer(1, 6))
    
    story.append(PageBreak())

    # Sections with images
    for section in sections:
        section_id = f"section_{sections.index(section) + 1}"
        story.append(Paragraph(f'<a name="{section_id}"/>{section.title}', heading_style))
        story.append(Spacer(1, 12))

        # Insert images for this section's category
        if section.section_id in images_by_category:
            log_with_session(f"✓ Adding {len(images_by_category[section.section_id])} images to section: {section.section_id}", session_id)
            
            for img_info in images_by_category[section.section_id]:
                img_path_str = img_info.get("img_path", "")
                caption = img_info.get("caption", "")
                
                # Resolve image path relative to image_base_dir (directory of classified_images.json)
                if image_base_dir:
                    full_img_path = Path(image_base_dir) / img_path_str
                else:
                    full_img_path = Path(img_path_str)
                
                if full_img_path.exists():
                    try:
                        img = RLImage(str(full_img_path), width=6*inch, height=4*inch)
                        img.hAlign = 'CENTER'
                        story.append(img)
                        story.append(Spacer(1, 6))
                        # Add centered caption
                        centered_caption_style = ParagraphStyle(
                            'CenteredCaption',
                            parent=body_style,
                            alignment=1,
                            fontSize=10,
                            fontName='Helvetica-Oblique'  # Italic font
                        )
                        story.append(Paragraph(caption, centered_caption_style))
                        story.append(Spacer(1, 12))
                        log_with_session(f"✓ Added image: {full_img_path.name}", session_id)
                    except Exception as e:
                        logger.warning(f"Failed to add image {full_img_path}: {e}")
                else:
                    log_with_session(f"⚠️ Image not found: {full_img_path}", session_id, logging.WARNING)

        # Content paragraphs
        paragraphs = section.content.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            is_table, table_data = parse_markdown_table(para)
            
            if is_table and table_data and len(table_data) >= 2:
                try:
                    available_width = 6.5 * inch
                    num_cols = len(table_data[0])
                    col_widths = [available_width / num_cols] * num_cols
                    
                    wrapped_table_data = []
                    for row_idx, row in enumerate(table_data):
                        wrapped_row = []
                        for cell in row:
                            if row_idx == 0:  # Header row
                                wrapped_row.append(Paragraph(f"<b>{cell}</b>", table_cell_style))
                            else:
                                wrapped_row.append(Paragraph(cell, table_cell_style))
                        wrapped_table_data.append(wrapped_row)
                    
                    table = Table(wrapped_table_data, colWidths=col_widths, repeatRows=1)
                    table.setStyle(TableStyle([
                        # Header row
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('TOPPADDING', (0, 0), (-1, 0), 12),
                        
                        # Data rows
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        
                        # Grid
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#2980b9')),
                        
                        # Padding
                        ('LEFTPADDING', (0, 0), (-1, -1), 8),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 1), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                    ]))
                    
                    story.append(table)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    logger.warning(f"Failed to render table: {e}")
                    # Fallback: render as text
                    for line in para.split('\n'):
                        if line.strip() and not re.match(r'^\|[\s\-:|]+\|$', line):
                            story.append(Paragraph(sanitize_for_reportlab(line), body_style))
            
            # Check if it's a bullet list
            elif para.startswith(('- ', '• ', '* ')):
                lines = para.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith(('- ', '• ', '* ')):
                        line = re.sub(r'^[-•*]\s+', '', line)
                        story.append(Paragraph(f"• {sanitize_for_reportlab(line)}", body_style))
                    elif line:
                        story.append(Paragraph(f"  {sanitize_for_reportlab(line)}", body_style))
                story.append(Spacer(1, 6))
            
            # Check if it's a numbered list
            elif re.match(r'^\d+\.', para):
                lines = para.split('\n')
                for line in lines:
                    line = line.strip()
                    if re.match(r'^\d+\.', line):
                        story.append(Paragraph(sanitize_for_reportlab(line), body_style))
                    elif line:
                        story.append(Paragraph(f"  {sanitize_for_reportlab(line)}", body_style))
                story.append(Spacer(1, 6))
            
            # Regular paragraph
            else:
                for line in para.split('\n'):
                    line = line.strip()
                    if line:
                        story.append(Paragraph(sanitize_for_reportlab(line), body_style))
                        story.append(Spacer(1, 4))
        
        story.append(Spacer(1, 24))
        story.append(PageBreak())

    try:
        # Build PDF - this will apply the template to all pages
        doc.build(story, onFirstPage=add_page_decorations, onLaterPages=add_page_decorations)
        log_with_session(f"✅ PDF report created: {output_path}", session_id)
    except Exception as e:
        logger.error(f"Failed to build PDF: {e}")
        raise


def extract_location(input_file: str, session_id: str = None) -> str:
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            sample_text = f.read()[:5000]
        
        if not sample_text.strip():
            return "Unspecified location in India"
    except Exception as e:
        log_with_session(f"❌ Failed to read OCR file: {e}", session_id, logging.ERROR)
        return "Unspecified location in India"

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Initialize both models
    primary_model = genai.GenerativeModel("gemini-2.5-flash")
    fallback_model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    prompt = f"""Extract ONLY the primary city/place from the text for a road/highway/bridge project in India.

Text:
{sample_text}

Respond with just the location name, nothing else.
Location:"""
    
    # Try primary model
    try:
        location = primary_model.generate_content(prompt).text.strip()
        if not location or len(location) < 3:
            return "Unspecified location in India"
        return location
    except Exception as e:
        log_with_session(f"⚠️ Location extraction failed with primary model, trying fallback: {e}", session_id, logging.WARNING)
        
        # Try fallback model
        try:
            location = fallback_model.generate_content(prompt).text.strip()
            if not location or len(location) < 3:
                return "Unspecified location in India"
            log_with_session(f"✅ Location extracted with fallback model", session_id)
            return location
        except Exception as fallback_e:
            log_with_session(f"❌ Location extraction failed with both models: {fallback_e}", session_id, logging.ERROR)
            return "Unspecified location in India"


def enhance_location(location: str, session_id: str = None) -> str:
    """Enhance location details using search"""
    if location == "Unspecified location in India":
        return location

    log_with_session(f"🔍 Enhancing location: {location}", session_id)
    search_tool = GoogleSearchTool(session_id=session_id)
    results = search_tool.search(f"{location} district state India location details")
    context = "\n".join([r.get("snippet", "") for r in results.get("results", [])[:3]])

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Initialize both models
    primary_model = genai.GenerativeModel("gemini-2.5-flash")
    fallback_model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    prompt = f"""Enhance location into format 'City, District, State' for India.

Original: {location}

Search Results:
{context}

Respond with just the enhanced location in format: City, District, State
Enhanced Location:"""
    
    # Try primary model
    try:
        enhanced = primary_model.generate_content(prompt).text.strip()
        if enhanced and "," in enhanced:
            return enhanced
        return location
    except Exception as e:
        log_with_session(f"⚠️ Location enhancement failed with primary model, trying fallback: {e}", session_id, logging.WARNING)
        
        # Try fallback model
        try:
            enhanced = fallback_model.generate_content(prompt).text.strip()
            if enhanced and "," in enhanced:
                log_with_session(f"✅ Location enhanced with fallback model", session_id)
                return enhanced
            return location
        except Exception as fallback_e:
            log_with_session(f"❌ Location enhancement failed with both models: {fallback_e}", session_id, logging.ERROR)
            return location


def generate_inception_report(
    classified_file: str,
    output_path: str,
    ocr_file: Optional[str] = None,
    session_id: Optional[str] = None,
    progress_store: Optional[Dict[str, Dict[str, float]]] = None,
    stream_callback: Optional[Callable[[str, str], None]] = None
) -> bool:
    try:
        def progress_callback(value: float):
            if progress_store and session_id and session_id in progress_store:
                progress_store[session_id]["report"] = value

        progress_callback(5.0)

        # Extract location
        if ocr_file and Path(ocr_file).exists():
            location = extract_location(ocr_file, session_id)
            location = enhance_location(location, session_id)  # Added location enhancement
        else:
            location = "Unspecified location in India"
        
        log_with_session(f"📍 Report location: {location}", session_id)
        progress_callback(10.0)

        # Load classified data
        classified_path = Path(classified_file)
        try:
            with open(classified_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log_with_session(f"❌ Failed to load classified file: {e}", session_id, logging.ERROR)
            return False

        # Parse categories
        categories_data = data.get("categories", {})
        categorized_chunks = {}
        
        for category, chunks in categories_data.items():
            if isinstance(chunks, list):
                categorized_chunks[category] = [
                    TextChunk(text=chunk.get("text", "")) 
                    for chunk in chunks if isinstance(chunk, dict)
                ]
        
        progress_callback(15.0)

        # Generate sections
        generator = ContentGenerator(location, progress_callback=progress_callback, session_id=session_id, stream_callback=stream_callback)
        
        section_categories = [
            "executive_summary", "introduction", "site_appreciation", "methodology",
            "task_assignment", "cross_sections", "design_standards", "work_programme",
            "development", "quality_assurance", "checklists", "summary_conclusion", "compliances"
        ]
        
        sections = []
        total_sections = len(section_categories) + 5
        
        for idx, cat in enumerate(section_categories):
            chunks = categorized_chunks.get(cat, [TextChunk(text="No input provided.")])
            section = generator.generate_section(cat, chunks)
            sections.append(section)
            
            progress = 15.0 + (idx + 1) / total_sections * 70.0
            progress_callback(progress)

        # Generate appendices
        appendix_types = [
            "appendix_irc_codes", "appendix_monsoon", "appendix_equipment",
            "appendix_testing", "appendix_compliance_matrix"
        ]
        
        for idx, appendix_type in enumerate(appendix_types):
            appendix = generator.generate_appendix(appendix_type)
            sections.append(appendix)
            
            progress = 85.0 + (idx + 1) / len(appendix_types) * 10.0
            progress_callback(progress)

        # Load image data from classified_images.json in the same directory as classified_file
        progress_callback(95.0)
        image_data = None
        image_base_dir = classified_path.parent

        image_json_path = image_base_dir / "classified_images.json"
        
        if image_json_path.exists():
            try:
                with open(image_json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                # Extract the list of image entries
                if isinstance(raw_data, dict) and "images" in raw_data:
                    image_data = raw_data["images"]
                elif isinstance(raw_data, list):
                    # Fallback if old format was used
                    image_data = raw_data
                else:
                    image_data = []
                    log_with_session("⚠️ Unexpected format in classified_images.json", session_id, logging.WARNING)

                log_with_session(f"📸 Loaded {len(image_data)} images from {image_json_path}", session_id)
            except Exception as e:
                log_with_session(f"⚠️ Failed to load image data: {e}", session_id, logging.WARNING)
                image_data = []
        else:
            image_data = []

        # Create PDF with images
        create_pdf_report(
            sections, 
            location, 
            output_path, 
            image_data=image_data, 
            image_base_dir=str(image_base_dir),
            session_id=session_id
        )
        
        progress_callback(100.0)
        log_with_session(f"✅ Inception report generated successfully: {output_path}", session_id)
        return True

    except Exception as e:
        log_with_session(f"❌ Report generation failed: {e}", session_id, logging.ERROR)
        return False


# if __name__ == "__main__":
#     import sys
#     
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
#     
#     # Default files in current directory
#     classified_file = "classified.json"
#     ocr_file = "ocr_output.txt"
#     output_file = "inception_report.pdf"
#     
#     # Check if files exist
#     if not Path(classified_file).exists():
#         print(f"❌ Error: {classified_file} not found in current directory")
#         sys.exit(1)
#     
#     if not Path(ocr_file).exists():
#         print(f"⚠️  Warning: {ocr_file} not found, will use default location")
#         ocr_file = None
#     
#     print(f"📄 Classified: {classified_file}")
#     print(f"📝 OCR File:   {ocr_file if ocr_file else 'Not provided'}")
#     print(f"📋 Output:     {output_file}")
#     print()
#     
#     start_time = time.time()
#     
#     success = generate_inception_report(
#         classified_file=classified_file,
#         output_path=output_file,
#         ocr_file=ocr_file,
#         session_id=None,
#         progress_store=None
#     )
#     
#     elapsed = time.time() - start_time
#     
#     if success:
#         print(f"\n✅ Report generated successfully in {elapsed:.1f}s")
#         print(f"   Output saved to: {output_file}")
#     else:
#         print(f"\n❌ Report generation failed")
#         sys.exit(1)