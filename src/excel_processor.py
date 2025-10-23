"""
Excel Sheet Classifier with LLM-Powered Data Cleaning
Classifies Excel sheets and uses Gemini 2.5 Flash to extract cleaned, structured content.
Uses the same API format as classifier.py with google.generativeai library.
"""

import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime
import os
import logging
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

import google.generativeai as genai

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# CONFIGURATION
# ============================================================================

# DPR Categories for Indian road/highway/bridge projects
EXCEL_CATEGORIES = {
    "executive_summary": "High-level project overview, objectives, key highlights for Indian road/highway/bridge projects",
    "introduction": "Project background, location details, stakeholders, NH/SH details, introductory visuals",
    "site_appreciation": "Site location, terrain, soil conditions, geographical features, existing infrastructure, site photographs",
    "methodology": "Construction methodology, IRC standards, work execution approach, technical processes for road/bridge construction",
    "task_assignment": "Project organization structure, contractor teams, labour deployment, responsibility matrix",
    "cross_sections": "Road cross-sections, bridge drawings, pavement designs, structural drawings as per IRC/MORT&H specifications",
    "design_standards": "IRC codes (IRC:6, IRC:21, IRC:112), MORT&H specifications, Indian standards (IS codes), design criteria",
    "work_programme": "Project timeline, construction schedule, bar charts, milestone charts, activity sequences",
    "development": "Stage construction, phasing plans, traffic diversion schemes, development sequence",
    "quality_assurance": "Quality control plans, material testing, IRC compliance checks, field testing procedures",
    "checklists": "Construction checklists, safety checklists, quality inspection formats, compliance verification",
    "summary_conclusion": "Project summary, achievements, recommendations, completion status, way forward",
    "compliances": "NHAI/MoRTH compliance, environmental clearances, statutory approvals, IRC compliance certificates"
}

# ============================================================================
# GEMINI API FUNCTIONS
# ============================================================================

class GeminiClassifier:
    """Gemini-based Excel sheet classifier using google.generativeai library"""
    
    def __init__(self):
        """Initialize Gemini models"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            raise ValueError("GOOGLE_API_KEY not found in .env")
        
        genai.configure(api_key=api_key)
        
        # Primary and fallback models
        self.primary_model_name = "gemini-2.5-flash"
        self.fallback_model_name = "gemini-2.5-flash-lite"
        
        try:
            self.model = genai.GenerativeModel(self.primary_model_name)
            self.fallback_model = genai.GenerativeModel(self.fallback_model_name)
            logger.info(f"GeminiClassifier initialized with {self.primary_model_name} (fallback: {self.fallback_model_name})")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini models: {e}")
            raise
    
    def call_gemini(self, prompt: str, max_retries: int = 3, use_fallback: bool = True) -> str:
        """
        Call Gemini API with automatic fallback
        
        Args:
            prompt: Input prompt for the model
            max_retries: Number of retry attempts per model
            use_fallback: Whether to use fallback model if primary fails
            
        Returns:
            Model response text
        """
        # Try primary model first
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "rateLimit" in error_str:
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limited on {self.primary_model_name}. Waiting {wait_time}s... (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Primary model error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        break
        
        # Fallback to weaker model if enabled
        if use_fallback:
            logger.warning(f"⚠️ Primary model failed, trying fallback: {self.fallback_model_name}")
            
            for attempt in range(max_retries):
                try:
                    response = self.fallback_model.generate_content(prompt)
                    logger.info(f"✓ Successfully used fallback model")
                    return response.text.strip()
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "quota" in error_str.lower() or "rateLimit" in error_str:
                        wait_time = 10 * (attempt + 1)
                        logger.warning(f"Rate limited on {self.fallback_model_name}. Waiting {wait_time}s... (attempt {attempt + 1})")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"Fallback model error: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
        
        logger.error("All models failed")
        return ""

# Initialize global classifier
gemini_classifier = None

def get_classifier() -> GeminiClassifier:
    """Get or create global classifier instance"""
    global gemini_classifier
    if gemini_classifier is None:
        gemini_classifier = GeminiClassifier()
    return gemini_classifier

# ============================================================================
# DATA EXTRACTION AND CLEANING WITH LLM
# ============================================================================

def extract_sheet_data_for_llm(df: pd.DataFrame, sheet_name: str) -> str:
    """
    Extract complete sheet data in a format suitable for LLM processing
    """
    if df.empty:
        return f"Sheet: {sheet_name}\nEmpty sheet"
    
    data_parts = []
    
    # Basic info
    data_parts.append(f"SHEET: {sheet_name}")
    data_parts.append(f"DIMENSIONS: {df.shape[0]} rows × {df.shape[1]} columns")
    data_parts.append("=" * 50)
    
    # Column headers
    data_parts.append("COLUMNS:")
    for col_idx, col_name in enumerate(df.columns):
        col_name_str = str(col_name) if pd.notna(col_name) else f"Column_{col_idx}"
        data_parts.append(f"  [{col_idx}] {col_name_str}")
    data_parts.append("")
    
    # Complete data
    data_parts.append("COMPLETE DATA:")
    data_parts.append("-" * 50)
    
    # Include all rows (limit to 500 rows for very large sheets)
    max_rows = min(500, len(df))
    for idx in range(max_rows):
        row = df.iloc[idx]
        row_data = []
        for col_idx, val in enumerate(row):
            if pd.notna(val) and str(val).strip():
                # Clean the value but keep it complete
                val_str = str(val).strip()
                # Replace newlines with spaces for better formatting
                val_str = val_str.replace('\n', ' | ')
                row_data.append(f"[{col_idx}]{val_str}")
            else:
                row_data.append(f"[{col_idx}](empty)")
        
        data_parts.append(f"Row {idx}: {' | '.join(row_data)}")
    
    if len(df) > max_rows:
        data_parts.append(f"... ({len(df) - max_rows} more rows truncated)")
    
    return "\n".join(data_parts)

def clean_sheet_data_with_llm(raw_data: str, sheet_name: str, category: str) -> str:
    """
    Use Gemini 2.5 Flash to clean and structure the sheet data into meaningful markdown
    """
    prompt = f"""You are an expert data analyst processing Indian highway DPR (Detailed Project Report) Excel sheets.

RAW EXCEL DATA:
{raw_data}

CATEGORY: {category}
SHEET NAME: {sheet_name}

TASK: Extract the actual meaningful content from this Excel sheet and structure it properly for a DPR report.

CRITICAL INSTRUCTIONS:
1. IGNORE the "Unnamed:" columns and empty cells - they are Excel formatting artifacts
2. Look for the actual data tables and meaningful content
3. Extract proper key-value pairs and table data
4. Remove all redundant headers and formatting lines
5. Focus on the real project information, not the column structure
6. If you see repeated content, extract it only once
7. Look for patterns like: S.No, Item, Details, Client Name, etc.

EXPECTED OUTPUT FORMAT:
# {sheet_name}

## Project Information
- **Client:** [extract client name]
- **Project Title:** [extract project title]
- **Location:** [extract location details]
- **Length:** [extract road length]

## Key Details
[Create proper tables or bullet points from the actual data]

## Important Notes
[Extract any important notes, revisions, or approvals]

DO NOT:
- Create tables for each "Unnamed" column
- Repeat the same information multiple times
- Include empty cells or formatting artifacts
- Copy the raw column structure

Focus only on the meaningful project data that would be valuable for a DPR report.

Now extract and clean the data:"""

    try:
        classifier = get_classifier()
        cleaned_content = classifier.call_gemini(prompt)
        
        if not cleaned_content:
            # Fallback: create basic structure from raw data
            return f"# {sheet_name}\n\n## Data\n\n{raw_data[:2000]}..."
        
        return cleaned_content
    except Exception as e:
        logger.error(f"Error cleaning sheet data: {e}")
        return f"# {sheet_name}\n\n## Data\n\n{raw_data[:2000]}..."

# ============================================================================
# SHEET CLASSIFICATION
# ============================================================================

def classify_sheet_with_llm(sheet_name: str, sample_content: str) -> str:
    """
    Use Gemini 2.5 Flash to classify the sheet
    """
    categories_text = "\n".join([f"- {k}: {v}" for k, v in EXCEL_CATEGORIES.items()])
    
    prompt = f"""You are classifying Excel sheets from Indian highway/road/bridge DPR documents.

Sheet Name: {sheet_name}

Sample Content:
{sample_content}

Available Categories:
{categories_text}

Return ONLY the category key (e.g., 'executive_summary', 'methodology', etc.) that best matches this sheet.
Return only the category name, nothing else."""

    try:
        classifier = get_classifier()
        category = classifier.call_gemini(prompt)
        
        # Clean and validate
        category = category.replace('"', '').replace("'", "").strip().lower()
        category = category.split('\n')[0].strip()
        
        # Validate category
        if category in EXCEL_CATEGORIES:
            return category
        
        # Try fuzzy matching
        for key in EXCEL_CATEGORIES.keys():
            if key in category or category in key:
                return key
        
        return "executive_summary"
        
    except Exception as e:
        logger.error(f"Error classifying sheet: {e}")
        return "executive_summary"

def classify_sheet_rule_based(sheet_name: str) -> str:
    """
    Rule-based classification fallback
    """
    sheet_lower = sheet_name.lower()
    
    keywords_map = {
        "executive_summary": ["executive", "summary", "overview", "abstract"],
        "introduction": ["introduction", "intro", "background", "preface"],
        "site_appreciation": ["site", "location", "appreciation", "photographs"],
        "methodology": ["methodology", "method", "approach", "procedure"],
        "task_assignment": ["task", "assignment", "organization", "team"],
        "cross_sections": ["cross section", "drawing", "design", "section"],
        "design_standards": ["standard", "code", "irc", "specification", "is code"],
        "work_programme": ["programme", "program", "schedule", "timeline", "chart"],
        "development": ["development", "stage", "phase", "construction"],
        "quality_assurance": ["quality", "qa", "qc", "testing", "assurance"],
        "checklists": ["checklist", "check list", "inspection", "verification"],
        "summary_conclusion": ["conclusion", "summary", "recommendation"],
        "compliances": ["compliance", "approval", "clearance", "statutory"]
    }
    
    for category, keywords in keywords_map.items():
        for keyword in keywords:
            if keyword in sheet_lower:
                return category
    
    return "executive_summary"

# ============================================================================
# SHEET PROCESSING
# ============================================================================

def get_sheet_sample_content(df: pd.DataFrame, max_chars: int = 1000) -> str:
    """
    Get a representative sample of sheet content for classification
    """
    if df.empty:
        return "Empty sheet"
    
    # Get first few rows as text
    sample_rows = []
    for idx in range(min(5, len(df))):
        row_values = [str(val) for val in df.iloc[idx] if pd.notna(val) and str(val).strip()]
        if row_values:
            sample_rows.append(" | ".join(row_values[:5]))
    
    sample_text = "\n".join(sample_rows)
    return sample_text[:max_chars]

def process_single_sheet(excel_path: str, sheet_name: str, use_llm: bool = True) -> Dict:
    """
    Process a single sheet: read, classify, and clean with LLM
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        
        if df.empty:
            return {"status": "empty", "sheet_name": sheet_name}
        
        # Get sample content for classification
        sample_content = get_sheet_sample_content(df)
        
        # Classify the sheet
        if use_llm:
            category = classify_sheet_with_llm(sheet_name, sample_content)
        else:
            category = classify_sheet_rule_based(sheet_name)
        
        # Extract complete data for LLM cleaning
        raw_data = extract_sheet_data_for_llm(df, sheet_name)
        
        # Clean the data with LLM
        if use_llm:
            cleaned_content = clean_sheet_data_with_llm(raw_data, sheet_name, category)
        else:
            # Simple fallback for no-LLM mode
            cleaned_content = f"# {sheet_name}\n\n## Raw Data\n\n{raw_data[:2000]}..."
        
        return {
            "status": "success",
            "sheet_name": sheet_name,
            "category": category,
            "content": cleaned_content,
            "dimensions": f"{df.shape[0]}x{df.shape[1]}"
        }
        
    except Exception as e:
        logger.error(f"Error processing sheet '{sheet_name}': {e}")
        return {
            "status": "error",
            "sheet_name": sheet_name,
            "error": str(e)
        }

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_excel_file(excel_path: str, use_llm: bool = True) -> Optional[List[Dict]]:
    """
    Process entire Excel file and return classified data
    """
    try:
        print(f"\n{'='*70}")
        print(f"📊 Excel Sheet Classifier (Gemini 2.5 Flash)")
        print(f"{'='*70}")
        print(f"📁 Input: {excel_path}")
        print(f"🤖 LLM Classification: {'Enabled' if use_llm else 'Disabled (Rule-based)'}")
        print(f"{'='*70}\n")
        
        if not Path(excel_path).exists():
            print(f"❌ File not found: {excel_path}")
            return None
        
        excel_file = pd.ExcelFile(excel_path)
        total_sheets = len(excel_file.sheet_names)
        print(f"✅ Found {total_sheets} sheets\n")
        
        category_content_map = {}
        successful = 0
        failed = 0
        
        for sheet_idx, sheet_name in enumerate(excel_file.sheet_names, 1):
            print(f"[{sheet_idx}/{total_sheets}] 📝 Processing: '{sheet_name}'")
            
            result = process_single_sheet(excel_path, sheet_name, use_llm)
            
            if result["status"] == "success":
                print(f"    📊 Size: {result['dimensions']}")
                
                category = result["category"]
                sheet_content = result["content"]
                
                if category not in category_content_map:
                    category_content_map[category] = []
                
                category_content_map[category].append({
                    "sheet_name": result["sheet_name"],
                    "content": sheet_content
                })
                
                print(f"    ✅ Category: {category}")
                print(f"    🧹 Gemini 2.5 Flash cleaning completed")
                successful += 1
                
            elif result["status"] == "empty":
                print(f"    ⚠️  Skipped: Empty sheet")
            else:
                print(f"    ❌ Error: {result['error']}")
                failed += 1
            
            print()
        
        # Create list of category objects
        output_data = []
        
        for category, sheets in category_content_map.items():
            output_data.append({
                "category": category,
                "sheets": sheets  # list of { "sheet_name": ..., "content": ... }
            })
        
        print(f"{'='*70}")
        print(f"✅ Processing complete!")
        print(f"   Total sheets: {total_sheets}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Categories created: {len(output_data)}")
        print(f"{'='*70}\n")
        
        return output_data
        
    except Exception as e:
        logger.error(f"Critical error processing Excel file: {e}", exc_info=True)
        return None

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_classified_data(data: List[Dict], output_path: str = "excel_classified.json") -> str:
    """
    Save classified data to JSON file with proper error handling
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Test JSON serialization first
        try:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            print(f"✅ JSON serialization successful: {len(json_str)} characters")
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
            # Try to save with problematic characters handled
            json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        print(f"💾 Saved: {output_path}")
        
        # Verify file was written correctly
        file_size = output_file.stat().st_size
        print(f"📁 File size: {file_size} bytes")
        
        print_summary(data)
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error saving file: {e}", exc_info=True)
        return ""


def print_summary(data: List[Dict]):
    """
    Print classification summary
    """
    print(f"\n{'='*70}")
    print(f"📊 CLASSIFICATION SUMMARY")
    print(f"{'='*70}")
    
    print("📑 CATEGORIES CREATED:\n")
    
    for category_obj in data:
        category = category_obj["category"]
        sheets = category_obj["sheets"]  # This is now a list of sheets
        sheet_count = len(sheets)
        
        # Calculate total content length for this category
        content_length = sum(len(sheet["content"]) for sheet in sheets)
        
        print(f"  📁 {category.upper().replace('_', ' ')}: {sheet_count} sheet(s), {content_length} chars")
        
        # Print individual sheets in this category
        for sheet in sheets:
            print(f"     └─ {sheet['sheet_name']} ({len(sheet['content'])} chars)")
    
    total_sheets = sum(len(item['sheets']) for item in data)
    total_size = sum(len(sheet['content']) for item in data for sheet in item['sheets'])
    
    print(f"\n📊 Total sheets: {total_sheets}")
    print(f"📊 Total content size: {total_size} characters")
    print(f"💾 Output format: List of category objects with sheets array")
    print(f"{'='*70}\n")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(
        description="Classify Excel sheets and clean data with Gemini 2.5 Flash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set API key in .env file first
  # GOOGLE_API_KEY=your-api-key-here
  
  # Run the classifier
  python excel_classifier_gemini.py input.xlsx
  python excel_classifier_gemini.py input.xlsx -o output.json
  python excel_classifier_gemini.py input.xlsx --no-llm
        """
    )
    
    parser.add_argument('excel_file', help='Path to Excel file to classify')
    parser.add_argument('-o', '--output', default='excel_classified.json', help='Output JSON file path')
    parser.add_argument('--no-llm', action='store_true', help='Use rule-based classification instead of LLM')
    
    args = parser.parse_args()
    
    # Validate API key if using LLM
    if not args.no_llm:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ Error: GOOGLE_API_KEY not found in environment variables")
            print("   Create a .env file with: GOOGLE_API_KEY=your-api-key")
            print("   Or use: --no-llm for rule-based classification")
            return 1
    
    use_llm = not args.no_llm
    classified_data = process_excel_file(args.excel_file, use_llm=use_llm)
    
    if classified_data:
        save_classified_data(classified_data, args.output)
        print(f"\n✅ Classification complete! Output saved to: {args.output}")
    else:
        print("\n❌ Classification failed")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit(main())