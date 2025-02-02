import argparse
import os
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
from tqdm import tqdm
import google.generativeai as genai
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_safe_filename(title: str, max_length: int = 200) -> str:
    """
    Create a safe filename from a manuscript title, handling length limits.
    
    Args:
        title: Original manuscript title
        max_length: Maximum length for the filename (default 200 to stay well under filesystem limits)
    
    Returns:
        A safe filename that:
        - Contains only alphanumeric chars plus space, period, underscore, hyphen
        - Is under the specified length limit
        - Maintains uniqueness using a hash suffix for long titles
    """
    # First convert to safe characters
    safe_title = "".join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' 
                        for c in title)
    
    # If under length limit, return as is
    if len(safe_title) <= max_length:
        return safe_title
        
    # For long titles, truncate and add hash to ensure uniqueness
    hash_suffix = hashlib.md5(title.encode()).hexdigest()[:8]
    truncated = safe_title[:max_length - len(hash_suffix) - 1]
    return f"{truncated}_{hash_suffix}"

class RateLimiter:
    """Rate limiter for API requests respecting both per-minute and daily limits."""
    
    def __init__(self):
        self.minute_limit = 150000
        self.day_limit = 1500000000
        self.minute_tokens = 15000000
        self.day_tokens = 1500000000
        self.last_minute = datetime.now()
        self.last_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def wait_if_needed(self) -> None:
        """Check limits and wait if necessary."""
        now = datetime.now()
        
        # Check and reset daily limit
        if now.date() > self.last_day.date():
            self.day_tokens = self.day_limit
            self.last_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check and reset minute limit
        minutes_elapsed = (now - self.last_minute).total_seconds() / 60
        if minutes_elapsed >= 1:
            self.minute_tokens = self.minute_limit
            self.last_minute = now
        
        # Wait if we're out of tokens
        if self.minute_tokens <= 0:
            sleep_time = 60 - (now - self.last_minute).total_seconds()
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.minute_tokens = self.minute_limit
                self.last_minute = datetime.now()
        
        if self.day_tokens <= 0:
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            sleep_time = (tomorrow - now).total_seconds()
            logger.info(f"Daily limit reached. Waiting until midnight ({sleep_time:.1f} seconds)")
            time.sleep(sleep_time)
            self.day_tokens = self.day_limit
            self.day_tokens = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Consume tokens
        self.minute_tokens -= 1
        self.day_tokens -= 1

class TranscriptionError(Exception):
    """Custom exception for transcription-related errors."""
    pass

def extract_json_from_response(text: str) -> Dict:
    """
    Extracts JSON from a text, using a hybrid approach of standard JSON parsing and regex section-based extraction.
    It first attempts standard parsing, and then tries targeted extraction with section-specific formatting.
    """
    default_response = {
        'transcription': '',
        'revised_transcription': '',
        'summary': text,
        'keywords': [],
        'marginalia': [],
        'confidence': 0,
        'transcription_notes': 'Failed to parse JSON response',
        'content_notes': 'Original response preserved in summary field'
    }

    try:
        # 1. Standard JSON Parsing Attempt
        try:
           parsed = fix_json_and_load(text)
           if validate_keys(parsed):
             return parsed
           else:
             logger.error(f"Standard JSON parse failed key validation. Full text:\n{text}")
        except json.JSONDecodeError as e:
            logger.error(f"Standard JSON decode error: {e}. Full text:\n{text}")

        # 2. Section-Based Extraction Attempt
        extracted = extract_and_clean_sections(text)
        if extracted and validate_keys(extracted):
            return extracted
        else:
            logger.error(f"Section-based extraction failed. Full text:\n{text}")

    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {e}. Full text:\n{text}")
    
    return default_response

def extract_and_clean_sections(text: str) -> Dict:
    """Extracts and cleans specific sections from text using robust JSON parsing."""
    
    section_patterns = {
        'transcription': create_robust_pattern('transcription'),
        'revised_transcription': create_robust_pattern('revised_transcription'),
        'summary': r'"summary"\s*:\s*"?(.*?)"?(?=\s*,\s*"(?:keywords|marginalia|confidence|transcription_notes|content_notes)"|\s*\})',
        'keywords': r'"keywords"\s*:\s*\[(.*?)\]',
        'marginalia': r'"marginalia"\s*:\s*\[(.*?)\]',
        'confidence': r'"confidence"\s*:\s*(\d+)',
        'transcription_notes': r'"transcription_notes"\s*:\s*"?(.*?)"?(?=\s*,\s*"content_notes"|\s*\})',
        'content_notes': r'"content_notes"\s*:\s*"?(.*?)"?(?=\s*\})'
    }
    
    extracted = {}
    for key, pattern in section_patterns.items():
        try:
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                
                if key in ('transcription', 'revised_transcription'):
                    # Use string literal parsing for transcription fields
                    value = parse_json_string(value)
                elif key in ('keywords', 'marginalia'):
                    # Split and clean arrays
                    value = [
                        item.strip().strip('"').strip("'") 
                        for item in re.findall(r'"([^"]*)"', value) if item.strip()
                    ]
                elif key == 'confidence':
                    try:
                        value = float(value)
                    except:
                        value = 0
                else:
                    # Clean string values
                    value = re.sub(r'\\(["\\])', r'\1', value)
                
                extracted[key] = value
                
        except Exception as e:
            logger.warning(f"Error parsing section {key}: {e}")
    
    return extracted

def create_robust_pattern(field_name: str) -> str:
    """Creates a robust pattern for extracting JSON fields."""
    return fr'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)"'

def parse_json_string(text: str) -> str:
    """
    Parses a JSON string value, properly handling escapes and special characters.
    
    Args:
        text: The string to parse, without surrounding quotes
        
    Returns:
        Properly decoded string with preserved line breaks and special characters
    """
    try:
        # Handle escaped characters properly
        decoded = text.encode('utf-8').decode('unicode-escape')
        
        # Preserve intentional line breaks and structural marks
        decoded = decoded.replace('\\n', '\n')
        decoded = decoded.replace('\\|', '|')
        
        return decoded
    except Exception as e:
        logger.warning(f"Error parsing JSON string: {e}")
        return text
    
def fix_json_and_load(json_string: str) -> Dict:
    """Attempts to fix common JSON formatting errors before loading."""
    
    # Remove trailing commas in objects
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    
    # Unescapes escaped forward slashes
    json_string = json_string.replace('\\/', '/')
    
    # Handle single quotes by replacing them with double quotes (if that's not in the string already)
    if "'" in json_string and '"' not in json_string:
        json_string = json_string.replace("'", '"')
    
    # Remove leading '```json\n{'
    json_string = re.sub(r'^```(?:json)?\s*\n\s*{', '{', json_string, flags=re.DOTALL)
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Try a more aggressive fix and then load
        json_string = re.sub(r'([{\[,])\s*([}\]\,])', r'\1\2', json_string)
        json_string = re.sub(r'([{\[,])\s*([}\]\,])', r'\1\2', json_string)
        try:
           return json.loads(json_string)
        except:
            raise

def validate_keys(data: Dict) -> bool:
    """Validates that the extracted JSON contains the necessary keys."""
    required_keys = ['transcription', 'revised_transcription', 'summary']
    return all(key in data for key in required_keys)

def format_malformed_response(text: str, model: genai.GenerativeModel) -> Dict:
    """Attempt to format malformed response using Gemini Pro."""
    # First check if we actually have valid JSON in code blocks
    formatted = extract_json_from_response(text)
    if formatted.get('transcription') and formatted.get('transcription') != text:
        return formatted
        
    prompt = f"""Format this transcription response as valid JSON with this structure, outputting nothing but the correct json:
    {{
        "transcription": "initial transcription with markup",
        "revised_transcription": "refined version after context review",
        "summary": "brief content description",
        "keywords": ["theme1", "theme2"],
        "marginalia": ["location: content"],
        "confidence": number (0-100),
        "transcription_notes": "challenges and resolutions",
        "content_notes": "scholarly observations"
    }}

    Original response:
    {text}
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1},
            safety_settings={
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        formatted = extract_json_from_response(response.text)
        if formatted.get('transcription') and formatted.get('transcription') != text:
            return formatted
            
    except Exception as e:
        logger.error(f"Failed to format response: {str(e)}")
    
    # If all else fails, return a basic structure with the original text
    return {
        'transcription': text,
        'revised_transcription': text,
        'summary': 'Automatic formatting failed',
        'keywords': [],
        'marginalia': [],
        'confidence': 0,
        'transcription_notes': f'Failed to format response',
        'content_notes': 'Original response preserved in transcription field'
    }
    
class ManuscriptProcessor:
    """Handles the processing of manuscript pages with rate limiting."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.model = self._initialize_model()
    
    def _initialize_model(self) -> genai.GenerativeModel:
        """Initialize and configure the Gemini model."""
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-pro")
    
    def _initialize_model(self) -> genai.GenerativeModel:
        """Initialize and configure the Gemini model."""
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-pro")
    
    def _construct_prompt(self, metadata: Dict, page_number: int,
                         previous_page: Optional[Dict], next_page: Optional[Dict], notes: str) -> str:
        """Create the analysis prompt for a manuscript page."""
        return f"""As an expert paleographer examining this manuscript page, provide a detailed analysis 
        following these guidelines:

        ANALYSIS GUIDELINES:

        CONTEXT:
        Metadata: {json.dumps(metadata, indent=2)}
        Page: {page_number}
        Previous Content: {previous_page['revised_transcription'] if previous_page else 'Not available'}
        Following Content: {next_page['transcription'] if next_page else 'Not available'}
        Notes: {notes if notes else 'Not given'}

        Begin by examining the metadata to understand:
        - The manuscript's purpose and historical context
        - Expected content type and organization
        - Known authors, sources, or attributions
        This context should inform your entire analysis, especially name identification and terminology.

        ANALYSIS GUIDELINES:

        1. Page Layout and Type:
        First determine the page's basic arrangement:
        - Text layout (single column, double column, etc.)
        - Number of lines per column
        - Text blocks and their relationships
        - Presence of ruling or frames
        - Headers, footers, or running text

        Then classify the page type:
        - Primary text page (describe arrangement)
        - Special element (binding, flyleaf, illustration)
        - Mixed content (note how text and decoration interact)
        This organization determines your transcription approach.

        2. Transcription Approach:
        Initial transcription:
        For each text block in order:
        - Start new blocks with ||
        - Record each line, marking breaks with |
        - Follow the text's visual arrangement
        - Use [brackets] for uncertain readings
        - Use <angle brackets> for editorial additions
        - Mark illegible text with {{...}}
        - Maintain original letter forms and spelling
        Note column changes and uncertain readings in transcription_notes

        Revised transcription:
        Working with the text's structure:
        - Use context to correct likely misreadings
        - Apply historical and language knowledge
        - Standardize letter forms (e.g., 's' for 'ſ')
        - Resolve abbreviations using context
        - Choose readings that align with:
        * Document's language and period
        * Names from metadata
        * Surrounding context
        - Preserve text block organization
        Document significant revisions in transcription_notes

        3. Summary Writing:
        Create a clear narrative of the page's content that could be read in sequence with other page summaries to understand the manuscript. Focus on:
        - What actually happens or is discussed
        - How ideas or narratives progress
        - Who speaks or is discussed
        - What sources are quoted or referenced
        - Which topics begin or conclude
        The summary should help readers follow the text's development while supporting search and navigation, 
        so note headers and marks for possible beginning and ends of sections. 
        Note significant interpretive decisions in content_notes.
        Avoid:
        - "This page contains..." or "This page shows..."
        - Generic content descriptions or information that applies to the whole manuscript

        4. Keyword Selection:
        Choose terms researchers would use to find this content across manuscripts:
        - Historical figures (standard English forms)
        - Key concepts and themes
        - Places and events
        - Text genres or types
        Base keywords on:
        - The specific content of this page
        - Standard terminology of the period
        - Likely research interests
        Document any uncertainty about name identification in content_notes

        5. Marginalia Documentation:
        Record both text and visual elements in margins:
        "[location]: [content] ([function/type])"
        - Note location (top, bottom, left, right, interlinear)
        - Describe content (text or visual elements)
        - Indicate function (correction, commentary, decoration)
        Include significant margin features in content_notes

        Confidence Score:
        - Higher (80-100): Clear text, standard content
        - Medium (60-80): Some uncertainty in readings
        - Lower (below 60): Significant interpretation required
        Base this on the certainty of your readings and interpretations

        Return ONLY a JSON object with this structure:
        {{
            "transcription": "initial transcription with markup (empty string for non-text pages)",
            "revised_transcription": "refined version after context review (empty string for non-text pages)",
            "summary": "brief description following the guidelines above",
            "keywords": ["terms following the purpose guidelines above"],
            "marginalia": ["location: content"],
            "confidence": number (0-100),
            "transcription_notes": "challenges and resolutions",
            "content_notes": "scholarly observations and contextual information, as well as descriptions of illustrations and artistic elements."
        }}"""
    
    def process_page(self, image_path: str, metadata: Dict, page_number: int,
                    previous_page: Optional[Dict] = None,
                    next_page: Optional[Dict] = None, notes: str = None) -> Dict:
        """Processes a single page, optionally replacing an existing transcription.

        Args:
            image_path: Path to the image of the page.
            metadata: Manuscript metadata.
            page_number: The page number.
            previous_page: Transcription data for the previous page (optional).
            next_page: Transcription data for the next page (optional).
            notes: Data for improving the transcription (optional)
        
        Returns:
            A dictionary containing the transcription results for the page.
        """

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                self.rate_limiter.wait_if_needed()
                image = Image.open(image_path)
                prompt = self._construct_prompt(metadata, page_number, previous_page, next_page, notes)

                response = self.model.generate_content(
                    [prompt, image],
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )

                result = extract_json_from_response(response.text)
                result['page_number'] = page_number
                return result

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2**retry_count * 30
                    logger.warning(f"Error processing page {page_number}, attempt {retry_count}. "
                                   f"Waiting {wait_time} seconds before retry. Error: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to process page {page_number} after {max_retries} attempts: {str(e)}")
                    return {
                        'page_number': page_number,
                        'error': str(e),
                        'transcription': '',
                        'revised_transcription': '',
                        'summary': '',
                        'keywords': [],
                        'marginalia': [],
                        'confidence': 0,
                        'transcription_notes': f'Failed after {max_retries} attempts: {str(e)}',
                        'content_notes': ''
                    }

def process_manuscript(manuscript_path: str, output_dir: str,
                      processor: ManuscriptProcessor, replace_existing: bool = False) -> Optional[Dict]:
    """Process all pages in a manuscript directory.  Replaces existing pages if replace_existing is True."""

    try:
        # Load metadata
        metadata_path = os.path.join(manuscript_path, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_path}")  # More specific error message
        return None

    try:
        # Setup paths and files
        manuscript_title = metadata.get('Title', os.path.basename(manuscript_path))
        safe_title = create_safe_filename(manuscript_title)

        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', 'webp'}
        image_files = sorted(
            f for f in os.listdir(manuscript_path)
            if Path(f).suffix.lower() in image_extensions
        )

        manuscript_output_dir = os.path.join(output_dir, safe_title)
        os.makedirs(manuscript_output_dir, exist_ok=True)

        # Check for existing transcription
        output_path = os.path.join(manuscript_output_dir, 'transcription.json')
        existing_results = None
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            logger.info(f"Found existing transcription with {len(existing_results.get('pages', {}))} pages")

            if not replace_existing:  # Log what is being skipped
                num_skipped = len([page for page in existing_results.get('pages', {}).values() if 'error' not in page])
                if num_skipped:
                    logger.info(f"Skipping {num_skipped} existing transcribed pages without errors (use --replace to overwrite)")

        except FileNotFoundError:
            pass  # File does not exist yet. Handle normally.

        except Exception as e:
            logger.error(f"Error loading existing transcription: {e}")


        # Initialize results. If replace_existing is True then discard existing_results
        results = existing_results if existing_results and not replace_existing else {
            'manuscript_title': manuscript_title,
            'metadata': metadata,
            'pages': {},
            'total_pages': len(image_files),
            'successful_pages': 0
        }


        # Always create a 20-character display title
        title_length = 20
        display_title = manuscript_title[:17] + '...' if len(manuscript_title) > title_length else manuscript_title.ljust(title_length)
        
        # Process pages
        with tqdm(total=len(image_files), desc=display_title, unit='page') as pbar:  # Simplified tqdm

            for page_number in range(1, len(image_files) + 1):
                try:
                    previous_page = results['pages'].get(str(page_number - 1)) if page_number > 1 else None
                    next_page = results['pages'].get(str(page_number + 1)) if page_number < len(image_files) else None
                    existing_page = results['pages'].get(str(page_number))

                    if existing_page and not replace_existing and 'error' not in existing_page:
                        page_result = existing_page  # Skip if not replacing and page is good.

                    else:  # Always transcribe if there is no existing or it should be replaced.
                        page_result = processor.process_page(
                            os.path.join(manuscript_path, image_files[page_number - 1]),
                            metadata,
                            page_number,
                            previous_page,
                            next_page,
                            notes=""

                        )
                    results['pages'][str(page_number)] = page_result

                    if 'error' not in page_result:
                        if not existing_page:  # Only increment if this is a new success
                            results['successful_pages'] += 1

                    pbar.update(1)

                    # Save progress after each page
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    logger.error(f"Error processing page {page_number}: {e}")

        return results

    except Exception as e:
        logger.error(f"Failed to process manuscript at {manuscript_path}: {e}")
        return None

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process medieval manuscripts using Gemini Vision API.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        default='data/raw',
        help='Directory containing manuscript folders'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/transcripts',
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--manuscripts',
        type=int,
        help='Number of manuscripts to process (default: process all)'
    )
    
    parser.add_argument(
        '--api-key',
        help='Gemini API key (alternative to environment variable)'
    )
    
    return parser.parse_args()

def main():
    pass

if __name__ == "__main__":
    main()