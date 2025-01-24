import argparse
import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
from tqdm import tqdm
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API requests respecting both per-minute and daily limits."""
    
    def __init__(self):
        self.minute_limit = 15
        self.day_limit = 1500
        self.minute_tokens = 15
        self.day_tokens = 1500
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
    """Extract and validate JSON from model response."""
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response: {text[:200]}...")
    
    return {
        'transcription': '',
        'revised_transcription': '',
        'summary': 'Failed to parse response',
        'keywords': [],
        'marginalia': [],
        'confidence': 0,
        'transcription_notes': 'Failed to parse response',
        'content_notes': 'Failed to extract valid JSON response'
    }

def format_malformed_response(text: str, model: genai.GenerativeModel) -> Dict:
    """Attempt to format malformed response using Gemini Pro."""
    prompt = f"""Format this transcription response as valid JSON with this structure:
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
        if not formatted.get('transcription'):
            raise TranscriptionError("Failed to format response")
        return formatted
        
    except Exception as e:
        logger.error(f"Failed to format response: {str(e)}")
        return {
            'transcription': text,
            'revised_transcription': text,
            'summary': 'Automatic formatting failed',
            'keywords': [],
            'marginalia': [],
            'confidence': 0,
            'transcription_notes': f'Failed to format response: {str(e)}',
            'content_notes': 'Original unformatted response preserved in transcription field'
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
    
    def _construct_prompt(self, metadata: Dict, page_number: int,
                         previous_page: Optional[Dict], next_page: Optional[Dict]) -> str:
        """Create the analysis prompt for a manuscript page."""
        return f"""As an expert paleographer examining this manuscript page, provide a detailed analysis 
        following these guidelines:

        MANUSCRIPT CONTEXT:
        Metadata: {json.dumps(metadata, indent=2)}
        Page Number: {page_number}
        Previous Content: {previous_page['revised_transcription'] if previous_page else 'Not available'}
        Following Content: {next_page['transcription'] if next_page else 'Not available'}

        TRANSCRIPTION GUIDELINES:
        1. Main Text:
           - Preserve original line breaks with |
           - Mark paragraph breaks with ||
           - Use [brackets] for uncertain readings
           - Use <angle brackets> for editorial additions
           - Mark illegible text with {{...}}

        2. Marginalia:
           - Start each entry with specific location
           - Note contemporary vs. later additions
           - Include both text and visual elements

        3. Process:
           - Initial careful reading
           - Document uncertain sections
           - Compare with context
           - Refine readings
           - Note remaining uncertainties

        Return ONLY a JSON object with this structure:
        {{
            "transcription": "initial transcription with markup",
            "revised_transcription": "refined version after context review",
            "summary": "brief content description",
            "keywords": ["theme1", "theme2", "subject1"],
            "marginalia": ["location: content"],
            "confidence": number (0-100),
            "transcription_notes": "challenges and resolutions",
            "content_notes": "scholarly observations"
        }}"""
    
    def process_page(self, image_path: str, metadata: Dict, page_number: int,
                    previous_page: Optional[Dict] = None,
                    next_page: Optional[Dict] = None) -> Dict:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.rate_limiter.wait_if_needed()
                image = Image.open(image_path)
                prompt = self._construct_prompt(metadata, page_number, previous_page, next_page)
                
                response = self.vision_model.generate_content(
                    [prompt, image],
                    generation_config={"temperature": 0.2},
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                result = extract_json_from_response(response.text)
                
                # If JSON parsing failed, try to format with text model
                if not result.get('transcription'):
                    self.rate_limiter.wait_if_needed()
                    result = format_malformed_response(response.text, self.text_model)
                
                result['page_number'] = page_number
                return result
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count * 30
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
                      processor: ManuscriptProcessor) -> Optional[Dict]:
    """Process all pages in a manuscript directory."""
    try:
        # Load metadata
        metadata_path = os.path.join(manuscript_path, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Setup paths and files
        manuscript_title = metadata.get('Title', os.path.basename(manuscript_path))
        safe_title = "".join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' 
                            for c in manuscript_title)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = sorted(
            f for f in os.listdir(manuscript_path)
            if Path(f).suffix.lower() in image_extensions
        )
        
        manuscript_output_dir = os.path.join(output_dir, safe_title)
        os.makedirs(manuscript_output_dir, exist_ok=True)
        
        # Initialize results
        results = {
            'manuscript_title': manuscript_title,
            'metadata': metadata,
            'pages': [],
            'total_pages': len(image_files),
            'successful_pages': 0,
            'failed_pages': []
        }
        
        # Process pages
        with tqdm(total=len(image_files), desc=f"Processing {manuscript_title}", unit="page") as pbar:
            idx = 0
            while idx < len(image_files):
                try:
                    # Get context
                    previous_page = results['pages'][idx-1] if idx > 0 else None
                    next_page = None
                    
                    # Process page
                    page_result = processor.process_page(
                        os.path.join(manuscript_path, image_files[idx]),
                        metadata,
                        idx + 1,
                        previous_page,
                        next_page
                    )
                    
                    # Update results
                    if len(results['pages']) <= idx:
                        results['pages'].append(page_result)
                    else:
                        results['pages'][idx] = page_result
                    
                    if 'error' not in page_result:
                        results['successful_pages'] += 1
                        pbar.set_postfix(successful=f"{results['successful_pages']}/{idx+1}")
                        idx += 1
                    else:
                        if idx + 1 not in results['failed_pages']:
                            results['failed_pages'].append(idx + 1)
                        pbar.set_postfix(failed=f"{len(results['failed_pages'])} pages")
                    
                    pbar.update(1)
                    
                    # Save progress
                    output_path = os.path.join(manuscript_output_dir, 'transcription.json')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    
                except Exception as e:
                    logger.error(f"Error processing page {idx + 1}: {e}")
                    if idx + 1 not in results['failed_pages']:
                        results['failed_pages'].append(idx + 1)
                    idx += 1
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process manuscript at {manuscript_path}: {e}")
        return None

def batch_process_manuscripts(
    input_dir: str = 'data/raw',
    output_dir: str = 'data/transcripts',
    manuscript_limit: Optional[int] = None
) -> List[Dict]:
    """
    Process manuscripts in the input directory.
    
    Args:
        input_dir: Directory containing manuscript folders
        output_dir: Directory for output files
        manuscript_limit: Maximum number of manuscripts to process (None for all)
    
    Returns:
        List of processing results for each manuscript
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        processor = ManuscriptProcessor()
        
        # Get list of valid manuscript directories
        manuscripts = [item for item in sorted(os.listdir(input_dir))
                      if os.path.isdir(os.path.join(input_dir, item)) and
                      os.path.exists(os.path.join(input_dir, item, 'metadata.json'))]
        
        # Apply manuscript limit if specified
        if manuscript_limit is not None:
            manuscripts = manuscripts[:manuscript_limit]
            logger.info(f"Processing {manuscript_limit} of {len(manuscripts)} available manuscripts")
        
        batch_results = []
        for item in tqdm(manuscripts, desc="Processing manuscripts", unit="manuscript"):
            try:
                result = process_manuscript(
                    os.path.join(input_dir, item),
                    output_dir,
                    processor
                )
                
                if result:
                    batch_results.append(result)
                    
                    # Save batch progress
                    batch_results_path = os.path.join(output_dir, 'batch_results.json')
                    with open(batch_results_path, 'w', encoding='utf-8') as f:
                        json.dump(batch_results, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"Failed to process manuscript {item}: {e}")
        
        return batch_results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return []

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
    """Main entry point for manuscript processing."""
    try:
        args = parse_args()
        
        # Set API key if provided
        if args.api_key:
            os.environ['GEMINI_API_KEY'] = args.api_key
        
        # Validate inputs
        if not os.path.isdir(args.input_dir):
            raise ValueError(f"Input directory does not exist: {args.input_dir}")
        
        if args.manuscripts is not None and args.manuscripts <= 0:
            raise ValueError("Number of manuscripts must be positive")
        
        # Process manuscripts
        results = batch_process_manuscripts(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            manuscript_limit=args.manuscripts
        )
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total manuscripts processed: {len(results)}")
        
        total_pages = sum(result['total_pages'] for result in results)
        successful_pages = sum(result['successful_pages'] for result in results)
        print(f"Total pages processed: {successful_pages}/{total_pages}")
        
        for result in results:
            print(f"\nManuscript: {result['manuscript_title']}")
            print(f"Pages processed: {result['successful_pages']}/{result['total_pages']}")
            if result['failed_pages']:
                print(f"Failed pages: {result['failed_pages']}")
                
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()