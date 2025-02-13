import argparse
from datetime import datetime
import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from PIL import Image
import asyncio
import google.generativeai as genai
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSection(NamedTuple):
    """A section of the main body text."""
    name: str  # e.g., "column_1", "paragraph_1", etc.
    text: str

class Illustration(NamedTuple):
    """An illustration or decorative element."""
    location: str  # e.g., "top_margin", "folio_12v", etc.
    description: str
    dimensions: Optional[str] = None  # e.g., "50x75mm"

class MarginalNote(NamedTuple):
    """A marginal annotation or note."""
    location: str  # e.g., "left_margin", "bottom_margin"
    text: str
    hand: Optional[str] = None  # Different from main scribe?

class Note(NamedTuple):
    """Additional textual content like headers, footers, or special text."""
    type: str  # e.g., "header", "footer", "chapter_number"
    text: str
    location: Optional[str] = None

@dataclass
class TranscriptionResult:
    """Container for transcription results."""
    body: List[TextSection]
    illustrations: Optional[List[Illustration]] = None
    marginalia: Optional[List[MarginalNote]] = None
    notes: Optional[List[Note]] = None
    language: str = ""
    transcription_notes: str = ""

def create_robust_pattern(field_name: str) -> str:
    """Creates a robust pattern for extracting JSON fields with proper Unicode support."""
    return fr'"{field_name}"\s*:\s*"((?:\\.|[^"\\])*?)"'

def extract_array_field(text: str, field_name: str) -> List[str]:
    """Extracts an array field from text, handling various formats."""
    array_pattern = fr'"{field_name}"\s*:\s*\[(.*?)\]'
    match = re.search(array_pattern, text, re.DOTALL)
    if match:
        items = re.findall(r'["\']((?:\\.|[^"\'\\])*?)["\']', match.group(1))
        return [item.strip() for item in items if item.strip()]
    return []

def extract_object_array(text: str, field_name: str, required_fields: List[str]) -> List[Dict]:
    """Extracts an array of objects with specified required fields."""
    array_pattern = fr'"{field_name}"\s*:\s*\[(.*?)\](?=\s*,\s*"|\s*}})'
    match = re.search(array_pattern, text, re.DOTALL)
    if not match:
        return []

    array_text = match.group(1)
    objects = []
    
    # Match individual objects in the array
    object_matches = re.finditer(r'{(.*?)}', array_text, re.DOTALL)
    
    for obj_match in object_matches:
        obj_text = obj_match.group(1)
        extracted_obj = {}
        
        for field in required_fields:
            field_match = re.search(create_robust_pattern(field), obj_text)
            if field_match:
                extracted_obj[field] = field_match.group(1)
        
        if all(field in extracted_obj for field in required_fields):
            objects.append(extracted_obj)
    
    return objects

def extract_json_from_response(text: str) -> Dict:
    """Extracts structured transcription data from text response."""
    # Remove code block markers if present
    text = re.sub(r'^```(json)?\s*|\s*```\s*$', '', text, flags=re.MULTILINE)
    
    # Try standard JSON parsing first
    try:
        parsed = json.loads(text)
        if validate_transcription_data(parsed):
            return parsed
    except json.JSONDecodeError:
        logger.info("Direct JSON parsing failed, attempting field extraction")
    
    # Initialize empty result
    result = {}
    extraction_success = False
    
    try:
        # Extract all possible content arrays
        content_arrays = {
            'body': ['name', 'text'],
            'illustrations': ['location', 'description'],
            'marginalia': ['location', 'text'],
            'notes': ['type', 'text']
        }
        
        for array_name, required_fields in content_arrays.items():
            extracted = extract_object_array(text, array_name, required_fields)
            if extracted:
                result[array_name] = extracted
                extraction_success = True
        
        # Extract required string fields
        notes_match = re.search(create_robust_pattern('transcription_notes'), text)
        if notes_match:
            result['transcription_notes'] = notes_match.group(1)
            extraction_success = True
        else:
            # transcription_notes is required - if missing, use fallback
            logger.error("Failed to extract required transcription_notes")
            return create_fallback_response(text)
        
        # Extract optional language field
        language_match = re.search(create_robust_pattern('language'), text)
        if language_match:
            result['language'] = language_match.group(1)
        else:
            result['language'] = ""  # Empty string is valid for language
        
        # Ensure we have minimal valid content
        if extraction_success and validate_transcription_data(result):
            # Add empty lists for any missing content arrays
            for array_name in content_arrays.keys():
                if array_name not in result:
                    result[array_name] = []
            return result
        
        logger.error("Failed to extract sufficient transcription data")
        return create_fallback_response(text)
        
    except Exception as e:
        logger.error(f"Error during JSON extraction: {e}")
        return create_fallback_response(text)
    
def validate_transcription_data(data: Dict) -> bool:
    """
    Validates only that the types are correct for any fields that exist.
    Does not require any fields to be present or have specific subfields.
    """
    # Must be a dictionary
    if not isinstance(data, dict):
        return False
    
    # Validate body is a list of dictionaries if present
    if 'body' in data:
        if not isinstance(data['body'], list):
            return False
        for section in data['body']:
            if not isinstance(section, dict):
                return False
    
    # Validate illustrations is a list of dictionaries if present
    if 'illustrations' in data:
        if not isinstance(data['illustrations'], list):
            return False
        for item in data['illustrations']:
            if not isinstance(item, dict):
                return False
    
    # Validate marginalia is a list of dictionaries if present
    if 'marginalia' in data:
        if not isinstance(data['marginalia'], list):
            return False
        for item in data['marginalia']:
            if not isinstance(item, dict):
                return False
    
    # Validate notes is a list of dictionaries if present
    if 'notes' in data:
        if not isinstance(data['notes'], list):
            return False
        for item in data['notes']:
            if not isinstance(item, dict):
                return False
    
    # Validate language is a string if present
    if 'language' in data:
        if not isinstance(data['language'], str):
            return False
    
    # Validate transcription_notes is a string if present
    if 'transcription_notes' in data:
        if not isinstance(data['transcription_notes'], str):
            return False
    
    return True

def create_fallback_response(text: str) -> Dict:
    """Creates a basic valid response structure containing the original text."""
    return {
        'body': [{
            'name': 'unstructured_content',
            'text': text
        }],
        'illustrations': [],
        'marginalia': [],
        'notes': [],
        'language': 'unknown',
        'transcription_notes': 'Failed to parse structured response'
    }

class PageTranscriber:
    def __init__(self, prompt_path: str = "prompts/transcription_prompt_staged.txt"):
        """Initialize the transcriber with Gemini API and prompt template."""
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-pro-exp")
        self.backup_model = genai.GenerativeModel("gemini-1.5-pro")

    def _convert_to_named_tuple(self, data: Dict, tuple_class: type) -> NamedTuple:
        """Convert a dictionary to the appropriate named tuple type."""
        field_names = tuple_class._fields
        field_values = [data.get(field) for field in field_names]
        return tuple_class(*field_values)

    def _parse_transcription_response(self, response_text: str) -> TranscriptionResult:
        """Parse the model's response into a structured transcription result."""
        try:
            data = extract_json_from_response(response_text)
            
            # Convert body sections to TextSection tuples
            body = [TextSection(section['name'], section['text']) 
                   for section in data['body']]
            
            # Convert illustrations if present
            illustrations = None
            if data.get('illustrations'):
                illustrations = [self._convert_to_named_tuple(ill, Illustration) 
                               for ill in data['illustrations']]
            
            # Convert marginalia if present
            marginalia = None
            if data.get('marginalia'):
                marginalia = [self._convert_to_named_tuple(note, MarginalNote) 
                            for note in data['marginalia']]
            
            # Convert additional notes if present
            notes = None
            if data.get('notes'):
                notes = [self._convert_to_named_tuple(note, Note) 
                        for note in data['notes']]
            
            return TranscriptionResult(
                body=body,
                illustrations=illustrations,
                marginalia=marginalia,
                notes=notes,
                language=data.get('language', ''),
                transcription_notes=data.get('transcription_notes', '')
            )
            
        except Exception as e:
            logger.error(f"Error parsing transcription response: {e}")
            return TranscriptionResult(
                body=[TextSection(
                    name="error_content",
                    text=f"Failed to parse response: {str(e)}\nOriginal text: {response_text}"
                )],
                language="unknown",
                transcription_notes=f"Error during parsing: {str(e)}"
            )

    def _serialize_result(self, result: TranscriptionResult) -> Dict:
        """Convert TranscriptionResult to a JSON-serializable dictionary."""
        return {
            'body': [{'name': section.name, 'text': section.text} 
                    for section in result.body],
            'illustrations': [dict(ill._asdict()) for ill in (result.illustrations or [])],
            'marginalia': [dict(note._asdict()) for note in (result.marginalia or [])],
            'notes': [dict(note._asdict()) for note in (result.notes or [])],
            'language': result.language,
            'transcription_notes': result.transcription_notes
        }

    async def transcribe_page(self, manuscript_dir: str, page_number: int, notes: str = "") -> TranscriptionResult:
        """Transcribe a single manuscript page with context from adjacent pages."""
        manuscript_path = Path(manuscript_dir)
        max_retries = 2
        retry_delay = 30  # seconds
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Load metadata
                metadata_path = manuscript_path / 'standard_metadata.json'
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Get image files
                image_dir = manuscript_path / 'images'
                image_files = sorted([f for f in image_dir.iterdir() 
                                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
                
                if not image_files:
                    raise ValueError(f"No image files found in {image_dir}")
                
                if page_number < 1 or page_number > len(image_files):
                    raise ValueError(f"Invalid page number {page_number}. Valid range: 1-{len(image_files)}")
                
                # Process the page
                image = Image.open(image_files[page_number - 1])
                metadata_json = json.dumps(metadata, indent=2).replace("{", "{{").replace("}", "}}")
                prompt = self.prompt_template.format(
                    metadata=metadata_json,
                    page_number=page_number,
                    total_pages=len(image_files),
                    notes=notes if notes else ""
                )
                if attempt > 0: #If it is not the first attempt
                    response = self.backup_model.generate_content(
                        [prompt, image],
                        generation_config={"temperature": 0.3},
                        safety_settings={
                            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                else:
                    response = self.model.generate_content(
                        [prompt, image],
                        generation_config={"temperature": 0.3},
                        safety_settings={
                            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                
                result = self._parse_transcription_response(response.text)
                
                # Check specifically for the failure indicator in transcription_notes
                if result.transcription_notes == "Failed to parse structured response":
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed to parse response. Retrying in {retry_delay*(attempt+1)} seconds...")
                        logger.info(f"Response: {result}")
                        await asyncio.sleep(retry_delay*(attempt+1))
                        continue
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed to parse structured response")
                else:
                    return result
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")
                    return TranscriptionResult(
                        body=[TextSection(
                            name="transcription_error",
                            text=f"Failed to transcribe page after {max_retries + 1} attempts: {str(e)}"
                        )],
                        language="unknown",
                        transcription_notes=f"Transcription error after {max_retries + 1} attempts: {str(e)}"
                    )
        
        # If we get here, we've exhausted all retries
        return TranscriptionResult(
            body=[TextSection(
                name="transcription_error",
                text=f"Failed to get valid transcription after {max_retries + 1} attempts"
            )],
            language="unknown",
            transcription_notes=f"Failed to get valid transcription after {max_retries + 1} attempts"
        )

async def main():
    parser = argparse.ArgumentParser(description='Transcribe manuscript pages')
    parser.add_argument('dir', help='Path to manuscript directory')
    parser.add_argument('--page', type=int, help='Specific page to transcribe')
    parser.add_argument('--prompt', default='prompts/transcription_prompt_staged.txt',
                       help='Path to prompt template file')
    parser.add_argument('--notes', default='', 
                       help='Additional notes or context for analysis')
    parser.add_argument('--replace', action='store_true',
                       help='Replace existing transcriptions')
    args = parser.parse_args()
    
    try:
        manuscript_path = Path(args.dir)
        if not manuscript_path.is_dir():
            raise ValueError(f"Not a directory: {manuscript_path}")

        # Initialize transcriber
        transcriber = PageTranscriber(args.prompt)
        transcript_path = manuscript_path / 'transcript.json'
        
        # Load or create transcript
        if transcript_path.exists() and not args.replace:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        else:
            # Read metadata for new transcript
            with open(manuscript_path / 'standard_metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Initialize new transcript
            transcript = {
                'manuscript_title': metadata.get('title', 'Untitled Manuscript'),
                'metadata': metadata,
                'pages': {},
                'last_updated': datetime.now().isoformat(),
                'successful_pages': 0,
                'failed_pages': []
            }

        # Get list of image files
        image_files = sorted([
            f for f in (manuscript_path / 'images').iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        ])
        transcript['total_pages'] = len(image_files)

        if args.page:
            # Process single page
            if args.page < 1 or args.page > len(image_files):
                raise ValueError(f"Invalid page number. Valid range: 1-{len(image_files)}")
            
            # Load existing transcript if available
            existing_transcript = None
            if transcript_path.exists():
                try:
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        existing_transcript = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not load existing transcript: {e}")
            
            # If we're not replacing and the page exists, skip it
            if not args.replace and existing_transcript and str(args.page) in existing_transcript['pages']:
                logger.info(f"Skipping page {args.page} (already exists)")
                return
            
            # If we have an existing transcript and we're replacing a page,
            # update our working transcript with all existing pages
            if existing_transcript:
                transcript['pages'] = existing_transcript['pages']
                transcript['successful_pages'] = existing_transcript.get('successful_pages', 0)
                transcript['failed_pages'] = existing_transcript.get('failed_pages', [])
            
            # Process the requested page
            result = await transcriber.transcribe_page(manuscript_path, args.page, args.notes)
            transcript['pages'][str(args.page)] = transcriber._serialize_result(result)
            
            # Update metadata
            if result.transcription_notes == "Failed to parse structured response":
                if args.page not in transcript['failed_pages']:
                    transcript['failed_pages'].append(args.page)
            else:
                if args.page in transcript['failed_pages']:
                    transcript['failed_pages'].remove(args.page)
                transcript['successful_pages'] += 1

            transcript['last_updated'] = datetime.now().isoformat()
            
            # Save the updated transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Completed transcription of page {args.page}")
            
        else:
            # Process all pages with progress bar
            from tqdm import tqdm
            pages_to_process = []
            
            # Determine which pages need processing
            for page_num in range(1, len(image_files) + 1):
                if args.replace or str(page_num) not in transcript['pages']:
                    pages_to_process.append(page_num)
            
            if not pages_to_process:
                logger.info("No pages to process (use --replace to force reprocessing)")
                return
            
            for page_num in tqdm(pages_to_process, desc="Transcribing pages"):
                try:
                    result = await transcriber.transcribe_page(manuscript_path, page_num, args.notes)
                    transcript['pages'][str(page_num)] = transcriber._serialize_result(result)
                    
                    # Update metadata and save after each page
                    if result.transcription_notes == "Failed to parse structured response":
                        if page_num not in transcript['failed_pages']:
                            transcript['failed_pages'].append(page_num)
                    else:
                        if page_num in transcript['failed_pages']:
                            transcript['failed_pages'].remove(page_num)
                        transcript['successful_pages'] += 1
                    
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        json.dump(transcript, f, indent=2, ensure_ascii=False)
                        
                except Exception as e:
                    logger.error(f"Failed to transcribe page {page_num}: {e}")
                    transcript['failed_pages'].append(page_num)
                    
                    # Save after failures too
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcript saved to {transcript_path}")
                    
    except Exception as e:
        logger.error(f"Error in transcription process: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())