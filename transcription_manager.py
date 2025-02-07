from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Optional, AsyncIterator, List
import asyncio
from gemini_transcribe import ManuscriptProcessor  # Ensure this import is correct
from summarizer import ManuscriptSummarizer  # Ensure this import is correct
import os

# Configure logging (consistent with manuscript_server.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionManager:
    def __init__(self, catalogue_dir: str = "data/catalogue"):
        """Initialize the transcription manager."""
        self.catalogue_dir = Path(catalogue_dir)
        logger.info(f"TranscriptionManager initialized with catalogue_dir: {self.catalogue_dir}")

        # Initialize processors
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")

        self.manuscript_processor = ManuscriptProcessor()
        self.summarizer = ManuscriptSummarizer(api_key)

        # Track active transcriptions (this is where the state is stored)
        self.active_transcriptions: Dict[str, Dict] = {}
        logger.info("TranscriptionManager initialization complete.")


    async def transcribe_page(self, manuscript_id: str, page_number: int, notes: str = None) -> Dict:
        """Transcribes a single page (no changes needed here)."""
        try:
            manuscript_dir = self.catalogue_dir / manuscript_id
            image_dir = manuscript_dir / 'images'

            # Get all files in the images directory
            image_files = sorted(f for f in image_dir.iterdir() if f.is_file())

            if not image_files:
                raise ValueError(f"No image files found in {image_dir}")

            if page_number < 1 or page_number > len(image_files):
                raise ValueError(f"Invalid page number: {page_number}. Valid range: 1-{len(image_files)}")

            metadata = self._load_metadata(manuscript_id)

            transcription_path = manuscript_dir / 'transcription.json'
            previous_page = None
            next_page = None

            if transcription_path.exists():
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                    pages = trans_data.get('pages', {})
                    if str(page_number - 1) in pages:
                        previous_page = pages[str(page_number - 1)]
                    if str(page_number + 1) in pages:
                        next_page = pages[str(page_number + 1)]

            result = self.manuscript_processor.process_page(
                str(image_files[page_number - 1]),
                metadata,
                page_number,
                previous_page,
                next_page,
                notes
            )

            await self._update_transcription_file(manuscript_id, page_number, result)
            return result

        except Exception as e:
            logger.error(f"Error transcribing page {page_number}: {e}")
            return {
                'error': str(e),
                'page_number': page_number
            }

    async def transcribe_manuscript(self, manuscript_id: str, notes: str = None) -> AsyncIterator[Dict]:
        """Manages the transcription of an entire manuscript (with detailed logging)."""
        logger.info(f"1. transcribe_manuscript called for ID: {manuscript_id}, notes: {notes}")

        if manuscript_id in self.active_transcriptions:
            logger.info(f"2. Manuscript {manuscript_id} already being transcribed.")
            yield {'status': 'already_running'}  # Return immediately if already running
            return

        manuscript_dir = self.catalogue_dir / manuscript_id
        image_dir = manuscript_dir / 'images'
        logger.info(f"3. Checking manuscript directory: {manuscript_dir}, image directory: {image_dir}")

        if not manuscript_dir.exists() or not image_dir.exists():
            logger.error(f"4. Manuscript directory or image directory not found: {manuscript_dir}")
            yield {'status': 'error', 'error': 'Manuscript not found'}
            return

        total_pages = len(list(image_dir.iterdir()))
        status_info = {
            'total_pages': total_pages,
            'successful_pages': 0,
            'failed_pages': []
        }
        logger.info(f"5. Initial status info: {status_info}")

        try:
            # Initialize tracking (store transcription state)
            self.active_transcriptions[manuscript_id] = {
                'started_at': datetime.now().isoformat(),
                'current_page': 0,  # Track the current page being processed
                'status': 'running'
            }
            logger.info(f"6. Transcription started for {manuscript_id}")

            # Load existing transcription data (if any)
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            processed_pages = set()  # Keep track of already processed pages
            logger.info(f"7. Checking for existing transcription: {transcription_path}")
            if transcription_path.exists():
                logger.info("8. Transcription file exists.")
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                    # Only add pages that were successfully transcribed
                    processed_pages = {
                        int(page) for page, data in trans_data.get('pages', {}).items()
                        if 'error' not in data  # Check for 'error' key
                    }
                    status_info['successful_pages'] = len(processed_pages)
            else:
                logger.info("10. Transcription file does not exist.")

            logger.info(f"11. Loaded existing data. Processed pages: {processed_pages}, Total Pages: {total_pages}")

            # Loop through each page and transcribe it
            for page_number in range(1, total_pages + 1):
                logger.info(f"12. Starting loop iteration for page: {page_number}")
                if page_number in processed_pages:
                    logger.info(f"13. Page {page_number} already processed, skipping.")
                    continue  # Skip already processed pages

                # Update current page in active_transcriptions
                self.active_transcriptions[manuscript_id]['current_page'] = page_number

                logger.info(f"14. Starting transcription for page {page_number}")
                result = await self.transcribe_page(manuscript_id, page_number, notes)  # Await the result
                # logger.info(f"15. Transcription result for page {page_number}: {result}")

                if 'error' not in result:
                    logger.info("16. Page transcribed successfully.")
                    status_info['successful_pages'] += 1
                else:
                    logger.info(f"17. Page transcription failed: {result['error']}")
                    status_info['failed_pages'].append(page_number)


                logger.info(f"18. Yielding status update for page {page_number}")
                yield {  # Yield a status update (for SSE)
                    'status': 'in_progress',
                    'page': page_number,  # Current page number
                    **status_info  # Include total_pages, successful_pages, failed_pages
                }
                logger.info(f"19. Yielded status update for page {page_number}")

            # After all pages are processed (or if an error occurs)
            if status_info['successful_pages'] > 0:
                logger.info(f"20. Generating summary for {manuscript_id}")
                await self.generate_summary(manuscript_id) # Await summary
                logger.info(f"21. Summary generated for {manuscript_id}")

            logger.info(f"22. Transcription complete for {manuscript_id}")
            yield {  # Yield a final 'completed' status
                'status': 'complete',
                **status_info
            }

        except Exception as e:
            logger.exception(f"23. Error transcribing manuscript: {e}")  # Use logger.exception
            yield {
                'status': 'error',
                'error': str(e),
                **status_info
            }
        finally:
            logger.info(f"24. Removing {manuscript_id} from active transcriptions")
            self.active_transcriptions.pop(manuscript_id, None)  # Remove from active transcriptions

    async def generate_summary(self, manuscript_id: str) -> Dict:
        """Generates a summary (no changes needed here)."""
        try:
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            logger.info(f"transcription path: {transcription_path}")
            self.summarizer.update_transcription(transcription_path)
            return {'status': 'success'}

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {'error': str(e)}

    def get_transcription_status(self, manuscript_id: str) -> Optional[Dict]:
        """Gets the current transcription status (no changes needed here)."""
        logger.info(f"get_transcription_status called for {manuscript_id}")
        status = self.active_transcriptions.get(manuscript_id)
        logger.info(f"Returning status: {status}")
        return status

    def _load_metadata(self, manuscript_id: str) -> Dict:
        """Loads metadata (no changes needed here)."""
        metadata_path = self.catalogue_dir / manuscript_id / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)


    async def _update_transcription_file(self, manuscript_id: str,
                                       page_number: int, page_data: Dict) -> None:
        """Updates the transcription file (no changes needed here)."""
        transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
        transcription_path.parent.mkdir(exist_ok=True)

        try:
            # Load existing data or create new
            if transcription_path.exists():
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                image_dir = self.catalogue_dir / manuscript_id / 'images'
                data = {
                    'manuscript_id': manuscript_id,
                    'metadata': self._load_metadata(manuscript_id),
                    'pages': {},
                    'total_pages': len(list(image_dir.iterdir())),
                    'successful_pages': 0,
                    'failed_pages': []
                }

            # Update page data
            data['pages'][str(page_number)] = page_data

            # Update success/failure counts
            if 'error' not in page_data:
                data['successful_pages'] = len(data['pages'].keys())
                if page_number in data['failed_pages']:
                    data['failed_pages'].remove(page_number)
            else:
                if page_number not in data['failed_pages']:
                    data['failed_pages'].append(page_number)

            # Update timestamp
            data['last_updated'] = datetime.now().isoformat()

            # Write updated data
            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error updating transcription file: {e}")
            raise