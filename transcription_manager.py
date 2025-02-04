from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Optional, AsyncIterator, List
import asyncio
from gemini_transcribe import ManuscriptProcessor
from summarizer import ManuscriptSummarizer
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionManager:
    def __init__(self, catalogue_dir: str = "data/catalogue"):
        """Initialize the transcription manager."""
        self.catalogue_dir = Path(catalogue_dir)
        
        # Initialize processors
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
            
        self.manuscript_processor = ManuscriptProcessor()
        self.summarizer = ManuscriptSummarizer(api_key)
        
        # Track active transcriptions
        self.active_transcriptions: Dict[str, Dict] = {}
        
    async def transcribe_page(self, manuscript_id: str, page_number: int) -> Dict:
        """
        Transcribe a single page from a manuscript.
        
        Args:
            manuscript_id: The manuscript identifier
            page_number: The page number to transcribe
            
        Returns:
            Dictionary containing transcription results or error information
        """
        try:
            manuscript_dir = self.catalogue_dir / manuscript_id
            image_dir = manuscript_dir / 'images'
            
            # Ensure proper order of image files
            image_files = sorted(image_dir.glob('*.[jJpP][nNiI][gGfF]'))
            if page_number < 1 or page_number > len(image_files):
                raise ValueError(f"Invalid page number: {page_number}")
            
            # Process the page
            result = self.manuscript_processor.process_page(
                str(image_files[page_number - 1]),
                self._load_metadata(manuscript_id),
                page_number
            )
            
            # Update transcription file
            await self._update_transcription_file(manuscript_id, page_number, result)
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing page {page_number}: {e}")
            return {
                'error': str(e),
                'page_number': page_number
            }
    
    async def transcribe_manuscript(self, manuscript_id: str) -> AsyncIterator[Dict]:
        """
        Manage the transcription of an entire manuscript.
        
        Args:
            manuscript_id: The manuscript identifier
            
        Yields:
            Status updates as transcription progresses
        """
        if manuscript_id in self.active_transcriptions:
            yield {'status': 'already_running'}
            return
        
        try:
            # Initialize tracking
            self.active_transcriptions[manuscript_id] = {
                'started_at': datetime.now().isoformat(),
                'current_page': 0,
                'status': 'running'
            }
            
            manuscript_dir = self.catalogue_dir / manuscript_id
            image_dir = manuscript_dir / 'images'
            image_files = sorted(image_dir.glob('*.[jJpP][nNiI][gGfF]'))
            total_pages = len(image_files)
            
            successful_pages = 0
            failed_pages: List[int] = []
            
            # Process each page
            for page_number in range(1, total_pages + 1):
                self.active_transcriptions[manuscript_id]['current_page'] = page_number
                
                result = await self.transcribe_page(manuscript_id, page_number)
                
                if 'error' not in result:
                    successful_pages += 1
                else:
                    failed_pages.append(page_number)
                
                yield {
                    'status': 'in_progress',
                    'page': page_number,
                    'total_pages': total_pages,
                    'successful_pages': successful_pages,
                    'failed_pages': failed_pages
                }
            
            # Generate summary if we have successful transcriptions
            if successful_pages > 0:
                await self.generate_summary(manuscript_id)
            
            yield {
                'status': 'complete',
                'total_pages': total_pages,
                'successful_pages': successful_pages,
                'failed_pages': failed_pages
            }
            
        except Exception as e:
            logger.error(f"Error transcribing manuscript: {e}")
            yield {
                'status': 'error',
                'error': str(e)
            }
            
        finally:
            self.active_transcriptions.pop(manuscript_id, None)
    
    async def generate_summary(self, manuscript_id: str) -> Dict:
        """
        Generate a summary for a transcribed manuscript.
        
        Args:
            manuscript_id: The manuscript identifier
            
        Returns:
            Dictionary containing summary status or error information
        """
        try:
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            self.summarizer.update_transcription(transcription_path)
            return {'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {'error': str(e)}
    
    def get_transcription_status(self, manuscript_id: str) -> Optional[Dict]:
        """Get the current status of an active transcription."""
        return self.active_transcriptions.get(manuscript_id)
    
    def _load_metadata(self, manuscript_id: str) -> Dict:
        """Load manuscript metadata."""
        metadata_path = self.catalogue_dir / manuscript_id / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def _update_transcription_file(self, manuscript_id: str, 
                                       page_number: int, page_data: Dict) -> None:
        """
        Update the transcription file with new page data.
        
        Args:
            manuscript_id: The manuscript identifier
            page_number: The page number being updated
            page_data: The new page transcription data
        """
        transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
        transcription_path.parent.mkdir(exist_ok=True)
        
        try:
            # Load existing data or create new
            if transcription_path.exists():
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {
                    'manuscript_id': manuscript_id,
                    'metadata': self._load_metadata(manuscript_id),
                    'pages': {},
                    'total_pages': len(list(
                        (self.catalogue_dir / manuscript_id / 'images').glob('*.[jJpP][nNiI][gGfF]')
                    )),
                    'successful_pages': 0,
                    'failed_pages': []
                }
            
            # Update page data
            data['pages'][str(page_number)] = page_data
            
            # Update success/failure counts
            if 'error' not in page_data:
                if str(page_number) not in data['pages']:
                    data['successful_pages'] += 1
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