from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, AsyncIterator, Any
import logging
import asyncio
import json

from transcriber import PageTranscriber
from processor import ManuscriptProcessor

logger = logging.getLogger(__name__)

class CatalogueEventType(Enum):
    # Transcription events
    TRANSCRIPTION_STARTED = "transcription_started"
    TRANSCRIPTION_PAGE_COMPLETE = "transcription_page_complete"
    TRANSCRIPTION_COMPLETE = "transcription_complete"
    TRANSCRIPTION_ERROR = "transcription_error"
    
    # Processing status events
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETE = "processing_complete"
    PROCESSING_ERROR = "processing_error"
    
    # General events
    METADATA_UPDATED = "metadata_updated"
    ERROR_OCCURRED = "error_occurred"

@dataclass
class CatalogueEvent:
    type: CatalogueEventType
    manuscript_id: str
    timestamp: datetime
    data: Dict[str, Any]
    error: Optional[str] = None

class ManuscriptCatalogue:
    """
    Central manager for manuscript processing and access.
    Coordinates between transcription and cataloguing components while
    managing state through events.
    """
    
    def __init__(self, catalogue_dir: str = "data/catalogue"):
        """Initialize the manuscript catalogue with components and event system."""
        self.catalogue_dir = Path(catalogue_dir)
        if not self.catalogue_dir.exists():
            raise FileNotFoundError(f"Catalogue directory not found: {self.catalogue_dir}")
        
        # Initialize components
        self.transcriber = PageTranscriber()
        self.processor = ManuscriptProcessor()
        
        # Event and state management
        self._event_handlers: List[Callable[[CatalogueEvent], None]] = []
        self._active_processes: Dict[str, Dict] = {}
        
        # Initialize manuscript listings
        self.manuscript_listings: Dict[str, Dict] = {}
        self._load_manuscript_listings()
    
    def subscribe(self, handler: Callable[[CatalogueEvent], None]):
        """Subscribe to catalogue events."""
        self._event_handlers.append(handler)
    
    def unsubscribe(self, handler: Callable[[CatalogueEvent], None]):
        """Remove a subscription."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
    
    def _emit_event(self, event: CatalogueEvent):
        """Emit an event to all subscribers."""
        logger.debug(f"Emitting event: {event}")
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def _load_manuscript_listings(self):
        """Load all manuscript listings into memory."""
        self.manuscript_listings.clear()
        for manuscript_dir in self.catalogue_dir.iterdir():
            if not manuscript_dir.is_dir():
                continue
                
            try:
                # Read metadata
                metadata_path = manuscript_dir / 'standard_metadata.json'
                if not metadata_path.exists():
                    continue
                    
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                manuscript_id = manuscript_dir.name
                
                # Get transcription status
                status = self.get_transcription_status(manuscript_id)
                
                self.manuscript_listings[manuscript_id] = {
                    'manuscript_id': manuscript_id,
                    'metadata': metadata,
                    'total_pages': self.get_page_count(manuscript_id),
                    **status
                }
                
            except Exception as e:
                logger.error(f"Error loading manuscript {manuscript_dir.name}: {e}")
                continue

    def get_manuscript_listings(self) -> Dict[str, Dict]:
        """
        Get complete information for all manuscripts in the catalogue.
        
        Returns:
            Dict mapping manuscript_id to manuscript information containing:
            - Full metadata from standard_metadata.json
            - manuscript_id
            - total_pages (from image count)
            - transcription_status (not_started, in_progress, complete)
            - transcribed_pages (count of successfully transcribed pages)
            - last_updated (timestamp of last modification)
        """
        return self.manuscript_listings.copy()  # Return a copy to prevent external modification

    def refresh_manuscript(self, manuscript_id: str) -> None:
        """
        Refresh the listing for a specific manuscript.
        Should be called after any operation that modifies the manuscript.
        """
        if not self.manuscript_exists(manuscript_id):
            return
            
        try:
            metadata_path = self.catalogue_dir / manuscript_id / 'standard_metadata.json'
            if not metadata_path.exists():
                self.manuscript_listings.pop(manuscript_id, None)
                return
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            status = self.get_transcription_status(manuscript_id)
            
            self.manuscript_listings[manuscript_id] = {
                'manuscript_id': manuscript_id,
                'metadata': metadata,
                'total_pages': self.get_page_count(manuscript_id),
                **status
            }
            
            self._emit_event(CatalogueEvent(
                type=CatalogueEventType.METADATA_UPDATED,
                manuscript_id=manuscript_id,
                timestamp=datetime.now(),
                data=self.manuscript_listings[manuscript_id]
            ))
            
        except Exception as e:
            logger.error(f"Error refreshing manuscript {manuscript_id}: {e}")
            self.manuscript_listings.pop(manuscript_id, None)
            
    def manuscript_exists(self, manuscript_id: str) -> bool:
        """Check if a manuscript exists in the catalogue."""
        if manuscript_id not in self.manuscript_listings:
            # Do a filesystem check and refresh if found
            manuscript_dir = self.catalogue_dir / manuscript_id
            if manuscript_dir.exists() and manuscript_dir.is_dir():
                self.refresh_manuscript(manuscript_id)
                return True
            return False
        return True
    
    def get_transcription_status(self, manuscript_id: str) -> Dict:
        """
        Get current transcription status for a manuscript.
        
        Returns:
            Dictionary with status information including:
            - current state (not_started, in_progress, complete)
            - progress information if in_progress
            - error information if failed
            - transcribed_pages count
            - last_updated timestamp
        """
        # Check active processes first
        if manuscript_id in self._active_processes:
            return {
                'status': 'in_progress',
                **self._active_processes[manuscript_id]
            }
            
        # Check transcription file
        transcript_path = self.catalogue_dir / manuscript_id / 'transcript.json'
        if not transcript_path.exists():
            return {
                'status': 'not_started',
                'transcribed_pages': 0
            }
            
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
                
            total_pages = self.get_page_count(manuscript_id)
            transcribed_pages = transcript.get('successful_pages', 0)
            
            return {
                'status': 'complete' if transcribed_pages == total_pages else 'partial',
                'transcribed_pages': transcribed_pages,
                'total_pages': total_pages,
                'last_updated': transcript.get('last_updated')
            }
            
        except Exception as e:
            logger.error(f"Error reading transcript for {manuscript_id}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def transcribe_manuscript(self, manuscript_id: str, notes: Optional[str] = None) -> None:
        """
        Transcribe all pages in a manuscript.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
            notes: Optional notes to guide transcription
        """
        if not self.manuscript_exists(manuscript_id):
            raise ValueError(f"Manuscript {manuscript_id} not found")
            
        if manuscript_id in self._active_processes:
            raise RuntimeError(f"Manuscript {manuscript_id} is already being processed")
        
        try:
            self._active_processes[manuscript_id] = {
                'type': 'transcription',
                'started_at': datetime.now().isoformat(),
                'current_page': 0,
                'completed_pages': 0
            }
            
            total_pages = self.get_page_count(manuscript_id)
            
            self._emit_event(CatalogueEvent(
                type=CatalogueEventType.TRANSCRIPTION_STARTED,
                manuscript_id=manuscript_id,
                timestamp=datetime.now(),
                data={'total_pages': total_pages}
            ))
            
            for page_num in range(1, total_pages + 1):
                self._active_processes[manuscript_id]['current_page'] = page_num
                
                try:
                    result = await self.transcriber.transcribe_page(
                        str(self.catalogue_dir / manuscript_id),
                        page_num,
                        notes
                    )
                    
                    self._active_processes[manuscript_id]['completed_pages'] += 1
                    
                    self._emit_event(CatalogueEvent(
                        type=CatalogueEventType.TRANSCRIPTION_PAGE_COMPLETE,
                        manuscript_id=manuscript_id,
                        timestamp=datetime.now(),
                        data={
                            'page_number': page_num,
                            'total_pages': total_pages,
                            'result': result
                        }
                    ))
                    
                except Exception as e:
                    self._emit_event(CatalogueEvent(
                        type=CatalogueEventType.TRANSCRIPTION_ERROR,
                        manuscript_id=manuscript_id,
                        timestamp=datetime.now(),
                        data={'page_number': page_num},
                        error=str(e)
                    ))
                    
            self._emit_event(CatalogueEvent(
                type=CatalogueEventType.TRANSCRIPTION_COMPLETE,
                manuscript_id=manuscript_id,
                timestamp=datetime.now(),
                data={'total_pages': total_pages}
            ))
            
        except Exception as e:
            self._emit_event(CatalogueEvent(
                type=CatalogueEventType.ERROR_OCCURRED,
                manuscript_id=manuscript_id,
                timestamp=datetime.now(),
                error=str(e)
            ))
            
        finally:
            self._active_processes.pop(manuscript_id, None)
    
    async def transcribe_page(self, manuscript_id: str, page_number: int, 
                            notes: Optional[str] = None) -> Dict:
        """
        Transcribe a single manuscript page.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
            page_number: Page number to transcribe (1-based)
            notes: Optional notes to guide transcription
        """
        if not self.manuscript_exists(manuscript_id):
            raise ValueError(f"Manuscript {manuscript_id} not found")
            
        total_pages = self.get_page_count(manuscript_id)
        if page_number < 1 or page_number > total_pages:
            raise ValueError(f"Invalid page number {page_number}. Valid range: 1-{total_pages}")
            
        result = await self.transcriber.transcribe_page(
            str(self.catalogue_dir / manuscript_id),
            page_number,
            notes
        )
        
        self._emit_event(CatalogueEvent(
            type=CatalogueEventType.TRANSCRIPTION_PAGE_COMPLETE,
            manuscript_id=manuscript_id,
            timestamp=datetime.now(),
            data={
                'page_number': page_number,
                'total_pages': total_pages,
                'result': result
            }
        ))
        
        return result
    
    def get_transcription(self, manuscript_id: str, page_number: Optional[int] = None) -> Dict:
        """
        Get existing transcription data.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
            page_number: Optional specific page number, if None returns all pages
        """
        transcript_path = self.catalogue_dir / manuscript_id / 'transcript.json'
        if not transcript_path.exists():
            return {}
            
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
            
        if page_number is not None:
            return transcript.get('pages', {}).get(str(page_number), {})
            
        return transcript
    
    def get_image_path(self, manuscript_id: str, page_number: int) -> Path:
        """Get filesystem path to page image."""
        image_dir = self.catalogue_dir / manuscript_id / 'images'
        image_files = sorted(image_dir.iterdir())
        
        if not image_files:
            raise FileNotFoundError(f"No images found for manuscript {manuscript_id}")
            
        if page_number < 1 or page_number > len(image_files):
            raise ValueError(f"Invalid page number {page_number}")
            
        return image_files[page_number - 1]
    
    def get_page_count(self, manuscript_id: str) -> int:
        """Get total number of pages in manuscript."""
        image_dir = self.catalogue_dir / manuscript_id / 'images'
        return len(list(image_dir.iterdir()))
    
    def validate_manuscript_structure(self, manuscript_id: str) -> List[str]:
        """
        Validate manuscript directory structure and required files.
        
        Returns:
            List of validation issues found, empty if valid
        """
        issues = []
        manuscript_dir = self.catalogue_dir / manuscript_id
        
        # Check basic structure
        if not manuscript_dir.exists():
            issues.append("Manuscript directory not found")
            return issues
            
        if not manuscript_dir.is_dir():
            issues.append("Manuscript path is not a directory")
            return issues
            
        # Check required files
        if not (manuscript_dir / 'standard_metadata.json').exists():
            issues.append("standard_metadata.json not found")
            
        # Check images directory
        image_dir = manuscript_dir / 'images'
        if not image_dir.exists() or not image_dir.is_dir():
            issues.append("images directory not found")
        else:
            image_files = list(image_dir.iterdir())
            if not image_files:
                issues.append("No image files found")
                
        return issues