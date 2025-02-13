from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

from transcription_manager import TranscriptionQueueManager

logger = logging.getLogger(__name__)

class ManuscriptCatalogue:
    """
    Central manager for manuscript access and transcription coordination.
    Maintains manuscript listings and coordinates transcription processes.
    """
    
    def __init__(self, catalogue_dir: str = "data/catalogue"):
        """Initialize the manuscript catalogue with components."""
        self.catalogue_dir = Path(catalogue_dir)
        if not self.catalogue_dir.exists():
            raise FileNotFoundError(f"Catalogue directory not found: {self.catalogue_dir}")
        self.status_log: List[Dict] = []
        # Initialize transcription manager
        self.transcription_manager = TranscriptionQueueManager(
            catalogue_dir=self.catalogue_dir,
            num_workers=6
        )
        
        # Initialize manuscript listings
        self.manuscript_listings: Dict[str, Dict] = {}
        self._load_manuscript_listings()
        
    def get_manuscript_listings(self) -> Dict[str, Dict]:
        """Get complete information for all manuscripts in the catalogue."""
        # Update status for any manuscripts being transcribed
        for manuscript_id in self.manuscript_listings:
            job_status = self.transcription_manager.get_job_status(manuscript_id)
            if job_status:
                self._refresh_manuscript(manuscript_id)
                
        return self.manuscript_listings.copy()

    def get_manuscript(self, manuscript_id: str) -> Optional[Dict]:
        """Get complete information for a specific manuscript."""
        if not self.manuscript_exists(manuscript_id):
            return None
        
        manuscript = {}
        # Get basic manuscript info
        manuscript['metadata'] = self.manuscript_listings[manuscript_id].copy()
        
        # Add page data if available
        transcript_path = self.catalogue_dir / manuscript_id / 'transcript.json'
        if transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
                pages = {}
                for page_num, transcription in transcript.get('pages', {}).items():
                    pages[page_num] = {
                        'page_number': int(page_num),
                        'transcription': transcription
                    }
                manuscript['pages'] = pages
        
        return manuscript

    def manuscript_exists(self, manuscript_id: str) -> bool:
        """Check if a manuscript exists in the catalogue."""
        if manuscript_id not in self.manuscript_listings:
            # Do a filesystem check and refresh if found
            manuscript_dir = self.catalogue_dir / manuscript_id
            if manuscript_dir.exists() and manuscript_dir.is_dir():
                self._refresh_manuscript(manuscript_id)
                return True
            return False
        return True
    
    def get_transcription_status(self, manuscript_id: str) -> Dict:
        """Get current transcription status for a manuscript."""
        # Don't check manuscript_exists since that can cause recursion
        
        job_status = self.transcription_manager.get_job_status(manuscript_id)
        if not job_status:
            # Check if we have any transcribed pages
            transcript_path = self.catalogue_dir / manuscript_id / 'transcript.json'
            if transcript_path.exists():
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = json.load(f)
                    return {
                        'status': 'completed' if len(transcript.get('pages', {})) == transcript.get('total_pages', 0) else 'partial',
                        'transcribed_pages': len(transcript.get('pages', {})),
                        'failed_pages': transcript.get('failed_pages', []),
                        'total_pages': transcript.get('total_pages', 0),
                        'last_updated': transcript.get('last_updated')
                    }
            return {'status': 'not_started'}
            
        return {
            'status': job_status.status.value,
            'transcribed_pages': job_status.completed_pages,
            'failed_pages': job_status.failed_pages,
            'total_pages': job_status.total_pages,
            'started_at': job_status.started_at.isoformat() if job_status.started_at else None,
            'completed_at': job_status.completed_at.isoformat() if job_status.completed_at else None,
            'error': job_status.error
        }
    
    def request_transcription(self, manuscript_id: str, notes: str = "", priority: int = 1) -> bool:
        """Request transcription of a manuscript."""
        if not self.manuscript_exists(manuscript_id):
            raise ValueError(f"Manuscript {manuscript_id} not found")
            
        return self.transcription_manager.request_transcription(
            manuscript_id=manuscript_id,
            notes=notes,
            priority=priority
        )

    def get_pending_requests(self) -> List[Dict]:
        """Get list of pending transcription requests."""
        list_of_transcription_requests = self.transcription_manager.get_pending_requests()
        requests = []
        for request in list_of_transcription_requests:
            requests.append({
                "manuscript_id": request.manuscript_id,
                "requested_at": request.requested_at.isoformat(),
                "notes": request.notes,
                "priority": request.priority})
        return requests
    
    def approve_request(self, manuscript_id: str) -> bool:
        """Approve a pending transcription request."""
        return self.transcription_manager.approve_request(manuscript_id)

    def reject_request(self, manuscript_id: str) -> bool:
        """Reject a pending transcription request."""
        return self.transcription_manager.reject_request(manuscript_id)

    def start_transcription(self, manuscript_id: str, priority: int = 1) -> bool:
        """Start transcription of a manuscript."""
        if not self.manuscript_exists(manuscript_id):
            raise ValueError(f"Manuscript {manuscript_id} not found")
            
        success = self.transcription_manager.queue_manuscript(manuscript_id, priority)
        if success:
            self._refresh_manuscript(manuscript_id)
        return success

    def get_transcription(self, manuscript_id: str, page_number: Optional[int] = None) -> Dict:
        """Get transcription data for manuscript pages."""
        transcript_path = self.catalogue_dir / manuscript_id / 'transcript.json'
        if not transcript_path.exists():
            return {}
            
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
            
        if page_number is not None:
            # Return single page data
            transcription = transcript.get('pages', {}).get(str(page_number), {})
            return {
                'page_number': page_number,
                'transcription': transcription if transcription else None
            }
        
        # Return all pages
        pages = {}
        for page_num, transcription in transcript.get('pages', {}).items():
            pages[page_num] = {
                'page_number': int(page_num),
                'transcription': transcription
            }
        return pages

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
    
    def validate_manuscript(self, manuscript_id: str) -> List[str]:
        """Validate manuscript directory structure and required files."""
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

    def _refresh_manuscript(self, manuscript_id: str) -> None:
        """Refresh manuscript listing after changes."""
        try:
            metadata_path = self.catalogue_dir / manuscript_id / 'standard_metadata.json'
            if not metadata_path.exists():
                self.manuscript_listings.pop(manuscript_id, None)
                return
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Get status from queue manager first to avoid recursion
            status = {'status': 'not_started'}
            job_status = self.transcription_manager.get_job_status(manuscript_id)
            if job_status:
                status = {
                    'status': job_status.status.value,
                    'transcribed_pages': job_status.completed_pages,
                    'failed_pages': job_status.failed_pages,
                    'total_pages': job_status.total_pages,
                    'started_at': job_status.started_at.isoformat() if job_status.started_at else None,
                    'completed_at': job_status.completed_at.isoformat() if job_status.completed_at else None,
                    'error': job_status.error
                }
            else:
                # Check transcript file directly
                transcript_path = self.catalogue_dir / manuscript_id / 'transcript.json'
                if transcript_path.exists():
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript = json.load(f)
                        status = {
                            'status': 'completed' if len(transcript.get('pages', {})) == transcript.get('total_pages', 0) else 'partial',
                            'transcribed_pages': len(transcript.get('pages', {})),
                            'failed_pages': transcript.get('failed_pages', []),
                            'total_pages': transcript.get('total_pages', 0),
                            'last_updated': transcript.get('last_updated')
                        }
            
            self.manuscript_listings[manuscript_id] = {
                'id': manuscript_id,
                'total_pages': self.get_page_count(manuscript_id),
                **metadata,
                'transcription_status': status
            }
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'manuscript_id': manuscript_id,
                'status': status
            }
            self.status_log.append(log_entry)
            
        except Exception as e:
            logger.error(f"Error refreshing manuscript {manuscript_id}: {e}")
            self.manuscript_listings.pop(manuscript_id, None)

    def _load_manuscript_listings(self) -> None:
        """Load all manuscript listings into memory."""
        self.manuscript_listings.clear()
        for manuscript_dir in self.catalogue_dir.iterdir():
            if not manuscript_dir.is_dir():
                continue
                
            try:
                manuscript_id = manuscript_dir.name
                self._refresh_manuscript(manuscript_id)
                
            except Exception as e:
                logger.error(f"Error loading manuscript {manuscript_dir.name}: {e}")
                continue

    def get_recent_status_updates(self, since: Optional[datetime] = None) -> List[Dict]:
        """Get recent status updates, optionally filtered by timestamp."""
        if since:
            return [
                entry for entry in self.status_log 
                if datetime.fromisoformat(entry['timestamp']) > since
            ]
        return self.status_log.copy()

    def get_manuscript_status_history(self, manuscript_id: str, since: Optional[datetime] = None) -> List[Dict]:
        """Get status history for a specific manuscript."""
        return [
            entry for entry in self.status_log 
            if entry['manuscript_id'] == manuscript_id
            and (not since or datetime.fromisoformat(entry['timestamp']) > since)
        ]