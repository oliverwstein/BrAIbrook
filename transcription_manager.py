from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import queue
import threading
import time
import asyncio

from pydantic import BaseModel

from transcriber import PageTranscriber

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Create console handler with formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class TranscriptionStatus(Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUESTED = "requested"

@dataclass
class TranscriptionRequest:
    """Represents a pending transcription request."""
    manuscript_id: str
    requested_at: datetime
    notes: str = ""
    priority: int = 1
    pages: Optional[List[int]] = None
    total_pages: int = 0  # Add this to track total pages in manuscript
    
@dataclass
class TranscriptionJob:
    manuscript_id: str
    priority: int
    total_pages: int
    completed_pages: int = 0
    failed_pages: List[int] = None
    status: TranscriptionStatus = TranscriptionStatus.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    request: Optional[TranscriptionRequest] = None
    pages_to_process: Optional[List[int]] = None

    def __post_init__(self):
        if self.failed_pages is None:
            self.failed_pages = []

    def __lt__(self, other):
        # For PriorityQueue ordering
        return self.priority < other.priority

class TranscriptionQueueManager:
    """Manages manuscript transcription jobs and worker threads."""

    def __init__(self, catalogue_dir: Path, num_workers: int = 2):
        """Initialize the transcription queue manager.
        
        Args:
            catalogue_dir: Path to the manuscript catalogue directory
            num_workers: Number of worker threads to create
        """
        logger.info(f"Opening the Scriptorium with {num_workers} monks")
        self.catalogue_dir = Path(catalogue_dir)
        self.monk_names = ["Alcuin", "Bede", "Cassiodorus", "Dunstan", "Eadfrith", "Felix", "Gildas", "Hugh"]
        if not self.catalogue_dir.exists():
            raise ValueError(f"Catalogue directory not found: {self.catalogue_dir}")
            
        self.num_workers = num_workers
        self.job_queue = queue.PriorityQueue()
        self.active_jobs: Dict[str, TranscriptionJob] = {}
        self.completed_jobs: Dict[str, TranscriptionJob] = {}
        self.pending_requests: Dict[str, TranscriptionRequest] = {}
        self.worker_threads: List[threading.Thread] = []
        self.should_stop = threading.Event()
        self.jobs_lock = threading.Lock()
        
        # Create and start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Brother {self.monk_names[i]}",
                daemon=True
            )
            self.worker_threads.append(worker)
            worker.start()

    def request_transcription(
        self, 
        manuscript_id: str, 
        notes: str = "", 
        priority: int = 1,
        pages: Optional[List[int]] = None
    ) -> bool:
        """Create a new transcription request."""
        with self.jobs_lock:
            if manuscript_id in self.pending_requests:
                return False
            
            # Validate manuscript and pages
            try:
                manuscript_dir = self.catalogue_dir / manuscript_id
                if not manuscript_dir.exists():
                    raise ValueError(f"Manuscript directory not found: {manuscript_id}")
                
                image_dir = manuscript_dir / 'images'
                if not image_dir.exists():
                    raise ValueError(f"Images directory not found: {manuscript_id}")
                
                total_pages = len([f for f in image_dir.iterdir() 
                                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
                
                if pages:
                    if not all(1 <= p <= total_pages for p in pages):
                        raise ValueError(f"Invalid page numbers. Valid range: 1-{total_pages}")
                    pages = sorted(list(set(pages)))  # Remove duplicates and sort
                
                request = TranscriptionRequest(
                    manuscript_id=manuscript_id,
                    requested_at=datetime.now(),
                    notes=notes,
                    priority=priority,
                    pages=pages,
                    total_pages=total_pages
                )
                self.pending_requests[manuscript_id] = request
                return True
                
            except Exception as e:
                logger.error(f"Error creating transcription request: {e}")
                raise

    def queue_manuscript(
        self, 
        manuscript_id: str, 
        priority: int = 1,
        pages: Optional[List[int]] = None,
        notes: str = ""
    ) -> bool:
        """Add a manuscript to the transcription queue."""
        page_info = f" (pages {pages})" if pages else " (all pages)"
        logger.info(f"Manuscript {manuscript_id} has been sent to the Scriptorium with priority {priority}{page_info}")
        
        with self.jobs_lock:
            if manuscript_id in self.active_jobs:
                logger.warning(f"Manuscript {manuscript_id} is already being transcribed")
                return False
                
            try:
                # Validate manuscript directory
                manuscript_dir = self.catalogue_dir / manuscript_id
                if not manuscript_dir.exists():
                    raise ValueError(f"Manuscript directory not found: {manuscript_id}")
                
                # Count total pages
                image_dir = manuscript_dir / 'images'
                if not image_dir.exists():
                    raise ValueError(f"Images directory not found: {manuscript_id}")
                    
                image_files = [f for f in image_dir.iterdir() 
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
                total_pages = len(image_files)
                
                if total_pages == 0:
                    raise ValueError(f"No image files found for manuscript: {manuscript_id}")

                # Load existing transcript if available to get current progress
                transcript_path = manuscript_dir / 'transcript.json'
                existing_pages = set()
                failed_pages = []
                if transcript_path.exists():
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript = json.load(f)
                        existing_pages = {int(p) for p in transcript.get('pages', {})}
                        failed_pages = transcript.get('failed_pages', [])

                # Validate requested pages if specified
                if pages:
                    if not all(1 <= p <= total_pages for p in pages):
                        raise ValueError(f"Invalid page numbers. Valid range: 1-{total_pages}")
                    pages = sorted(list(set(pages)))  # Remove duplicates and sort
                
                # Create request
                request = TranscriptionRequest(
                    manuscript_id=manuscript_id,
                    requested_at=datetime.now(),
                    notes=notes,
                    priority=priority,
                    pages=pages,
                    total_pages=total_pages
                )
                
                # Initialize job with current progress
                job = TranscriptionJob(
                    manuscript_id=manuscript_id,
                    priority=priority,
                    total_pages=total_pages,
                    completed_pages=len(existing_pages - set(failed_pages)),
                    failed_pages=failed_pages,
                    request=request,
                    pages_to_process=pages
                )
                
                self.active_jobs[manuscript_id] = job
                self.job_queue.put(job)
                
                logger.info(f"Successfully queued manuscript {manuscript_id}{page_info}")
                return True
                
            except Exception as e:
                logger.error(f"Error queueing manuscript {manuscript_id}: {str(e)}")
                return False
    
    def get_pending_requests(self) -> List[TranscriptionRequest]:
        """Get all pending transcription requests."""
        return list(self.pending_requests.values())
    
    def approve_request(self, manuscript_id: str) -> bool:
        """Approve a pending transcription request."""
        with self.jobs_lock:
            request = self.pending_requests.pop(manuscript_id, None)
            if not request:
                return False
            
            # Create new job
            try:
                # First verify the manuscript still exists/is valid
                manuscript_dir = self.catalogue_dir / manuscript_id
                if not manuscript_dir.exists():
                    logger.error(f"Manuscript directory not found: {manuscript_id}")
                    return False

                # Count total pages
                image_dir = manuscript_dir / 'images'
                if not image_dir.exists():
                    logger.error(f"Images directory not found for manuscript: {manuscript_id}")
                    return False
                    
                image_files = [f for f in image_dir.iterdir() 
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
                total_pages = len(image_files)
                
                if total_pages == 0:
                    logger.error(f"No image files found for manuscript: {manuscript_id}")
                    return False

                # Create and queue the job
                job = TranscriptionJob(
                    manuscript_id=manuscript_id,
                    priority=request.priority,
                    total_pages=total_pages,
                    request=request  # Keep reference to original request
                )
                self.active_jobs[manuscript_id] = job
                self.job_queue.put(job)
                
                logger.info(f"Successfully queued approved request for {manuscript_id} with {total_pages} pages")
                return True
                
            except Exception as e:
                logger.error(f"Error queueing approved request for {manuscript_id}: {str(e)}", exc_info=True)
                return False
    
    def reject_request(self, manuscript_id: str) -> bool:
        """Reject a pending transcription request."""
        with self.jobs_lock:
            if manuscript_id not in self.pending_requests:
                return False
            self.pending_requests.pop(manuscript_id)
            return True

    def get_job_status(self, manuscript_id: str) -> Optional[TranscriptionJob]:
        """Get current status of a transcription job or request."""
        # First check active and completed jobs
        status = (self.active_jobs.get(manuscript_id) or 
                 self.completed_jobs.get(manuscript_id))
        
        # If no job found, check if there's a pending request
        if not status and manuscript_id in self.pending_requests:
            request = self.pending_requests[manuscript_id]
            # Create a job object to represent the request status
            status = TranscriptionJob(
                manuscript_id=manuscript_id,
                priority=request.priority,
                total_pages=0,  # We don't know the total pages yet
                status=TranscriptionStatus.REQUESTED,
                request=request
            )
        
        if status:
            if status.status.value != "in_progress":
                logger.debug(f"Retrieved status for {manuscript_id}: {status.status.value}")
        return status

    def _worker_loop(self):
        """Main worker thread loop processing transcription jobs."""
        worker_name = threading.current_thread().name
        logger.info(f"{worker_name} begins their daily devotions")
        
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            transcriber = PageTranscriber()
        except Exception as e:
            logger.error(f"{worker_name} cannot find his quill: {str(e)}", exc_info=True)
            return
        
        while not self.should_stop.is_set():
            try:
                try:
                    job = self.job_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                manuscript_id = job.manuscript_id
                manuscript_dir = self.catalogue_dir / manuscript_id
                transcript_path = manuscript_dir / 'transcript.json'
                
                # Update job status to in progress
                with self.jobs_lock:
                    if job.status == TranscriptionStatus.QUEUED:
                        job.status = TranscriptionStatus.IN_PROGRESS
                        job.started_at = datetime.now()
                        page_info = f" (pages {job.pages_to_process})" if job.pages_to_process else " (all pages)"
                        logger.info(f"{worker_name} has begun transcribing {manuscript_id}{page_info}")
                
                try:
                    # Initialize or load transcript file
                    if transcript_path.exists():
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            transcript = json.load(f)
                    else:
                        with open(manuscript_dir / 'standard_metadata.json', 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        transcript = {
                            'manuscript_id': manuscript_id,
                            'title': metadata.get('title', 'Untitled'),
                            'pages': {},
                            'successful_pages': 0,
                            'failed_pages': [],
                            'total_pages': job.total_pages,
                            'last_updated': datetime.now().isoformat()
                        }
                    
                    # Determine pages to process
                    pages_to_process = job.pages_to_process if job.pages_to_process else range(1, job.total_pages + 1)
                    total_pages_to_process = len(pages_to_process)
                    
                    logger.info(f"{worker_name} will transcribe {total_pages_to_process} pages of {manuscript_id}")
                    
                    # Process each requested page
                    for page_num in pages_to_process:
                        if self.should_stop.is_set():
                            logger.info(f"{worker_name} has been called to prayer and stopped transcribing {manuscript_id}")
                            break
                            
                        # Skip already transcribed pages unless specifically requested
                        if str(page_num) in transcript['pages'] and not job.pages_to_process:
                            job.completed_pages += 1
                            continue
                        
                        try:
                            logger.info(f"{worker_name} is transcribing page {page_num} of {manuscript_id}")
                            result = loop.run_until_complete(
                                transcriber.transcribe_page(
                                    str(manuscript_dir),
                                    page_num,
                                    job.request.notes if job.request else ""
                                )
                            )
                            
                            # Check if this was already a successfully transcribed page
                            was_previously_transcribed = (
                                str(page_num) in transcript['pages'] and 
                                page_num not in transcript['failed_pages']
                            )
                            
                            # Handle transcription result
                            if result.transcription_notes == "Failed to parse structured response":
                                logger.warning(f"{worker_name} has spoiled his scribing of page {page_num} of {manuscript_id}")
                                if page_num not in transcript['failed_pages']:
                                    transcript['failed_pages'].append(page_num)
                                if page_num not in job.failed_pages:
                                    job.failed_pages.append(page_num)
                                # If it was previously successful but now failed, decrement the count
                                if was_previously_transcribed:
                                    transcript['successful_pages'] = max(0, transcript['successful_pages'] - 1)
                            else:
                                logger.debug(f"{worker_name} has transcribed page {page_num} of {manuscript_id}")
                                if page_num in transcript['failed_pages']:
                                    transcript['failed_pages'].remove(page_num)
                                # Only increment counters if this is a new success
                                if not was_previously_transcribed:
                                    transcript['successful_pages'] += 1
                                    job.completed_pages += 1
                                
                                # Update transcript regardless of previous status
                                transcript['pages'][str(page_num)] = {
                                    'body': [{'name': s.name, 'text': s.text} for s in result.body],
                                    'illustrations': [dict(i._asdict()) for i in (result.illustrations or [])],
                                    'marginalia': [dict(m._asdict()) for m in (result.marginalia or [])],
                                    'notes': [dict(n._asdict()) for n in (result.notes or [])],
                                    'language': result.language,
                                    'transcription_notes': result.transcription_notes
                                }
                            
                            # Save progress after each page
                            transcript['last_updated'] = datetime.now().isoformat()
                            with open(transcript_path, 'w', encoding='utf-8') as f:
                                json.dump(transcript, f, indent=2, ensure_ascii=False)
                            
                        except Exception as e:
                            logger.error(f"{worker_name} cannot decipher page {page_num} of {manuscript_id}: {str(e)}")
                            if page_num not in transcript['failed_pages']:
                                transcript['failed_pages'].append(page_num)
                            if page_num not in job.failed_pages:
                                job.failed_pages.append(page_num)
                            
                            # Save after failures too
                            transcript['last_updated'] = datetime.now().isoformat()
                            with open(transcript_path, 'w', encoding='utf-8') as f:
                                json.dump(transcript, f, indent=2, ensure_ascii=False)
                    
                    # Update job status upon completion
                    with self.jobs_lock:
                        job.completed_at = datetime.now()
                        
                        # Count all successfully transcribed pages
                        successfully_transcribed = len([p for p in transcript['pages'] 
                                                    if int(p) not in transcript['failed_pages']])
                        job.completed_pages = successfully_transcribed
                        
                        # Determine completion status
                        if job.pages_to_process:
                            # For partial transcription, check if requested pages are done
                            remaining_pages = [p for p in job.pages_to_process 
                                            if str(p) not in transcript['pages'] 
                                            or p in transcript['failed_pages']]
                            is_complete = not remaining_pages
                        else:
                            # For full transcription, check if all pages are done
                            is_complete = successfully_transcribed == job.total_pages
                        
                        if is_complete:
                            job.status = TranscriptionStatus.COMPLETED
                            logger.info(f"{worker_name} completed transcription of {manuscript_id}")
                        else:
                            job.status = TranscriptionStatus.PARTIAL
                            logger.info(f"{worker_name} partially completed transcription of {manuscript_id} "
                                    f"({successfully_transcribed}/{job.total_pages} pages)")
                        
                        self.completed_jobs[manuscript_id] = job
                        self.active_jobs.pop(manuscript_id, None)
                    
                except Exception as e:
                    logger.error(f"{worker_name} has made a grave error with {manuscript_id}: {str(e)}", exc_info=True)
                    with self.jobs_lock:
                        job.error = str(e)
                        job.status = TranscriptionStatus.FAILED
                        job.completed_at = datetime.now()
                        self.completed_jobs[manuscript_id] = job
                        self.active_jobs.pop(manuscript_id, None)
                
            except Exception as e:
                logger.error(f"{worker_name} is at a loss: {str(e)}", exc_info=True)
                continue
        
        # Clean up event loop
        try:
            loop.stop()
            loop.close()
        except Exception as e:
            logger.error(f"{worker_name} has gone mad and refuses to stop: {str(e)}", exc_info=True)
            
    def _get_canonical_hour(self) -> str:
        """Get the appropriate canonical hour based on current time."""
        hour = datetime.now().hour
        if hour < 3: return "Matins"
        if hour < 6: return "Lauds"
        if hour < 9: return "Prime"
        if hour < 12: return "Terce"
        if hour < 15: return "Sext"
        if hour < 18: return "None"
        if hour < 21: return "Vespers"
        return "Compline"
    
    def shutdown(self):
        """Gracefully shut down the queue manager."""
        logger.info(f"The scriptorium bell rings for {self._get_canonical_hour()}...")
        self.should_stop.set()
        
        for worker in self.worker_threads:
            logger.debug(f"Waiting for {worker.name} to finish")
            worker.join(timeout=5)
            if worker.is_alive():
                logger.warning(f"{worker.name} seems to have dozed off at their desk.")
            else:
                logger.debug(f"{worker.name} has carefully stored their manuscripts and retired.")
        
        logger.info("The Scriptorium sits silent.")