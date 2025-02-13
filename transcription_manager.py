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

    def queue_manuscript(self, manuscript_id: str, priority: int = 1) -> bool:
        """Add a manuscript to the transcription queue.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
            priority: Priority level (lower numbers = higher priority)
            
        Returns:
            bool: True if manuscript was queued successfully
        """
        logger.info(f"Manuscript {manuscript_id} has been sent to the Scriptorium with priority {priority}")
        
        with self.jobs_lock:
            # Check if manuscript is already being processed
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
                    raise ValueError(f"Images directory not found for manuscript: {manuscript_id}")
                    
                image_files = [f for f in image_dir.iterdir() 
                             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
                total_pages = len(image_files)
                
                if total_pages == 0:
                    raise ValueError(f"No image files found for manuscript: {manuscript_id}")
                
                # Create job and add to queue
                job = TranscriptionJob(
                    manuscript_id=manuscript_id,
                    priority=priority,
                    total_pages=total_pages
                )
                self.active_jobs[manuscript_id] = job
                self.job_queue.put(job)
                
                logger.info(f"Successfully queued manuscript {manuscript_id} with {total_pages} pages")
                return True
                
            except Exception as e:
                logger.error(f"Error queueing manuscript {manuscript_id}: {str(e)}", exc_info=True)
                return False

    def get_job_status(self, manuscript_id: str) -> Optional[TranscriptionJob]:
        """Get current status of a transcription job."""
        status = (self.active_jobs.get(manuscript_id) or 
                 self.completed_jobs.get(manuscript_id))
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
        
        # Initialize transcriber
        try:
            transcriber = PageTranscriber()
        except Exception as e:
            logger.error(f"{worker_name} cannot find his quill: {str(e)}", exc_info=True)
            return
        
        while not self.should_stop.is_set():
            try:
                # Get next job from queue with timeout
                try:
                    job = self.job_queue.get(timeout=1)
                    logger.debug(f"{worker_name} has been assigned manuscript {job.manuscript_id}")
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
                        logger.info(f"{worker_name} has begun transcribing {manuscript_id}")
                
                try:
                    # Initialize or load transcript file
                    if transcript_path.exists():
                        # logger.debug(f"Loading existing transcript for {manuscript_id}")
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            transcript = json.load(f)
                    else:
                        # logger.debug(f"Creating new transcript for {manuscript_id}")
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
                    
                    # Process each page
                    for page_num in range(1, job.total_pages + 1):
                        if self.should_stop.is_set():
                            logger.info(f"{worker_name} has been called to prayer and stopped transcribing {manuscript_id}")
                            break
                            
                        # Skip already transcribed pages
                        if str(page_num) in transcript['pages']:
                            # logger.debug(f"Skipping already transcribed page {page_num} of {manuscript_id}")
                            job.completed_pages += 1
                            continue
                        
                        try:
                            logger.info(f"{worker_name} is transcribing page {page_num} of {manuscript_id}")
                            result = loop.run_until_complete(
                                transcriber.transcribe_page(
                                    str(manuscript_dir),
                                    page_num
                                )
                            )
                            
                            # Handle failed transcription
                            if result.transcription_notes == "Failed to parse structured response":
                                logger.warning(f"{worker_name} has spoiled his scribing of {page_num} of {manuscript_id} and shall try again.")
                                if page_num not in transcript['failed_pages']:
                                    transcript['failed_pages'].append(page_num)
                                if page_num not in job.failed_pages:
                                    job.failed_pages.append(page_num)
                            else:
                                logger.debug(f"{worker_name} has transcribed page {page_num} of {manuscript_id}")
                                if page_num in transcript['failed_pages']:
                                    transcript['failed_pages'].remove(page_num)
                                transcript['successful_pages'] += 1
                                job.completed_pages += 1
                                
                                # Add to transcript
                                transcript['pages'][str(page_num)] = {
                                    'body': [{'name': s.name, 'text': s.text} for s in result.body],
                                    'illustrations': [dict(i._asdict()) for i in (result.illustrations or [])],
                                    'marginalia': [dict(m._asdict()) for m in (result.marginalia or [])],
                                    'notes': [dict(n._asdict()) for n in (result.notes or [])],
                                    'language': result.language,
                                    'transcription_notes': result.transcription_notes
                                }
                            
                            # Save progress
                            transcript['last_updated'] = datetime.now().isoformat()
                            with open(transcript_path, 'w', encoding='utf-8') as f:
                                json.dump(transcript, f, indent=2, ensure_ascii=False)
                            
                        except Exception as e:
                            logger.error(f"{worker_name} cannot decipher page {page_num} of {manuscript_id} and will move on: {str(e)}", 
                                       exc_info=True)
                            if page_num not in transcript['failed_pages']:
                                transcript['failed_pages'].append(page_num)
                            if page_num not in job.failed_pages:
                                job.failed_pages.append(page_num)
                            
                            # Save after failures too
                            transcript['last_updated'] = datetime.now().isoformat()
                            with open(transcript_path, 'w', encoding='utf-8') as f:
                                json.dump(transcript, f, indent=2, ensure_ascii=False)
                    
                    # Update job status to completed
                    with self.jobs_lock:
                        job.completed_at = datetime.now()
                        job.status = TranscriptionStatus.COMPLETED
                        self.completed_jobs[manuscript_id] = job
                        self.active_jobs.pop(manuscript_id, None)
                        logger.info(f"{worker_name} completed manuscript {manuscript_id}")
                    
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