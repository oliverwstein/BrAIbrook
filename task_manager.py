from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import json
import logging
import asyncio
import threading
import time
from google.cloud import firestore
import firebase_admin
from firebase_admin import firestore as admin_firestore

from transcriber import PageTranscriber

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUESTED = "requested"

@dataclass
class TaskRequest:
    """Represents a pending task request."""
    manuscript_id: str
    task_type: str  # e.g., "transcription"
    requested_at: datetime
    notes: str = ""
    priority: int = 1
    pages: Optional[List[int]] = None
    
@dataclass
class TaskJob:
    manuscript_id: str
    task_type: str
    priority: int
    completed_pages: int = 0
    failed_pages: List[int] = field(default_factory=list)
    status: TaskStatus = TaskStatus.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    request: Optional[TaskRequest] = None
    pages_to_process: Optional[List[int]] = None

    def __lt__(self, other):
        return self.priority < other.priority

class TaskManager:
    """Manages manuscript processing tasks and worker threads."""

    def __init__(self, num_workers: int = 2):
        """Initialize the task manager with Firestore and worker threads."""
        logger.info(f"Initializing task manager with {num_workers} workers")
        
        # Initialize Firebase Admin SDK
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            raise
            
        self.db = firestore.Client()
        self.num_workers = num_workers
        self.job_queue = asyncio.Queue()
        self.active_jobs: Dict[str, TaskJob] = {}
        self.completed_jobs: Dict[str, TaskJob] = {}
        self.pending_requests: Dict[str, TaskRequest] = {}
        self.worker_threads: List[asyncio.Task] = []
        self.should_stop = asyncio.Event()
        self.jobs_lock = asyncio.Lock()
        
        # Initialize task processors
        self.transcriber = PageTranscriber()
        
        # Register task handlers
        self.task_handlers = {
            'transcription': self.transcription_task
        }
        
        # Create and start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(
                self._worker_loop(),
                name=f"Worker-{i}"
            )
            self.worker_threads.append(worker)

    async def request_task(self, request: TaskRequest) -> bool:
        """Create a new task request."""
        if request.task_type not in self.task_handlers:
            raise ValueError(f"Unknown task type: {request.task_type}")
            
        request_key = f"{request.manuscript_id}_{request.task_type}"
        
        async with self.jobs_lock:
            if request_key in self.pending_requests:
                return False
            
            # Create request document in Firestore
            request_ref = self.db.collection('task_requests').document(request_key)
            request_ref.set({
                'manuscript_id': request.manuscript_id,
                'task_type': request.task_type,
                'requested_at': admin_firestore.SERVER_TIMESTAMP,
                'notes': request.notes,
                'priority': request.priority,
                'pages': request.pages,
                'status': TaskStatus.REQUESTED.value
            })
            
            self.pending_requests[request_key] = request
            return True

    async def queue_task(self, request: TaskRequest) -> bool:
        """Add a task to the queue."""
        if request.task_type not in self.task_handlers:
            raise ValueError(f"Unknown task type: {request.task_type}")
            
        task_key = f"{request.manuscript_id}_{request.task_type}"
        
        async with self.jobs_lock:
            if task_key in self.active_jobs:
                return False
                
            # Create job document in Firestore
            job_ref = self.db.collection('tasks').document(task_key)
            
            # Get manuscript metadata for total pages
            manuscript_ref = self.db.collection('manuscripts').document(request.manuscript_id)
            manuscript_doc = await manuscript_ref.get()
            if not manuscript_doc.exists:
                raise ValueError(f"Manuscript {request.manuscript_id} not found")
            
            metadata = manuscript_doc.to_dict()
            total_pages = metadata.get('total_pages', 0)
            
            # Validate pages if specified
            pages_to_process = request.pages
            if pages_to_process:
                if not all(1 <= p <= total_pages for p in pages_to_process):
                    raise ValueError(f"Invalid page numbers. Valid range: 1-{total_pages}")
                pages_to_process = sorted(list(set(pages_to_process)))
            
            # Create job
            job = TaskJob(
                manuscript_id=request.manuscript_id,
                task_type=request.task_type,
                priority=request.priority,
                request=request,
                pages_to_process=pages_to_process
            )
            
            # Update Firestore
            job_ref.set({
                'manuscript_id': job.manuscript_id,
                'task_type': job.task_type,
                'status': job.status.value,
                'priority': job.priority,
                'created_at': admin_firestore.SERVER_TIMESTAMP,
                'pages_to_process': job.pages_to_process
            })
            
            # Add to queue and tracking
            self.active_jobs[task_key] = job
            await self.job_queue.put(job)
            
            return True

    async def get_job_status(self, manuscript_id: str, task_type: str) -> Optional[TaskJob]:
        """Get current status of a task."""
        task_key = f"{manuscript_id}_{task_type}"
        
        # Check active jobs
        if task_key in self.active_jobs:
            return self.active_jobs[task_key]
            
        # Check completed jobs
        if task_key in self.completed_jobs:
            return self.completed_jobs[task_key]
            
        # Check Firestore
        job_ref = self.db.collection('tasks').document(task_key)
        job_doc = await job_ref.get()
        
        if not job_doc.exists:
            # Check for pending request
            request_ref = self.db.collection('task_requests').document(task_key)
            request_doc = await request_ref.get()
            
            if request_doc.exists:
                request_data = request_doc.to_dict()
                return TaskJob(
                    manuscript_id=manuscript_id,
                    task_type=task_type,
                    status=TaskStatus.REQUESTED,
                    priority=request_data.get('priority', 1),
                    request=TaskRequest(**request_data)
                )
            return None
            
        job_data = job_doc.to_dict()
        return TaskJob(**job_data)

    async def _worker_loop(self):
        """Main worker loop processing tasks."""
        while not self.should_stop.is_set():
            try:
                # Get next job from queue
                job = await self.job_queue.get()
                
                # Get appropriate task handler
                handler = self.task_handlers.get(job.task_type)
                if not handler:
                    logger.error(f"No handler found for task type: {job.task_type}")
                    continue
                
                # Process job
                try:
                    await self._process_job(job, handler)
                except Exception as e:
                    logger.error(f"Error processing job: {e}")
                    await self._update_job_status(job, TaskStatus.FAILED, error=str(e))
                    
            except Exception as e:
                logger.error(f"Worker error: {e}")
                continue

    async def _process_job(self, job: TaskJob, handler):
        """Process a single job using its handler."""
        job_key = f"{job.manuscript_id}_{job.task_type}"
        
        try:
            # Update status to in progress
            await self._update_job_status(job, TaskStatus.IN_PROGRESS)
            
            # Get manuscript metadata
            manuscript_ref = self.db.collection('manuscripts').document(job.manuscript_id)
            metadata = (await manuscript_ref.get()).to_dict()
            total_pages = metadata.get('total_pages', 0)
            
            # Determine pages to process
            pages = job.pages_to_process or range(1, total_pages + 1)
            
            # Process each page
            for page_num in pages:
                try:
                    await handler(job.manuscript_id, page_num, job.request.notes if job.request else "")
                    job.completed_pages += 1
                    await self._update_job_status(job, TaskStatus.IN_PROGRESS)
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    job.failed_pages.append(page_num)
                    await self._update_job_status(job, TaskStatus.IN_PROGRESS)
            
            # Complete job
            final_status = TaskStatus.COMPLETED if not job.failed_pages else TaskStatus.FAILED
            await self._update_job_status(job, final_status)
            
            # Move to completed jobs
            async with self.jobs_lock:
                self.completed_jobs[job_key] = job
                self.active_jobs.pop(job_key, None)
                
        except Exception as e:
            logger.error(f"Job processing error: {e}")
            await self._update_job_status(job, TaskStatus.FAILED, error=str(e))

    async def transcription_task(self, manuscript_id: str, page_number: int, notes: str = "") -> None:
        """Handle transcription of a single page."""
        try:
            # Get transcription
            result = await self.transcriber.transcribe_page(manuscript_id, page_number, notes)
            
            @firestore.transactional
            def update_transaction(transaction):
                data_ref = self.db.collection('manuscripts').document(manuscript_id)
                transaction.set(data_ref, {
                    f'pages.{page_number}.transcription': self.transcriber._serialize_result(result),
                    f'pages.{page_number}.last_updated': admin_firestore.SERVER_TIMESTAMP
                }, merge=True)
            
            # Execute transaction
            transaction = self.db.transaction()
            await update_transaction(transaction)
            
        except Exception as e:
            logger.error(f"Transcription error for {manuscript_id} page {page_number}: {e}")
            raise

    async def _update_job_status(self, job: TaskJob, status: TaskStatus, error: Optional[str] = None):
        """Update job status in memory and Firestore."""
        job.status = status
        if error:
            job.error = error
            
        if status == TaskStatus.IN_PROGRESS and not job.started_at:
            job.started_at = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            job.completed_at = datetime.now()
            
        # Update Firestore with transaction
        job_key = f"{job.manuscript_id}_{job.task_type}"
        
        @firestore.transactional
        def update_transaction(transaction):
            job_ref = self.db.collection('tasks').document(job_key)
            transaction.set(job_ref, {
                'status': status.value,
                'completed_pages': job.completed_pages,
                'failed_pages': job.failed_pages,
                'error': error,
                'started_at': job.started_at,
                'completed_at': job.completed_at
            }, merge=True)
        
        # Execute transaction
        transaction = self.db.transaction()
        await update_transaction(transaction)

    async def shutdown(self):
        """Gracefully shut down the task manager."""
        logger.info("Shutting down task manager...")
        await self.should_stop.set()
        
        # Wait for all workers to complete
        await asyncio.gather(*self.worker_threads, return_exceptions=True)