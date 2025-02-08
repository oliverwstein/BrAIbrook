"""
Server implementation for the Leystar manuscript system.
Provides REST API endpoints for manuscript management and real-time updates via SSE.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import asyncio
from typing import AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from catalogue import ManuscriptCatalogue, CatalogueEventType, CatalogueEvent
from pydantic import BaseModel

class TranscriptionRequest(BaseModel):
    notes: str = ""
    replace_existing: bool = False
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(title="Leystar Manuscript API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:4173",  # Vite preview
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize catalogue
catalogue = ManuscriptCatalogue()

@app.get("/")
async def index():
    """Root endpoint providing API information."""
    manuscripts = catalogue.get_manuscript_listings()
    return {
        'name': 'Leystar Manuscript API',
        'version': '1.0',
        'endpoints': {
            'GET /manuscripts': 'List all manuscripts',
            'GET /manuscripts/{id}/info': 'Get manuscript information',
            'GET /manuscripts/{id}/pages': 'Get manuscript pages',
            'GET /manuscripts/{id}/pages/{number}': 'Get page information',
            'GET /manuscripts/{id}/pages/{number}/image': 'Get page image',
            'POST /manuscripts/{id}/transcribe': 'Start manuscript transcription',
            'POST /manuscripts/{id}/pages/{number}/transcribe': 'Transcribe specific page',
            'GET /manuscripts/{id}/status': 'Get transcription status'
        },
        'total_manuscripts': len(manuscripts)
    }

@app.get("/manuscripts")
async def list_manuscripts():
    """List all manuscripts with their basic metadata."""
    try:
        manuscripts = catalogue.get_manuscript_listings()
        return JSONResponse(content=manuscripts)
    except Exception as e:
        logger.error(f"Error listing manuscripts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve manuscript list")

@app.get("/manuscripts/{manuscript_id}/info")
async def get_manuscript_info(manuscript_id: str):
    """Get detailed information for a specific manuscript."""
    try:
        # Get manuscript info directly from catalogue
        manuscript = catalogue.get_manuscript_listings().get(manuscript_id)
        if not manuscript:
            # Get all available IDs for the error message
            available_manuscripts = list(catalogue.get_manuscript_listings().keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Manuscript not found. Available manuscripts: {available_manuscripts}"
            )
            
        return JSONResponse(content=manuscript)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error retrieving manuscript {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve manuscript data")

@app.get("/manuscripts/{manuscript_id}/pages")
async def get_manuscript_pages(manuscript_id: str):
    """Get complete page data for a specific manuscript."""
    if not catalogue.manuscript_exists(manuscript_id):
        raise HTTPException(status_code=404, detail="Manuscript not found")
        
    try:
        pages = catalogue.get_transcription(manuscript_id)
        return JSONResponse(content=pages)
    except Exception as e:
        logger.error(f"Error retrieving pages for {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve manuscript pages")

@app.get("/manuscripts/{manuscript_id}/pages/{page_number}")
async def get_page(manuscript_id: str, page_number: int):
    """Get data for a specific manuscript page."""
    if not catalogue.manuscript_exists(manuscript_id):
        raise HTTPException(status_code=404, detail="Manuscript not found")
        
    try:
        page_data = catalogue.get_transcription(manuscript_id, page_number)
        if not page_data:
            raise HTTPException(status_code=404, detail="Page not found")
        return JSONResponse(content=page_data)
    except Exception as e:
        logger.error(f"Error retrieving page {page_number} from {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve page data")

@app.get("/manuscripts/{manuscript_id}/pages/{page_number}/image")
async def get_page_image(manuscript_id: str, page_number: int):
    """Get the image file for a specific manuscript page."""
    if not catalogue.manuscript_exists(manuscript_id):
        raise HTTPException(status_code=404, detail="Manuscript not found")
        
    try:
        image_path = catalogue.get_image_path(manuscript_id, page_number)
        return FileResponse(image_path)
    except Exception as e:
        logger.error(f"Error retrieving image for page {page_number} from {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")

@app.post("/manuscripts/{manuscript_id}/transcribe")
async def transcribe_manuscript(manuscript_id: str):
    """Start or continue transcription of a complete manuscript."""
    if not catalogue.manuscript_exists(manuscript_id):
        raise HTTPException(status_code=404, detail="Manuscript not found")

    # Check if transcription is already running
    status = catalogue.get_transcription_status(manuscript_id)
    if status and status['status'] == 'in_progress':
        return JSONResponse(
            content={'status': 'already_running', 'message': 'Transcription already in progress'},
            status_code=409
        )

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events from catalogue events."""
        queue = asyncio.Queue()
        
        def event_handler(event: CatalogueEvent):
            if event.manuscript_id == manuscript_id:
                queue.put_nowait(event)
        
        # Subscribe to catalogue events
        catalogue.subscribe(event_handler)
        
        try:
            # Start transcription in background task
            transcription_task = asyncio.create_task(
                catalogue.transcribe_manuscript(manuscript_id)
            )
            
            while True:
                try:
                    # Wait for next event
                    event = await queue.get()
                    
                    # Convert event to SSE format
                    yield f"data: {json.dumps(event.data)}\n\n"
                    
                    # Check if transcription is complete
                    if event.type in {
                        CatalogueEventType.TRANSCRIPTION_COMPLETE,
                        CatalogueEventType.TRANSCRIPTION_ERROR
                    }:
                        break
                        
                except asyncio.CancelledError:
                    break
                    
        finally:
            # Clean up
            catalogue.unsubscribe(event_handler)
            if not transcription_task.done():
                transcription_task.cancel()
    
    return EventSourceResponse(event_generator())

@app.post("/manuscripts/{manuscript_id}/pages/{page_number}/transcribe")
async def transcribe_page(manuscript_id: str, page_number: int):
    """Transcribe a single manuscript page."""
    if not catalogue.manuscript_exists(manuscript_id):
        raise HTTPException(status_code=404, detail="Manuscript not found")
        
    try:
        result = await catalogue.transcribe_page(manuscript_id, page_number)
        return JSONResponse(content={
            'status': 'success',
            'data': result,
            'manuscript': catalogue.get_transcription_status(manuscript_id)
        })
    except Exception as e:
        logger.error(f"Error transcribing page {page_number} of {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to transcribe page")

@app.get("/manuscripts/{manuscript_id}/status")
async def manuscript_status(manuscript_id: str):
    """Get real-time status of manuscript transcription process."""
    if not catalogue.manuscript_exists(manuscript_id):
        raise HTTPException(status_code=404, detail="Manuscript not found")

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events for manuscript status."""
        queue = asyncio.Queue()
        
        def event_handler(event: CatalogueEvent):
            if event.manuscript_id == manuscript_id:
                queue.put_nowait(event)
        
        # Subscribe to catalogue events
        catalogue.subscribe(event_handler)
        
        try:
            # Send initial status
            status = catalogue.get_transcription_status(manuscript_id)
            yield f"data: {json.dumps(status)}\n\n"
            
            while True:
                try:
                    event = await queue.get()
                    yield f"data: {json.dumps(event.data)}\n\n"
                except asyncio.CancelledError:
                    break
                    
        finally:
            catalogue.unsubscribe(event_handler)
    
    return EventSourceResponse(event_generator())

# Start server
if __name__ == "__main__":
    import uvicorn
    print("Starting Leystar Manuscript Server...")
    print("API documentation will be available at http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")