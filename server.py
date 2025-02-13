"""
Server implementation for the Leystar manuscript system.
Provides REST API endpoints for manuscript management and transcription tracking.
"""

from contextlib import asynccontextmanager
import json
import logging
from datetime import datetime
from pathlib import Path
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from catalogue import ManuscriptCatalogue

class TranscriptionRequest(BaseModel):
    notes: str = ""
    priority: int = 1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create background tasks
    task = asyncio.create_task(update_transcription_status())
    yield
    # Shutdown: Cancel background tasks
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Add this endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            # For now, we'll just echo them back
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:4173",  # Vite preview
        "http://127.0.0.1:4173",
        "http://localhost:8000",  # FastAPI server
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Initialize catalogue
catalogue = ManuscriptCatalogue()

# Track manuscripts being transcribed
transcribing_manuscripts: Dict[str, Dict] = {}

async def update_transcription_status():
    """Background task to update status of transcribing manuscripts."""
    while True:
        try:
            # Get list of manuscripts we're tracking
            manuscript_ids = list(transcribing_manuscripts.keys())
            
            for manuscript_id in manuscript_ids:
                # Get current status
                status = catalogue.get_transcription_status(manuscript_id)
                
                # Remove completed/failed manuscripts from tracking
                if status['status'] in ['completed', 'error']:
                    manuscript = transcribing_manuscripts.pop(manuscript_id, None)
                    if manuscript:
                        # Notify connected clients about completion
                        await manager.broadcast(json.dumps({
                            'type': 'transcription_update',
                            'manuscript_id': manuscript_id,
                            'data': status
                        }))
                else:
                    # Update tracked status and notify clients
                    transcribing_manuscripts[manuscript_id] = status
                    await manager.broadcast(json.dumps({
                        'type': 'transcription_update',
                        'manuscript_id': manuscript_id,
                        'data': status
                    }))
                    
        except Exception as e:
            logger.error(f"Error updating transcription status: {e}")
            
        await asyncio.sleep(5)  # Update every 5 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks when server starts."""
    asyncio.create_task(update_transcription_status())

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
            'GET /transcription/status': 'Get all active transcription statuses',
            'GET /transcription/status/{id}': 'Get specific active transcription status',
            'GET /transcription/history': 'Get status history for all manuscripts',
            'GET /transcription/history?since={timestamp}': 'Get status history since timestamp',
            'GET /transcription/history/{id}': 'Get status history for specific manuscript',
            'GET /transcription/history/{id}?since={timestamp}': 'Get manuscript status history since timestamp'
        },
        'total_manuscripts': len(manuscripts)
    }

@app.get("/manuscripts")
def list_manuscripts():
    """List all manuscripts with their basic metadata."""
    try:
        manuscripts = catalogue.get_manuscript_listings()
        return JSONResponse(content=manuscripts)
    except Exception as e:
        logger.error(f"Error listing manuscripts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve manuscript list")

@app.get("/manuscripts/{manuscript_id}/info")
def get_manuscript_info(manuscript_id: str):
    """Get detailed information for a specific manuscript."""
    try:
        manuscript = catalogue.get_manuscript(manuscript_id)
        if not manuscript:
            raise HTTPException(
                status_code=404,
                detail=f"Manuscript {manuscript_id} not found"
            )
        return JSONResponse(content=manuscript)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving manuscript {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve manuscript data")

@app.get("/manuscripts/{manuscript_id}/pages")
def get_manuscript_pages(manuscript_id: str):
    """Get complete page data for a specific manuscript."""
    try:
        pages = catalogue.get_transcription(manuscript_id)
        return JSONResponse(content=pages)
    except Exception as e:
        logger.error(f"Error retrieving pages for {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve manuscript pages")

@app.get("/manuscripts/{manuscript_id}/pages/{page_number}")
def get_page(manuscript_id: str, page_number: int):
    """Get data for a specific manuscript page."""
    try:
        page_data = catalogue.get_transcription(manuscript_id, page_number)
        if not page_data:
            raise HTTPException(status_code=404, detail="Page not found")
        return JSONResponse(content=page_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving page {page_number} from {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve page data")

@app.get("/manuscripts/{manuscript_id}/pages/{page_number}/image")
def get_page_image(manuscript_id: str, page_number: int):
    """Get the image file for a specific manuscript page."""
    try:
        image_path = catalogue.get_image_path(manuscript_id, page_number)
        return FileResponse(image_path)
    except Exception as e:
        logger.error(f"Error retrieving image for page {page_number} from {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")

@app.post("/manuscripts/{manuscript_id}/transcribe")
def transcribe_manuscript(manuscript_id: str, request: TranscriptionRequest):
    """Start transcription of a complete manuscript."""
    try:
        success = catalogue.start_transcription(
            manuscript_id=manuscript_id,
            priority=request.priority
        )
        
        if success:
            # Add to tracking
            status = catalogue.get_transcription_status(manuscript_id)
            transcribing_manuscripts[manuscript_id] = status
            
            return JSONResponse(content={
                'status': 'started',
                'manuscript_id': manuscript_id
            })
        else:
            return JSONResponse(
                content={'status': 'already_running'},
                status_code=409
            )
            
    except Exception as e:
        logger.error(f"Error starting transcription for {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start transcription")

@app.get("/transcription/status")
def get_all_transcription_status():
    """Get status of all currently transcribing manuscripts."""
    return JSONResponse(content=transcribing_manuscripts)

@app.get("/transcription/status/{manuscript_id}")
def get_manuscript_transcription_status(manuscript_id: str):
    """Get transcription status for a specific manuscript if being transcribed."""
    status = transcribing_manuscripts.get(manuscript_id)
    if not status:
        raise HTTPException(
            status_code=404,
            detail="Manuscript not currently being transcribed"
        )
    return JSONResponse(content=status)

@app.get("/transcription/history")
def get_status_history(since: Optional[str] = None):
    """Get status history for all manuscripts, optionally filtered by time."""
    try:
        since_dt = datetime.fromisoformat(since) if since else None
        history = catalogue.get_recent_status_updates(since_dt)
        return JSONResponse(content=history)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    except Exception as e:
        logger.error(f"Error getting status history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status history")

@app.get("/transcription/history/{manuscript_id}")
def get_manuscript_status_history(manuscript_id: str, since: Optional[str] = None):
    """Get status history for a specific manuscript, optionally filtered by time."""
    try:
        since_dt = datetime.fromisoformat(since) if since else None
        history = catalogue.get_manuscript_status_history(manuscript_id, since_dt)
        if not history and not catalogue.manuscript_exists(manuscript_id):
            raise HTTPException(status_code=404, detail="Manuscript not found")
        return JSONResponse(content=history)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status history for {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status history")
    
if __name__ == "__main__":
    import uvicorn
    print("Starting Leystar Manuscript Server...")
    print("API documentation will be available at http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")