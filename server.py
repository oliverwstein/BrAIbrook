"""
Server implementation for the Leystar manuscript system.
Provides REST API endpoints for manuscript management and transcription tracking.
"""

from contextlib import asynccontextmanager
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends, status, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from jose import JWTError, jwt


from catalogue import ManuscriptCatalogue

# Load environment variables
load_dotenv()

# Security constants
SECRET_KEY = os.getenv("JWT_SECRET")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

class TranscriptionRequestBody(BaseModel):
    notes: str = ""
    priority: int = 1
    pages: str = ""  # Optional string like "1,2,3" or "1-5" or empty for all pages


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

# --- CORS Configuration ---
# Use allow_origins=["*"] for development simplicity (but NOT for production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://ley-star.loca.lt"  # Your frontend localtunnel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_admin(token: Optional[str] = Cookie(None, alias="auth_token")):
    """Check if the current request has valid admin credentials."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication token provided"
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        is_admin: bool = payload.get("is_admin", False)
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to perform this action"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return True

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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_host = request.client.host
    logger.info(f"Request from IP: {client_host}")
    response = await call_next(request)
    return response

@app.on_event("startup")
async def startup_event():
    """Start background tasks when server starts."""
    asyncio.create_task(update_transcription_status())

@app.get("/admin/status")
async def check_admin_status(token: Optional[str] = Cookie(None, alias="auth_token")):
    """Check current admin authentication status."""
    if not token:
        return JSONResponse(content={"is_admin": False})
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        is_admin = payload.get("is_admin", False)
        return JSONResponse(content={"is_admin": is_admin})
    except JWTError:
        return JSONResponse(content={"is_admin": False})
    
@app.post("/admin/login")
async def login(password: str = Form(...)):
    if password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"is_admin": True}, expires_delta=access_token_expires
    )
    
    response = JSONResponse(content={"message": "Login successful"})
    response.set_cookie(
        key="auth_token",
        value=access_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return response

@app.post("/admin/logout")
async def logout():
    response = JSONResponse(content={"message": "Logout successful"})
    response.delete_cookie(key="auth_token")
    return response

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


@app.get("/admin/transcription-requests")
async def list_transcription_requests(is_admin: bool = Depends(get_current_admin)):
    """Get list of pending transcription requests."""
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin access required"
        )
    
    try:
        requests = catalogue.get_pending_requests()
        return JSONResponse(content=requests)
    except Exception as e:
        logger.error(f"Error listing transcription requests: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve requests")

@app.post("/admin/transcription-requests/{manuscript_id}/approve")
async def approve_transcription_request(
    manuscript_id: str,
    is_admin: bool = Depends(get_current_admin)
):
    """Approve a pending transcription request."""
    if not is_admin:
        raise HTTPException(
            status_code=401,
            detail="Admin access required"
        )
    
    try:
        success = catalogue.approve_request(manuscript_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail="No pending request found for this manuscript"
            )
        
        # Add to tracking
        status = catalogue.get_transcription_status(manuscript_id)
        transcribing_manuscripts[manuscript_id] = status
        
        # Notify connected clients
        await manager.broadcast(json.dumps({
            'type': 'transcription_update',
            'manuscript_id': manuscript_id,
            'data': status
        }))
        
        return JSONResponse(content={
            'status': 'approved',
            'manuscript_id': manuscript_id
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving request for {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve request")

@app.post("/admin/transcription-requests/{manuscript_id}/reject")
async def reject_transcription_request(
    manuscript_id: str,
    is_admin: bool = Depends(get_current_admin)
):
    """Reject a pending transcription request."""
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin access required"
        )
    
    try:
        success = catalogue.reject_request(manuscript_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail="No pending request found for this manuscript"
            )
            
        # Get updated status and notify clients
        status = catalogue.get_transcription_status(manuscript_id)
        await manager.broadcast(json.dumps({
            'type': 'transcription_update',
            'manuscript_id': manuscript_id,
            'data': status
        }))
        
        return JSONResponse(content={
            'status': 'rejected',
            'manuscript_id': manuscript_id
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting request for {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reject request")

@app.post("/manuscripts/{manuscript_id}/transcribe")
async def transcribe_manuscript(
    manuscript_id: str, 
    request: TranscriptionRequestBody,
    token: Optional[str] = Cookie(None, alias="auth_token")
):
    """Start or request transcription of a manuscript."""
    try:
        # Parse pages string if provided
        pages_to_process = None
        if request.pages.strip():
            try:
                # Handle both comma-separated list and ranges
                pages = []
                for part in request.pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        pages.extend(range(start, end + 1))
                    else:
                        pages.append(int(part))
                pages_to_process = sorted(list(set(pages)))  # Remove duplicates and sort
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid page format. Use comma-separated numbers or ranges (e.g., '1,2,3' or '1-5')"
                )

        # Check admin status
        is_admin = False
        if token:
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                is_admin = payload.get("is_admin", False)
            except JWTError:
                pass

        # Log the request
        page_info = f" (pages {request.pages})" if request.pages.strip() else " (all pages)"
        logger.info(
            f"Received transcription request for {manuscript_id}{page_info} "
            f"(admin: {is_admin}, priority: {request.priority})"
        )
        
        if not is_admin:
            logger.info(f"Processing non-admin request for {manuscript_id}")
            try:
                success = catalogue.request_transcription(
                    manuscript_id=manuscript_id,
                    notes=request.notes,
                    priority=request.priority,
                    pages=pages_to_process
                )
                logger.info(f"Request transcription result for {manuscript_id}: {success}")
            except Exception as e:
                logger.error(f"Error in request_transcription for {manuscript_id}: {str(e)}")
                raise
            
            if success:
                # Get status and notify clients about the request
                try:
                    status = catalogue.get_transcription_status(manuscript_id)
                    logger.info(f"Got status for requested manuscript {manuscript_id}: {status}")
                    await manager.broadcast(json.dumps({
                        'type': 'transcription_update',
                        'manuscript_id': manuscript_id,
                        'data': status
                    }))
                except Exception as e:
                    logger.error(f"Error broadcasting status for {manuscript_id}: {str(e)}")
                    raise
                
                return JSONResponse(
                    content={
                        'status': 'request_submitted',
                        'manuscript_id': manuscript_id,
                        'message': 'Transcription request submitted for approval'
                    },
                    status_code=202
                )
            else:
                return JSONResponse(
                    content={'status': 'already_requested'},
                    status_code=409
                )
            
        # For admins, proceed with direct transcription
        logger.info(f"Processing admin transcription request for {manuscript_id}{page_info}")
        success = catalogue.start_transcription(
            manuscript_id=manuscript_id,
            priority=request.priority,
            pages=pages_to_process,
            notes=request.notes  # Pass through the notes for admin transcriptions
        )
        
        if success:
            # Add to tracking
            status = catalogue.get_transcription_status(manuscript_id)
            transcribing_manuscripts[manuscript_id] = status
            
            # Notify clients
            await manager.broadcast(json.dumps({
                'type': 'transcription_update',
                'manuscript_id': manuscript_id,
                'data': status
            }))
            
            logger.info(f"Successfully started transcription for {manuscript_id}{page_info}")
            return JSONResponse(content={
                'status': 'started',
                'manuscript_id': manuscript_id
            })
        else:
            logger.warning(f"Transcription already in progress for {manuscript_id}")
            return JSONResponse(
                content={'status': 'already_running'},
                status_code=409
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling transcription for {manuscript_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process transcription")
    
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