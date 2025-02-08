"""
Manuscript API Server

This module implements a Flask server providing REST API endpoints for manuscript
management, including transcription, summarization, and image access. It integrates
with the ManuscriptCatalogue and TranscriptionManager components to provide a complete
manuscript processing system.

The server supports both synchronous operations (like retrieving manuscript data) and
asynchronous operations (like transcription and summarization) with real-time status
updates via Server-Sent Events (SSE).
"""

from datetime import datetime
import json
from typing import AsyncIterator, Dict
from pathlib import Path
import logging
import asyncio

from flask import Flask, Response, jsonify, request, send_file, stream_with_context
from flask_cors import CORS

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import re

from manuscript_catalogue import ManuscriptCatalogue
from transcription_manager import TranscriptionManager

# --- Qdrant and Sentence Transformer Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'  # Use the *same* model you used for creating the embeddings!
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
COLLECTION_NAME = 'manuscripts'
CHUNK_TYPE = 'paragraph'

# --- Initialize Qdrant Client and Sentence Transformer ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer(MODEL_NAME)
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Configure CORS for development environments
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize core components
catalogue = ManuscriptCatalogue()
transcription_manager = TranscriptionManager()

@app.route('/')
def index():
    """
    Root endpoint providing API information and available endpoints.
    
    Returns:
        JSON response with API version, endpoints, and manuscript count
    """
    return jsonify({
        'name': 'Leystar Manuscript API',
        'version': '1.0',
        'endpoints': {
            'GET /manuscripts': 'List all manuscripts',
            'GET /manuscripts/{id}/info': 'Get manuscript catalogue info',
            'GET /manuscripts/{id}/pages': 'Get manuscript data by page',
            'GET /manuscripts/{id}/pages/{number}': 'Get page information',
            'GET /manuscripts/{id}/pages/{number}/image': 'Get page image',
            'POST /manuscripts/{id}/transcribe': 'Start manuscript transcription',
            'POST /manuscripts/{id}/pages/{number}/transcribe': 'Transcribe specific page',
            'POST /manuscripts/{id}/summarize': 'Generate manuscript summary'
        },
        'total_manuscripts': len(catalogue.get_manuscript_listings())
    })

@app.route('/manuscripts')
def list_manuscripts():
    """
    List all manuscripts with their basic metadata.
    
    Returns:
        JSON array of manuscript listings or error response
    """
    try:
        manuscripts = catalogue.get_manuscript_listings()
        return jsonify(manuscripts)
    except Exception as e:
        logger.error(f"Error listing manuscripts: {e}")
        return jsonify({'error': 'Failed to retrieve manuscript list'}), 500

@app.route('/manuscripts/<manuscript_id>/info')
def get_manuscript_info(manuscript_id: str):
    """
    Retrieve detailed information for a specific manuscript.
    
    Args:
        manuscript_id: Unique identifier for the manuscript
    
    Returns:
        JSON response with manuscript information or error details
    """
    try:
        catalogue.refresh_manuscript(manuscript_id)  # Always get fresh data
        manuscript = catalogue.get_manuscript_info(manuscript_id)
        
        if manuscript:
            return jsonify(manuscript)
        
        # Return available manuscripts if requested one not found
        available_manuscripts = catalogue.get_manuscript_listings()
        return jsonify({
            'error': f'Manuscript "{manuscript_id}" not found',
            'available_manuscripts': [
                {
                    'id': ms.id,
                    'title': ms.title,
                    'total_pages': ms.total_pages
                } for ms in available_manuscripts
            ]
        }), 404
    
    except Exception as e:
        logger.error(f"Error retrieving manuscript {manuscript_id}: {e}")
        return jsonify({
            'error': 'Failed to retrieve manuscript data',
            'details': str(e)
        }), 500

@app.route('/manuscripts/<manuscript_id>/pages')
def get_manuscript_pages(manuscript_id: str):
    """
    Get complete page data for a specific manuscript.
    
    Args:
        manuscript_id: Unique identifier for the manuscript
    
    Returns:
        JSON response with all page data or error details
    """
    try:
        if not catalogue.manuscript_listings.get(manuscript_id):
            return jsonify({'error': 'Manuscript not found'}), 404

        catalogue.refresh_manuscript(manuscript_id)  # Always get fresh data
        manuscript_pages = catalogue.get_manuscript_pages(manuscript_id)
        
        if manuscript_pages:
            return jsonify(manuscript_pages)
            
        logger.warning(f"{manuscript_id} is not transcribed")
        return jsonify({})
        
    except Exception as e:
        logger.error(f"Error retrieving manuscript {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to retrieve manuscript data'}), 500

@app.route('/manuscripts/<manuscript_id>/pages/<int:page_number>')
def get_page(manuscript_id: str, page_number: int):
    """
    Retrieve data for a specific manuscript page.
    
    Args:
        manuscript_id: Unique identifier for the manuscript
        page_number: Page number to retrieve (1-based)
    
    Returns:
        JSON response with page data or redirect information
    """
    try:
        catalogue.refresh_manuscript(manuscript_id)  # Always get fresh data
        manuscript = catalogue.get_manuscript_info(manuscript_id)
        if not manuscript:
            return jsonify({'error': 'Manuscript not found'}), 404
        
        total_pages = manuscript.total_pages
        if page_number < 1 or page_number > total_pages:
            page_data, _ = catalogue.get_page_data(manuscript_id, 1)
            return jsonify({
                'error': f'Requested page {page_number} is invalid',
                'redirect': {
                    'page': 1,
                    'total_pages': total_pages
                },
                'page_data': page_data
            }), 404
        
        page_data, _ = catalogue.get_page_data(manuscript_id, page_number)
        if page_data:
            return jsonify(page_data)
        
        # Fall back to first page if requested page has no data
        page_data, _ = catalogue.get_page_data(manuscript_id, 1)
        return jsonify({
            'error': f'Page {page_number} data not found',
            'redirect': {
                'page': 1,
                'total_pages': total_pages
            },
            'page_data': page_data
        }), 404
    
    except Exception as e:
        logger.error(f"Error retrieving page {page_number} from {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to retrieve page data'}), 500

@app.route('/manuscripts/<manuscript_id>/pages/<int:page_number>/image')
def get_page_image(manuscript_id: str, page_number: int):
    """
    Retrieve the image file for a specific manuscript page.
    
    Args:
        manuscript_id: Unique identifier for the manuscript
        page_number: Page number to retrieve (1-based)
    
    Returns:
        Image file response or error details
    """
    try:
        _, image_path = catalogue.get_page_data(manuscript_id, page_number)
        if image_path and image_path.exists():
            return send_file(image_path)
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving image for page {page_number} from {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to retrieve image'}), 500

@app.route('/manuscripts/<manuscript_id>/transcribe', methods=['POST'])
async def transcribe_manuscript(manuscript_id: str):
    """
    Start or continue transcription of a complete manuscript.
    
    Args:
        manuscript_id: Unique identifier for the manuscript
    
    Returns:
        Server-Sent Events stream with transcription progress updates
    """
    logger.info(f"transcribe_manuscript ROUTE called with manuscript_id: {manuscript_id}")

    try:
        catalogue.refresh_manuscript(manuscript_id)  # Always get fresh data
        if not catalogue.get_manuscript_info(manuscript_id):
            logger.warning(f"Manuscript not found: {manuscript_id}")
            return jsonify({'error': 'Manuscript not found'}), 404

        if transcription_manager.get_transcription_status(manuscript_id):
            logger.info(f"Transcription already in progress for: {manuscript_id}")
            return jsonify({
                'status': 'already_running',
                'message': 'Transcription already in progress'
            }), 409

        request_data = request.get_json()
        notes = request_data.get('notes', '')
        logger.info(f"Notes received: {notes}")

        async def generate_updates():
            async for status in transcription_manager.transcribe_manuscript(manuscript_id, notes):
                logger.info(f"Yielding status update: {status}")
                catalogue.refresh_manuscript(manuscript_id)  # Update catalogue data
                yield f"data: {json.dumps(status)}\n\n"

        async def run_in_thread():
            return [item async for item in generate_updates()]

        updates = await asyncio.to_thread(lambda: asyncio.run(run_in_thread()))

        def generate_response():
            for update in updates:
                yield update

        return Response(
            stream_with_context(generate_response()),
            mimetype='text/event-stream'
        )

    except Exception as e:
        logger.exception(f"Error starting manuscript transcription for {manuscript_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/manuscripts/<manuscript_id>/pages/<int:page_number>/transcribe', methods=['POST'])
async def transcribe_page(manuscript_id: str, page_number: int):
    """
    Transcribe a single manuscript page.
    
    Args:
        manuscript_id: Unique identifier for the manuscript
        page_number: Page number to transcribe (1-based)
    
    Returns:
        JSON response with transcription result or error details
    """
    try:
        catalogue.refresh_manuscript(manuscript_id)  # Always get fresh data
        manuscript = catalogue.get_manuscript_info(manuscript_id)
        if not manuscript:
            return jsonify({'error': 'Manuscript not found'}), 404
            
        if page_number < 1 or page_number > manuscript.total_pages:
            return jsonify({
                'error': f'Invalid page number for manuscript with {manuscript.total_pages} pages'
            }), 400

        request_data = request.get_json() if request.is_json else {}
        notes = request_data.get('notes')

        result = await transcription_manager.transcribe_page(manuscript_id, page_number, notes)
        
        if 'error' in result:
            return jsonify({
                'status': 'error',
                'error': result['error']
            }), 500

        # Get updated manuscript info after transcription
        catalogue.refresh_manuscript(manuscript_id)
        updated_manuscript = catalogue.get_manuscript_info(manuscript_id)
        
        return jsonify({
            'status': 'success',
            'data': result,
            'manuscript': {
                'transcription_info': {
                    'successful_pages': updated_manuscript.transcription_info.successful_pages,
                    'failed_pages': updated_manuscript.transcription_info.failed_pages,
                    'last_updated': updated_manuscript.transcription_info.last_updated
                }
            }
        })

    except Exception as e:
        logger.error(f"Error transcribing page {page_number} of {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to transcribe page'}), 500

@app.route('/manuscripts/<manuscript_id>/status', methods=['GET'])
def manuscript_status(manuscript_id: str):
    """
    Get real-time status of manuscript transcription process.
    
    Args:
        manuscript_id: Unique identifier for the manuscript
    
    Returns:
        Server-Sent Events stream with status updates
    """
    logger.info(f"Received status request for manuscript: {manuscript_id}")

    catalogue.refresh_manuscript(manuscript_id)  # Always get fresh data
    if not catalogue.get_manuscript_info(manuscript_id):
        logger.warning(f"Manuscript not found: {manuscript_id}")
        return jsonify({'error': 'Manuscript not found'}), 404

    def generate_status():
        status = transcription_manager.get_transcription_status(manuscript_id)
        if status is None:
            manuscript_dir = transcription_manager.catalogue_dir / manuscript_id
            transcription_path = manuscript_dir / 'transcription.json'
            
            if transcription_path.exists():
                with open(transcription_path, "r") as f:
                    transcription = json.load(f)
                    total_pages = transcription.get("total_pages")
                    successful_pages = transcription.get("successful_pages")
                    
                if total_pages == successful_pages:
                    yield f"data: {json.dumps({
                        'status': 'completed',
                        'total_pages': total_pages,
                        'successful_pages': successful_pages,
                        'failed_pages': []
                    })}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'not_started'})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'not_started'})}\n\n"
        else:
            yield f"data: {json.dumps(status)}\n\n"

    return Response(generate_status(), mimetype='text/event-stream')
    
@app.route('/manuscripts/<manuscript_id>/summarize', methods=['POST'])
async def generate_summary(manuscript_id: str):
    """
    Generate or update summary for a manuscript.
    
    Args:
        manuscript_id: Unique identifier for the manuscript
    
    Returns:
        JSON response with summary generation result
    """
    try:
        catalogue.refresh_manuscript(manuscript_id)  # Always get fresh data
        manuscript = catalogue.get_manuscript_info(manuscript_id)
        if not manuscript:
            return jsonify({'error': 'Manuscript not found'}), 404

        result = await transcription_manager.generate_summary(manuscript_id)
        
        if 'error' in result:
            return jsonify({
                'status': 'error',
                'error': result['error']
            }), 500

        # Refresh catalogue data after summary generation
        catalogue.refresh_manuscript(manuscript_id)
        
        return jsonify({
            'status': 'success',
            'message': 'Summary generated successfully'
        })

    except Exception as e:
        logger.error(f"Error generating summary for {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to generate summary'}), 500

# --- Helper function for chunking ---
def chunk_text(text, chunk_type='paragraph'):
    if chunk_type == 'sentence':
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    elif chunk_type == 'paragraph':
        return text.split('\n\n')
    else:
        raise ValueError("Invalid chunk_type")

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" parameter'}), 400

        query = data['query']
        search_type = data.get('search_type', 'page')  # Default to page-level
        top_k = data.get('top_k', 10)
        manuscript_id = data.get('manuscript_id')
        page_number = data.get('page_number')
        min_score = data.get('min_score')

        query_embedding = model.encode([query])[0]

        # --- Build the Filter ---
        filter_conditions = []

        if search_type == "manuscript":
            filter_conditions.append(models.FieldCondition(key="type", match=models.MatchValue(value="summary")))
        elif search_type == "page":
            # Search across sections, page summaries, marginalia, and content notes:
            filter_conditions.append(
                models.FieldCondition(key="type", match=models.MatchAny(any=["section", "page_summary", "marginalia", "content_notes"]))
            )
        elif search_type == "toc":
            filter_conditions.append(models.FieldCondition(key="type", match=models.MatchValue(value="toc_entry")))

        # Add more search types if needed, like "all"
        elif search_type == "all":
          filter_conditions.append(
                models.FieldCondition(key="type", match=models.MatchAny(any=["section", "page_summary", "marginalia", "content_notes","summary","toc_entry"]))
            )
        else:
            return jsonify({'error': 'Invalid search_type'}), 400


        if manuscript_id:
            filter_conditions.append(models.FieldCondition(key="manuscript_id", match=models.MatchValue(value=manuscript_id)))
        if page_number:
            filter_conditions.append(models.FieldCondition(key="page_number", match=models.MatchValue(value=int(page_number))))

        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        # --- Perform the Search ---
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=min_score
        )

        # --- Format Results ---
        results = []
        for point in search_result:
            results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)