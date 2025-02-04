import json

from flask import Flask, Response, jsonify, send_file
from manuscript_catalogue import ManuscriptCatalogue
from transcription_manager import TranscriptionManager
from flask_cors import CORS
import logging
import asyncio
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
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
    """Root endpoint providing API information."""
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
    """List all manuscripts with their metadata."""
    try:
        manuscripts = catalogue.get_manuscript_listings()
        return jsonify(manuscripts)
    except Exception as e:
        logger.error(f"Error listing manuscripts: {e}")
        return jsonify({'error': 'Failed to retrieve manuscript list'}), 500

@app.route('/manuscripts/<manuscript_id>/info')
def get_manuscript_info(manuscript_id):
    """
    Retrieve catalogue information for a specific manuscript.
    If the manuscript is not found, return a list of available manuscripts.
    
    Args:
        manuscript_id (str): Unique identifier for the manuscript
    
    Returns:
        JSON response with either manuscript information or a list of available manuscripts
    """
    try:
        manuscript = catalogue.get_manuscript_info(manuscript_id)
        if manuscript:
            return jsonify(manuscript)
        
        # If manuscript not found, retrieve and return the list of available manuscripts
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
        }), 5000

@app.route('/manuscripts/<manuscript_id>/pages')
def get_manuscript_pages(manuscript_id):
    """Get complete information for a specific manuscript."""
    try:
        manuscript = catalogue.get_manuscript_pages(manuscript_id)
        if manuscript:
            return jsonify(manuscript)
        return jsonify({'error': 'Manuscript not found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving manuscript {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to retrieve manuscript data'}), 500

@app.route('/manuscripts/<manuscript_id>/pages/<int:page_number>')
def get_page(manuscript_id, page_number):
    """
    Retrieve transcription data for a specific page of a manuscript.
    If the requested page is invalid, redirect to the first page.
    
    Args:
        manuscript_id (str): Unique identifier for the manuscript
        page_number (int): Requested page number
    
    Returns:
        JSON response with page data or a redirect to the first page
    """
    try:
        # Attempt to get the manuscript info to determine total pages
        manuscript = catalogue.get_manuscript_info(manuscript_id)
        if not manuscript:
            return jsonify({'error': 'Manuscript not found'}), 404
        
        # Validate page number
        total_pages = manuscript.total_pages
        if page_number < 1 or page_number > total_pages:
            # If page is out of range, return first page data with a redirect hint
            page_data, image_path = catalogue.get_page_data(manuscript_id, 1)
            return jsonify({
                'error': f'Requested page {page_number} is invalid',
                'redirect': {
                    'page': 1,
                    'total_pages': total_pages
                },
                'page_data': page_data
            }), 404
        
        # Retrieve the requested page data
        page_data, image_path = catalogue.get_page_data(manuscript_id, page_number)
        if page_data:
            return jsonify(page_data)
        
        # If no page data found, fall back to first page
        page_data, image_path = catalogue.get_page_data(manuscript_id, 1)
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
def get_page_image(manuscript_id, page_number):
    """Get the image for a specific page."""
    try:
        _, image_path = catalogue.get_page_data(manuscript_id, page_number)
        if image_path and image_path.exists():
            return send_file(image_path)
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving image for page {page_number} from {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to retrieve image'}), 500

@app.route('/manuscripts/<manuscript_id>/transcribe', methods=['POST'])
def transcribe_manuscript(manuscript_id):
    """Start transcription of a manuscript."""
    try:
        # Get manuscript info first
        manuscript = catalogue.get_manuscript_view(manuscript_id)
        if not manuscript:
            return jsonify({'error': 'Manuscript not found'}), 404

        # Check if transcription is already running
        if transcription_manager.get_transcription_status(manuscript_id):
            return jsonify({
                'status': 'already_running',
                'message': 'Transcription already in progress'
            }), 409

        # Start transcription and return initial status
        async def start_transcription():
            async for status in transcription_manager.transcribe_manuscript(manuscript_id):
                yield f"data: {json.dumps(status)}\n\n"

        return Response(
            start_transcription(),
            mimetype='text/event-stream'
        )

    except Exception as e:
        logger.error(f"Error starting transcription for {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to start transcription'}), 500

@app.route('/manuscripts/<manuscript_id>/pages/<int:page_number>/transcribe', methods=['POST'])
async def transcribe_page(manuscript_id, page_number):
    """Transcribe a specific page."""
    try:
        # Verify manuscript and page exist
        manuscript = catalogue.get_manuscript_view(manuscript_id)
        if not manuscript:
            return jsonify({'error': 'Manuscript not found'}), 404
            
        if page_number < 1 or page_number > manuscript.total_pages:
            return jsonify({'error': 'Invalid page number'}), 400

        # Transcribe the page
        result = await transcription_manager.transcribe_page(manuscript_id, page_number)
        
        if 'error' in result:
            return jsonify({
                'status': 'error',
                'error': result['error']
            }), 500

        return jsonify({
            'status': 'success',
            'data': result
        })

    except Exception as e:
        logger.error(f"Error transcribing page {page_number} of {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to transcribe page'}), 500

@app.route('/manuscripts/<manuscript_id>/summarize', methods=['POST'])
async def generate_summary(manuscript_id):
    """Generate a summary for a manuscript."""
    try:
        # Verify manuscript exists
        manuscript = catalogue.get_manuscript_view(manuscript_id)
        if not manuscript:
            return jsonify({'error': 'Manuscript not found'}), 404

        # Generate summary
        result = await transcription_manager.generate_summary(manuscript_id)
        
        if 'error' in result:
            return jsonify({
                'status': 'error',
                'error': result['error']
            }), 500

        return jsonify({
            'status': 'success',
            'message': 'Summary generated successfully'
        })

    except Exception as e:
        logger.error(f"Error generating summary for {manuscript_id}: {e}")
        return jsonify({'error': 'Failed to generate summary'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)