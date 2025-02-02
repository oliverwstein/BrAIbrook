import asyncio
from datetime import datetime
import time
import traceback
from flask import Flask, Response, jsonify, send_file, request
from flask_cors import CORS
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from threading import Thread
from gemini_transcribe import create_safe_filename
from search import init_search_engine
from threading import enumerate as enumerate_threads
import psutil

# Set up logging to be more visible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # This ensures our logging config takes precedence
)
logger = logging.getLogger(__name__)

# Ensure logging is visible
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",    # Default Vite dev server
            "http://127.0.0.1:5173",    # Alternative Vite dev server
            "http://localhost:4173",    # Vite preview
            "http://127.0.0.1:4173",    # Alternative Vite preview
            "http://localhost:3000",    # Alternative dev port
            "http://127.0.0.1:3000",    # Alternative dev port
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True     # If you need to handle cookies/auth
    }
})

class ManuscriptServer:
    def __init__(self, raw_dir: str = "data/raw", transcripts_dir: str = "data/transcripts"):
        logger.info("Starting ManuscriptServer initialization...")
        
        self.raw_dir = Path(raw_dir).absolute()
        self.transcripts_dir = Path(transcripts_dir).absolute()
        
        if not self.raw_dir.exists():
            logger.error(f"Raw directory does not exist: {self.raw_dir}")
            raise FileNotFoundError(f"Raw directory not found: {self.raw_dir}")
        if not self.transcripts_dir.exists():
            logger.error(f"Transcripts directory does not exist: {self.transcripts_dir}")
            raise FileNotFoundError(f"Transcripts directory not found: {self.transcripts_dir}")
            
        # Initialize unified manuscript tracking
        self.manuscripts = self._initialize_manuscripts()
        self.transcription_info = {}  # Track ongoing transcriptions

    async def status_stream(self, title: str):
        """Stream status updates for a manuscript's transcription progress."""
        safe_title = create_safe_filename(title)
        transcript_path = self.transcripts_dir / safe_title / 'transcription.json'
        last_mtime = None
        last_status = None

        while True:
            try:
                if transcript_path.exists():
                    with open(transcript_path, 'r') as f:
                        data = json.load(f)
                        pages = data.get('pages', {})
                        successful_pages = len([p for p in pages.values() 
                                            if not p.get('error')])
                        status = {
                            'status': 'in_progress',
                            'successful_pages': successful_pages,
                            'total_pages': data.get('total_pages', 0),
                            'failed_pages': data.get('failed_pages', [])
                        }

                        if status != last_status:
                            # Update manuscript data with current progress
                            self.manuscripts[title].update({
                                'transcribed': successful_pages >= status['total_pages'],
                                'transcript_file': transcript_path,
                                'transcription_info': {
                                    'successful_pages': successful_pages,
                                    'failed_pages': status['failed_pages'],
                                    'last_updated': datetime.now().isoformat()
                                }
                            })
                            yield {'data': json.dumps(status)}
                            last_status = status.copy()

                            if successful_pages >= status['total_pages']:
                                yield {'data': json.dumps({**status, 'status': 'completed'})}
                                break

                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error reading transcript for {title}: {e}")
                await asyncio.sleep(1)
                
    def start_transcription(self, title: str, manuscript_folder: Path) -> None:
        def transcribe():
            try:
                from gemini_transcribe import ManuscriptProcessor, process_manuscript
                processor = ManuscriptProcessor()
                safe_title = create_safe_filename(title)
                transcript_path = self.transcripts_dir / safe_title / 'transcription.json'

                # Initialize status
                self.transcription_info[title] = {
                    'status': 'in_progress',
                    'started_at': datetime.now().isoformat(),
                    'successful_pages': 0
                }

                # Start transcription process
                result = process_manuscript(
                    str(manuscript_folder),
                    str(self.transcripts_dir),
                    processor
                )

                if result:
                    # Monitor and update progress
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        pages = data.get('pages', {})
                        successful_pages = len(pages)
                        
                        # Update both status trackers
                        progress_info = {
                            'successful_pages': successful_pages,
                            'failed_pages': data.get('failed_pages', []),
                            'last_updated': datetime.now().isoformat()
                        }
                        
                        self.manuscripts[title].update({
                            'transcribed': True,
                            'transcript_file': transcript_path,
                            'transcription_info': progress_info
                        })

                        self.transcription_info[title] = {
                            'status': 'completed',
                            'completed_at': datetime.now().isoformat(),
                            **progress_info
                        }
                else:
                    self.transcription_info[title] = {
                        'status': 'failed',
                        'error': 'Transcription returned no results',
                        'failed_at': datetime.now().isoformat()
                    }

            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                self.transcription_info[title] = {
                    'status': 'failed',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }

        Thread(target=transcribe, daemon=True).start()

    def generate_manuscript_summary(self, title: str) -> Dict:
        """Generate a summary and table of contents for a manuscript, updating its data."""
        info = self.get_manuscript_info(title)
        if not info:
            raise ValueError('Manuscript not found')

        if not info.get('transcribed'):
            raise ValueError('Cannot generate summary for untranscribed manuscript')

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError('API key not configured')

        safe_title = create_safe_filename(title)
        transcript_path = self.transcripts_dir / safe_title / 'transcription.json'

        if not transcript_path.exists():
            raise ValueError('Transcription file not found')

        # Initialize summarizer and generate summary
        from summarizer import ManuscriptSummarizer
        summarizer = ManuscriptSummarizer(api_key)
        summarizer.update_transcription(transcript_path)

        # Read the updated transcription and update manuscript data
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcription = json.load(f)
            
        # Update the manuscript information with new summary data
        self.manuscripts[title].update({
            'summary': transcription.get('summary', {}),
            'table_of_contents': transcription.get('table_of_contents', [])
        })
        
        return {
            'summary': transcription.get('summary', {}),
            'table_of_contents': transcription.get('table_of_contents', [])
        }
    
    def get_page_data(self, title: str, page_number: int) -> Tuple[Optional[Dict], Optional[str]]:
        """Get transcription data and image path for a specific page."""
        if title not in self.manuscripts:
            return None, None

        info = self.manuscripts[title]
        page_idx = page_number - 1

        # Get image path
        image_files = sorted([f for f in info['raw_folder'].iterdir() 
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
        
        image_path = str(image_files[page_idx]) if page_idx < len(image_files) else None

        # Get transcription data if available
        page_data = None
        if info['transcribed']:
            try:
                with open(info['transcript_file'], 'r', encoding='utf-8') as f:
                    transcription = json.load(f)
                
                if str(page_number) in transcription.get('pages', {}):
                   page_data = transcription['pages'][str(page_number)]
            except Exception as e:
                logger.error(f"Error loading transcription data: {e}")

        return page_data, image_path

    def _initialize_manuscripts(self) -> Dict[str, Dict]:
        """Create unified mapping of all manuscripts with their status and data locations."""
        logger.info("Starting unified manuscript initialization...")
        manuscripts = {}

        # First, scan raw directory for all manuscripts
        for folder in self.raw_dir.iterdir():
            if not folder.is_dir():
                continue

            try:
                metadata_file = folder / 'metadata.json'
                if not metadata_file.exists():
                    continue

                # Load metadata
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                title = metadata.get('Title')
                record_id = metadata.get('Record ID')

                if not title or not record_id:
                    logger.warning(f"Missing title or record ID in {folder}")
                    continue

                # Count images
                image_files = sorted([f for f in folder.iterdir() 
                                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])

                # Initialize basic manuscript info
                manuscripts[title] = {
                    'record_id': record_id,
                    'metadata': metadata,
                    'raw_folder': folder,
                    'total_pages': len(image_files),
                    'transcribed': False,
                    'transcription_info': None,
                    'summary': {},
                    'table_of_contents': []
                }
                
            except Exception as e:
                logger.error(f"Error processing raw folder {folder}: {e}")

        # Then, scan transcripts directory and update transcription status
        for transcript_file in self.transcripts_dir.glob("*/transcription.json"):
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcription_data = json.load(f)
                
                title = transcription_data.get('manuscript_title')
                if not title or title not in manuscripts:
                    continue

                manuscripts[title].update({
                    'transcribed': True,
                    'transcript_file': transcript_file,
                    'transcription_info': {
                        'successful_pages': len(transcription_data.get('pages', {})),
                        'failed_pages': transcription_data.get('failed_pages', []),
                        'last_updated': transcription_data.get('last_updated', None)
                    },
                    'pages': transcription_data.get('pages', {}),
                    'summary': transcription_data.get('summary', {}),
                    'table_of_contents': transcription_data.get('table_of_contents', [])
                })

            except Exception as e:
                logger.error(f"Error processing transcript file {transcript_file}: {e}")

        logger.info(f"Manuscript initialization complete. Found {len(manuscripts)} manuscripts")
        return manuscripts

    def list_manuscripts(self) -> list:
        """Get a list of all manuscripts with their status."""
        return [{
            'title': title,
            'record_id': info['record_id'],
            'total_pages': info['total_pages'],
            'transcribed': info['transcribed'],
            'transcription_info': {
                'successful_pages': info['transcription_info']['successful_pages'],
                'failed_pages': info['transcription_info']['failed_pages'],
                'last_updated': info['transcription_info']['last_updated']
            } if info['transcribed'] else None,
            'summary': info.get('summary', {}),
            'table_of_contents': info.get('table_of_contents', [])
        } for title, info in self.manuscripts.items()]

    def get_manuscript_info(self, title: str) -> Optional[Dict]:
        """Get detailed information about a manuscript."""
        if title not in self.manuscripts:
            return None
        
        info = self.manuscripts[title]
        return {
            'title': title,
            'record_id': info.get('record_id', ''),
            'metadata': info.get('metadata', {}),
            'total_pages': info.get('total_pages', 0),
            'transcribed': info.get('transcribed', False),
            'transcription_info': info.get('transcription_info', None),
            'summary': info.get('summary', {}),
            'table_of_contents': info.get('table_of_contents', [])
        }
        

# Initialize server
try:
    logger.info("Initializing ManuscriptServer...")
    manuscript_server = ManuscriptServer()
    logger.info("ManuscriptServer initialization complete")
    logger.info("Initializing search engine...")
    search_engine = init_search_engine()
    search_engine.index_manuscripts(manuscript_server.manuscripts)
    logger.info("Search engine initialization complete")
except Exception as e:
    logger.error(f"Failed to initialize ManuscriptServer: {e}")
    raise

@app.errorhandler(Exception)
def handle_error(error):
    response = jsonify({'error': str(error)})
    response.status_code = 500
    return response

@app.route('/')
def index():
    """Root route that shows available endpoints and manuscripts."""
    manuscripts = manuscript_server.list_manuscripts()
    endpoints = {
        'List all manuscripts': '/manuscripts',
        'Get manuscript info': '/manuscripts/<title>',
        'Get page transcription': '/manuscripts/<title>/pages/<page_number>',
        'Get page image': '/manuscripts/<title>/pages/<page_number>/image'
    }
    
    return jsonify({
        'available_endpoints': endpoints,
        'available_manuscripts': manuscripts
    })

@app.route('/manuscripts', methods=['GET'])
def list_manuscripts():
    """List all manuscripts with their transcription status."""
    logger.info("Handling /manuscripts request")
    return jsonify(manuscript_server.list_manuscripts())

@app.route('/manuscripts/<path:title>', methods=['GET'])
def manuscript_info(title):
    """Get detailed information about a specific manuscript."""
    logger.info(f"Handling /manuscripts/{title} request")
    info = manuscript_server.get_manuscript_info(title)
    if info:
        return jsonify(info)
    return jsonify({'error': 'Manuscript not found'}), 404

@app.route('/manuscripts/<path:title>/transcription', methods=['GET'])
def get_manuscript_transcription(title):
    """Get complete transcription data for a manuscript."""
    info = manuscript_server.get_manuscript_info(title)
    if not info:
        return jsonify({'error': 'Manuscript not found'}), 404
        
    try:
        # Get transcript file path
        safe_title = create_safe_filename(title)
        transcript_path = manuscript_server.transcripts_dir / safe_title / 'transcription.json'
        
        if not transcript_path.exists():
            return jsonify({'error': 'No transcription available'}), 404
            
        # Load and return the complete transcription data
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
            return jsonify(transcription_data)
            
    except Exception as e:
        logger.error(f"Error loading transcription for {title}: {e}")
        return jsonify({'error': f'Failed to load transcription: {str(e)}'}), 500
    
@app.route('/manuscripts/<path:title>/pages/<int:page>', methods=['GET'])
def get_page(title, page):
    """Get transcription data for a specific page."""
    logger.info(f"Handling /manuscripts/{title}/pages/{page} request")
    page_data, _ = manuscript_server.get_page_data(title, page)
    if page_data:
        return jsonify(page_data)
    return jsonify({'error': 'Page not found'}), 404

@app.route('/manuscripts/<path:title>/pages/<int:page>/image', methods=['GET'])
def get_page_image(title, page):
    """Get the image for a specific page."""
    logger.info(f"Handling /manuscripts/{title}/pages/{page}/image request")
    _, image_path = manuscript_server.get_page_data(title, page)
    if image_path:
        return send_file(image_path)
    return jsonify({'error': 'Image not found'}), 404

@app.route('/manuscripts/<path:title>/transcribe', methods=['POST'])
def transcribe_manuscript(title):
    """Start transcription of a specific manuscript."""
    logger.info(f"Handling transcribe request for {title}")
    
    info = manuscript_server.get_manuscript_info(title)
    if not info:
        return jsonify({'error': 'Manuscript not found'}), 404

    # Check if transcription is complete by comparing page counts
    if info.get('transcription_info'):
        successful_pages = info['transcription_info'].get('successful_pages', 0)
        total_pages = info['total_pages']
        
        if successful_pages == total_pages:
            return jsonify({
                'status': 'complete',
                'message': f'Manuscript already fully transcribed ({successful_pages}/{total_pages} pages)'
            }), 400
    
    # Check if transcription is already in progress
    if title in manuscript_server.transcription_info:
        status = manuscript_server.transcription_info[title]['status']
        if status == 'in_progress':
            current_progress = manuscript_server.transcription_info[title].get('current_page', 0)
            return jsonify({
                'status': 'in_progress',
                'message': f'Transcription already in progress (page {current_progress + 1}/{info["total_pages"]})'
            })
    
    # Start transcription in background
    manuscript_folder = manuscript_server.manuscripts[title]['raw_folder']
    manuscript_server.start_transcription(title, manuscript_folder)
    
    return jsonify({
        'status': 'started',
        'message': f'Starting transcription of {info["total_pages"]} pages'
    })

@app.route('/manuscripts/<path:title>/pages/<int:page>/transcribe', methods=['POST'])
def transcribe_page(title, page):
    """
    Transcribe a specific page of a manuscript using the Gemini Vision AI.

    Request JSON body can include optional:
    - notes: Additional context or notes for transcription
    - previous_page: Page context for previous page (optional)
    - next_page: Page context for next page (optional)

    Args:
        title (str): The manuscript title
        page (int): The page number to transcribe

    Returns:
        JSON response with transcription results or error details
    """
    try:
        # Validate manuscript exists
        manuscript_info = manuscript_server.get_manuscript_info(title)
        if not manuscript_info:
            return jsonify({'error': 'Manuscript not found'}), 404
        
        # Get request data, with fallbacks
        request_data = request.get_json(silent=True) or {}
        notes = request_data.get('notes', '')
        
        # Locate the manuscript's raw image folder
        raw_folder = manuscript_server.manuscripts[title]['raw_folder']
        
        # Find image files, sorted to match page numbering
        image_files = sorted([f for f in raw_folder.iterdir() 
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
        
        # Validate page number
        if page < 1 or page > len(image_files):
            return jsonify({'error': 'Page number out of range'}), 400
        
        # Get the specific image path for the requested page
        image_path = str(image_files[page - 1])
        
        # Import the transcription processor
        from gemini_transcribe import ManuscriptProcessor
        
        # Initialize the processor
        processor = ManuscriptProcessor()
        
        # Prepare context for previous and next pages
        existing_results = manuscript_server.manuscripts[title].get('pages', {})
        
        # Retrieve previous page context if it exists
        previous_page = request_data.get('previous_page')
        if not previous_page and page > 1:
            previous_page = existing_results.get(str(page - 1))
        
        # Retrieve next page context if it exists
        next_page = request_data.get('next_page')
        if not next_page and page < len(image_files):
            next_page = existing_results.get(str(page + 1))
        
        # Process the specific page
        page_result = processor.process_page(
            image_path,
            manuscript_info['metadata'],
            page,
            previous_page,
            next_page,
            notes
        )
        
        # Update the transcription file
        safe_title = create_safe_filename(title)
        transcript_path = manuscript_server.transcripts_dir / safe_title / 'transcription.json'
        
        # Load existing transcription data or create new
        if transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
        else:
            transcription_data = {
                'manuscript_title': title,
                'metadata': manuscript_info['metadata'],
                'pages': {},
                'total_pages': len(image_files),
                'successful_pages': 0,
                'failed_pages': []
            }
        
        # Update the specific page in the transcription data
        transcription_data['pages'][str(page)] = page_result
        
        # Update successful/failed page tracking
        if 'error' not in page_result:
            if str(page) not in transcription_data['pages'] or 'error' in transcription_data['pages'][str(page)]:
                transcription_data['successful_pages'] += 1
                
                # Remove from failed pages if previously marked
                if page in transcription_data.get('failed_pages', []):
                    transcription_data['failed_pages'].remove(page)
        else:
            # Add to failed pages if not already there
            if page not in transcription_data.get('failed_pages', []):
                transcription_data.setdefault('failed_pages', []).append(page)
        
        # Save updated transcription data
        os.makedirs(transcript_path.parent, exist_ok=True)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)
        
        # Update manuscript status
        manuscript_server.manuscripts[title].update({
            'transcribed': transcription_data['successful_pages'] == transcription_data['total_pages'],
            'transcript_file': transcript_path,
            'transcription_info': {
                'successful_pages': transcription_data['successful_pages'],
                'failed_pages': transcription_data.get('failed_pages', []),
                'last_updated': datetime.now().isoformat()
            }
        })
        
        return jsonify(page_result)

    except Exception as e:
        logger.error(f"Error transcribing page {page} of {title}: {e}")
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500
    
@app.route('/manuscripts/<path:title>/generate-summary', methods=['POST'])
def generate_manuscript_summary(title):
    """Route handler for generating manuscript summaries."""
    try:
        result = manuscript_server.generate_manuscript_summary(title)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return jsonify({'error': f'Summary generation failed: {str(e)}'}), 500
    
@app.route('/manuscripts/<path:title>/status', methods=['GET'])
def manuscript_status_stream(title):
    def event_stream():
        while True:
            if title in manuscript_server.manuscripts:
                safe_title = create_safe_filename(title)
                transcript_path = manuscript_server.transcripts_dir / safe_title / 'transcription.json'
                
                if transcript_path.exists():
                    try:
                        with open(transcript_path, 'r') as f:
                            data = json.load(f)
                            pages = data.get('pages', {})
                            successful_pages = len([p for p in pages.values() if not p.get('error')])
                            status = {
                                'status': 'in_progress',
                                'successful_pages': successful_pages,
                                'total_pages': data.get('total_pages', 0),
                                'failed_pages': data.get('failed_pages', [])
                            }
                            
                            yield f"data: {json.dumps(status)}\n\n"
                    except Exception as e:
                        logger.error(f"Error reading transcript: {e}")
            time.sleep(1)
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/search/manuscripts', methods=['GET'])
def search_manuscripts():
    """Search manuscripts using metadata."""
    try:
        query = request.args.get('q', '')
        num_results = int(request.args.get('limit', 10))
        
        logger.info(f"Received search request - query: {query}, limit: {num_results}")
        
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
            
        search_engine = init_search_engine()
        logger.info("Search engine initialized")
        
        try:
            results = search_engine.search_manuscripts(query, num_results=num_results)
            logger.info(f"Search completed successfully with {len(results)} results")
        except Exception as search_error:
            logger.error(f"Search engine error: {str(search_error)}")
            logger.error(f"Search engine error traceback:", exc_info=True)
            return jsonify({'error': f'Search engine error: {str(search_error)}'}), 500

        try:
            response = jsonify({
                'query': query,
                'num_results': len(results),
                'results': results
            })
            logger.info("Response successfully jsonified")
            return response
        except Exception as json_error:
            logger.error(f"JSON serialization error: {str(json_error)}")
            logger.error("JSON serialization error traceback:", exc_info=True)
            return jsonify({'error': f'JSON serialization error: {str(json_error)}'}), 500
            
    except Exception as e:
        logger.error(f"Route handler error: {str(e)}")
        logger.error("Route handler error traceback:", exc_info=True)
        return jsonify({'error': f'Route handler error: {str(e)}'}), 500

@app.route('/search/pages', methods=['GET'])
def search_pages():
    """Search pages across all manuscripts or within a specific manuscript."""
    query = request.args.get('q', '')
    num_results = int(request.args.get('limit', 10))
    manuscript_title = request.args.get('manuscript')  # Optional
    
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
        
    search_engine = init_search_engine()
    results = search_engine.search_pages(
        query, 
        num_results=num_results,
        manuscript_title=manuscript_title
    )
    
    return jsonify({
        'query': query,
        'manuscript': manuscript_title,
        'num_results': len(results),
        'results': results
    })

@app.route('/manuscripts/<path:title>/similar', methods=['GET'])
def similar_manuscripts(title):
    """Find manuscripts similar to a given one."""
    num_results = int(request.args.get('limit', 5))
    
    search_engine = init_search_engine()
    results = search_engine.get_similar_manuscripts(title, num_results=num_results)
    
    return jsonify({
        'manuscript': title,
        'num_similar': len(results),
        'similar_manuscripts': results
    })

@app.route('/manuscripts/<path:title>/pages/<int:page>/similar', methods=['GET'])
def similar_pages(title, page):
    """Find pages similar to a given one."""
    num_results = int(request.args.get('limit', 5))
    
    search_engine = init_search_engine()
    results = search_engine.get_similar_pages(title, page, num_results=num_results)
    
    return jsonify({
        'manuscript': title,
        'page': page,
        'num_similar': len(results),
        'similar_pages': results
    })

@app.route('/manuscripts/updates', methods=['GET'])
def manuscript_updates():
    def event_stream():
        while True:
            # Check for manuscript updates
            for title, status in manuscript_server.transcription_info.items():
                if status['status'] == 'in_progress':
                    yield f"data: {json.dumps({'title': title, 'status': status})}\n\n"
            time.sleep(1)
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/search/status', methods=['GET'])
def search_status():
    """Get current status of the search engine."""
    search_engine = init_search_engine()
    return jsonify(search_engine.get_status())

@app.route('/manuscripts/<path:title>/pages/<int:page>/status')
def page_status_stream(title, page):
    def event_stream():
        while True:
            page_data, _ = manuscript_server.get_page_data(title, page)
            if page_data:
                yield f"data: {json.dumps(page_data)}\n\n"
            time.sleep(1)
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/system/threads', methods=['GET'])
def get_thread_info():
    """Get information about all threads in the application."""
    current_process = psutil.Process()
    
    # Get system-wide thread limits
    system_info = {
        'cpu_count': psutil.cpu_count(logical=True),
        'max_threads_soft_limit': None,
        'max_threads_hard_limit': None
    }
    
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        system_info['max_threads_soft_limit'] = soft
        system_info['max_threads_hard_limit'] = hard
    except (ImportError, AttributeError):
        # Windows or system doesn't support this
        pass

    # Get all threads for this process
    threads_info = []
    process_threads = current_process.threads()
    python_threads = list(enumerate_threads())
    
    for thread in process_threads:
        thread_data = {
            'thread_id': thread.id,
            'cpu_time_user': thread.user_time,
            'cpu_time_system': thread.system_time,
            'status': 'unknown'  # Default status
        }
        
        # Try to match with Python thread objects for more info
        for py_thread in python_threads:
            if py_thread.ident == thread.id:
                thread_data.update({
                    'name': py_thread.name,
                    'daemon': py_thread.daemon,
                    'alive': py_thread.is_alive(),
                    'status': 'alive' if py_thread.is_alive() else 'stopped'
                })
                
                # Add info about transcription tasks if it's a transcription thread
                if py_thread.name.startswith('Thread-') and manuscript_server.transcription_info:
                    for title, status in manuscript_server.transcription_info.items():
                        if status.get('status') == 'in_progress':
                            thread_data['task'] = f"Transcribing: {title}"
                            thread_data['progress'] = status.get('current_page', 0)
                            break
                break
        
        threads_info.append(thread_data)

    return jsonify({
        'system': system_info,
        'process': {
            'pid': current_process.pid,
            'total_threads': len(threads_info),
            'memory_usage': current_process.memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent': current_process.cpu_percent()
        },
        'threads': threads_info
    })

@app.route('/manuscripts/transcribe', methods=['POST'])
def start_transcriptions():
    """Start transcribing a specified number of untranscribed manuscripts."""
    try:
        # Get number of manuscripts to transcribe from request, default to 1
        num_to_start = request.json.get('count', 1)
        
        # Get manuscripts that need transcription
        manuscripts = manuscript_server.list_manuscripts()
        to_transcribe = []
        
        for manuscript in manuscripts:
            title = manuscript['title']
            total_pages = manuscript['total_pages']
            successful_pages = (manuscript.get('transcription_info', {}).get('successful_pages', 0) 
                              if manuscript.get('transcription_info') else 0)
            
            # Add if needs transcription and not already in progress
            if successful_pages < total_pages:
                if (title not in manuscript_server.transcription_info or 
                    manuscript_server.transcription_info[title].get('status') != 'in_progress'):
                    to_transcribe.append({
                        'title': title,
                        'remaining_pages': total_pages - successful_pages
                    })
        
        # Sort by remaining pages (descending)
        to_transcribe.sort(key=lambda x: x['remaining_pages'], reverse=True)
        
        # Start requested number of transcriptions
        started = []
        for item in to_transcribe[:num_to_start]:
            title = item['title']
            try:
                manuscript_folder = manuscript_server.manuscripts[title]['raw_folder']
                manuscript_server.start_transcription(title, manuscript_folder)
                started.append(title)
                logger.info(f"Started transcription of {title} ({item['remaining_pages']} pages remaining)")
            except Exception as e:
                logger.error(f"Failed to start transcription for {title}: {e}")
        
        return jsonify({
            'status': 'success',
            'started': started,
            'count': len(started),
            'remaining_manuscripts': len(to_transcribe) - len(started)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
    
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000)