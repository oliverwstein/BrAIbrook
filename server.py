from datetime import datetime
from flask import Flask, jsonify, send_file, request
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from threading import Thread
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
        self.transcription_status = {}  # Track ongoing transcriptions

    
    def start_transcription(self, title: str, manuscript_folder: Path) -> None:
        """Run transcription in a background thread."""
        def transcribe():
            try:
                from gemini_transcribe import ManuscriptProcessor, process_manuscript
                processor = ManuscriptProcessor()
                
                # Update status to in_progress
                self.transcription_status[title] = {'status': 'in_progress', 'started_at': datetime.now().isoformat()}
                
                # Process manuscript
                result = process_manuscript(
                    str(manuscript_folder),
                    str(self.transcripts_dir),
                    processor
                )
                
                if result:
                    # Create transcript file path
                    transcript_file = self.transcripts_dir / result['manuscript_title'] / 'transcription.json'
                    
                    # Update manuscript tracking
                    self.manuscripts[title].update({
                        'transcribed': True,
                        'transcript_file': transcript_file,
                        'transcription_info': {
                            'successful_pages': result['successful_pages'],
                            'failed_pages': result.get('failed_pages', []),
                            'last_updated': datetime.now().isoformat()
                        }
                    })
                    
                    # Load transcription data
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcription_data = json.load(f)
                        self.manuscripts[title]['transcription_data'] = transcription_data
                    
                    # Update status to completed
                    self.transcription_status[title] = {
                        'status': 'completed',
                        'completed_at': datetime.now().isoformat(),
                        'successful_pages': result['successful_pages'],
                        'failed_pages': result.get('failed_pages', [])
                    }
                else:
                    self.transcription_status[title] = {
                        'status': 'failed',
                        'error': 'Transcription returned no results',
                        'failed_at': datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                self.transcription_status[title] = {
                    'status': 'failed',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }
        
        # Start transcription in background thread
        Thread(target=transcribe, daemon=True).start()

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
                    'transcription_info': None
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
                        'successful_pages': transcription_data.get('successful_pages', 0),
                        'failed_pages': transcription_data.get('failed_pages', []),
                        'last_updated': transcription_data.get('last_updated', None)
                    }
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
            'transcription_status': {
                'successful_pages': info['transcription_info']['successful_pages'],
                'failed_pages': info['transcription_info']['failed_pages'],
                'last_updated': info['transcription_info']['last_updated']
            } if info['transcribed'] else None
        } for title, info in self.manuscripts.items()]

    def get_manuscript_info(self, title: str) -> Optional[Dict]:
        """Get detailed information about a manuscript."""
        if title not in self.manuscripts:
            return None
        
        info = self.manuscripts[title]
        return {
            'title': title,
            'record_id': info['record_id'],
            'metadata': info['metadata'],
            'total_pages': info['total_pages'],
            'transcribed': info['transcribed'],
            'transcription_info': info['transcription_info']
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
                if page_idx < len(transcription['pages']):
                    page_data = transcription['pages'][page_idx]
            except Exception as e:
                logger.error(f"Error loading transcription data: {e}")

        return page_data, image_path

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
    if title in manuscript_server.transcription_status:
        status = manuscript_server.transcription_status[title]['status']
        if status == 'in_progress':
            current_progress = manuscript_server.transcription_status[title].get('current_page', 0)
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

# Add status check endpoint
@app.route('/manuscripts/<path:title>/transcribe/status', methods=['GET'])
def transcription_status(title):
    """Get the status of an ongoing or completed transcription."""
    if title not in manuscript_server.transcription_status:
        return jsonify({'status': 'not_started'})
    
    return jsonify(manuscript_server.transcription_status[title])

@app.route('/search', methods=['GET'])
def search_manuscripts():
    """Search manuscripts using keyword query."""
    query = request.args.get('q', '')
    num_results = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
        
    search_engine = init_search_engine()
    results = search_engine.search(query, num_results=num_results)
    
    return jsonify({
        'query': query,
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

@app.route('/search/status', methods=['GET'])
def search_status():
    """Get current status of the search engine."""
    search_engine = init_search_engine()
    return jsonify(search_engine.get_status())

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
                if py_thread.name.startswith('Thread-') and manuscript_server.transcription_status:
                    for title, status in manuscript_server.transcription_status.items():
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
            successful_pages = (manuscript.get('transcription_status', {}).get('successful_pages', 0) 
                              if manuscript.get('transcription_status') else 0)
            
            # Add if needs transcription and not already in progress
            if successful_pages < total_pages:
                if (title not in manuscript_server.transcription_status or 
                    manuscript_server.transcription_status[title].get('status') != 'in_progress'):
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