from flask import Flask, jsonify, send_file, redirect, url_for
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

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
        
        # Convert to absolute paths for clarity in logging
        self.raw_dir = Path(raw_dir).absolute()
        self.transcripts_dir = Path(transcripts_dir).absolute()
        self.manuscript_map = {}
        self.transcriptions = {}
        
        logger.info(f"Raw directory path: {self.raw_dir}")
        logger.info(f"Transcripts directory path: {self.transcripts_dir}")
        
        # Check if directories exist
        if not self.raw_dir.exists():
            logger.error(f"Raw directory does not exist: {self.raw_dir}")
            raise FileNotFoundError(f"Raw directory not found: {self.raw_dir}")
        if not self.transcripts_dir.exists():
            logger.error(f"Transcripts directory does not exist: {self.transcripts_dir}")
            raise FileNotFoundError(f"Transcripts directory not found: {self.transcripts_dir}")
            
        self._initialize_manuscript_map()

    def _initialize_manuscript_map(self) -> None:
        """Create mapping between manuscript titles and their data locations."""
        logger.info("Starting manuscript map initialization...")
        
        # List all transcription.json files
        transcript_files = list(self.transcripts_dir.glob("*/transcription.json"))
        logger.info(f"Found {len(transcript_files)} potential transcript files")
        
        for transcript_file in transcript_files:
            try:
                logger.info(f"Processing transcript file: {transcript_file}")
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    manuscript_title = data.get('manuscript_title')
                    record_id = data.get('metadata', {}).get('Record ID')
                    
                    logger.info(f"Found manuscript title: {manuscript_title}, record ID: {record_id}")
                    
                    if manuscript_title and record_id:
                        raw_folder = self.raw_dir / f"34-{record_id}"
                        logger.info(f"Looking for raw folder: {raw_folder}")
                        
                        if raw_folder.exists():
                            self.manuscript_map[manuscript_title] = {
                                'record_id': record_id,
                                'transcript_file': transcript_file,
                                'raw_folder': raw_folder
                            }
                            logger.info(f"Successfully mapped manuscript: {manuscript_title}")
                        else:
                            logger.warning(f"Raw folder not found: {raw_folder}")
                    else:
                        logger.warning(f"Missing title or record ID in {transcript_file}")
            except Exception as e:
                logger.error(f"Error processing {transcript_file}: {e}")

        logger.info(f"Manuscript map initialization complete. Found {len(self.manuscript_map)} manuscripts:")
        for title, info in self.manuscript_map.items():
            logger.info(f"- {title} (Record ID: {info['record_id']})")

    def _load_transcription(self, manuscript_title: str) -> Optional[Dict]:
        """Load transcription data for a manuscript if not already cached."""
        if manuscript_title not in self.manuscript_map:
            logger.warning(f"Manuscript not found: {manuscript_title}")
            return None
            
        if manuscript_title not in self.transcriptions:
            try:
                transcript_file = self.manuscript_map[manuscript_title]['transcript_file']
                logger.info(f"Loading transcription from {transcript_file}")
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    self.transcriptions[manuscript_title] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading transcription for {manuscript_title}: {e}")
                return None
                
        return self.transcriptions[manuscript_title]

    def get_page_data(self, manuscript_title: str, page_number: int) -> Tuple[Optional[Dict], Optional[str]]:
        """Get transcription data and image path for a specific page."""
        logger.info(f"Retrieving page {page_number} for manuscript: {manuscript_title}")
        
        transcription = self._load_transcription(manuscript_title)
        if not transcription:
            return None, None

        try:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(transcription['pages']):
                logger.warning(f"Page {page_number} out of range for {manuscript_title}")
                return None, None
                
            page_data = transcription['pages'][page_idx]
            
            raw_folder = self.manuscript_map[manuscript_title]['raw_folder']
            image_files = sorted([f for f in raw_folder.iterdir() 
                                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
            
            if page_idx < len(image_files):
                logger.info(f"Found image for page {page_number}: {image_files[page_idx]}")
                return page_data, str(image_files[page_idx])
            
            logger.warning(f"No image found for page {page_number}")
            return page_data, None
            
        except Exception as e:
            logger.error(f"Error getting page data for {manuscript_title}, page {page_number}: {e}")
            return None, None

    def get_manuscript_info(self, manuscript_title: str) -> Optional[Dict]:
        """Get basic information about a manuscript."""
        transcription = self._load_transcription(manuscript_title)
        if not transcription:
            return None
            
        return {
            'title': transcription['manuscript_title'],
            'metadata': transcription['metadata'],
            'total_pages': transcription['total_pages'],
            'successful_pages': transcription['successful_pages'],
            'failed_pages': transcription.get('failed_pages', [])
        }

    def list_manuscripts(self) -> list:
        """Get a list of all available manuscripts."""
        manuscripts = []
        for title in self.manuscript_map:
            info = self.get_manuscript_info(title)
            if info:
                manuscripts.append({
                    'title': title,
                    'record_id': self.manuscript_map[title]['record_id'],
                    'total_pages': info['total_pages'],
                    'successful_pages': info['successful_pages'],
                    'failed_pages': info['failed_pages']
                })
        return manuscripts

# Initialize server
try:
    logger.info("Initializing ManuscriptServer...")
    manuscript_server = ManuscriptServer()
    logger.info("ManuscriptServer initialization complete")
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
    """List all available manuscripts with basic information."""
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

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000)