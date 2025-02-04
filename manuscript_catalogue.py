from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ListingInfo:
    """Basic manuscript information for the manuscript list view."""
    id: str
    title: str
    total_pages: int
    transcribed_pages: int
    last_updated: Optional[str]

@dataclass
class TranscriptionStatus:
    """Status information for transcription progress."""
    successful_pages: int
    failed_pages: List[int]
    last_updated: str

@dataclass
class PageTranscription:
    """Transcription data for a single page."""
    transcription: str
    revised_transcription: str
    summary: str
    keywords: List[str]
    marginalia: List[str]
    confidence: float
    content_notes: str

@dataclass
class ManuscriptView:
    """Complete manuscript information for the viewer."""
    id: str
    title: str
    metadata: Dict
    total_pages: int
    transcription_status: Optional[TranscriptionStatus]
    summary: Optional[Dict]
    table_of_contents: Optional[List]

class ManuscriptCatalogue:
    def __init__(self, catalogue_dir: str = "data/catalogue"):
        self.catalogue_dir = Path(catalogue_dir)
        if not self.catalogue_dir.exists():
            raise FileNotFoundError(f"Catalogue directory not found: {self.catalogue_dir}")
        
        # Cache of basic manuscript information
        self.manuscript_listings: Dict[str, ListingInfo] = {}
        self._load_listings()
        
        # Cache for manuscript viewer data
        self._viewer_cache: Dict[str, ManuscriptView] = {}
        # Cache for page transcriptions
        self._page_cache: Dict[str, Dict[int, PageTranscription]] = {}
        
    def _load_listings(self) -> None:
        """Load basic information for manuscript listing."""
        for manuscript_dir in self.catalogue_dir.iterdir():
            if not manuscript_dir.is_dir():
                continue
                
            try:
                metadata_path = manuscript_dir / 'metadata.json'
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                manuscript_id = manuscript_dir.name
                image_dir = manuscript_dir / 'images'
                total_pages = len(list(image_dir.glob('*.[jJpP][nNiI][gGfF]')))
                
                transcription_path = manuscript_dir / 'transcription.json'
                transcribed_pages = 0
                last_updated = None
                
                if transcription_path.exists():
                    with open(transcription_path, 'r', encoding='utf-8') as f:
                        trans_data = json.load(f)
                        transcribed_pages = trans_data.get('successful_pages', 0)
                        total_pages = trans_data.get('successful_pages', 0)
                        last_updated = trans_data.get('last_updated')
                
                self.manuscript_listings[manuscript_id] = ListingInfo(
                    id=manuscript_id,
                    title=metadata.get('Title', ''),
                    total_pages=total_pages,
                    transcribed_pages=transcribed_pages,
                    last_updated=last_updated
                )
                
            except Exception as e:
                logger.error(f"Error loading listing for {manuscript_dir.name}: {e}")
    
    def get_manuscript_listings(self) -> List[ListingInfo]:
        """Get manuscript information for the list view."""
        return list(self.manuscript_listings.values())
    
    def get_manuscript_info(self, manuscript_id: str) -> Optional[ManuscriptView]:
        """Get manuscript information (less the transcript) for the viewer."""
        if manuscript_id not in self.manuscript_listings:
            return None
            
        if manuscript_id not in self._viewer_cache:
            try:
                # Load metadata
                metadata_path = self.catalogue_dir / manuscript_id / 'metadata.json'
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Load transcription data if available
                transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
                transcription_status = None
                summary = None
                table_of_contents = None
                
                if transcription_path.exists():
                    with open(transcription_path, 'r', encoding='utf-8') as f:
                        trans_data = json.load(f)
                        transcription_status = TranscriptionStatus(
                            successful_pages=trans_data.get('successful_pages', 0),
                            failed_pages=trans_data.get('failed_pages', []),
                            last_updated=trans_data.get('last_updated', '')
                        )
                        summary = trans_data.get('summary')
                        table_of_contents = trans_data.get('table_of_contents')
                
                self._viewer_cache[manuscript_id] = ManuscriptView(
                    id=manuscript_id,
                    title=metadata.get('Title', ''),
                    metadata=metadata,
                    total_pages=self.manuscript_listings[manuscript_id].total_pages,
                    transcription_status=transcription_status,
                    summary=summary,
                    table_of_contents=table_of_contents
                )
                
            except Exception as e:
                logger.error(f"Error loading viewer data for {manuscript_id}: {e}")
                return None
                
        return self._viewer_cache[manuscript_id]
    
    def get_manuscript_pages(self, manuscript_id: str) -> Optional[Dict[int, PageTranscription]]:
        """Get all page transcriptions for a manuscript.
        
        Args:
            manuscript_id: The manuscript identifier
            
        Returns:
            Dictionary mapping page numbers to PageTranscription objects,
            or None if manuscript not found
        """
        if manuscript_id not in self.manuscript_listings:
            return None
            
        try:
            # Check if pages are already cached
            if manuscript_id in self._page_cache:
                return self._page_cache[manuscript_id]
                
            # Load transcription file
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            if not transcription_path.exists():
                return {}
                
            with open(transcription_path, 'r', encoding='utf-8') as f:
                trans_data = json.load(f)
                
            pages = {}
            for page_num_str, page_data in trans_data.get('pages', {}).items():
                page_num = int(page_num_str)
                pages[page_num] = PageTranscription(
                    transcription=page_data.get('transcription', ''),
                    revised_transcription=page_data.get('revised_transcription', ''),
                    summary=page_data.get('summary', ''),
                    keywords=page_data.get('keywords', []),
                    marginalia=page_data.get('marginalia', []),
                    confidence=page_data.get('confidence', 0.0),
                    content_notes=page_data.get('content_notes', '')
                )
                
            # Cache the results
            self._page_cache[manuscript_id] = pages
            return pages
            
        except Exception as e:
            logger.error(f"Error loading pages for {manuscript_id}: {e}")
            return None
        
    def get_page_data(self, manuscript_id: str, page_number: int) -> Tuple[Optional[PageTranscription], Optional[Path]]:
        """Get page transcription and image path.
        
        Args:
            manuscript_id: The manuscript identifier
            page_number: The page number to request (1-based indexing)
            
        Returns:
            Tuple of (PageTranscription or None, Path to image or None)
        """
        try:
            if manuscript_id not in self.manuscript_listings:
                logger.error(f"Manuscript {manuscript_id} not found in listings")
                return None, None
                
            # Get image path
            image_dir = self.catalogue_dir / manuscript_id / 'images'
            if not image_dir.exists():
                logger.error(f"Image directory not found: {image_dir}")
                return None, None
                
            # Get sorted list of image files with case-insensitive extension matching
            image_files = sorted([
                f for f in image_dir.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
            ])
            
            if not image_files:
                logger.error(f"No image files found in {image_dir}")
                return None, None
                
            # Validate page number (1-based indexing)
            if page_number < 1 or page_number > len(image_files):
                logger.error(f"Invalid page number {page_number}. Valid range: 1-{len(image_files)}")
                return None, None
                
            image_path = image_files[page_number - 1]
            
            # Check page cache first
            if manuscript_id in self._page_cache:
                if page_number in self._page_cache[manuscript_id]:
                    return self._page_cache[manuscript_id][page_number], image_path
            
            # Load transcription if available
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            if transcription_path.exists():
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                    
                    # Look for page data using string key
                    page_data = trans_data.get('pages', {}).get(str(page_number))
                    
                    if page_data:
                        transcription = PageTranscription(
                            transcription=page_data.get('transcription', ''),
                            revised_transcription=page_data.get('revised_transcription', ''),
                            summary=page_data.get('summary', ''),
                            keywords=page_data.get('keywords', []),
                            marginalia=page_data.get('marginalia', []),
                            confidence=float(page_data.get('confidence', 0.0)),
                            content_notes=page_data.get('content_notes', '')
                        )
                        
                        # Update cache
                        if manuscript_id not in self._page_cache:
                            self._page_cache[manuscript_id] = {}
                        self._page_cache[manuscript_id][page_number] = transcription
                        
                        return transcription, image_path
            
            # If we get here, we have an image but no transcription
            logger.info(f"No transcription found for page {page_number} of {manuscript_id}")
            return None, image_path
            
        except Exception as e:
            logger.error(f"Error in get_page_data for {manuscript_id} page {page_number}: {e}")
            return None, None