"""
ManuscriptCatalogue Module

This module provides functionality for managing and accessing manuscript data, including
listings, metadata, transcriptions, and associated images. It implements caching mechanisms
for performance optimization while ensuring data consistency.

The module uses a directory-based storage structure where each manuscript has its own
directory containing metadata, transcription data, and image files.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ListingInfo:
    """
    Basic manuscript information for list view display.
    
    Attributes:
        id: Unique identifier for the manuscript
        title: Display title of the manuscript
        total_pages: Total number of pages in the manuscript
        transcribed_pages: Number of successfully transcribed pages
        last_updated: Timestamp of the most recent update (ISO format)
    """
    id: str
    title: str
    total_pages: int
    transcribed_pages: int
    last_updated: Optional[str]

@dataclass
class TranscriptionInfo:
    """
    Information about the transcription progress and status.
    
    Attributes:
        successful_pages: Count of successfully transcribed pages
        failed_pages: List of page numbers that failed transcription
        last_updated: Timestamp of the most recent transcription (ISO format)
    """
    successful_pages: int
    failed_pages: List[int]
    last_updated: str

@dataclass
class PageTranscription:
    """
    Complete transcription data for a single manuscript page.
    
    Attributes:
        transcription: Raw transcription with original markup
        revised_transcription: Cleaned and standardized transcription
        summary: Brief description of page contents
        keywords: List of relevant search terms and topics
        marginalia: List of marginal notes and annotations
        confidence: Confidence score of the transcription (0-100)
        content_notes: Additional notes about the page content
        transcription_notes: Additional notes about the page transcription
    """
    transcription: str
    revised_transcription: str
    summary: str
    keywords: List[str]
    marginalia: List[str]
    confidence: float
    content_notes: str
    transcription_notes: str

@dataclass
class ManuscriptView:
    """
    Complete manuscript information for detailed viewing.
    
    Attributes:
        id: Unique identifier for the manuscript
        title: Display title of the manuscript
        metadata: Dictionary of all manuscript metadata
        total_pages: Total number of pages
        transcription_info: Information about transcription status
        summary: Overall manuscript summary if available
        table_of_contents: Structured table of contents if available
    """
    id: str
    title: str
    metadata: Dict
    total_pages: int
    transcription_info: Optional[TranscriptionInfo]
    summary: Optional[Dict]
    table_of_contents: Optional[List]

class ManuscriptCatalogue:
    """
    Manages access to manuscript data with caching capabilities.
    
    This class provides methods to access manuscript listings, detailed information,
    transcriptions, and associated images. It implements caching mechanisms to
    optimize performance while ensuring data consistency.
    """

    def __init__(self, catalogue_dir: str = "data/catalogue"):
        """
        Initialize the manuscript catalogue.
        
        Args:
            catalogue_dir: Base directory containing manuscript data
                         (default: "data/catalogue")
        
        Raises:
            FileNotFoundError: If catalogue directory doesn't exist
        """
        self.catalogue_dir = Path(catalogue_dir)
        if not self.catalogue_dir.exists():
            raise FileNotFoundError(f"Catalogue directory not found: {self.catalogue_dir}")
        
        # Initialize cache structures
        self.manuscript_listings: Dict[str, ListingInfo] = {}
        self._viewer_cache: Dict[str, ManuscriptView] = {}
        self._page_cache: Dict[str, Dict[int, PageTranscription]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Load initial listings
        self._load_listings()
    
    def _load_listings(self) -> None:
        """
        Load basic information for all manuscripts in the catalogue.
        
        This method scans the catalogue directory and loads basic information
        for each manuscript, including title, page count, and transcription status.
        Errors for individual manuscripts are logged but don't stop the process.
        """
        for manuscript_dir in self.catalogue_dir.iterdir():
            if not manuscript_dir.is_dir():
                continue
                
            try:
                metadata_path = manuscript_dir / 'metadata.json'
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                manuscript_id = manuscript_dir.name
                self._load_manuscript_listing(manuscript_id, metadata)
                
            except Exception as e:
                logger.error(f"Error loading listing for {manuscript_dir.name}: {e}")

    def _load_manuscript_listing(self, manuscript_id: str, metadata: Dict) -> None:
        """
        Load or update listing information for a single manuscript.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
            metadata: Pre-loaded metadata dictionary
        """
        try:
            image_dir = self.catalogue_dir / manuscript_id / 'images'
            total_pages = len(list(image_dir.iterdir()))
            
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            transcribed_pages = 0
            last_updated = None
            
            if transcription_path.exists():
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                    transcribed_pages = trans_data.get('successful_pages', 0)
                    last_updated = trans_data.get('last_updated')
            
            self.manuscript_listings[manuscript_id] = ListingInfo(
                id=manuscript_id,
                title=metadata.get('Title', ''),
                total_pages=total_pages,
                transcribed_pages=transcribed_pages,
                last_updated=last_updated
            )
        except Exception as e:
            logger.error(f"Error in _load_manuscript_listing for {manuscript_id}: {e}")

    def refresh_manuscript(self, manuscript_id: str) -> None:
        """
        Refresh all cached data for a specific manuscript.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
        """
        try:
            metadata_path = self.catalogue_dir / manuscript_id / 'metadata.json'
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Clear caches
            self._viewer_cache.pop(manuscript_id, None)
            self._page_cache.pop(manuscript_id, None)
            self._cache_timestamps.pop(manuscript_id, None)
            
            # Reload listing
            self._load_manuscript_listing(manuscript_id, metadata)
            
        except Exception as e:
            logger.error(f"Error refreshing manuscript {manuscript_id}: {e}")

    def get_manuscript_listings(self) -> List[ListingInfo]:
        """
        Get basic information for all manuscripts.
        
        Returns:
            List of ListingInfo objects for all manuscripts
        """
        return list(self.manuscript_listings.values())
    
    def get_manuscript_info(self, manuscript_id: str) -> Optional[ManuscriptView]:
        """
        Get detailed information for a specific manuscript.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
        
        Returns:
            ManuscriptView object if found, None otherwise
        """
        if manuscript_id not in self.manuscript_listings:
            return None
        
        try:
            # Initialize cache-related variables
            current_mtime = 0
            needs_refresh = False
            
            # Check transcription file status within a protected block
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            try:
                if transcription_path.exists():
                    current_mtime = transcription_path.stat().st_mtime
                    cached_mtime = self._cache_timestamps.get(manuscript_id, 0)
                    needs_refresh = current_mtime > cached_mtime
            except (FileNotFoundError, OSError) as e:
                logger.debug(f"File access error for {manuscript_id}: {e}")
                needs_refresh = True

            # Refresh cache if needed
            if needs_refresh:
                self.refresh_manuscript(manuscript_id)
                
            # Load fresh data if not cached
            if manuscript_id not in self._viewer_cache:
                metadata_path = self.catalogue_dir / manuscript_id / 'metadata.json'
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                transcription_info = None
                summary = None
                table_of_contents = None
                
                if transcription_path.exists():
                    with open(transcription_path, 'r', encoding='utf-8') as f:
                        trans_data = json.load(f)
                        transcription_info = TranscriptionInfo(
                            successful_pages=len(trans_data.get('pages', {})),
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
                    transcription_info=transcription_info,
                    summary=summary,
                    table_of_contents=table_of_contents
                )
                
                if current_mtime > 0:
                    self._cache_timestamps[manuscript_id] = current_mtime
            
            return self._viewer_cache[manuscript_id]
                
        except Exception as e:
            logger.error(f"Error in get_manuscript_info for {manuscript_id}: {e}")
            return None

    def get_manuscript_pages(self, manuscript_id: str) -> Optional[Dict[int, PageTranscription]]:
        """
        Get transcription data for all pages of a manuscript.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
        
        Returns:
            Dictionary mapping page numbers to PageTranscription objects,
            or None if manuscript not found
        """
        if manuscript_id not in self.manuscript_listings:
            return None
        
        try:
            # Check if we need to refresh cache
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            if transcription_path.exists():
                current_mtime = transcription_path.stat().st_mtime
                cached_mtime = self._cache_timestamps.get(manuscript_id, 0)
                
                if current_mtime > cached_mtime:
                    self._page_cache.pop(manuscript_id, None)
                    self._cache_timestamps[manuscript_id] = current_mtime
            
            # Return cached data if available
            if manuscript_id in self._page_cache:
                return self._page_cache[manuscript_id]
            
            # Load fresh data
            if transcription_path.exists():
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
                        content_notes=page_data.get('content_notes', ''),
                        transcription_notes=page_data.get('transcription_notes', '')
                    )
                
                self._page_cache[manuscript_id] = pages
                return pages
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in get_manuscript_pages for {manuscript_id}: {e}")
            return None

    def get_page_data(self, manuscript_id: str, page_number: int) -> Tuple[Optional[PageTranscription], Optional[Path]]:
        """
        Get transcription and image path for a specific page.
        
        Args:
            manuscript_id: Unique identifier for the manuscript
            page_number: Page number (1-based indexing)
        
        Returns:
            Tuple of (PageTranscription object or None, Path to image or None)
        """
        try:
            if manuscript_id not in self.manuscript_listings:
                return None, None
            
            # Get image path
            image_dir = self.catalogue_dir / manuscript_id / 'images'
            image_files = sorted([
                f for f in image_dir.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
            ])
            
            if not image_files:
                logger.error(f"No image files found in {image_dir}")
                return None, None
            
            if page_number < 1 or page_number > len(image_files):
                logger.error(f"Invalid page number {page_number}. Valid range: 1-{len(image_files)}")
                return None, None
            
            image_path = image_files[page_number - 1]
            
            # Check if we need to refresh cache
            transcription_path = self.catalogue_dir / manuscript_id / 'transcription.json'
            if transcription_path.exists():
                current_mtime = transcription_path.stat().st_mtime
                cached_mtime = self._cache_timestamps.get(manuscript_id, 0)
                
                if current_mtime > cached_mtime:
                    self._page_cache.pop(manuscript_id, None)
                    self._cache_timestamps[manuscript_id] = current_mtime
            
            # Get page data
            pages = self.get_manuscript_pages(manuscript_id)
            if pages is not None and page_number in pages:
                return pages[page_number], image_path
            
            return None, image_path
            
        except Exception as e:
            logger.error(f"Error in get_page_data for {manuscript_id} page {page_number}: {e}")
            return None, None