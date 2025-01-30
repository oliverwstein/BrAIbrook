from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ManuscriptSearchEngine:
    """Search engine for manuscript collection supporting both manuscript and page-level search."""
    
    def __init__(self):
        # Initialize vectorizers for different search modes
        self.manuscript_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.9,    # Ignore terms that appear in more than 90% of docs
            min_df=1       # Ignore terms that appear in fewer than 1 docs
        )
        
        self.page_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,   # More lenient for pages
            min_df=1       # Include unique terms in pages
        )
        
        self.manuscripts: Dict[str, Dict] = {}
        self.manuscript_matrix = None
        self.page_matrix = None
        self.page_map: List[Tuple[str, int]] = []  # [(manuscript_title, page_number), ...]
        self.last_indexed = None
    
    def _preprocess_text(self, text: Union[str, List[str]]) -> str:
        """Preprocess text for indexing."""
        if isinstance(text, list):
            text = ' '.join(text)
        if not isinstance(text, str):
            return ''
            
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _prepare_manuscript_doc(self, metadata: Dict) -> str:
        """Prepare a manuscript-level document for indexing."""
        doc_parts = []
        
        # Add title with extra weight
        if 'Title' in metadata:
            doc_parts.extend([self._preprocess_text(metadata['Title'])] * 3)
        
        # Add all other metadata fields
        for field, value in metadata.items():
            if field != 'Title':  # Skip title as it's already added
                doc_parts.append(self._preprocess_text(value))
        
        return ' '.join(doc_parts)
    
    def _prepare_page_doc(self, page: Dict) -> str:
        """Prepare a page-level document for indexing."""
        content_parts = []
        
        # Add transcriptions with high weight
        if page.get('transcription'):
            content_parts.extend([self._preprocess_text(page['transcription'])] * 2)
        if page.get('revised_transcription'):
            content_parts.append(self._preprocess_text(page['revised_transcription']))
            
        # Add summary with medium weight
        if page.get('summary'):
            content_parts.extend([self._preprocess_text(page['summary'])] * 2)
            
        # Add keywords with high weight
        if page.get('keywords'):
            content_parts.extend([self._preprocess_text(page['keywords'])] * 3)
            
        # Add marginalia
        if page.get('marginalia'):
            content_parts.append(self._preprocess_text(page['marginalia']))
            
        # Add notes
        if page.get('content_notes'):
            content_parts.append(self._preprocess_text(page['content_notes']))
        if page.get('transcription_notes'):
            content_parts.append(self._preprocess_text(page['transcription_notes']))
            
        return ' '.join(content_parts)
    
    def index_manuscripts(self, manuscripts: Dict[str, Dict]) -> None:
        """Index manuscripts and their pages for searching."""
        logger.info(f"Starting indexing of {len(manuscripts)} manuscripts")
        self.manuscripts = manuscripts
        
        # Prepare manuscript-level documents
        manuscript_docs = []
        for title, info in manuscripts.items():
            if 'metadata' not in info:
                logger.warning(f"No metadata found for manuscript: {title}")
                continue
            doc = self._prepare_manuscript_doc(info['metadata'])
            manuscript_docs.append(doc)
        
        # Create manuscript TF-IDF matrix
        self.manuscript_matrix = self.manuscript_vectorizer.fit_transform(manuscript_docs)
        
        # Prepare page-level documents
        page_docs = []
        self.page_map = []
        
        for title, info in manuscripts.items():
            if not info.get('transcribed') or 'pages' not in info:
                continue
                
            for page in info['pages']:
                if not page.get('error'):  # Skip failed pages
                    doc = self._prepare_page_doc(page)
                    page_docs.append(doc)
                    self.page_map.append((title, page.get('page_number', 0)))
        
        # Create page TF-IDF matrix if we have any pages
        if page_docs:
            self.page_matrix = self.page_vectorizer.fit_transform(page_docs)
        else:
            logger.warning("No transcribed pages found to index")
            self.page_matrix = None
            
        self.last_indexed = datetime.now()
        logger.info(f"Indexing complete. {len(manuscript_docs)} manuscripts and {len(page_docs)} pages indexed")
    
    def _get_manuscript_info(self, idx: int) -> Dict:
        """Get manuscript info for a given index."""
        title = list(self.manuscripts.keys())[idx]
        info = self.manuscripts[title]
        return {
            'title': title,
            'score': None,  # Will be filled in by search
            'metadata': info.get('metadata', {}),
            'transcribed': info.get('transcribed', False),
            'total_pages': info.get('total_pages', 0)
        }
    
    def _get_page_info(self, idx: int, score: float) -> Dict:
        """Get page info for search results."""
        title, page_number = self.page_map[idx]
        manuscript = self.manuscripts[title]
        page = next((p for p in manuscript['pages'] 
                    if p.get('page_number') == page_number), {})
        
        return {
            'manuscript_title': title,
            'page_number': page_number,
            'score': float(score),
            'summary': page.get('summary', ''),
            'keywords': page.get('keywords', []),
            'marginalia': page.get('marginalia', []),
            'preview': page.get('revised_transcription', 
                              page.get('transcription', ''))[:200] + '...',
            'confidence': page.get('confidence', 0)
        }
    
    def search_manuscripts(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search manuscripts using metadata."""
        if self.manuscript_matrix is None:
            logger.error("Search engine not initialized. Call index_manuscripts first.")
            return []
        
        # Transform query
        query_vector = self.manuscript_vectorizer.transform([self._preprocess_text(query)])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.manuscript_matrix).flatten()
        
        # Get top results
        results = []
        top_indices = np.argsort(similarities)[::-1][:num_results]
        
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                result = self._get_manuscript_info(idx)
                result['score'] = float(similarities[idx])
                results.append(result)
        
        return results
    
    def search_pages(self, query: str, num_results: int = 10, 
                    manuscript_title: Optional[str] = None) -> List[Dict]:
        """
        Search pages across all manuscripts or within a specific manuscript.
        
        Args:
            query: Search query
            num_results: Maximum number of results to return
            manuscript_title: Optional manuscript to restrict search to
        """
        if self.page_matrix is None:
            logger.error("No pages indexed. Call index_manuscripts first.")
            return []
            
        # Transform query
        query_vector = self.page_vectorizer.transform([self._preprocess_text(query)])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.page_matrix).flatten()
        
        # Filter for specific manuscript if requested
        if manuscript_title:
            mask = np.array([title == manuscript_title for title, _ in self.page_map])
            similarities = similarities * mask
        
        # Get top results
        results = []
        top_indices = np.argsort(similarities)[::-1]
        
        for idx in top_indices:
            if similarities[idx] > 0 and len(results) < num_results:
                result = self._get_page_info(idx, similarities[idx])
                results.append(result)
        
        return results
    
    def get_similar_manuscripts(self, title: str, num_results: int = 5) -> List[Dict]:
        """Find manuscripts similar to a given one."""
        if title not in self.manuscripts:
            logger.error(f"Manuscript not found: {title}")
            return []
        
        # Get the document index
        doc_idx = list(self.manuscripts.keys()).index(title)
        
        # Calculate similarities
        similarities = cosine_similarity(
            self.manuscript_matrix[doc_idx:doc_idx+1], 
            self.manuscript_matrix
        ).flatten()
        
        # Get top results (excluding the query document)
        results = []
        top_indices = np.argsort(similarities)[::-1]
        
        for idx in top_indices:
            if len(results) >= num_results:
                break
            if idx != doc_idx and similarities[idx] > 0:
                result = self._get_manuscript_info(idx)
                result['score'] = float(similarities[idx])
                results.append(result)
        
        return results
    
    def get_similar_pages(self, manuscript_title: str, page_number: int, 
                         num_results: int = 5) -> List[Dict]:
        """Find pages similar to a given one."""
        if not self.page_matrix:
            logger.error("No pages indexed")
            return []
            
        # Find the page index
        try:
            page_idx = self.page_map.index((manuscript_title, page_number))
        except ValueError:
            logger.error(f"Page not found: {manuscript_title} page {page_number}")
            return []
        
        # Calculate similarities
        similarities = cosine_similarity(
            self.page_matrix[page_idx:page_idx+1],
            self.page_matrix
        ).flatten()
        
        # Get top results (excluding the query page)
        results = []
        top_indices = np.argsort(similarities)[::-1]
        
        for idx in top_indices:
            if len(results) >= num_results:
                break
            if idx != page_idx and similarities[idx] > 0:
                result = self._get_page_info(idx, similarities[idx])
                results.append(result)
        
        return results
    
    def get_status(self) -> Dict:
        """Get current status of the search engine."""
        return {
            'num_manuscripts': len(self.manuscripts),
            'num_pages_indexed': len(self.page_map) if self.page_map else 0,
            'manuscript_vocabulary_size': len(self.manuscript_vectorizer.vocabulary_) 
                if self.manuscript_matrix is not None else 0,
            'page_vocabulary_size': len(self.page_vectorizer.vocabulary_)
                if self.page_matrix is not None else 0,
            'last_indexed': self.last_indexed.isoformat() if self.last_indexed else None
        }

# Global instance for server.py to use
_search_engine = None

def init_search_engine() -> ManuscriptSearchEngine:
    """Initialize or return existing search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = ManuscriptSearchEngine()
    return _search_engine

def get_search_engine() -> Optional[ManuscriptSearchEngine]:
    """Get the current search engine instance."""
    return _search_engine