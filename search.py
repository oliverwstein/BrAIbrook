from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ManuscriptSearchEngine:
    """Search engine for manuscript collection using TF-IDF."""
    
    def __init__(self):
        self.manuscripts: Dict[str, Dict] = {}
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            max_df=0.9,  # Ignore terms that appear in more than 90% of docs
            min_df=2,    # Ignore terms that appear in fewer than 2 docs
        )
        self.tfidf_matrix = None
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
    
    def _prepare_document(self, metadata: Dict) -> str:
        """Prepare a document for indexing by combining all metadata fields."""
        doc_parts = []
        
        # Add title with extra weight (repeated three times for higher importance)
        if 'Title' in metadata:
            doc_parts.extend([self._preprocess_text(metadata['Title'])] * 3)
        
        # Add all other metadata fields
        for field, value in metadata.items():
            if field != 'Title':  # Skip title as it's already added
                doc_parts.append(self._preprocess_text(value))
        
        return ' '.join(doc_parts)
    
    def index_manuscripts(self, manuscripts: Dict[str, Dict]) -> None:
        """Index manuscripts for searching."""
        logger.info(f"Starting indexing of {len(manuscripts)} manuscripts")
        self.manuscripts = manuscripts
        
        # Prepare documents for vectorization
        documents = []
        for title, info in manuscripts.items():
            if 'metadata' not in info:
                logger.warning(f"No metadata found for manuscript: {title}")
                continue
            doc = self._prepare_document(info['metadata'])
            documents.append(doc)
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.last_indexed = datetime.now()
        
        logger.info(f"Indexing complete. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
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
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search manuscripts using TF-IDF similarity."""
        if self.tfidf_matrix is None or not self.manuscripts:
            logger.error("Search engine not initialized. Call index_manuscripts first.")
            return []
        
        # Preprocess query
        query = self._preprocess_text(query)
        
        # Transform query to TF-IDF space
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:num_results]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                result = self._get_manuscript_info(idx)
                result['score'] = float(similarities[idx])
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
            self.tfidf_matrix[doc_idx:doc_idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top results (excluding the query document)
        top_indices = np.argsort(similarities)[::-1]
        results = []
        
        for idx in top_indices:
            if len(results) >= num_results:
                break
            if idx != doc_idx and similarities[idx] > 0:
                result = self._get_manuscript_info(idx)
                result['score'] = float(similarities[idx])
                results.append(result)
        
        return results
    
    def get_status(self) -> Dict:
        """Get current status of the search engine."""
        return {
            'num_manuscripts': len(self.manuscripts),
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.tfidf_matrix is not None else 0,
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