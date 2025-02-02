from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenType(Enum):
    """Types of tokens in a boolean query."""
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    TERM = auto()
    PHRASE = auto()

@dataclass
class Token:
    """Represents a token in the boolean query."""
    type: TokenType
    value: str = ""

class QueryParser:
    """Parses boolean search queries into tokens."""
    
    def __init__(self):
        self.operators = {
            'AND': TokenType.AND,
            'OR': TokenType.OR,
            'NOT': TokenType.NOT,
            '&&': TokenType.AND,
            '||': TokenType.OR,
            '!': TokenType.NOT,
            '-': TokenType.NOT
        }
    
    def tokenize(self, query: str) -> List[Token]:
        """Convert a query string into a list of tokens."""
        # If query doesn't contain any operators, return it as a simple term
        if not any(op in query.upper() for op in self.operators):
            return [Token(TokenType.TERM, query.strip())]
            
        tokens = []
        i = 0
        query = query.strip()
        
        while i < len(query):
            char = query[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
                
            # Handle parentheses
            if char == '(':
                tokens.append(Token(TokenType.LPAREN))
                i += 1
                continue
            if char == ')':
                tokens.append(Token(TokenType.RPAREN))
                i += 1
                continue
                
            # Handle quoted phrases
            if char == '"':
                phrase = []
                i += 1  # Skip opening quote
                while i < len(query) and query[i] != '"':
                    phrase.append(query[i])
                    i += 1
                if i < len(query):  # Skip closing quote
                    i += 1
                tokens.append(Token(TokenType.PHRASE, ''.join(phrase)))
                continue
                
            # Handle operators
            operator = self._match_operator(query[i:])
            if operator:
                tokens.append(Token(self.operators[operator]))
                i += len(operator)
                continue
                
            # Handle terms
            term = self._match_term(query[i:])
            tokens.append(Token(TokenType.TERM, term))
            i += len(term)
            
        return self._normalize_tokens(tokens)
    
    def _match_operator(self, text: str) -> Union[str, None]:
        """Match an operator at the start of text."""
        for op in sorted(self.operators.keys(), key=len, reverse=True):
            if text.upper().startswith(op.upper()):
                return op
        return None
    
    def _match_term(self, text: str) -> str:
        """Match a term until the next operator or whitespace."""
        term = []
        for char in text:
            if char.isspace() or char in '()':
                break
            if any(text[len(term):].upper().startswith(op.upper()) 
                   for op in self.operators.keys()):
                break
            term.append(char)
        return ''.join(term)
    
    def _normalize_tokens(self, tokens: List[Token]) -> List[Token]:
        """Add implicit AND operators and handle consecutive NOT operators."""
        normalized = []
        for i, token in enumerate(tokens):
            if i > 0:
                prev = tokens[i-1]
                # Add implicit AND between terms/phrases
                if (token.type in {TokenType.TERM, TokenType.PHRASE, TokenType.NOT, TokenType.LPAREN} and
                    prev.type in {TokenType.TERM, TokenType.PHRASE, TokenType.RPAREN}):
                    normalized.append(Token(TokenType.AND))
            normalized.append(token)
        return normalized

class BooleanSearchEngine:
    """Implements boolean search operations on TF-IDF matrices."""
    
    def __init__(self, tfidf_matrix, vectorizer):
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        self.parser = QueryParser()
    
    def search(self, query: str, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute a boolean search query and return matching document indices and scores.
        
        Args:
            query: Boolean search query
            threshold: Similarity threshold for matching terms
            
        Returns:
            Tuple of (matches, scores) where matches is a boolean mask and scores
            are the similarity scores for matched documents
        """
        tokens = self.parser.tokenize(query)
        matches = self._evaluate_tokens(tokens, threshold)
        
        # Calculate similarity scores for the original query
        query_vector = self.vectorizer.transform([' '.join(
            t.value for t in tokens if t.type in {TokenType.TERM, TokenType.PHRASE}
        )])
        scores = np.asarray(query_vector.dot(self.tfidf_matrix.T).todense())[0]
        
        return matches, scores
    
    def _evaluate_tokens(self, tokens: List[Token], threshold: float) -> np.ndarray:
        """Evaluate a list of tokens using shunting yard algorithm."""
        if len(tokens) == 1 and tokens[0].type == TokenType.TERM:
            return self._get_term_matches(tokens[0].value, threshold)
            
        output = []
        operators = []
        
        precedence = {
            TokenType.NOT: 3,
            TokenType.AND: 2,
            TokenType.OR: 1
        }
        
        for token in tokens:
            if token.type in {TokenType.TERM, TokenType.PHRASE}:
                # Transform term into document matches
                matches = self._get_term_matches(token.value, threshold)
                output.append(matches)
            elif token.type == TokenType.LPAREN:
                operators.append(token)
            elif token.type == TokenType.RPAREN:
                while operators and operators[-1].type != TokenType.LPAREN:
                    output.append(self._evaluate_operator(operators.pop(), output))
                if operators:  # Remove LPAREN
                    operators.pop()
            else:  # Operator
                while (operators and
                       operators[-1].type != TokenType.LPAREN and
                       precedence.get(operators[-1].type, 0) >= precedence.get(token.type, 0)):
                    output.append(self._evaluate_operator(operators.pop(), output))
                operators.append(token)
        
        # Process remaining operators
        while operators:
            output.append(self._evaluate_operator(operators.pop(), output))
        
        return output[-1] if output else np.zeros(self.tfidf_matrix.shape[0], dtype=bool)
    
    def _get_term_matches(self, term: str, threshold: float) -> np.ndarray:
        """Get boolean mask of documents matching a term."""
        query_vector = self.vectorizer.transform([term])
        similarities = np.asarray(query_vector.dot(self.tfidf_matrix.T).todense())[0]
        return similarities >= threshold
    
    def _evaluate_operator(self, operator: Token, stack: List[np.ndarray]) -> np.ndarray:
        """Apply boolean operator to values on the stack."""
        if operator.type == TokenType.NOT:
            operand = stack.pop()
            return ~operand
        else:
            right = stack.pop()
            left = stack.pop()
            if operator.type == TokenType.AND:
                return left & right
            else:  # OR
                return left | right

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
        
        # Initialize boolean search engines
        self.manuscript_boolean_engine = None
        self.page_boolean_engine = None
        
    
    def _preprocess_text(self, text: Union[str, List[str]]) -> str:
        """Preprocess text for indexing."""
        if isinstance(text, list):
            text = ' '.join(str(text))
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
        
        # Initialize manuscript boolean search engine
        self.manuscript_boolean_engine = BooleanSearchEngine(
            self.manuscript_matrix,
            self.manuscript_vectorizer
        )
        
        # Prepare page-level documents
        page_docs = []
        self.page_map = []
        
        for title, info in manuscripts.items():
            if not info.get('transcribed') or 'pages' not in info:
                continue
            for page_number, page_data in info['pages'].items():
                if not page_data.get('error'):  # Skip failed pages
                    doc = self._prepare_page_doc(page_data)
                    page_docs.append(doc)
                    self.page_map.append((title, int(page_number)))
        
        # Create page TF-IDF matrix if we have any pages
        if page_docs:
            self.page_matrix = self.page_vectorizer.fit_transform(page_docs)
            # Initialize page boolean search engine
            self.page_boolean_engine = BooleanSearchEngine(
                self.page_matrix,
                self.page_vectorizer
            )
        else:
            logger.warning("No transcribed pages found to index")
            self.page_matrix = None
            self.page_boolean_engine = None
            
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
            'total_pages': info.get('total_pages', 0),
            'pages': info.get('pages', {}),
            'summary': info.get('summary', {}),
            'table_of_contents': info.get('table_of_contents', {})
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
        """
        Search manuscripts using boolean query syntax.
        
        Supports:
        - AND, OR, NOT operators (can use &&, ||, ! or - as alternatives)
        - Parentheses for grouping
        - Quoted phrases
        - Implicit AND between terms
        
        Examples:
        - "medieval AND manuscript"
        - "illuminated OR decorated"
        - "church NOT monastery"
        - "latin && (verse || poetry)"
        - "\"lord's prayer\" NOT psalter"
        """
        if self.manuscript_matrix is None:
            logger.error("Search engine not initialized. Call index_manuscripts first.")
            return []
        
        # For simple queries without operators, use traditional search
        if not any(op in query.upper() for op in self.manuscript_boolean_engine.parser.operators):
            query_vector = self.manuscript_vectorizer.transform([query])
            similarities = np.asarray(query_vector.dot(self.manuscript_matrix.T).todense())[0]
            results = []
            top_indices = np.argsort(similarities)[::-1][:num_results]
            
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include relevant results
                    result = self._get_manuscript_info(idx)
                    result['score'] = float(similarities[idx])
                    results.append(result)
            
            return results
        
        # For complex queries with operators, use boolean search
        try:
            matches, similarities = self.manuscript_boolean_engine.search(query, threshold=0.01)  # Lower threshold
            
            # Get top results
            results = []
            matched_indices = np.where(matches)[0]
            if len(matched_indices) > 0:
                matched_scores = similarities[matched_indices]
                sorted_indices = matched_indices[np.argsort(matched_scores)[::-1]]
                
                for idx in sorted_indices[:num_results]:
                    if similarities[idx] > 0:  # Only include relevant results
                        result = self._get_manuscript_info(idx)
                        result['score'] = float(similarities[idx])
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during manuscript search: {e}")
            # Fall back to simple search if boolean parsing fails
            query_vector = self.manuscript_vectorizer.transform([query])
            similarities = np.asarray(query_vector.dot(self.manuscript_matrix.T).todense())[0]
            results = []
            top_indices = np.argsort(similarities)[::-1][:num_results]
            
            for idx in top_indices:
                if similarities[idx] > 0:
                    result = self._get_manuscript_info(idx)
                    result['score'] = float(similarities[idx])
                    results.append(result)
            
            return results
    
    def search_pages(self, query: str, num_results: int = 10, 
                    manuscript_title: Optional[str] = None) -> List[Dict]:
        """
        Search pages using boolean query syntax across all manuscripts or within a specific manuscript.
        
        Args:
            query: Boolean search query
            num_results: Maximum number of results to return
            manuscript_title: Optional manuscript to restrict search to
        """
        if self.page_matrix is None:
            logger.error("No pages indexed. Call index_manuscripts first.")
            return []
            
        try:
            # Use boolean search engine
            matches, similarities = self.page_boolean_engine.search(query)
            
            # Filter for specific manuscript if requested
            if manuscript_title:
                mask = np.array([title == manuscript_title for title, _ in self.page_map])
                matches = matches & mask
            
            # Get top results
            results = []
            matched_indices = np.where(matches)[0]
            matched_scores = similarities[matched_indices]
            sorted_indices = matched_indices[np.argsort(matched_scores)[::-1]]
            
            for idx in sorted_indices:
                if len(results) >= num_results:
                    break
                if similarities[idx] > 0:
                    results.append(self._get_page_info(idx, float(similarities[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during page search: {e}")
            # Fall back to simple search
            query_vector = self.page_vectorizer.transform([query])
            similarities = np.asarray(query_vector.dot(self.page_matrix.T).todense())[0]
            
            if manuscript_title:
                mask = np.array([title == manuscript_title for title, _ in self.page_map])
                similarities = similarities * mask
            
            results = []
            top_indices = np.argsort(similarities)[::-1]
            
            for idx in top_indices:
                if similarities[idx] > 0 and len(results) < num_results:
                    results.append(self._get_page_info(idx, float(similarities[idx])))
            
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
                result = self._get_page_info(idx, float(similarities[idx]))
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