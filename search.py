from elasticsearch import Elasticsearch, helpers
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import json
from pathlib import Path

class ManuscriptSearchEngine:
    """Handles all Elasticsearch operations for the manuscript database."""
    
    def __init__(self, host: str = 'localhost', port: int = 9200, index_name: str = 'manuscripts'):
        """Initialize the Elasticsearch connection and index."""
        self.es = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = index_name
        self.logger = logging.getLogger(__name__)
        
        # Create index if it doesn't exist
        if not self.es.indices.exists(index=self.index_name):
            self._create_index()

    def _create_index(self) -> None:
        """Create the Elasticsearch index with appropriate mappings."""
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "record_id": {"type": "keyword"},
                    "metadata": {
                        "properties": {
                            "date": {"type": "text"},
                            "language": {"type": "keyword"},
                            "origin": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                            "scribe": {"type": "text"},
                            "material": {"type": "keyword"},
                            "dimensions": {"type": "text"},
                            "folios": {"type": "integer"},
                            "custom_fields": {"type": "object"}
                        }
                    },
                    "transcribed": {"type": "boolean"},
                    "transcription_info": {
                        "properties": {
                            "successful_pages": {"type": "integer"},
                            "failed_pages": {"type": "integer"},
                            "last_updated": {"type": "date"}
                        }
                    },
                    "pages": {
                        "type": "nested",
                        "properties": {
                            "page_number": {"type": "integer"},
                            "transcription": {"type": "text", "analyzer": "standard"},
                            "revised_transcription": {"type": "text", "analyzer": "standard"},
                            "summary": {"type": "text"},
                            "keywords": {"type": "keyword"},
                            "marginalia": {"type": "text"},
                            "confidence": {"type": "float"},
                            "transcription_notes": {"type": "text"},
                            "content_notes": {"type": "text"}
                        }
                    }
                }
            },
            "settings": {
                "analysis": {
                    "analyzer": {
                        "folded_text": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            }
        }
        
        self.es.indices.create(index=self.index_name, body=mapping)
        self.logger.info(f"Created index '{self.index_name}' with manuscript mappings")

    def index_manuscript(self, manuscript_data: Dict[str, Any]) -> bool:
        """Index a manuscript and its pages."""
        try:
            # Prepare the document
            doc = {
                "title": manuscript_data.get("title"),
                "record_id": manuscript_data.get("record_id"),
                "metadata": manuscript_data.get("metadata", {}),
                "transcribed": manuscript_data.get("transcribed", False),
                "transcription_info": manuscript_data.get("transcription_info", {}),
                "pages": manuscript_data.get("pages", [])
            }
            
            # Index the document
            self.es.index(
                index=self.index_name,
                id=doc["record_id"],
                body=doc
            )
            
            self.logger.info(f"Successfully indexed manuscript: {doc['title']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error indexing manuscript: {str(e)}")
            return False

    def search_manuscripts(self, 
                         query: str, 
                         fields: Optional[List[str]] = None,
                         page_content: bool = False,
                         filters: Optional[Dict[str, Any]] = None,
                         size: int = 10) -> Dict[str, Any]:
        """
        Search manuscripts with various options.
        
        Args:
            query: Search query string
            fields: Specific fields to search in
            page_content: Whether to search in page content
            filters: Dictionary of filters to apply
            size: Maximum number of results to return
        """
        try:
            # Build the query
            search_fields = fields if fields else ["title^2", "metadata.*"]
            if page_content:
                search_fields.extend(["pages.transcription", "pages.revised_transcription"])

            must_queries = [{
                "multi_match": {
                    "query": query,
                    "fields": search_fields,
                    "type": "best_fields",
                    "operator": "and"
                }
            }]

            # Add filters if provided
            if filters:
                for field, value in filters.items():
                    if value is not None:
                        must_queries.append({"match": {field: value}})

            # Construct the full query
            body = {
                "query": {
                    "bool": {
                        "must": must_queries
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "pages.transcription": {},
                        "pages.revised_transcription": {}
                    }
                },
                "size": size
            }

            # Execute search
            results = self.es.search(index=self.index_name, body=body)
            
            # Format results
            formatted_results = []
            for hit in results["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "manuscript": hit["_source"],
                    "highlights": hit.get("highlight", {})
                }
                formatted_results.append(result)

            return {
                "total": results["hits"]["total"]["value"],
                "results": formatted_results
            }

        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return {"total": 0, "results": [], "error": str(e)}

    def get_manuscript(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific manuscript by record ID."""
        try:
            result = self.es.get(index=self.index_name, id=record_id)
            return result["_source"]
        except Exception as e:
            self.logger.error(f"Error retrieving manuscript {record_id}: {str(e)}")
            return None

    def update_manuscript(self, record_id: str, update_data: Dict[str, Any]) -> bool:
        """Update specific fields of a manuscript."""
        try:
            self.es.update(
                index=self.index_name,
                id=record_id,
                body={"doc": update_data}
            )
            return True
        except Exception as e:
            self.logger.error(f"Error updating manuscript {record_id}: {str(e)}")
            return False

    def delete_manuscript(self, record_id: str) -> bool:
        """Delete a manuscript from the index."""
        try:
            self.es.delete(index=self.index_name, id=record_id)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting manuscript {record_id}: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the manuscript collection."""
        try:
            stats = {
                "total_manuscripts": 0,
                "transcribed_manuscripts": 0,
                "total_pages": 0,
                "languages": set(),
                "origins": set(),
                "date_range": {"min": None, "max": None}
            }
            
            # Use the count API for basic counts
            stats["total_manuscripts"] = self.es.count(index=self.index_name)["count"]
            
            # Use aggregations for detailed statistics
            aggs = {
                "transcribed_count": {
                    "filter": {"term": {"transcribed": True}}
                },
                "languages": {
                    "terms": {"field": "metadata.language"}
                },
                "origins": {
                    "terms": {"field": "metadata.origin.keyword"}
                },
                "total_pages": {
                    "sum": {"field": "metadata.folios"}
                }
            }
            
            results = self.es.search(
                index=self.index_name,
                body={"size": 0, "aggs": aggs}
            )
            
            # Process aggregation results
            stats["transcribed_manuscripts"] = results["aggregations"]["transcribed_count"]["doc_count"]
            stats["total_pages"] = int(results["aggregations"]["total_pages"]["value"])
            stats["languages"] = [bucket["key"] for bucket in results["aggregations"]["languages"]["buckets"]]
            stats["origins"] = [bucket["key"] for bucket in results["aggregations"]["origins"]["buckets"]]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}

def init_search_engine(host: str = 'localhost', port: int = 9200) -> ManuscriptSearchEngine:
    """Initialize and return a configured search engine instance."""
    return ManuscriptSearchEngine(host=host, port=port)