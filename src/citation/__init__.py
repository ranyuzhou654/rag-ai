# src/citation/__init__.py
"""Citation management module for RAG-AI system"""

from .citation_manager import CitationManager, CitationSource, document_to_citation_source

__all__ = ['CitationManager', 'CitationSource', 'document_to_citation_source']