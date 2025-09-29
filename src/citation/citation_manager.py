# src/citation/citation_manager.py
"""
Comprehensive citation and source tracking system for RAG-AI
Provides academic-grade citation generation and source traceability
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import re
from urllib.parse import urlparse
from loguru import logger


@dataclass
class CitationSource:
    """Citation source data structure"""
    id: str
    title: str
    authors: List[str]
    source_type: str  # arxiv, journal, conference, blog, website
    published_date: Optional[datetime] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    journal_name: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    venue: Optional[str] = None
    
    # Additional metadata
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # For tracking usage in responses
    usage_count: int = 0
    last_cited: Optional[datetime] = None


class CitationManager:
    """
    Enterprise-grade citation management system
    - Generates multiple citation formats (APA, MLA, BibTeX, IEEE)
    - Tracks source usage and popularity
    - Provides source verification and link generation
    - Supports academic integrity compliance
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.citations_db_path = data_dir / "citations_database.json"
        self.usage_stats_path = data_dir / "citation_usage_stats.json"
        
        # Create directories if needed
        self.data_dir.mkdir(exist_ok=True)
        
        # Load existing citations and stats
        self.citations_db: Dict[str, CitationSource] = self._load_citations_db()
        self.usage_stats: Dict = self._load_usage_stats()
    
    def _load_citations_db(self) -> Dict[str, CitationSource]:
        """Load citations database from disk"""
        if self.citations_db_path.exists():
            try:
                with open(self.citations_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                citations = {}
                for cite_id, cite_data in data.items():
                    # Convert datetime strings back to datetime objects
                    if cite_data.get('published_date'):
                        cite_data['published_date'] = datetime.fromisoformat(cite_data['published_date'])
                    if cite_data.get('last_cited'):
                        cite_data['last_cited'] = datetime.fromisoformat(cite_data['last_cited'])
                    
                    citations[cite_id] = CitationSource(**cite_data)
                
                logger.info(f"📚 Loaded {len(citations)} citations from database")
                return citations
            except Exception as e:
                logger.error(f"❌ Error loading citations database: {e}")
        
        return {}
    
    def _load_usage_stats(self) -> Dict:
        """Load citation usage statistics"""
        if self.usage_stats_path.exists():
            try:
                with open(self.usage_stats_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Error loading usage stats: {e}")
        
        return {
            'total_citations': 0,
            'popular_sources': {},
            'citation_trends': {},
            'daily_stats': {}
        }
    
    def save_citations_db(self):
        """Save citations database to disk"""
        try:
            data = {}
            for cite_id, citation in self.citations_db.items():
                cite_dict = citation.__dict__.copy()
                
                # Convert datetime objects to ISO strings for JSON serialization
                if cite_dict.get('published_date'):
                    cite_dict['published_date'] = cite_dict['published_date'].isoformat()
                if cite_dict.get('last_cited'):
                    cite_dict['last_cited'] = cite_dict['last_cited'].isoformat()
                
                data[cite_id] = cite_dict
            
            with open(self.citations_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved {len(data)} citations to database")
        except Exception as e:
            logger.error(f"❌ Error saving citations database: {e}")
    
    def save_usage_stats(self):
        """Save usage statistics to disk"""
        try:
            with open(self.usage_stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.usage_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving usage stats: {e}")
    
    def add_citation_source(self, source: CitationSource) -> bool:
        """Add or update a citation source"""
        try:
            self.citations_db[source.id] = source
            logger.info(f"📚 Added citation source: {source.id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error adding citation source {source.id}: {e}")
            return False
    
    def get_citation_source(self, source_id: str) -> Optional[CitationSource]:
        """Retrieve a citation source by ID"""
        return self.citations_db.get(source_id)
    
    def record_citation_usage(self, source_id: str, context: str = ""):
        """Record that a citation source was used in a response"""
        if source_id in self.citations_db:
            citation = self.citations_db[source_id]
            citation.usage_count += 1
            citation.last_cited = datetime.now()
            
            # Update usage statistics
            self.usage_stats['total_citations'] += 1
            
            today = datetime.now().strftime('%Y-%m-%d')
            if today not in self.usage_stats['daily_stats']:
                self.usage_stats['daily_stats'][today] = 0
            self.usage_stats['daily_stats'][today] += 1
            
            # Track popular sources
            if source_id not in self.usage_stats['popular_sources']:
                self.usage_stats['popular_sources'][source_id] = 0
            self.usage_stats['popular_sources'][source_id] += 1
            
            logger.debug(f"📊 Recorded citation usage for {source_id}")
    
    def generate_citation(self, source_id: str, format_type: str = "apa") -> Optional[str]:
        """Generate a formatted citation string"""
        citation = self.get_citation_source(source_id)
        if not citation:
            return None
        
        try:
            if format_type.lower() == "apa":
                return self._generate_apa_citation(citation)
            elif format_type.lower() == "mla":
                return self._generate_mla_citation(citation)
            elif format_type.lower() == "bibtex":
                return self._generate_bibtex_citation(citation)
            elif format_type.lower() == "ieee":
                return self._generate_ieee_citation(citation)
            elif format_type.lower() == "chicago":
                return self._generate_chicago_citation(citation)
            else:
                logger.warning(f"⚠️ Unknown citation format: {format_type}")
                return self._generate_apa_citation(citation)  # Default to APA
        except Exception as e:
            logger.error(f"❌ Error generating {format_type} citation for {source_id}: {e}")
            return None
    
    def _generate_apa_citation(self, citation: CitationSource) -> str:
        """Generate APA format citation"""
        # Format authors
        if citation.authors:
            if len(citation.authors) == 1:
                authors_str = citation.authors[0]
            elif len(citation.authors) == 2:
                authors_str = f"{citation.authors[0]} & {citation.authors[1]}"
            elif len(citation.authors) <= 20:
                authors_str = ", ".join(citation.authors[:-1]) + f", & {citation.authors[-1]}"
            else:
                authors_str = ", ".join(citation.authors[:19]) + ", ... " + citation.authors[-1]
        else:
            authors_str = "Unknown Author"
        
        # Format year
        year = citation.published_date.year if citation.published_date else "n.d."
        
        # Format title
        title = citation.title
        
        if citation.source_type == "arxiv":
            return f"{authors_str} ({year}). {title}. arXiv preprint arXiv:{citation.arxiv_id}."
        elif citation.source_type == "journal":
            journal_info = f"{citation.journal_name}"
            if citation.volume:
                journal_info += f", {citation.volume}"
            if citation.pages:
                journal_info += f", {citation.pages}"
            return f"{authors_str} ({year}). {title}. {journal_info}."
        elif citation.source_type == "conference":
            return f"{authors_str} ({year}). {title}. In {citation.venue}."
        elif citation.source_type == "blog":
            return f"{authors_str} ({year}). {title}. Retrieved from {citation.url}"
        else:
            return f"{authors_str} ({year}). {title}."
    
    def _generate_mla_citation(self, citation: CitationSource) -> str:
        """Generate MLA format citation"""
        # Format authors
        if citation.authors:
            if len(citation.authors) == 1:
                authors_str = citation.authors[0]
            elif len(citation.authors) == 2:
                authors_str = f"{citation.authors[0]} and {citation.authors[1]}"
            else:
                authors_str = f"{citation.authors[0]} et al."
        else:
            authors_str = "Unknown Author"
        
        title = f'"{citation.title}"'
        
        if citation.source_type == "arxiv":
            return f'{authors_str}. {title} arXiv preprint arXiv:{citation.arxiv_id}, {citation.published_date.year if citation.published_date else "n.d."}.'
        elif citation.source_type == "journal":
            journal_info = citation.journal_name
            if citation.volume:
                journal_info += f", vol. {citation.volume}"
            year = citation.published_date.year if citation.published_date else "n.d."
            return f'{authors_str}. {title} {journal_info}, {year}.'
        else:
            return f'{authors_str}. {title} {citation.published_date.year if citation.published_date else "n.d."}.'
    
    def _generate_bibtex_citation(self, citation: CitationSource) -> str:
        """Generate BibTeX format citation"""
        # Generate a safe key
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', citation.id)
        
        # Determine entry type
        if citation.source_type == "arxiv":
            entry_type = "article"
        elif citation.source_type == "journal":
            entry_type = "article"
        elif citation.source_type == "conference":
            entry_type = "inproceedings"
        else:
            entry_type = "misc"
        
        # Build BibTeX entry
        bibtex = f"@{entry_type}{{{safe_id},\n"
        bibtex += f"  title={{{citation.title}}},\n"
        
        if citation.authors:
            authors = " and ".join(citation.authors)
            bibtex += f"  author={{{authors}}},\n"
        
        if citation.published_date:
            bibtex += f"  year={{{citation.published_date.year}}},\n"
        
        if citation.source_type == "arxiv" and citation.arxiv_id:
            bibtex += f"  journal={{arXiv preprint arXiv:{citation.arxiv_id}}},\n"
        elif citation.journal_name:
            bibtex += f"  journal={{{citation.journal_name}}},\n"
        elif citation.venue:
            bibtex += f"  booktitle={{{citation.venue}}},\n"
        
        if citation.volume:
            bibtex += f"  volume={{{citation.volume}}},\n"
        
        if citation.pages:
            bibtex += f"  pages={{{citation.pages}}},\n"
        
        if citation.url:
            bibtex += f"  url={{{citation.url}}},\n"
        
        bibtex += "}"
        
        return bibtex
    
    def _generate_ieee_citation(self, citation: CitationSource) -> str:
        """Generate IEEE format citation"""
        # Format authors
        if citation.authors:
            if len(citation.authors) == 1:
                authors_str = citation.authors[0]
            elif len(citation.authors) <= 6:
                authors_str = ", ".join(citation.authors[:-1]) + f", and {citation.authors[-1]}"
            else:
                authors_str = f"{citation.authors[0]} et al."
        else:
            authors_str = "Unknown Author"
        
        title = f'"{citation.title},"'
        year = citation.published_date.year if citation.published_date else "n.d."
        
        if citation.source_type == "arxiv":
            return f"{authors_str}, {title} arXiv preprint arXiv:{citation.arxiv_id}, {year}."
        elif citation.source_type == "journal":
            journal_info = f"*{citation.journal_name}*"
            if citation.volume:
                journal_info += f", vol. {citation.volume}"
            if citation.pages:
                journal_info += f", pp. {citation.pages}"
            return f"{authors_str}, {title} {journal_info}, {year}."
        else:
            return f"{authors_str}, {title} {year}."
    
    def _generate_chicago_citation(self, citation: CitationSource) -> str:
        """Generate Chicago format citation"""
        # Similar to APA but with different punctuation
        if citation.authors:
            if len(citation.authors) == 1:
                authors_str = citation.authors[0]
            else:
                authors_str = f"{citation.authors[0]} et al."
        else:
            authors_str = "Unknown Author"
        
        title = f'"{citation.title}."'
        year = citation.published_date.year if citation.published_date else "n.d."
        
        if citation.source_type == "arxiv":
            return f"{authors_str}. {title} arXiv preprint arXiv:{citation.arxiv_id} ({year})."
        elif citation.source_type == "journal":
            journal_info = citation.journal_name
            if citation.volume:
                journal_info += f" {citation.volume}"
            if citation.pages:
                journal_info += f" ({year}): {citation.pages}"
            else:
                journal_info += f" ({year})"
            return f"{authors_str}. {title} {journal_info}."
        else:
            return f"{authors_str}. {title} {year}."
    
    def generate_source_links(self, source_id: str) -> Dict[str, str]:
        """Generate clickable links to original sources"""
        citation = self.get_citation_source(source_id)
        if not citation:
            return {}
        
        links = {}
        
        # Original URL
        if citation.url:
            links['original'] = citation.url
        
        # DOI link
        if citation.doi:
            links['doi'] = f"https://doi.org/{citation.doi}"
        
        # ArXiv links
        if citation.arxiv_id:
            links['arxiv_abs'] = f"https://arxiv.org/abs/{citation.arxiv_id}"
            links['arxiv_pdf'] = f"https://arxiv.org/pdf/{citation.arxiv_id}.pdf"
        
        # Publisher link
        if citation.publisher and citation.source_type == "journal":
            # Common publisher links (can be expanded)
            publisher_urls = {
                'IEEE': 'https://ieeexplore.ieee.org',
                'ACM': 'https://dl.acm.org',
                'Springer': 'https://link.springer.com',
                'Elsevier': 'https://www.sciencedirect.com'
            }
            if citation.publisher in publisher_urls:
                links['publisher'] = publisher_urls[citation.publisher]
        
        return links
    
    def get_popular_sources(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently cited sources"""
        popular = sorted(
            self.usage_stats['popular_sources'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return popular[:limit]
    
    def get_citation_statistics(self) -> Dict:
        """Get comprehensive citation statistics"""
        total_sources = len(self.citations_db)
        total_citations = self.usage_stats['total_citations']
        
        # Source type breakdown
        source_types = {}
        for citation in self.citations_db.values():
            source_type = citation.source_type
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        # Recent activity (last 7 days)
        recent_activity = {}
        from datetime import datetime, timedelta
        last_week = datetime.now() - timedelta(days=7)
        
        for date_str, count in self.usage_stats['daily_stats'].items():
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                if date >= last_week:
                    recent_activity[date_str] = count
            except ValueError:
                continue
        
        return {
            'total_sources': total_sources,
            'total_citations': total_citations,
            'source_types': source_types,
            'recent_activity': recent_activity,
            'average_citations_per_source': total_citations / max(total_sources, 1)
        }
    
    def verify_source_accessibility(self, source_id: str) -> Dict[str, bool]:
        """Verify that source links are accessible"""
        citation = self.get_citation_source(source_id)
        if not citation:
            return {}
        
        links = self.generate_source_links(source_id)
        accessibility = {}
        
        # This is a placeholder - in a full implementation, you would
        # actually check HTTP status codes for each link
        for link_type, url in links.items():
            try:
                # Simplified check - just verify URL format
                parsed = urlparse(url)
                accessibility[link_type] = bool(parsed.netloc and parsed.scheme)
            except:
                accessibility[link_type] = False
        
        return accessibility
    
    def export_bibliography(self, source_ids: List[str], format_type: str = "apa") -> str:
        """Export a bibliography for multiple sources"""
        bibliography = []
        
        for source_id in source_ids:
            citation_text = self.generate_citation(source_id, format_type)
            if citation_text:
                bibliography.append(citation_text)
        
        if format_type.lower() == "bibtex":
            return "\n\n".join(bibliography)
        else:
            # For other formats, sort alphabetically and number
            bibliography.sort()
            return "\n".join(f"{i+1}. {citation}" for i, citation in enumerate(bibliography))
    
    def cleanup_unused_sources(self, days_threshold: int = 90):
        """Remove sources that haven't been cited in the specified time period"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        unused_sources = []
        for source_id, citation in self.citations_db.items():
            if not citation.last_cited or citation.last_cited < cutoff_date:
                unused_sources.append(source_id)
        
        for source_id in unused_sources:
            del self.citations_db[source_id]
            logger.info(f"🗑️ Removed unused citation source: {source_id}")
        
        logger.info(f"🧹 Cleaned up {len(unused_sources)} unused citation sources")
        return len(unused_sources)


# Utility function to convert documents to citation sources
def document_to_citation_source(document) -> CitationSource:
    """Convert a Document object to a CitationSource object"""
    from datetime import timedelta
    
    # Determine source type based on document source
    source_type_mapping = {
        'arxiv': 'arxiv',
        'huggingface': 'arxiv',  # Many HF papers are actually arXiv papers
        'blog_google_ai': 'blog',
        'blog_openai': 'blog',
        'blog_bair': 'blog',
        'blog_deepmind': 'blog'
    }
    
    source_type = source_type_mapping.get(document.source, 'website')
    
    return CitationSource(
        id=document.id,
        title=document.title,
        authors=document.authors or [],
        source_type=source_type,
        published_date=document.published_date,
        url=document.url,
        doi=document.doi,
        arxiv_id=document.arxiv_id,
        abstract=document.abstract,
        keywords=document.keywords or []
    )