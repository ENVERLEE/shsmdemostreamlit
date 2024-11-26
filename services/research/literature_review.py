from typing import List, Dict, Any
from core.models import ResearchProject, Reference
from core.exceptions import ResearchException
from core.types import ResearchStatus
from services.llm.service import LLMService
from services.embedding.service import EmbeddingService
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import requests
import json

class LiteratureReviewService:
    def __init__(self, llm_service: LLMService, embedding_service: EmbeddingService):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.perplexity_api_key = self._load_api_key()
        self.perplexity_api_url = "https://api.perplexity.ai/chat/completions"

    def _load_api_key(self) -> str:
        """Load Perplexity API key from environment"""
        import os
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ResearchException("Perplexity API key not found in environment")
        return api_key

    async def collect_initial_papers(self, query: str, limit: int = 100) -> List[Reference]:
        try:
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            messages = [
                {
                    "role": "system",
                    "content": "Return academic papers in a structured format. For each paper include: Title, Authors (separated by commas), Year, Abstract, Citations count, Journal name, DOI, and URL. Separate papers with '---'. Format each paper as follows:\nTitle: [title]\nAuthors: [authors]\nYear: [year]\nAbstract: [abstract]\nCitations: [count]\nJournal: [journal]\nDOI: [doi]\nURL: [url]"
                },
                {
                    "role": "user",
                    "content": f"Find detailed academic papers about: {query}"
                }
            ]
            
            request_body = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": messages,
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 2048,
                "return_citations": True,
                "search_domain_filter": ["arxiv.org", "scholar.google.com", "science.org"],
                "stream": False
            }
            
            response = requests.post(
                self.perplexity_api_url,
                headers=headers,
                json=request_body
            )
            
            if response.status_code != 200:
                raise ResearchException(f"Perplexity API error: {response.text}")
                
            response_data = response.json()
            papers_data = self._parse_assistant_response(response_data["choices"][0]["message"]["content"])
            citations = response_data.get("citations", [])
            
            papers = []
            for paper_data in papers_data[:limit]:
                paper_text = f"{paper_data['title']}. {paper_data.get('abstract', '')}"
                paper_embedding = await self.embedding_service.create_embeddings(paper_text)
                query_embedding = await self.embedding_service.create_embeddings(query)
                
                relevance_score = np.dot(
                    paper_embedding[0].vector,
                    query_embedding[0].vector
                )
                
                paper = Reference(
                    title=paper_data["title"].strip(),
                    authors=[author.strip() for author in paper_data.get("authors", "").split(",")],
                    year=int(paper_data.get("year", 0)),
                    abstract=paper_data.get("abstract", "").strip(),
                    citation_count=int(paper_data.get("citations", "0").replace(",", "")),
                    journal_impact_factor=self._get_journal_impact_factor(paper_data.get("journal", "")),
                    relevance_score=float(relevance_score),
                    url=paper_data.get("url", "").strip(),
                    doi=paper_data.get("doi", "").strip()
                )
                papers.append(paper)
                
            return papers
            
        except Exception as e:
            raise ResearchException(f"Error collecting papers: {str(e)}")
            
    def _parse_assistant_response(self, content: str) -> List[dict]:
        """
        Parse the assistant's response text into structured paper data using sophisticated regex patterns.
        Handles various edge cases and format variations in the response.
        """
        import re
        
        # Clean and normalize the content
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        content = content.replace('\n', ' ').strip()
        
        # Sophisticated regex patterns for each field
        patterns = {
            'paper_separator': r'---',
            'title': r'Title:\s*(?P<title>(?:(?!Authors:|Year:|Abstract:|Citations:|Journal:|DOI:|URL:).)+)',
            'authors': r'Authors:\s*(?P<authors>(?:(?!Year:|Abstract:|Citations:|Journal:|DOI:|URL:).)+)',
            'year': r'Year:\s*(?P<year>(?:(?!Abstract:|Citations:|Journal:|DOI:|URL:).)+)',
            'abstract': r'Abstract:\s*(?P<abstract>(?:(?!Citations:|Journal:|DOI:|URL:).)+)',
            'citations': r'Citations:\s*(?P<citations>(?:(?!Journal:|DOI:|URL:).)+)',
            'journal': r'Journal:\s*(?P<journal>(?:(?!DOI:|URL:).)+)',
            'doi': r'DOI:\s*(?P<doi>(?:(?!URL:).)+)',
            'url': r'URL:\s*(?P<url>.+?)(?=---|$)',
        }
        
        # Split content into individual paper sections
        papers_raw = re.split(patterns['paper_separator'], content)
        papers_data = []
        
        for paper_raw in papers_raw:
            if not paper_raw.strip():
                continue
                
            paper_data = {}
            
            # Extract each field using regex with error handling
            for field, pattern in patterns.items():
                if field == 'paper_separator':
                    continue
                    
                match = re.search(pattern, paper_raw, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(field).strip()
                    # Clean and normalize the extracted value
                    value = re.sub(r'\s+', ' ', value)
                    paper_data[field] = value
                else:
                    # Handle missing fields gracefully
                    paper_data[field] = ""
            
            # Additional data cleaning and validation
            try:
                # Clean year
                year_str = re.sub(r'[^\d]', '', paper_data.get('year', ''))
                paper_data['year'] = int(year_str) if year_str and 1900 <= int(year_str) <= 2024 else 0
                
                # Clean citations
                citations_str = re.sub(r'[^\d]', '', paper_data.get('citations', ''))
                paper_data['citations'] = int(citations_str) if citations_str else 0
                
                # Clean DOI
                doi_match = re.search(r'10\.\d{4,9}/[-._;()/:\w]+', paper_data.get('doi', ''))
                paper_data['doi'] = doi_match.group(0) if doi_match else ""
                
                # Clean URL
                url_match = re.search(r'https?://\S+', paper_data.get('url', ''))
                paper_data['url'] = url_match.group(0) if url_match else ""
                
                # Validate and clean authors
                authors = paper_data.get('authors', '')
                authors = re.split(r',\s*(?:and\s+)?|\s+and\s+', authors)
                authors = [author.strip() for author in authors if author.strip()]
                paper_data['authors'] = ', '.join(authors)
                
            except Exception as e:
                logger.warning(f"Error cleaning paper data: {str(e)}")
                continue
            
            if paper_data.get('title') and paper_data.get('abstract'):
                papers_data.append(paper_data)
        
        return papers_data
    def _get_journal_impact_factor(self, journal_name: str) -> float:
        """Get journal impact factor from database"""
        # This would typically query a database of journal metrics
        # For now, using a simple dictionary of major journals
        impact_factors = {
            "Nature": 49.962,
            "Science": 47.728,
            "Cell": 41.582,
            "PNAS": 11.205,
            "PLoS ONE": 3.240,
            "Scientific Reports": 4.996
        }
        return impact_factors.get(journal_name, 1.0)  # Default to 1.0 if unknown

    async def analyze_research_gaps(self, papers: List[Reference]) -> List[Dict[str, Any]]:
        """Analyze research gaps from papers"""
        try:
            # Extract future work and limitations sections
            gap_texts = []
            for paper in papers:
                # Generate gap analysis prompt
                gap_prompt = f"""
                Based on this paper:
                Title: {paper.title}
                Abstract: {paper.abstract}
                Year: {paper.year}
                
                Identify potential research gaps by considering:
                1. Limitations mentioned in the paper
                2. Future work suggestions
                3. Methodological gaps
                4. Theoretical gaps
                5. Application gaps
                
                Format the response as a structured list of specific gaps.
                """
                
                gap_response = await self.llm_service.generate_research(gap_prompt)
                gap_texts.append({
                    "text": gap_response,
                    "paper": paper.title,
                    "year": paper.year,
                    "citations": paper.citation_count
                })
            
            # Create embeddings for gap analysis
            gap_embeddings = []
            for gap in gap_texts:
                embedding = await self.embedding_service.create_embeddings(gap["text"])
                gap_embeddings.append(embedding[0].vector)
            
            # Cluster similar gaps
            n_clusters = min(5, len(gap_embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(gap_embeddings)
            
            # Analyze each cluster
            research_gaps = []
            for i in range(n_clusters):
                cluster_gaps = [gap for j, gap in enumerate(gap_texts) if clusters[j] == i]
                
                # Sort gaps by citation count and recency
                cluster_gaps.sort(
                    key=lambda x: (x["citations"], x["year"]),
                    reverse=True
                )
                
                # Generate cluster summary
                summary_prompt = f"""
                Analyze these related research gaps:
                {json.dumps(cluster_gaps, indent=2)}
                
                Provide:
                1. A concise summary of the common gap theme
                2. The significance of this research gap
                3. Potential approaches to address it
                4. Required resources or expertise
                5. Potential challenges
                """
                
                cluster_summary = await self.llm_service.generate_research(summary_prompt)
                
                # Score the gap
                scores = await self._score_research_gap(
                    cluster_summary,
                    cluster_gaps
                )
                
                research_gaps.append({
                    "description": cluster_summary,
                    "papers": [gap["paper"] for gap in cluster_gaps],
                    "importance_score": scores["importance"],
                    "feasibility_score": scores["feasibility"],
                    "impact_score": scores["impact"],
                    "novelty_score": scores["novelty"],
                    "resource_requirements": scores["resources"],
                    "cluster_id": f"cluster_{i}"
                })
            
            return sorted(
                research_gaps,
                key=lambda x: (
                    x["importance_score"] * 
                    x["feasibility_score"] * 
                    x["impact_score"] * 
                    x["novelty_score"]
                ),
                reverse=True
            )
            
        except Exception as e:
            raise ResearchException(f"Error analyzing research gaps: {str(e)}")

    async def _score_research_gap(
        self,
        gap_summary: str,
        cluster_gaps: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Score a research gap using LLM"""
        try:
            scoring_prompt = f"""
            Evaluate this research gap and provide scores (0.0-1.0) for each criterion:

            Research Gap Summary:
            {gap_summary}

            Related Papers:
            {json.dumps([{
                "title": gap["paper"],
                "year": gap["year"],
                "citations": gap["citations"]
            } for gap in cluster_gaps], indent=2)}

            Score these aspects:
            1. Importance: How critical is addressing this gap?
            2. Feasibility: How feasible is it to address this gap?
            3. Impact: What potential impact would addressing this gap have?
            4. Novelty: How original/innovative is this research direction?
            5. Resources: How resource-intensive would this research be? (0.0 = very intensive, 1.0 = minimal resources)

            Format response as JSON with scores and brief justifications.
            """
            
            response = await self.llm_service.generate_research(scoring_prompt)
            
            # Parse JSON response
            scores = json.loads(response)
            
            # Validate scores
            required_fields = ["importance", "feasibility", "impact", "novelty", "resources"]
            for field in required_fields:
                if field not in scores:
                    raise ResearchException(f"Missing score field: {field}")
                if not isinstance(scores[field], (int, float)):
                    raise ResearchException(f"Invalid score format for {field}")
                if not 0 <= scores[field] <= 1:
                    raise ResearchException(f"Score out of range for {field}")
            
            return scores
            
        except Exception as e:
            raise ResearchException(f"Error scoring research gap: {str(e)}")

    def _normalize_citations(self, citations: int) -> float:
        """Normalize citation count to 0-1 scale"""
        if citations <= 0:
            return 0.0
        return min(1.0, citations / 1000)  # Assuming 1000+ citations is excellent

    def _normalize_impact_factor(self, impact_factor: float) -> float:
        """Normalize journal impact factor to 0-1 scale"""
        if impact_factor <= 0:
            return 0.0
        return min(1.0, impact_factor / 10)  # Assuming 10+ impact factor is excellent

    def _calculate_recency_score(self, year: int) -> float:
        """Calculate recency score based on publication year"""
        current_year = datetime.now().year  # Use actual current year
        years_old = current_year - year
        if years_old <= 0:
            return 1.0
        elif years_old >= 10:
            return 0.0
        return 1.0 - (years_old / 10)

    async def _generate_sample_papers(self, query: str, limit: int) -> List[Reference]:
        """Generate sample papers for development (remove in production)"""
        papers = []
        for i in range(limit):
            papers.append(Reference(
                title=f"Sample Paper {i}",
                authors=["Author A", "Author B"],
                year=2020 + (i % 5),
                citation_count=100 * (i % 10),
                journal_impact_factor=2.0 + (i % 8),
                relevance_score=0.8
            ))
        return papers

    async def evaluate_paper_quality(self, papers: List[Reference]) -> List[Reference]:
        """Evaluate paper quality based on multiple criteria"""
        try:
            for paper in papers:
                # Calculate quality score
                citation_score = self._normalize_citations(paper.citation_count or 0)
                impact_score = self._normalize_impact_factor(paper.journal_impact_factor or 0)
                year_score = self._calculate_recency_score(paper.year)
                
                # Get relevance score using embeddings
                paper_embedding = await self.embedding_service.create_embeddings(
                    f"{paper.title}. {', '.join(paper.authors)}"
                )
                paper.embedding = paper_embedding[0].vector
                
                # Calculate overall quality score
                paper.quality_score = (
                    citation_score * 0.3 +
                    impact_score * 0.3 +
                    year_score * 0.2 +
                    paper.relevance_score * 0.2
                )
            
            return sorted(papers, key=lambda x: x.quality_score, reverse=True)
        except Exception as e:
            raise ResearchException(f"Error evaluating papers: {str(e)}")
