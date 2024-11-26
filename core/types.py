from typing import TypeVar, Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

class ResearchStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class QualityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ResearchRequest(BaseModel):
    query: str
    context: Optional[str] = None
    max_iterations: Optional[int] = None
    quality_threshold: Optional[float] = None

class ResearchResult(BaseModel):
    query: str
    result: str
    confidence_score: float
    quality_score: float
    metadata: Dict[str, Any]
    status: ResearchStatus

class EmbeddingVector(BaseModel):
    text: str
    vector: List[float]
    metadata: Dict[str, Any]

class QualityMetrics(BaseModel):
    coherence_score: float
    relevance_score: float
    completeness_score: float
    overall_score: float

class ResearchPhase(str, Enum):
    PLANNING = "planning"
    METHODOLOGY = "methodology"
    QUALITY = "quality"
    OPTIMIZATION = "optimization"
    COMPLETED = "completed"
    FAILED = "failed"

class ValidationCriteria(str, Enum):
    METHODOLOGY = "methodology"
    DATA = "data"
    ANALYSIS = "analysis"
    RESULTS = "results"

class ResearchMetrics(BaseModel):
    methodology_score: float = Field(default=0.0, ge=0.0, le=1.0)
    data_score: float = Field(default=0.0, ge=0.0, le=1.0)
    analysis_score: float = Field(default=0.0, ge=0.0, le=1.0)
    results_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)

class ResearchGap(BaseModel):
    description: str
    importance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    feasibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cluster: Optional[str] = None

class Reference(BaseModel):
    title: str
    authors: List[str]
    year: int
    citation_count: Optional[int] = None
    journal_impact_factor: Optional[float] = None
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None

class TheoreticalFramework(BaseModel):
    concepts: Dict[str, str]  # concept_name: description
    relationships: List[Dict[str, Any]]  # list of concept relationships
    hierarchy: Dict[str, List[str]]  # parent_concept: [child_concepts]
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)

T = TypeVar('T')  # Generic type for flexible type hints
