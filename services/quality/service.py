from typing import Dict, Any, Optional
from core.exceptions import QualityControlException
from core.types import QualityLevel, QualityMetrics
from config.settings import QUALITY_CONFIG
from services.llm.service import LLMService

class QualityControlService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or QUALITY_CONFIG
        self.llm_service = LLMService()

    async def evaluate_quality(self, text: str) -> QualityMetrics:
        try:
            # Get quality scores from LLM
            scores = await self.llm_service.check_quality(text)
            
            metrics = QualityMetrics(
                coherence_score=scores["coherence"],
                relevance_score=scores["relevance"],
                completeness_score=scores["completeness"],
                overall_score=self._calculate_overall_score(scores)
            )
            
            return metrics
        except Exception as e:
            raise QualityControlException(f"Error evaluating quality: {str(e)}")

    def determine_quality_level(self, metrics: QualityMetrics) -> QualityLevel:
        try:
            overall_score = metrics.overall_score
            threshold = self.config["validation_threshold"]
            
            if overall_score >= threshold * 1.2:
                return QualityLevel.HIGH
            elif overall_score >= threshold:
                return QualityLevel.MEDIUM
            else:
                return QualityLevel.LOW
        except Exception as e:
            raise QualityControlException(f"Error determining quality level: {str(e)}")

    async def suggest_improvements(self, text: str, metrics: QualityMetrics) -> str:
        try:
            # Generate feedback based on metrics
            feedback = self._generate_feedback(metrics)
            
            # Get improvement suggestions from LLM
            improved_text = await self.llm_service.improve_text(text, feedback)
            
            return improved_text
        except Exception as e:
            raise QualityControlException(f"Error suggesting improvements: {str(e)}")

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "coherence": 0.3,
            "relevance": 0.3,
            "completeness": 0.2,
            "evidence": 0.2
        }
        
        overall_score = sum(
            scores[metric] * weight
            for metric, weight in weights.items()
        )
        
        return overall_score

    def _generate_feedback(self, metrics: QualityMetrics) -> str:
        """Generate feedback based on quality metrics."""
        feedback_points = []
        
        if metrics.coherence_score < self.config["min_confidence_score"]:
            feedback_points.append("Improve logical flow and coherence")
        
        if metrics.relevance_score < self.config["min_confidence_score"]:
            feedback_points.append("Enhance relevance to the topic")
        
        if metrics.completeness_score < self.config["min_confidence_score"]:
            feedback_points.append("Provide more comprehensive analysis")
        
        return ". ".join(feedback_points) if feedback_points else "Minor improvements suggested"
