from typing import Dict, Any, Optional
import requests
import json
from core.exceptions import LLMServiceException
from config.settings import LLM_CONFIG
from services.llm.prompts import RESEARCH_PROMPT, QUALITY_CHECK_PROMPT, IMPROVEMENT_PROMPT
from core.types import ResearchDirection, EvaluationCriteria

class LLMService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or LLM_CONFIG
        self.headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "HTTP-Referer": self.config.get("app_name", "SUHANGSSALMUK"),
            "X-Title": self.config.get("app_title", "SUHANGSSALMUK"),
        }
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = self.config["model_name"]

    async def generate_research(
        self, 
        query: str, 
        context: Optional[str] = None,
        direction: Optional[ResearchDirection] = None
    ) -> str:
        try:
            # 연구 방향에 따른 프롬프트 조정
            direction_guidance = ""
            if direction:
                direction_guidance = f"\nResearch Direction: {direction.value}\n"
                if direction == ResearchDirection.EXPLORATORY:
                    direction_guidance += "Focus on identifying variables and generating hypotheses."
                elif direction == ResearchDirection.DESCRIPTIVE:
                    direction_guidance += "Focus on detailed description and pattern analysis."
                elif direction == ResearchDirection.EXPLANATORY:
                    direction_guidance += "Focus on causal relationships and mechanisms."
                elif direction == ResearchDirection.EXPERIMENTAL:
                    direction_guidance += "Focus on experimental design and variable control."
                elif direction == ResearchDirection.THEORETICAL:
                    direction_guidance += "Focus on theoretical analysis and integration."

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": RESEARCH_PROMPT.format(
                            query=query, 
                            context=context or "",
                            direction_guidance=direction_guidance
                        )
                    },
                    {"role": "user", "content": query}
                ]
            }
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            if 'choices' in data and data['choices']:
                return data['choices'][0]['message']['content'].strip()
            else:
                raise LLMServiceException("No response from the model")
        except requests.RequestException as e:
            raise LLMServiceException(f"Error generating research: {str(e)}")

    async def check_quality(
        self, 
        text: str,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> Dict[str, float]:
        try:
            # 평가 기준에 따른 프롬프트 조정
            criteria_prompt = QUALITY_CHECK_PROMPT
            if evaluation_criteria:
                criteria_prompt = f"""Evaluate the quality based on these weighted criteria:
                1. Methodology (weight: {evaluation_criteria.methodology_weight})
                2. Innovation (weight: {evaluation_criteria.innovation_weight})
                3. Validity (weight: {evaluation_criteria.validity_weight})
                4. Reliability (weight: {evaluation_criteria.reliability_weight})

                Additional criteria: {evaluation_criteria.custom_criteria or 'None'}

                Text: {text}

                Provide numerical scores (0.0-1.0) for each criterion and detailed justification.
                """

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": criteria_prompt},
                ]
            }
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            if 'choices' in data and data['choices']:
                scores = self._parse_quality_scores(
                    data['choices'][0]['message']['content'],
                    evaluation_criteria
                )
                return scores
            else:
                raise LLMServiceException("No response from the model")
        except requests.RequestException as e:
            raise LLMServiceException(f"Error checking quality: {str(e)}")

    async def improve_text(
        self, 
        text: str, 
        feedback: str,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> str:
        try:
            improvement_guidance = ""
            if evaluation_criteria:
                improvement_guidance = f"""
                Consider these criteria weights:
                - Methodology: {evaluation_criteria.methodology_weight}
                - Innovation: {evaluation_criteria.innovation_weight}
                - Validity: {evaluation_criteria.validity_weight}
                - Reliability: {evaluation_criteria.reliability_weight}

                Minimum required scores:
                - Quality: {evaluation_criteria.min_quality_score}
                - Validity: {evaluation_criteria.required_validity_score}
                """

            prompt = IMPROVEMENT_PROMPT.format(
                text=text,
                feedback=feedback,
                improvement_guidance=improvement_guidance
            )
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt},
                ]
            }
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            if 'choices' in data and data['choices']:
                return data['choices'][0]['message']['content'].strip()
            else:
                raise LLMServiceException("No response from the model")
        except requests.RequestException as e:
            raise LLMServiceException(f"Error improving text: {str(e)}")

    def _parse_quality_scores(
        self, 
        response: str,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> Dict[str, float]:
        """Parse the quality check response to extract numerical scores."""
        try:
            scores = {
                "methodology": 0.0,
                "innovation": 0.0,
                "validity": 0.0,
                "reliability": 0.0,
                "overall": 0.0
            }

            for line in response.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    if key in scores:
                        try:
                            scores[key] = float(value.strip())
                        except ValueError:
                            pass

            # Apply weights if evaluation criteria is provided
            if evaluation_criteria:
                scores["overall"] = (
                    scores["methodology"] * evaluation_criteria.methodology_weight +
                    scores["innovation"] * evaluation_criteria.innovation_weight +
                    scores["validity"] * evaluation_criteria.validity_weight +
                    scores["reliability"] * evaluation_criteria.reliability_weight
                )
            else:
                scores["overall"] = sum(scores.values()) / len(scores)

            return scores
        except Exception as e:
            raise LLMServiceException(f"Error parsing quality scores: {str(e)}")