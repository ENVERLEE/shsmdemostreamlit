from typing import Dict, Any, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from core.exceptions import LLMServiceException
from config.settings import LLM_CONFIG
from services.llm.prompts import RESEARCH_PROMPT, QUALITY_CHECK_PROMPT, IMPROVEMENT_PROMPT

class LLMService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or LLM_CONFIG
        self.llm = ChatOpenAI(
            model_name=self.config["model_name"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        self.research_chain = LLMChain(llm=self.llm, prompt=RESEARCH_PROMPT)
        self.quality_chain = LLMChain(llm=self.llm, prompt=QUALITY_CHECK_PROMPT)
        self.improvement_chain = LLMChain(llm=self.llm, prompt=IMPROVEMENT_PROMPT)

    async def generate_research(self, query: str, context: Optional[str] = None) -> str:
        try:
            response = await self.research_chain.arun(
                query=query,
                context=context or ""
            )
            return response.strip()
        except Exception as e:
            raise LLMServiceException(f"Error generating research: {str(e)}")

    async def check_quality(self, text: str) -> Dict[str, float]:
        try:
            response = await self.quality_chain.arun(text=text)
            # Parse the response to extract numerical scores
            scores = self._parse_quality_scores(response)
            return scores
        except Exception as e:
            raise LLMServiceException(f"Error checking quality: {str(e)}")

    async def improve_text(self, text: str, feedback: str) -> str:
        try:
            response = await self.improvement_chain.arun(
                text=text,
                feedback=feedback
            )
            return response.strip()
        except Exception as e:
            raise LLMServiceException(f"Error improving text: {str(e)}")

    def _parse_quality_scores(self, response: str) -> Dict[str, float]:
        """Parse the quality check response to extract numerical scores."""
        try:
            scores = {
                "coherence": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "evidence": 0.0
            }
            # Implement parsing logic based on the response format
            # This is a simplified example
            for line in response.split("\n"):
                if ":" in line:
                    key, value = line.split(":")
                    key = key.strip().lower()
                    if key in scores:
                        try:
                            scores[key] = float(value.strip())
                        except ValueError:
                            pass
            return scores
        except Exception as e:
            raise LLMServiceException(f"Error parsing quality scores: {str(e)}")
