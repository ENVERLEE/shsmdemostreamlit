import asyncio
from typing import Optional
from services.research.service import ResearchService
from services.research.literature_review import LiteratureReviewService
from services.research.methodology import MethodologyService
from services.research.quality import QualityService
from services.llm.service import LLMService
from services.embedding.service import EmbeddingService
from core.types import ResearchRequest, ResearchStatus
from core.models import ResearchProject
from utils.helpers import save_json_file, ensure_directory
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import json
from datetime import datetime
import sys
import os


class ResearchAssistant:
    def __init__(self):
        self.research_service = ResearchService()
        self.output_dir = Path("output")
        ensure_directory(str(self.output_dir))

    async def process_research_request(
        self,
        query: str,
        context: Optional[str] = None,
        max_iterations: Optional[int] = None,
        quality_threshold: Optional[float] = None
    ):
        """Process a research request and save results."""
        try:
            # Create research request
            request = ResearchRequest(
                query=query,
                context=context,
                max_iterations=max_iterations,
                quality_threshold=quality_threshold
            )

            # Conduct research
            print(f"Starting research for query: {query}")
            research = await self.research_service.conduct_research(request)

            # Enhance with embeddings if research was successful
            if research.status == ResearchStatus.COMPLETED:
                research = await self.research_service.enhance_with_embeddings(research)

            # Save results
            output_file = self.output_dir / f"research_{research.id}.json"
            save_json_file(
                research.dict(),
                str(output_file)
            )

            print(f"Research completed. Results saved to: {output_file}")
            print(f"Status: {research.status}")
            print(f"Confidence Score: {research.confidence_score}")
            print(f"Quality Score: {research.quality_score}")
            
            return research

        except Exception as e:
            print(f"Error processing research request: {str(e)}")
            raise

async def get_user_input() -> tuple[str, Optional[str]]:
    """Get query and optional context from user input"""
    print("\n=== Research Assistant CLI ===")
    print("Enter your research query (or 'quit' to exit):")
    query = input("> ").strip()
    
    if query.lower() == 'quit':
        sys.exit(0)
    
    print("\nEnter additional context (optional, press Enter to skip):")
    context = input("> ").strip()
    return query, context if context else None

async def process_and_display_results(assistant: ResearchAssistant, query: str, context: Optional[str] = None):
    """Process research request and display results with quality feedback"""
    try:
        print("\nProcessing your research request...")
        research = await assistant.process_research_request(query=query, context=context)
        
        print("\n=== Research Results ===")
        print(f"Query: {research.query}")
        print(f"\nAnalysis:\n{research.result}")
        
        if research.quality_score:
            print("\n=== Quality Assessment ===")
            for criterion, score in research.quality_score.items():
                print(f"{criterion}: {score:.2f}")
        
        if research.improvements:
            print("\n=== Suggested Improvements ===")
            print(research.improvements)
        
        print("\nResearch saved to:", research.output_file)
        
    except Exception as e:
        print(f"\nError processing request: {str(e)}")

async def main():
    if os.environ.get('OPENAI_API_KEY') == None:
        open_ai_api_key = input("OPENAIAPIKEY를 입력해주세요: ")
        os.environ['OPENAI_API_KEY'] = open_ai_api_key
        assistant = ResearchAssistant()
    print("Welcome to Research Assistant CLI!")
    print("This tool will help you conduct research and analysis.")
    
    while True:
        query, context = await get_user_input()
        await process_and_display_results(assistant, query, context)
        print("\n-----------------------------------")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        os.environ.pop('OPENAI_API_KEY', None)
        print("\nExiting Research Assistant. Goodbye!")
