from typing import Dict, Any, List, Optional
from core.models import ResearchProject, TheoreticalFramework
from core.exceptions import ResearchException
from services.llm.service import LLMService
from services.embedding.service import EmbeddingService
import numpy as np
from scipy import stats
import json
from datetime import datetime

class MethodologyService:
    def __init__(self, llm_service: LLMService, embedding_service: EmbeddingService):
        self.llm_service = llm_service
        self.embedding_service = embedding_service

    async def design_theoretical_framework(
        self, 
        project: ResearchProject,
        references: List[Dict[str, Any]]
    ) -> TheoreticalFramework:
        """Design theoretical framework based on literature review"""
        try:
            # Extract key concepts and theories from references
            concepts = await self._extract_key_concepts(references)
            
            # Analyze relationships between concepts
            relationships = await self._analyze_concept_relationships(concepts, references)
            
            # Generate framework structure
            framework_prompt = f"""
            Based on these key concepts and their relationships:
            {json.dumps(relationships, indent=2)}
            
            Design a comprehensive theoretical framework that:
            1. Explains causal and correlational relationships between concepts
            2. Identifies dependent, independent, and mediating variables
            3. Specifies testable hypotheses
            4. Aligns with research objectives
            5. Integrates existing theories from the literature
            
            Research Question: {project.query}
            
            Format the response as JSON with:
            1. Framework description
            2. Key variables and their roles
            3. Hypotheses
            4. Theoretical foundations
            5. Measurement approach
            """
            
            framework_response = await self.llm_service.generate_research(framework_prompt)
            framework_data = json.loads(framework_response)
            
            # Create embeddings for framework components
            framework_text = f"""
            Description: {framework_data['description']}
            Variables: {json.dumps(framework_data['variables'])}
            Hypotheses: {json.dumps(framework_data['hypotheses'])}
            Foundations: {json.dumps(framework_data['theoretical_foundations'])}
            """
            framework_embedding = await self.embedding_service.create_embeddings(framework_text)
            
            return TheoreticalFramework(
                description=framework_data['description'],
                concepts=concepts,
                variables=framework_data['variables'],
                hypotheses=framework_data['hypotheses'],
                theoretical_foundations=framework_data['theoretical_foundations'],
                measurement_approach=framework_data['measurement_approach'],
                embedding=framework_embedding[0].vector
            )
            
        except Exception as e:
            raise ResearchException(f"Error designing theoretical framework: {str(e)}")

    async def evaluate_methodology(
        self,
        project: ResearchProject,
        sample_size: Optional[int] = None,
        effect_size: Optional[float] = None,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> Dict[str, Any]:
        """Evaluate methodology quality and calculate statistical power"""
        try:
            evaluation = {}
            
            # Statistical Power Analysis
            if sample_size and effect_size:
                power_analysis = self._calculate_power_analysis(
                    sample_size=sample_size,
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power
                )
                evaluation["power_analysis"] = power_analysis
            
            # Research Design Evaluation
            design_prompt = f"""
            Evaluate this research methodology:
            {project.methodology_description}
            
            Consider:
            1. Research Design
               - Appropriateness for research question
               - Control of confounding variables
               - Sampling strategy
               - Timeline feasibility
            
            2. Data Collection
               - Methods appropriateness
               - Measurement tools
               - Quality control procedures
               - Data validation approach
            
            3. Analysis Plan
               - Statistical methods
               - Data processing steps
               - Software and tools
               - Validation techniques
            
            4. Ethical Considerations
               - Participant protection
               - Data privacy
               - Informed consent
               - Risk mitigation
            
            Format response as JSON with scores (0-1) and detailed justifications.
            """
            
            design_response = await self.llm_service.generate_research(design_prompt)
            design_evaluation = json.loads(design_response)
            evaluation["design_evaluation"] = design_evaluation
            
            # Generate Improvement Suggestions
            suggestions_prompt = f"""
            Based on the methodology evaluation:
            {json.dumps(design_evaluation, indent=2)}
            
            And power analysis:
            {json.dumps(power_analysis, indent=2) if 'power_analysis' in evaluation else 'Not available'}
            
            Provide specific suggestions for:
            1. Research design improvements
            2. Data collection enhancements
            3. Analysis method refinements
            4. Quality control measures
            5. Timeline optimization
            
            Format suggestions as JSON with:
            - Category
            - Specific improvements
            - Implementation steps
            - Required resources
            - Expected impact (1-5)
            - Priority level (1-5)
            """
            
            suggestions = await self.llm_service.generate_research(suggestions_prompt)
            evaluation["improvement_suggestions"] = json.loads(suggestions)
            
            return evaluation
            
        except Exception as e:
            raise ResearchException(f"Error evaluating methodology: {str(e)}")

    def _calculate_power_analysis(
        self,
        sample_size: int,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> Dict[str, Any]:
        """Calculate statistical power analysis"""
        try:
            # Calculate required sample size for desired power
            required_n = stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power)
            required_n = int(np.ceil((required_n / effect_size) ** 2))
            
            # Calculate actual power with given sample size
            actual_power = stats.norm.cdf(
                effect_size * np.sqrt(sample_size) - stats.norm.ppf(1 - alpha/2)
            )
            
            # Calculate minimum detectable effect size
            min_effect_size = (stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power)) / np.sqrt(sample_size)
            
            return {
                "required_sample_size": required_n,
                "actual_sample_size": sample_size,
                "effect_size": effect_size,
                "min_detectable_effect": float(min_effect_size),
                "alpha": alpha,
                "target_power": power,
                "actual_power": float(actual_power),
                "is_sufficient": sample_size >= required_n,
                "recommendation": self._generate_power_recommendation(
                    sample_size,
                    required_n,
                    actual_power,
                    power
                )
            }
            
        except Exception as e:
            raise ResearchException(f"Error in power analysis: {str(e)}")

    def _generate_power_recommendation(
        self,
        actual_n: int,
        required_n: int,
        actual_power: float,
        target_power: float
    ) -> str:
        """Generate recommendation based on power analysis"""
        if actual_n >= required_n:
            if actual_power >= target_power + 0.1:
                return "Sample size is more than adequate. Consider reducing to save resources."
            else:
                return "Sample size is sufficient for the desired power."
        else:
            deficit = required_n - actual_n
            percent_increase = (deficit / actual_n) * 100
            return f"Increase sample size by {deficit} ({percent_increase:.1f}%) to achieve desired power."

    async def _extract_key_concepts(
        self,
        references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract key concepts from references"""
        try:
            # Prepare reference texts
            reference_texts = []
            for ref in references:
                text = f"""
                Title: {ref['title']}
                Abstract: {ref['abstract']}
                Key Findings: {ref.get('key_findings', 'Not available')}
                Methodology: {ref.get('methodology', 'Not available')}
                """
                reference_texts.append(text)
            
            # Generate concept extraction prompt
            extraction_prompt = f"""
            Extract key theoretical concepts from these research papers:
            {json.dumps(reference_texts, indent=2)}
            
            For each concept provide:
            1. Name
            2. Definition
            3. Related theories
            4. Measurement approaches
            5. Key relationships
            
            Format response as JSON array of concept objects.
            """
            
            concepts_response = await self.llm_service.generate_research(extraction_prompt)
            concepts = json.loads(concepts_response)
            
            # Create embeddings for concepts
            for concept in concepts:
                concept_text = f"""
                Name: {concept['name']}
                Definition: {concept['definition']}
                Theories: {', '.join(concept['related_theories'])}
                """
                concept_embedding = await self.embedding_service.create_embeddings(concept_text)
                concept['embedding'] = concept_embedding[0].vector
            
            return concepts
            
        except Exception as e:
            raise ResearchException(f"Error extracting concepts: {str(e)}")

    async def _analyze_concept_relationships(
        self,
        concepts: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze relationships between concepts"""
        try:
            relationships = []
            
            # Calculate similarity matrix
            n_concepts = len(concepts)
            similarity_matrix = np.zeros((n_concepts, n_concepts))
            
            for i in range(n_concepts):
                for j in range(i + 1, n_concepts):
                    similarity = np.dot(
                        concepts[i]['embedding'],
                        concepts[j]['embedding']
                    )
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
            
            # Generate relationship analysis prompt
            relationship_prompt = f"""
            Analyze relationships between these concepts:
            {json.dumps([{
                'name': c['name'],
                'definition': c['definition']
            } for c in concepts], indent=2)}
            
            Similarity Matrix:
            {similarity_matrix.tolist()}
            
            For each related concept pair, provide:
            1. Type of relationship (causal, correlational, mediating, etc.)
            2. Direction of relationship
            3. Strength of relationship
            4. Supporting evidence from literature
            5. Potential moderating variables
            
            Format response as JSON array of relationship objects.
            """
            
            relationships_response = await self.llm_service.generate_research(relationship_prompt)
            relationships = json.loads(relationships_response)
            
            # Sort relationships by strength
            relationships.sort(key=lambda x: x['strength'], reverse=True)
            
            return relationships
            
        except Exception as e:
            raise ResearchException(f"Error analyzing concept relationships: {str(e)}")

    async def validate_theoretical_framework(
        self,
        framework: TheoreticalFramework,
        references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate theoretical framework against existing literature"""
        try:
            # Create embeddings for validation
            framework_embedding = framework.embedding
            
            # Calculate similarity with existing frameworks
            similarities = []
            for ref in references:
                if "theoretical_framework" in ref:
                    ref_embedding = await self.embedding_service.create_embeddings(
                        ref["theoretical_framework"]
                    )
                    similarity = np.dot(framework_embedding, ref_embedding[0].vector)
                    similarities.append({
                        "reference": ref["title"],
                        "similarity": float(similarity)
                    })
            
            # Generate validation prompt
            validation_prompt = f"""
            Validate this theoretical framework:
            {framework.description}
            
            Variables:
            {json.dumps(framework.variables, indent=2)}
            
            Hypotheses:
            {json.dumps(framework.hypotheses, indent=2)}
            
            Similar Frameworks:
            {json.dumps(similarities, indent=2)}
            
            Evaluate:
            1. Theoretical foundations
            2. Construct validity
            3. Nomological validity
            4. Face validity
            5. Content validity
            
            Consider:
            1. Alignment with existing theories
            2. Internal consistency
            3. Explanatory power
            4. Testability of hypotheses
            5. Practical applicability
            
            Format response as JSON with scores and detailed justifications.
            """
            
            validation_response = await self.llm_service.generate_research(validation_prompt)
            validation_results = json.loads(validation_response)
            
            # Generate improvement suggestions if needed
            if any(score < 0.8 for score in validation_results['scores'].values()):
                suggestions_prompt = f"""
                Based on the validation results:
                {json.dumps(validation_results, indent=2)}
                
                Provide specific suggestions to improve:
                1. Theoretical foundations
                2. Construct relationships
                3. Hypothesis formulation
                4. Measurement approach
                5. Overall coherence
                
                Format suggestions as JSON with:
                - Category
                - Current issues
                - Specific improvements
                - Implementation steps
                - Expected impact
                """
                
                suggestions = await self.llm_service.generate_research(suggestions_prompt)
                validation_results['improvement_suggestions'] = json.loads(suggestions)
            
            return validation_results
            
        except Exception as e:
            raise ResearchException(f"Error validating framework: {str(e)}")
