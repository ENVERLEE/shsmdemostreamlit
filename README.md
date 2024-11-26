# Research Assistant Project

A comprehensive research assistant that leverages LangChain, LLMs, and embeddings to conduct high-quality research with iterative refinement and quality control.

## Features

- Asynchronous research processing
- Iterative refinement of research results
- Quality control with automatic improvement suggestions
- Embedding-based content enhancement
- Configurable research parameters
- Detailed research metrics and quality scoring

## Project Structure

```
research_project/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration settings
├── core/
│   ├── __init__.py
│   ├── models.py           # Core data models
│   ├── types.py            # Type definitions
│   └── exceptions.py       # Custom exceptions
├── services/
│   ├── __init__.py
│   ├── llm/               # Language model service
│   ├── embedding/         # Embedding service
│   ├── quality/           # Quality control service
│   └── research/          # Main research service
├── utils/
│   ├── __init__.py
│   └── helpers.py         # Utility functions
└── main.py                # Entry point
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the example research:

```bash
python main.py
```

Or use the ResearchAssistant class in your code:

```python
from main import ResearchAssistant
import asyncio

async def run_research():
    assistant = ResearchAssistant()
    research = await assistant.process_research_request(
        query="Your research query",
        context="Additional context",
        max_iterations=3,
        quality_threshold=0.8
    )
    print(research.result)

asyncio.run(run_research())
```

## Configuration

Adjust settings in `config/settings.py`:
- LLM parameters (model, temperature, etc.)
- Embedding parameters (model, chunk size, etc.)
- Quality control thresholds
- Research iteration limits

## Output

Research results are saved in the `output` directory as JSON files, containing:
- Research query and result
- Confidence and quality scores
- Metadata including embeddings
- Iteration history

## Error Handling

The project includes comprehensive error handling with custom exceptions for:
- Research processing errors
- LLM service errors
- Embedding service errors
- Quality control errors
- Configuration errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
