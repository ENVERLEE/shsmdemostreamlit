from typing import Dict, Any
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# LLM Settings
LLM_CONFIG = {
    "model_name": "qwen/qwen-2-7b-instruct:free",
    "temperature": 0.3,
    "max_tokens": 4096,
    "api_key": "place-holder",
}

EMBEDDING_CONFIG = {
    "model_name": "voyage-multilingual-02",
    "chunk_size": 8192,
    "chunk_overlap": 200,
    "api_key": "place-holder"
}

# Research Settings
RESEARCH_CONFIG = {
    "max_iterations": 5,
    "timeout": 300,
    "cache_results": True,
    "quality_threshold": 0.8,
}

# Quality Control Settings
QUALITY_CONFIG = {
    "min_confidence_score": 0.8,
    "validation_threshold": 0.7,
}

# Cache Settings
CACHE_CONFIG = {
    "enabled": True,
    "directory": str(BASE_DIR / "cache"),
    "ttl": 3600,  # Time to live in seconds
}
