from typing import List, Dict, Any, Optional
from voyageai import get_embedding, get_embeddings  # VoyageAI SDK 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.exceptions import EmbeddingServiceException
from core.models import EmbeddingData
from config.settings import EMBEDDING_CONFIG

class EmbeddingService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or EMBEDDING_CONFIG
        # VoyageAI API 키는 환경 변수나 설정에서 자동으로 로드됨
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )

    async def create_embeddings(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[EmbeddingData]:
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # VoyageAI를 사용하여 임베딩 생성
            embeddings = await get_embeddings(
                chunks,
                model=self.config["model_name"]
            )
            
            # Create EmbeddingData objects
            embedding_data = []
            for chunk, vector in zip(chunks, embeddings):
                data = EmbeddingData(
                    text=chunk,
                    vector=vector,
                    metadata=metadata or {}
                )
                embedding_data.append(data)
            return embedding_data
            
        except Exception as e:
            raise EmbeddingServiceException(f"Error creating embeddings: {str(e)}")

    async def similarity_search(
        self,
        query: str,
        embedding_data: List[EmbeddingData],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            # VoyageAI를 사용하여 쿼리 임베딩 생성
            query_embedding = await get_embedding(
                query,
                model=self.config["model_name"]
            )
            
            # Calculate similarity scores
            results = []
            for data in embedding_data:
                similarity = self._calculate_similarity(query_embedding, data.vector)
                results.append({
                    "text": data.text,
                    "similarity": similarity,
                    "metadata": data.metadata
                })
                
            # Sort by similarity and return top k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            raise EmbeddingServiceException(f"Error performing similarity search: {str(e)}")

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm1 * norm2)
        except Exception as e:
            raise EmbeddingServiceException(f"Error calculating similarity: {str(e)}")