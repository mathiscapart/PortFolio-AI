from ollama import Client 
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
from dotenv import load_dotenv

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            load_dotenv()
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        self.ollama_port = int(os.getenv("OLLAMA_PORT", 11434))
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "portfolio")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")

class EmbeddingModel:
    def __init__(self, model_name: str = "qwen3-embedding:0.6b", host: str = "localhost", port: int = 11434):
        self.model_name = model_name
        self.client = Client(host=f"{host}:{port}")
        self._dim = None

    def embed(self, text: str) -> list[float]:
        response = self.client.embed(input=text, model=self.model_name)
        return response.embeddings[0]
        
    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            response = self.client.embed(model=self.model_name, input="probe")
            self._dim = len(response.embeddings[0])
        return self._dim

class QdrantVectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)

    def add_embedding(self, collection_name: str, embeddings: list[str], payload: dict, embedding_model: EmbeddingModel):
        self.client.upsert(
                collection_name=collection_name,
                points=[
                PointStruct(
                        id=idx,
                        vector=embedding_model.embed(embedding),
                        payload=payload
                )
                for idx, embedding in enumerate(embeddings)
            ]
        )
    
    def create_collection(self, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )

def main():
    settings = Settings()
    qdrantClient = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)
    embedding_model = EmbeddingModel(model_name=settings.embedding_model, host=settings.ollama_host, port=settings.ollama_port)
    qdrantClient.create_collection(collection_name=settings.qdrant_collection, vector_size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
    try: 
        qdrantClient.add_embedding(collection_name=settings.qdrant_collection, embeddings=["This is a sample text to be embedded."], payload={"id": "text"}, embedding_model=embedding_model)
    except Exception as e:
        print(f"Error adding embedding: {e}")

if __name__ == "__main__":
    main()
