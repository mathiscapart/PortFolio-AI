import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class EmbeddingModel:
    def __init__(self, model_name: str = "qwen3-embedding:0.6b"):
        self.model_name = model_name
        self._dim = None

    def embed(self, text: str) -> list[float]:
        response = ollama.embed(input=text, model=self.model_name)
        return response.embeddings[0]
        
    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            response = ollama.embed(model=self.model_name, input="probe")
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
    qdrantClient = QdrantVectorStore(host="localhost", port=6333)
    embedding_model = EmbeddingModel()
    qdrantClient.create_collection(collection_name="sample_collection", vector_size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
    try: 
        qdrantClient.add_embedding(collection_name="sample_collection", embeddings=["This is a sample text to be embedded."], payload={"id": "text"}, embedding_model=embedding_model)
    except Exception as e:
        print(f"Error adding embedding: {e}")

if __name__ == "__main__":
    main()
