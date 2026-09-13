"""CLI d'interrogation et d'ingestion du RAG.

Usage :
    python -m backend.rag.cli "ma question"          -> affiche les k meilleurs chunks
    python -m backend.rag.cli --ingest backend/rag/corpus  -> ingère un répertoire
"""
import argparse

try:  # importe comme paquet (API de l'etape 5, tests)
    from backend.rag.main import EmbeddingModel, QdrantVectorStore, Settings, ingest_directory
except ImportError:  # execution a plat dans le conteneur (WORKDIR /app)
    from main import EmbeddingModel, QdrantVectorStore, Settings, ingest_directory


def _rechercher(question: str, k: int) -> None:
    settings = Settings()
    qdrant_store = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)
    embedding_model = EmbeddingModel(model_name=settings.embedding_model, host=settings.ollama_host, port=settings.ollama_port)

    resultats = qdrant_store.search(
        collection_name=settings.qdrant_collection,
        query=question,
        embedding_model=embedding_model,
        k=k,
    )
    for rang, chunk in enumerate(resultats, start=1):
        print(f"[{rang}] score={chunk['score']:.4f} source={chunk['source']} titre={chunk['titre']}")
        print(chunk["texte"])
        print()


def _ingerer(directory: str) -> None:
    settings = Settings()
    qdrant_store = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)
    embedding_model = EmbeddingModel(model_name=settings.embedding_model, host=settings.ollama_host, port=settings.ollama_port)

    qdrant_store.create_collection(
        collection_name=settings.qdrant_collection,
        vector_size=embedding_model.get_sentence_embedding_dimension(),
    )
    fichiers = ingest_directory(directory, qdrant_store, embedding_model, settings.qdrant_collection)
    print(f"{len(fichiers)} fichier(s) ingéré(s) dans la collection '{settings.qdrant_collection}'.")


def main():
    parser = argparse.ArgumentParser(description="Interrogation et ingestion du RAG.")
    parser.add_argument("question", nargs="?", help="Question à poser au RAG.")
    parser.add_argument("--ingest", metavar="DIR", help="Répertoire de fichiers .md à ingérer.")
    parser.add_argument("-k", type=int, default=5, help="Nombre de chunks à retourner (défaut : 5).")
    args = parser.parse_args()

    if args.ingest:
        _ingerer(args.ingest)
    elif args.question:
        _rechercher(args.question, args.k)
    else:
        parser.error("préciser une question ou --ingest <répertoire>.")


if __name__ == "__main__":
    main()
