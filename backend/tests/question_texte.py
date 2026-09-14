"""Pose une question écrite à l'assistant, hors production.

Même chaîne que `/voice` sans le micro ni la voix : recherche Qdrant, même
prompt système, mêmes options LLM. Sert à vérifier une réponse après une
modification du corpus. Pas un endpoint : rien n'est exposé.

Usage (Qdrant et Ollama lancés) :
    .venv-rocm\\Scripts\\python.exe -m backend.tests.question_texte "Où fait-il ses études ?"
    ... --k 8 --sources
"""
import argparse

from ollama import Client

from backend.api.main import (
    IDENTITE_SOURCE,
    IDENTITE_TITRE,
    K_EXTRAITS,
    OPTIONS_LLM,
    TIMEOUT_OLLAMA,
    ajouter_identite,
)
from backend.api.prompts import CONSIGNE_ORALE, SYSTEME, construire_prompt_utilisateur
from backend.rag.main import EmbeddingModel, QdrantVectorStore, Settings


def main():
    parser = argparse.ArgumentParser(description="Question écrite à l'assistant du portfolio.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=K_EXTRAITS, help=f"extraits transmis (défaut : {K_EXTRAITS}, comme /voice)")
    parser.add_argument("--sources", action="store_true", help="afficher les extraits retenus et leur score")
    args = parser.parse_args()

    settings = Settings()
    store = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)
    embedding = EmbeddingModel(
        model_name=settings.embedding_model, host=settings.ollama_host, port=settings.ollama_port
    )
    chunks = store.search(settings.qdrant_collection, args.question, embedding, args.k)
    chunks = ajouter_identite(chunks, store.get_chunk(settings.qdrant_collection, IDENTITE_SOURCE, IDENTITE_TITRE))

    messages = [
        {"role": "system", "content": SYSTEME + CONSIGNE_ORALE},
        {"role": "user", "content": construire_prompt_utilisateur(args.question, chunks)},
    ]
    ollama = Client(host=f"{settings.ollama_host}:{settings.ollama_port}", timeout=TIMEOUT_OLLAMA)
    for part in ollama.chat(
        model=settings.chat_model, messages=messages, stream=True, think=False, options=OPTIONS_LLM
    ):
        print(part.message.content, end="", flush=True)
    print()

    if args.sources:
        for rang, c in enumerate(chunks, start=1):
            score = "épinglé" if c["score"] is None else f"{c['score']:.3f}"
            print(f"[{rang}] {score} {c['source']} — {c.get('titre')}")


if __name__ == "__main__":
    main()
