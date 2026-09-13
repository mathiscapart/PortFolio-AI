"""API FastAPI du portfolio : health check et chat RAG en streaming (SSE).

`/chat` est en `def`, jamais `async def` : `search()` embarque un appel HTTP
bloquant vers Ollama (embedding forcé sur CPU). En `def`, FastAPI l'exécute
dans son threadpool ; en `async def`, chaque recherche gèlerait tous les flux
SSE en cours.

Les erreurs renvoyées au client sont génériques : le détail part dans les logs,
pas dans la réponse d'un visiteur anonyme.
"""
import json
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ollama import Client
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

try:  # importe comme paquet
    from backend.rag.main import EmbeddingModel, QdrantVectorStore, Settings
    from backend.api.prompts import SYSTEME, construire_prompt_utilisateur
except ImportError:  # exécution à plat
    from rag.main import EmbeddingModel, QdrantVectorStore, Settings
    from api.prompts import SYSTEME, construire_prompt_utilisateur

logger = logging.getLogger(__name__)

# Sans timeout, un Ollama qui accepte la connexion sans répondre fait pendre la
# requête indéfiniment et immobilise un thread du threadpool.
TIMEOUT_OLLAMA = int(os.getenv("OLLAMA_TIMEOUT", "60"))

app = FastAPI(title="PortFolio-AI API")

# Sans CORS, le front de l'étape 6 échoue au préflight. Jamais "*" : l'endpoint
# sera public derrière cloudflared.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    # Embedding sur CPU : sans borne haute, un corps de plusieurs Mo sature la
    # file Ollama et le threadpool.
    message: str = Field(min_length=1, max_length=2000)
    # 8 et non 5 : mesure sur le corpus, le chunk portant la reponse a
    # "ou a-t-elle fait ses etudes" arrive 6e. A k=5 il tombait hors
    # contexte et l'assistant refusait -- correctement, mais a tort.
    k: int = Field(default=8, ge=1, le=20)

    @field_validator("message")
    @classmethod
    def _non_vide(cls, valeur: str) -> str:
        if not valeur.strip():
            raise ValueError("le message ne peut pas être vide")
        return valeur


def _ollama(settings) -> Client:
    return Client(host=f"{settings.ollama_host}:{settings.ollama_port}", timeout=TIMEOUT_OLLAMA)


@app.get("/health")
def health():
    settings = Settings()
    problemes = []
    try:
        QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port).client.get_collections()
    except Exception as exc:
        logger.warning("qdrant injoignable : %s", exc)
        problemes.append("qdrant injoignable")
    try:
        _ollama(settings).list()
    except Exception as exc:
        logger.warning("ollama injoignable : %s", exc)
        problemes.append("ollama injoignable")
    if problemes:
        raise HTTPException(status_code=503, detail=problemes)
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatRequest):
    settings = Settings()
    qdrant_store = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)
    embedding_model = EmbeddingModel(
        model_name=settings.embedding_model, host=settings.ollama_host, port=settings.ollama_port
    )

    # Dans le try : sinon un Qdrant arrêté donne un 500 opaque. Et vérifié avant
    # d'ouvrir le flux, car après les en-têtes le statut ne peut plus changer.
    try:
        presente = qdrant_store.client.collection_exists(settings.qdrant_collection)
    except Exception as exc:
        logger.error("qdrant injoignable (collection_exists) : %s", exc)
        raise HTTPException(status_code=503, detail="Base vectorielle injoignable.") from exc
    if not presente:
        logger.error("collection '%s' absente", settings.qdrant_collection)
        raise HTTPException(status_code=503, detail="Corpus non indexé.")

    # Aucun seuil de score : mesuré, les populations se chevauchent (question
    # légitime à 0,296, hors sujet à 0,329). Aucune valeur ne sépare, donc
    # aucune n'est posée. Le refus tient par SYSTEME et le délimiteur.
    try:
        chunks = qdrant_store.search(
            collection_name=settings.qdrant_collection,
            query=body.message,
            embedding_model=embedding_model,
            k=body.k,
        )
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        logger.error("qdrant en erreur pendant la recherche : %s", exc)
        raise HTTPException(status_code=502, detail="Base vectorielle en erreur.") from exc
    except Exception as exc:
        logger.error("ollama en erreur pendant l'embedding : %s", exc)
        raise HTTPException(status_code=502, detail="Service d'embedding injoignable.") from exc

    messages = [
        {"role": "system", "content": SYSTEME},
        {"role": "user", "content": construire_prompt_utilisateur(body.message, chunks)},
    ]
    ollama_client = _ollama(settings)

    def flux():
        # Un octet avant l'appel Ollama : deux visiteurs simultanés sont
        # sérialisés (un seul slot GPU), et un silence de plus de ~100 s fait
        # couper le tunnel Cloudflare en 524.
        yield ": ping\n\n"
        try:
            for part in ollama_client.chat(model=settings.chat_model, messages=messages, stream=True):
                token = part.message.content
                if token:
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("ollama en erreur pendant la generation : %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': 'generation interrompue'}, ensure_ascii=False)}\n\n"
            return
        sources = [
            {"source": c["source"], "titre": c.get("titre"), "score": c["score"]} for c in chunks
        ]
        yield f"event: sources\ndata: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        flux(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
