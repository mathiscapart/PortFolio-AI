"""API FastAPI du portfolio : health check et chat RAG en streaming (SSE).

`/chat` est en `def`, jamais `async def` : `search()` embarque un appel HTTP
bloquant vers Ollama (embedding forcé sur CPU). En `def`, FastAPI l'exécute
dans son threadpool ; en `async def`, chaque recherche gèlerait tous les flux
SSE en cours.

Les erreurs renvoyées au client sont génériques : le détail part dans les logs,
pas dans la réponse d'un visiteur anonyme.
"""
import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ollama import Client
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from starlette.concurrency import run_in_threadpool

# Triton est absent des wheels ROCm Windows : le STT (Moshi) passe par
# torch.compile et leve TritonMissing sans repli en mode eager. Doit etre
# positionne avant l'import de backend.stt.model (cf. CLAUDE.md 4 bis).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import httpx

# torch et le STT (moshi) ne sont importes qu'au demarrage et dans /voice :
# l'API reste importable sans la pile GPU, ce qui permet de tester /chat et
# /health en CI sur un runner standard.
try:  # importe comme paquet
    from backend.rag.main import EmbeddingModel, QdrantVectorStore, Settings
    from backend.api.prompts import CONSIGNE_ORALE, SYSTEME, construire_prompt_utilisateur
except ImportError:  # exécution à plat
    from rag.main import EmbeddingModel, QdrantVectorStore, Settings
    from api.prompts import CONSIGNE_ORALE, SYSTEME, construire_prompt_utilisateur

logger = logging.getLogger(__name__)

# Sans timeout, un Ollama qui accepte la connexion sans répondre fait pendre la
# requête indéfiniment et immobilise un thread du threadpool.
TIMEOUT_OLLAMA = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# Sans num_ctx, Ollama prend 32768 : qwen3:8b monte a 9,16 Go et deborde de la
# VRAM des que le STT (2,5 Go) est charge -> 10,5 t/s. A 8192 : 5,86 Go, 54,8 t/s
# (mesure). 8 extraits de 600 tokens tiennent largement.
OPTIONS_LLM = {"num_ctx": 8192}

# Service TTS separe (backend/tts/server.py, `.venv-tts`) : Pocket TTS est 40x
# plus lent sous le torch ROCm de ce process que sous torch CPU.
TTS_URL = os.getenv("TTS_URL", "http://127.0.0.1:8001")

# Bornes d'une question vocale : 30 s d'audio (trames de 80 ms), 10 s sans
# aucun message du client.
TRAMES_MAX_QUESTION = 375
INACTIVITE_MAX_S = 10


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Charge une seule fois : plusieurs secondes pour le checkpoint Kyutai.
    # Jamais par requete, sous peine de payer ce cout a chaque session vocale.
    try:
        from backend.stt.model import charger_state
    except ImportError:  # exécution à plat
        from stt.model import charger_state
    app.state.stt_state, app.state.stt_trames_purge = charger_state()
    # Un seul GPU : une seconde session vocale doit etre refusee, pas mise en
    # attente. Pas besoin de verrou thread-safe : la boucle asyncio est
    # mono-thread et rien n'attend entre la lecture et l'ecriture de ce champ.
    app.state.session_vocale_active = False
    yield


# Pas de /docs ni /openapi.json : endpoint public, inutile d'y cartographier la surface.
app = FastAPI(
    title="PortFolio-AI API", lifespan=_lifespan, docs_url=None, redoc_url=None, openapi_url=None
)

ORIGINES_AUTORISEES = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]

# Sans CORS, le front de l'étape 6 échoue au préflight. Jamais "*" : l'endpoint
# sera public derrière cloudflared.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINES_AUTORISEES,
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
            for part in ollama_client.chat(
                model=settings.chat_model, messages=messages, stream=True, options=OPTIONS_LLM
            ):
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


# --- /voice : boucle vocale WebSocket -----------------------------------

_FIN_ITERATION = object()

# Coupe le texte du LLM aux frontieres de phrase pour lancer la synthese vocale
# de la premiere phrase des qu'elle est complete, sans attendre la reponse
# entiere : c'est ce qui tient le delai avant la premiere trame audio.
_FIN_PHRASE = re.compile(r"[.!?](?:\s|$)")


def _suivant(iterateur):
    """Avance un iterateur synchrone d'un cran. A appeler via
    `run_in_threadpool` : `next()` sur un flux Ollama est
    bloquant, et un WebSocket FastAPI est forcement `async def`."""
    try:
        return next(iterateur)
    except StopIteration:
        return _FIN_ITERATION


def _pcm16_vers_tensor(donnees: bytes, device) -> "torch.Tensor":
    """Convertit une trame PCM 16 bits mono recue du client en tenseur flottant
    [-1, 1] de forme (1, 1, echantillons), attendue par `InferenceState.step()`."""
    import torch

    entiers = torch.frombuffer(bytearray(donnees), dtype=torch.int16)
    flottants = entiers.to(device=device, dtype=torch.float32) / 32768.0
    return flottants[None, None, :]


def _extraire_phrases_completes(tampon: str) -> tuple[list[str], str]:
    """Coupe `tampon` en phrases terminees par un signe de ponctuation fort ;
    renvoie les phrases completes (a synthetiser tout de suite) et le reste
    (phrase en cours, incomplete)."""
    phrases = []
    reste = tampon
    while True:
        correspondance = _FIN_PHRASE.search(reste)
        if not correspondance:
            break
        fin = correspondance.end()
        phrases.append(reste[:fin].strip())
        reste = reste[fin:]
    return phrases, reste


async def _synthetiser_et_envoyer(websocket: WebSocket, phrase: str):
    """Genere l'audio d'une phrase et emet chaque trame des qu'elle est prete,
    sans attendre que la phrase entiere soit synthetisee."""
    reste = b""
    async with httpx.AsyncClient(base_url=TTS_URL, timeout=TIMEOUT_OLLAMA) as client:
        async with client.stream("POST", "/synthese", json={"texte": phrase}) as reponse:
            reponse.raise_for_status()
            async for bloc in reponse.aiter_bytes():
                # TCP peut couper au milieu d'un echantillon : le front lit des
                # Int16Array, qui exigent une longueur paire.
                bloc = reste + bloc
                coupe = len(bloc) - len(bloc) % 2
                reste = bloc[coupe:]
                if coupe:
                    await websocket.send_bytes(bloc[:coupe])


async def _gerer_session_vocale(websocket: WebSocket):
    settings = Settings()
    stt_state = app.state.stt_state
    await run_in_threadpool(stt_state.reinitialiser)

    # --- STT : transcription en direct, trame par trame (80 ms) ---
    # Une seule session a la fois : sans ces bornes, un socket ouvert et muet
    # (ou un visiteur qui ne clique jamais "Termine") bloquerait la voix pour
    # tout le monde, indefiniment.
    morceaux = []
    trames_recues = 0
    while trames_recues < TRAMES_MAX_QUESTION:
        try:
            message = await asyncio.wait_for(websocket.receive(), timeout=INACTIVITE_MAX_S)
        except asyncio.TimeoutError:
            await websocket.send_json({"type": "error", "message": "Session expirée."})
            return
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect()
        if message.get("bytes") is not None:
            # Mimi exige exactement frame_size echantillons : une trame hors
            # format ferait lever encode() et tuerait la session sans message.
            if len(message["bytes"]) != stt_state.frame_size * 2:
                continue
            trames_recues += 1
            trame = _pcm16_vers_tensor(message["bytes"], stt_state.device)
            texte = await run_in_threadpool(stt_state.step, trame)
            if texte:
                morceaux.append(texte)
                await websocket.send_json({"type": "transcript", "text": texte})
        elif message.get("text") is not None:
            try:
                charge = json.loads(message["text"])
            except json.JSONDecodeError:
                continue
            if charge.get("type") == "end":
                break

    # Purge le delai interne du modele (silence) : sans cette purge, les
    # derniers mots du visiteur restent bloques dans le pipeline et ne sont
    # jamais transcrits (cf. `audio_delay_seconds`, CLAUDE.md 4 bis).
    import torch

    silence = torch.zeros((1, 1, stt_state.frame_size), device=stt_state.device)
    for _ in range(app.state.stt_trames_purge):
        texte = await run_in_threadpool(stt_state.step, silence)
        if texte:
            morceaux.append(texte)
            await websocket.send_json({"type": "transcript", "text": texte})

    question = "".join(morceaux).strip()
    if not question:
        await websocket.send_json({"type": "error", "message": "Aucune parole detectee."})
        return

    # --- RAG : memes garde-fous que /chat ---
    qdrant_store = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)
    embedding_model = EmbeddingModel(
        model_name=settings.embedding_model, host=settings.ollama_host, port=settings.ollama_port
    )
    try:
        presente = await run_in_threadpool(qdrant_store.client.collection_exists, settings.qdrant_collection)
    except Exception as exc:
        logger.error("qdrant injoignable (collection_exists) : %s", exc)
        await websocket.send_json({"type": "error", "message": "Base vectorielle injoignable."})
        return
    if not presente:
        logger.error("collection '%s' absente", settings.qdrant_collection)
        await websocket.send_json({"type": "error", "message": "Corpus non indexé."})
        return

    try:
        chunks = await run_in_threadpool(
            qdrant_store.search, settings.qdrant_collection, question, embedding_model, 8
        )
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        logger.error("qdrant en erreur pendant la recherche : %s", exc)
        await websocket.send_json({"type": "error", "message": "Base vectorielle en erreur."})
        return
    except Exception as exc:
        logger.error("ollama en erreur pendant l'embedding : %s", exc)
        await websocket.send_json({"type": "error", "message": "Service d'embedding injoignable."})
        return

    messages = [
        {"role": "system", "content": SYSTEME + CONSIGNE_ORALE},
        {"role": "user", "content": construire_prompt_utilisateur(question, chunks)},
    ]
    ollama_client = _ollama(settings)

    # --- LLM en streaming, TTS phrase par phrase des qu'elle est complete ---
    # think=False : le raisonnement invisible de qwen3 coutait 9,2 s de silence
    # avant le premier token (mesure), inacceptable a l'oral.
    tampon = ""
    try:
        iterateur = ollama_client.chat(
            model=settings.chat_model, messages=messages, stream=True, think=False, options=OPTIONS_LLM
        )
        while True:
            part = await run_in_threadpool(_suivant, iterateur)
            if part is _FIN_ITERATION:
                break
            token = part.message.content
            if not token:
                continue
            await websocket.send_json({"type": "token", "text": token})
            tampon += token
            phrases, tampon = _extraire_phrases_completes(tampon)
            for phrase in phrases:
                await _synthetiser_et_envoyer(websocket, phrase)
        tampon = tampon.strip()
        if tampon:
            await _synthetiser_et_envoyer(websocket, tampon)
    except WebSocketDisconnect:
        raise  # visiteur parti : rien a lui signaler
    except Exception as exc:
        logger.error("generation ou synthese en erreur : %s", exc)
        await websocket.send_json({"type": "error", "message": "generation interrompue"})
        return

    sources = [{"source": c["source"], "titre": c.get("titre"), "score": c["score"]} for c in chunks]
    await websocket.send_json({"type": "sources", "sources": sources})


@app.websocket("/voice")
async def voice(websocket: WebSocket):
    """Boucle vocale complete : STT en direct -> RAG -> LLM en streaming ->
    TTS en streaming. Contrat detaille dans l'assignation de la tache, non
    reproduit ici.

    Une seule session a la fois : un seul GPU. Une session concurrente est
    refusee explicitement, jamais mise en attente ni traitee en parallele.
    Chaque appel bloquant (STT, embedding, Qdrant, iteration Ollama, TTS) part
    dans le threadpool via `run_in_threadpool` : sans ca, le premier visiteur
    gèlerait tous les autres flux, `/chat` compris.
    """
    # CORS ne s'applique pas aux WebSockets : sans ce controle, n'importe quel
    # site pourrait ouvrir /voice depuis le navigateur de ses visiteurs et
    # monopoliser la session unique. Un navigateur envoie toujours Origin ; un
    # client hors navigateur peut le forger de toute facon, il n'est pas vise.
    origine = websocket.headers.get("origin")
    if origine is not None and origine not in ORIGINES_AUTORISEES:
        await websocket.close(code=1008)
        return

    if app.state.session_vocale_active:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Une session vocale est déjà en cours."})
        await websocket.close()
        return

    app.state.session_vocale_active = True
    await websocket.accept()
    try:
        await _gerer_session_vocale(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        app.state.session_vocale_active = False
