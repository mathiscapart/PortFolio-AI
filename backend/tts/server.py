"""Service TTS, process separe dans `.venv-tts` (torch CPU).

Pocket TTS ne peut pas tourner dans le process de l'API : sous le torch ROCm de
`.venv-rocm`, il mesure 27,95x temps reel contre 0,70x sous torch CPU (meme
phrase, meme machine). L'API l'appelle en HTTP et relaie le PCM au fil de l'eau.

Lancement : `.venv-tts\\Scripts\\python -m uvicorn backend.tts.server:app --host 127.0.0.1 --port 8001`
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.tts.model import charger_modele, charger_voix, tensor_vers_pcm16


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.modele = charger_modele()
    app.state.voix = charger_voix(app.state.modele)
    yield


app = FastAPI(title="PortFolio-AI TTS", lifespan=_lifespan)


class RequeteSynthese(BaseModel):
    texte: str = Field(min_length=1, max_length=2000)


@app.post("/synthese")
def synthese(requete: RequeteSynthese):
    """PCM 16 bits mono 24 kHz, emis chunk par chunk des qu'il est genere.
    Generateur synchrone : Starlette l'itere dans son threadpool."""
    def flux():
        for chunk in app.state.modele.generate_audio_stream(app.state.voix, requete.texte):
            yield tensor_vers_pcm16(chunk)

    return StreamingResponse(flux(), media_type="application/octet-stream")
