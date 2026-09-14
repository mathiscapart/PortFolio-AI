"""Synthese vocale via Kyutai Pocket TTS.

Retenu contre Qwen3-TTS (architecture absente de `transformers`) et Chatterbox
(3,2 Go de VRAM) : Pocket TTS tourne entierement sur CPU (0 Go de VRAM), a
0,68x temps reel mesure, laissant toute la VRAM au LLM et au STT. Cf. CLAUDE.md
section 4 bis.
"""
import os

import numpy as np
import torch
from pocket_tts import TTSModel

# Voix du catalogue Kyutai, ou chemin vers un etat exporte (.safetensors) pour
# une voix clonee, ex. backend/tts/voix/mathis.safetensors (non versionne).
# Le clonage exige d'avoir accepte les conditions de kyutai/pocket-tts sur HF.
_VOIX = os.getenv("TTS_VOIX", "estelle")
_LANGUE = "french_24l"


def charger_modele() -> TTSModel:
    """Charge le modele une seule fois (au demarrage de l'API, pas par requete) :
    15 s de chargement mesures, inacceptable par requete."""
    return TTSModel.load_model(language=_LANGUE)


def charger_voix(modele: TTSModel):
    """Precalcule l'etat conditionne sur la voix retenue, reutilisable pour
    chaque generation (`generate_audio_stream` en fait une copie par defaut)."""
    return modele.get_state_for_audio_prompt(_VOIX)


def tensor_vers_pcm16(chunk: torch.Tensor) -> bytes:
    """Convertit un chunk audio flottant [-1, 1] de Pocket TTS en PCM 16 bits
    mono, le format attendu par le contrat WebSocket `/voice`."""
    echantillons = chunk.detach().cpu().numpy()
    echantillons = np.clip(echantillons, -1.0, 1.0)
    return (echantillons * 32767.0).astype(np.int16).tobytes()
