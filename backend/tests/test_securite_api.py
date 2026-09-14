"""Garde-fous de sécurité de l'API exposée publiquement."""
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.api import main as api_main

client = TestClient(api_main.app)


@pytest.mark.parametrize("chemin", ["/docs", "/redoc", "/openapi.json"])
def test_la_documentation_interactive_n_est_pas_exposee(chemin):
    assert client.get(chemin).status_code == 404


def test_voice_refuse_une_origine_etrangere():
    """CORS ne couvre pas les WebSockets : un site tiers ne doit pas pouvoir
    ouvrir /voice depuis le navigateur de ses visiteurs."""
    with pytest.raises(WebSocketDisconnect) as erreur:
        with client.websocket_connect("/voice", headers={"origin": "https://site-malveillant.example"}) as ws:
            ws.receive_json()
    assert erreur.value.code == 1008


def test_voice_ferme_proprement_la_connexion_en_fin_de_session(monkeypatch):
    """Sans trame de fermeture, Safari iOS voit une coupure anormale et émet
    un événement "error" tardif qui cassait la question suivante."""
    async def session_vide(websocket):
        await websocket.send_json({"type": "error", "message": "Aucune parole detectee."})

    monkeypatch.setattr(api_main, "_gerer_session_vocale", session_vide)
    monkeypatch.setattr(api_main.app.state, "session_vocale_active", False, raising=False)
    with client.websocket_connect("/voice", headers={"origin": "http://localhost:3000"}) as ws:
        ws.receive_json()
        message = ws.receive()
    assert message["type"] == "websocket.close"
    assert message["code"] == 1000
