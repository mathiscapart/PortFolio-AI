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
