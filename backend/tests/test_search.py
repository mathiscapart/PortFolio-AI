"""Tests de `QdrantVectorStore.search` (`backend/rag/main.py`).

`search` ne fait qu'assembler ce que lui renvoie `query_points` : on espionne
ce dernier pour rester déterministe, sans toucher à Qdrant ni Ollama.
"""
from types import SimpleNamespace

from backend.rag.main import QdrantVectorStore


class _EmbeddingModelFactice:
    """Renvoie un vecteur fixe, sans appeler Ollama."""

    def embed(self, text: str) -> list[float]:
        return [0.0, 1.0]


class _ClientEspion:
    """Remplace `QdrantClient` : `query_points` renvoie des points fabriqués,
    triés décroissant par score comme le ferait vraiment Qdrant."""

    def __init__(self, points):
        self._points = points
        self.derniers_kwargs = None

    def query_points(self, collection_name, query, limit):
        self.derniers_kwargs = {"collection_name": collection_name, "query": query, "limit": limit}
        points_tries = sorted(self._points, key=lambda p: p.score, reverse=True)
        return SimpleNamespace(points=points_tries[:limit])


def _point(payload, score):
    return SimpleNamespace(payload=payload, score=score)


def test_search_retourne_le_payload_enrichi_du_score():
    points = [_point({"texte": "chunk unique", "source": "doc.md"}, 0.42)]
    store = object.__new__(QdrantVectorStore)
    store.client = _ClientEspion(points)

    resultats = store.search(
        collection_name="test",
        query="une question",
        embedding_model=_EmbeddingModelFactice(),
        k=5,
    )

    assert resultats == [{"texte": "chunk unique", "source": "doc.md", "score": 0.42}]


def test_search_respecte_k_et_trie_par_score_decroissant():
    points = [
        _point({"texte": "faible"}, 0.1),
        _point({"texte": "fort"}, 0.9),
        _point({"texte": "moyen"}, 0.5),
    ]
    store = object.__new__(QdrantVectorStore)
    store.client = _ClientEspion(points)

    resultats = store.search(
        collection_name="test",
        query="une question",
        embedding_model=_EmbeddingModelFactice(),
        k=2,
    )

    assert [r["texte"] for r in resultats] == ["fort", "moyen"]
    assert [r["score"] for r in resultats] == [0.9, 0.5]
