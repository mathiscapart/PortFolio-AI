"""Tests de la dérivation des ids Qdrant (`backend/rag/main.py`).

Piège verrouillé ici : l'ancienne implémentation dérivait l'id d'un `enumerate`
local à chaque appel (`id=idx`), si bien qu'une deuxième ingestion écrasait les
points 0..n de la première au lieu de les mettre à jour. La version actuelle
dérive l'id d'un `uuid.uuid5(namespace, f"{source}#{index}")`, stable entre
deux appels et propre à chaque document.

Ces tests ne parlent ni à Qdrant ni à Ollama : `QdrantVectorStore` est
construit sans passer par `__init__` (qui instancierait un vrai `QdrantClient`
et déclencherait un appel réseau de vérification de version), et `client` est
remplacé directement par un espion qui capture les `PointStruct`.
"""
from backend.rag.chunking import chunk_markdown
from backend.rag.main import QdrantVectorStore


class _EmbeddingModelFactice:
    """Renvoie un vecteur fixe, sans appeler Ollama."""

    def embed(self, text: str) -> list[float]:
        return [0.0, 1.0]


class _ClientEspion:
    """Remplace `QdrantClient` : capture les appels `delete`/`upsert`, sans réseau."""

    def __init__(self):
        self.points_captures = []
        self.appels = []  # journal ordonné : ("delete", source) | ("upsert", None)

    def delete(self, collection_name, points_selector):
        source = points_selector.must[0].match.value
        self.appels.append(("delete", source))

    def upsert(self, collection_name, points):
        self.appels.append(("upsert", None))
        self.points_captures.extend(points)


def _ingerer(chunks: list[dict]) -> list:
    """Fait tourner `add_embedding` avec un client Qdrant espionné, retourne les points capturés."""
    store = object.__new__(QdrantVectorStore)
    store.client = _ClientEspion()
    store.add_embedding(
        collection_name="test",
        chunks=chunks,
        embedding_model=_EmbeddingModelFactice(),
    )
    return store.client.points_captures


def test_deux_ingestions_produisent_les_memes_ids():
    """Deux passes sur les mêmes chunks doivent produire les mêmes ids (idempotence)."""
    chunks = chunk_markdown("# Titre\n\nContenu de test.", source="doc.md")

    ids_premiere_passe = [p.id for p in _ingerer(chunks)]
    ids_deuxieme_passe = [p.id for p in _ingerer(chunks)]

    assert ids_premiere_passe == ids_deuxieme_passe


def test_ids_stables_meme_recalcules_depuis_zero():
    """L'id ne dépend que de (source, index), pas d'un compteur d'appel : le
    reconstruire depuis une nouvelle instance de `QdrantVectorStore` ne change rien."""
    chunks = chunk_markdown("# Titre\n\nContenu de test.", source="doc.md")

    ids_a = [p.id for p in _ingerer(chunks)]
    ids_b = [p.id for p in _ingerer(list(chunks))]

    assert ids_a == ids_b


def test_deux_documents_differents_ne_collisionnent_jamais():
    """Deux documents distincts ne doivent jamais produire le même id, même à index égal."""
    chunks_a = chunk_markdown("# Titre\n\nContenu A.", source="doc_a.md")
    chunks_b = chunk_markdown("# Titre\n\nContenu B.", source="doc_b.md")

    ids_a = {p.id for p in _ingerer(chunks_a)}
    ids_b = {p.id for p in _ingerer(chunks_b)}

    assert ids_a.isdisjoint(ids_b)


def test_ids_identiques_entre_lf_et_crlf():
    """Un même document en LF et en CRLF (core.autocrlf sous Windows) doit
    produire les mêmes uuid5 : avant la normalisation CRLF dans `chunk_markdown`,
    le front-matter ne matchait plus en CRLF, `source` retombait sur le
    paramètre d'appel et tous les `index` étaient décalés d'une unité, donc
    tous les ids dérivés changeaient. Une ingestion depuis une machine en LF
    puis une machine en CRLF doublait alors les points dans Qdrant."""
    texte_lf = "---\ntitre: T\nsource: doc.md\n---\n# Titre\n\nContenu de test.\n"
    texte_crlf = texte_lf.replace("\n", "\r\n")

    chunks_lf = chunk_markdown(texte_lf, source="ignore.md")
    chunks_crlf = chunk_markdown(texte_crlf, source="ignore.md")

    ids_lf = [p.id for p in _ingerer(chunks_lf)]
    ids_crlf = [p.id for p in _ingerer(chunks_crlf)]

    assert ids_lf == ids_crlf


def test_delete_precede_upsert_et_cible_les_sources_du_lot():
    """Finding 7 : chaque `source` du lot est purgée par `delete` avant l'`upsert`
    qui réinsère ses chunks à jour — c'est ce qui empêche les points orphelins
    d'une passe précédente (document raccourci) de survivre en base."""
    chunks = chunk_markdown("# Titre\n\nContenu de test.", source="doc.md")

    store = object.__new__(QdrantVectorStore)
    store.client = _ClientEspion()
    store.add_embedding(
        collection_name="test",
        chunks=chunks,
        embedding_model=_EmbeddingModelFactice(),
    )

    types_appels = [type_ for type_, _ in store.client.appels]
    assert types_appels == ["delete", "upsert"]

    sources_supprimees = {source for type_, source in store.client.appels if type_ == "delete"}
    assert sources_supprimees == {"doc.md"}


def test_ids_uniques_sur_tout_le_corpus_de_fixtures():
    """Sur les 3 fixtures (21 chunks), tous les ids générés sont uniques."""
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    tous_les_chunks = []
    for nom in ("fixture_animaux.md", "fixture_recettes.md", "fixture_terrain.md"):
        texte = (fixtures / nom).read_text(encoding="utf-8")
        tous_les_chunks.extend(chunk_markdown(texte, source=nom))

    ids = [p.id for p in _ingerer(tous_les_chunks)]
    assert len(ids) == len(tous_les_chunks)
    assert len(set(ids)) == len(ids)
