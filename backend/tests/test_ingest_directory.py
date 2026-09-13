"""Tests d'`ingest_directory` (`backend/rag/main.py`).

N'utilise que des répertoires temporaires (`tmp_path`) : ni `backend/rag/corpus/`
(squelette humain en cours de rédaction), ni `backend/rag/corpus_demo/` (données
de démo) ne sont touchés ou lus.
"""
from pathlib import Path

import pytest

from backend.rag.main import CorpusIncompletError, ingest_directory


class _EmbeddingModelFactice:
    """Renvoie un vecteur fixe, sans appeler Ollama."""

    def embed(self, text: str) -> list[float]:
        return [0.0, 1.0]


class _StoreEspion:
    """Remplace `QdrantVectorStore` : capture les lots passés à `add_embedding`, sans réseau."""

    def __init__(self):
        self.lots = []
        self.sources = []

    def add_embedding(self, collection_name, chunks, embedding_model, *args, **kwargs):
        self.lots.append(chunks)
        # `sources` doit être observé, pas seulement avalé : c'est lui qui cible
        # la purge des points périmés.
        self.sources.append(kwargs.get("sources"))


def test_fichier_avec_marqueur_leve_corpus_incomplet_error_en_le_nommant(tmp_path: Path):
    """Un fichier contenant encore `À REMPLIR` bloque toute l'ingestion et est
    nommé explicitement dans l'erreur : il ne doit jamais atteindre Qdrant."""
    fichier_incomplet = tmp_path / "experience.md"
    fichier_incomplet.write_text("# Expérience\n\nÀ REMPLIR\n", encoding="utf-8")

    with pytest.raises(CorpusIncompletError) as exc_info:
        ingest_directory(
            directory=str(tmp_path),
            qdrant_store=_StoreEspion(),
            embedding_model=_EmbeddingModelFactice(),
            collection_name="test",
        )

    assert "experience.md" in str(exc_info.value)


def test_repertoire_sans_marqueur_s_ingere_sans_exception(tmp_path: Path):
    """Un répertoire propre s'ingère de bout en bout, chaque fichier `.md`
    (hors README) produisant un lot de chunks envoyé au store."""
    (tmp_path / "a.md").write_text("# Titre A\n\nContenu A.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Titre B\n\nContenu B.\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Documentation du répertoire.\n", encoding="utf-8")

    store = _StoreEspion()
    fichiers = ingest_directory(
        directory=str(tmp_path),
        qdrant_store=store,
        embedding_model=_EmbeddingModelFactice(),
        collection_name="test",
    )

    assert [f.name for f in fichiers] == ["a.md", "b.md"]
    assert len(store.lots) == 2
    assert store.sources == [{"a.md"}, {"b.md"}]


def test_purge_ciblee_couvre_le_nom_de_fichier_et_la_source_du_front_matter(tmp_path: Path):
    """Le front-matter peut redéfinir `source`. Purger sur le seul nom de fichier
    ne matcherait alors aucun point déjà indexé : la purge serait un no-op
    silencieux, et les points périmés d'un document raccourci survivraient."""
    (tmp_path / "cv.md").write_text(
        "---\ntitre: T\nsource: parcours.md\n---\n\n# A\n\nDu contenu.\n", encoding="utf-8"
    )

    store = _StoreEspion()
    ingest_directory(
        directory=str(tmp_path),
        qdrant_store=store,
        embedding_model=_EmbeddingModelFactice(),
        collection_name="test",
    )

    assert store.sources == [{"cv.md", "parcours.md"}]
    # Toute source réellement insérée doit figurer dans la cible de purge.
    assert {c["source"] for c in store.lots[0]} <= store.sources[0]


def test_marqueur_detecte_quelle_que_soit_la_casse_et_la_normalisation():
    """Le marqueur doit resister a un editeur qui change la casse ou renormalise
    en NFD (A + U+0300, qui ne contient pas le A precompose)."""
    import unicodedata

    from backend.rag.main import _contient_marqueur

    assert _contient_marqueur("bla À REMPLIR bla")
    assert _contient_marqueur("bla à remplir bla")
    assert _contient_marqueur("bla " + unicodedata.normalize("NFD", "À REMPLIR") + " bla")
    # Limite assumee : la forme sans accent n'est pas detectee. Retirer les
    # diacritiques attraperait aussi une phrase francaise legitime ("a remplir
    # le formulaire") et bloquerait un vrai corpus.
    assert not _contient_marqueur("bla A REMPLIR bla")


def test_deux_fichiers_declarant_la_meme_source_sont_refuses_avant_toute_ecriture(tmp_path: Path):
    """Le front-matter peut redefinir `source` : deux fichiers portant la meme
    valeur s'effaceraient mutuellement, la purge du second supprimant les points
    du premier. L'erreur doit tomber avant le moindre appel Qdrant."""
    for nom in ("a.md", "b.md"):
        (tmp_path / nom).write_text(
            f"---\nsource: commun.md\n---\n\n# T\n\nTexte {nom}.\n", encoding="utf-8"
        )

    store = _StoreEspion()
    with pytest.raises(CorpusIncompletError) as exc_info:
        ingest_directory(
            directory=str(tmp_path),
            qdrant_store=store,
            embedding_model=_EmbeddingModelFactice(),
            collection_name="test",
        )

    assert "commun.md" in str(exc_info.value)
    assert store.lots == []


def test_un_embed_qui_echoue_ne_laisse_aucune_source_purgee():
    """Les embed sont payes avant la purge : sinon un `embed()` qui casse a
    mi-chemin laisse la source supprimee et jamais reinseree."""
    from backend.rag.main import QdrantVectorStore

    class _Client:
        def __init__(self):
            self.appels = []

        def delete(self, collection_name, points_selector):
            self.appels.append("delete")

        def upsert(self, collection_name, points):
            self.appels.append("upsert")

    class _EmbedQuiCasse:
        def __init__(self):
            self.n = 0

        def embed(self, texte):
            self.n += 1
            if self.n >= 2:
                raise RuntimeError("ollama tombe")
            return [0.0, 1.0]

    store = object.__new__(QdrantVectorStore)
    store.client = _Client()
    chunks = [{"texte": f"t{i}", "source": "x.md", "titre": None, "index": i} for i in range(3)]

    with pytest.raises(RuntimeError):
        store.add_embedding("col", chunks, _EmbedQuiCasse())

    assert store.client.appels == []
