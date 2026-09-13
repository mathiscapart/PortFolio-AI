"""Pertinence du retrieval : le chunk qui PORTE la reponse doit etre dans le contexte.

Le test qui manquait. La mesure precedente comparait `top1["source"]`, donc le
nom de FICHIER : le bon fichier avec la mauvaise section passait pour un succes.
C'est ainsi qu'un "4/4" a masque un refus legitime sur une question de reference.

Test d'integration : demande Qdrant peuple et Ollama joignables, sinon saute.
"""
import pytest

from backend.rag.main import EmbeddingModel, QdrantVectorStore, Settings

# Question -> fragment que le contexte DOIT contenir pour permettre la reponse.
REFERENCES = [
    ("Ou a-t-elle fait ses etudes ?", "INSA Lyon"),
    ("Quelle experience en detection d'anomalies ?", "capteurs"),
    ("Parle-t-elle anglais ?", "Anglais"),
    ("Quand a-t-elle rejoint Aubelis ?", "Aubelis"),
]

K_DEFAUT = 8


@pytest.fixture(scope="module")
def contexte():
    reglages = Settings()
    store = QdrantVectorStore(host=reglages.qdrant_host, port=reglages.qdrant_port)
    modele = EmbeddingModel(
        model_name=reglages.embedding_model,
        host=reglages.ollama_host,
        port=reglages.ollama_port,
    )
    try:
        if not store.client.collection_exists(reglages.qdrant_collection):
            pytest.skip("collection absente")
        if store.client.count(reglages.qdrant_collection).count == 0:
            pytest.skip("collection vide")
        modele.embed("sonde")
    except Exception as exc:
        pytest.skip(f"services indisponibles : {exc}")
    return reglages, store, modele


@pytest.mark.parametrize("question,fragment", REFERENCES)
def test_le_chunk_portant_la_reponse_est_dans_le_contexte(contexte, question, fragment):
    reglages, store, modele = contexte
    chunks = store.search(
        collection_name=reglages.qdrant_collection,
        query=question,
        embedding_model=modele,
        k=K_DEFAUT,
    )
    contenu = " ".join(c["texte"] for c in chunks)
    assert fragment in contenu, (
        f"{question!r} : aucun des {len(chunks)} extraits ne contient {fragment!r}. "
        "Le modele ne peut pas repondre, il refusera -- correctement, mais a tort."
    )
