from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
import os
import unicodedata
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Marqueur de rédaction laissé dans le corpus en cours d'écriture : un fichier
# qui le contient encore n'est pas prêt à être servi à un recruteur.
MARQUEUR_A_REMPLIR = "À REMPLIR"


class CorpusIncompletError(Exception):
    """Levée quand un fichier à ingérer contient encore le marqueur `À REMPLIR`."""

try:  # importe comme paquet (API de l'etape 5, tests)
    from backend.rag.chunking import chunk_markdown
except ImportError:  # execution a plat dans le conteneur (WORKDIR /app)
    from chunking import chunk_markdown

# Namespace fixe pour dériver des UUID5 stables à partir de (source, index) :
# une même paire produit toujours le même id, une ingestion répétée met donc
# à jour les mêmes points au lieu d'en écraser d'autres.
_POINT_ID_NAMESPACE = uuid.UUID("6f6e8f6e-6f60-4e30-9d1e-9a1f6c2b4a10")

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            load_dotenv()
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        self.ollama_port = int(os.getenv("OLLAMA_PORT", 11434))
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "portfolio")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
        # Lu ici et nulle part ailleurs : un os.getenv au niveau module
        # s'executerait avant load_dotenv() et ignorerait .env en silence.
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")

class EmbeddingModel:
    def __init__(self, model_name: str = "qwen3-embedding:0.6b", host: str = "localhost", port: int = 11434):
        self.model_name = model_name
        self.client = Client(host=f"{host}:{port}")
        self._dim = None

    def embed(self, text: str) -> list[float]:
        # num_gpu=0 force l'embedding sur CPU : le LLM (qwen3:8b, ~6,5 Go) doit
        # rester résident en VRAM, sinon chaque requête RAG l'évincerait pour
        # charger le modèle d'embedding, et le tour suivant repaierait
        # plusieurs secondes de rechargement.
        response = self.client.embed(input=text, model=self.model_name, options={"num_gpu": 0})
        return response.embeddings[0]

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            response = self.client.embed(model=self.model_name, input="probe", options={"num_gpu": 0})
            self._dim = len(response.embeddings[0])
        return self._dim

class QdrantVectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333, timeout: int = 10):
        # Sans timeout, un Qdrant qui accepte la connexion sans repondre fait
        # pendre la requete indefiniment et immobilise un thread du threadpool.
        self.client = QdrantClient(host=host, port=port, timeout=timeout)

    def add_embedding(
        self,
        collection_name: str,
        chunks: list[dict],
        embedding_model: EmbeddingModel,
        sources: set[str] | None = None,
    ):
        # Les ids sont dérivés de (source, index) : si un document raccourcit
        # (21 chunks -> 18), les anciens points 18, 19, 20 ne sont écrasés par
        # rien et restent en base avec un contenu périmé. On purge donc tous les
        # points de chaque source concernée avant de réinsérer.
        #
        # `sources` laisse l'appelant nommer explicitement les documents à
        # purger, y compris ceux qui ne produisent plus aucun chunk (fichier
        # vidé) : les déduire du seul contenu réinséré laisserait leurs points
        # périmés en base indéfiniment. À défaut, on les déduit des chunks.
        # Les embed sont payes AVANT la purge : un embed qui echoue a
        # mi-chemin laisserait sinon la source supprimee et jamais reinseree.
        points = [
            PointStruct(
                id=str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{chunk['source']}#{chunk['index']}")),
                vector=embedding_model.embed(chunk["texte"]),
                payload=chunk,
            )
            for chunk in chunks
        ]
        # Qdrant rejette un upsert vide (400 « Empty update request ») : un
        # document vidé n'a plus rien à réinsérer une fois purgé.
        for source in sources if sources is not None else {chunk["source"] for chunk in chunks}:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
            )
        if points:
            self.client.upsert(collection_name=collection_name, points=points)

    def create_collection(self, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )

    def search(self, collection_name: str, query: str, embedding_model: EmbeddingModel, k: int = 5) -> list[dict]:
        """Retourne les `k` chunks les plus proches de `query`, chacun enrichi de son `score`."""
        resultats = self.client.query_points(
            collection_name=collection_name,
            query=embedding_model.embed(query),
            limit=k,
        )
        return [{**point.payload, "score": point.score} for point in resultats.points]

    def get_chunk(self, collection_name: str, source: str, titre: str) -> dict | None:
        """Retourne le chunk désigné par sa source et son titre, ou None s'il n'existe pas."""
        points, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=source)),
                FieldCondition(key="titre", match=MatchValue(value=titre)),
            ]),
            limit=1,
        )
        return points[0].payload if points else None

def _normaliser(texte: str) -> str:
    return unicodedata.normalize("NFC", texte).casefold()


_MARQUEUR_NORMALISE = _normaliser(MARQUEUR_A_REMPLIR)


def _contient_marqueur(texte: str) -> bool:
    """Insensible a la casse et a la normalisation Unicode : un editeur qui
    sauvegarde en NFD ecrit A + U+0300, qui ne contient pas le A precompose."""
    return _MARQUEUR_NORMALISE in _normaliser(texte)


def ingest_directory(
    directory: str,
    qdrant_store: QdrantVectorStore,
    embedding_model: EmbeddingModel,
    collection_name: str,
) -> list[Path]:
    """Ingere tous les fichiers `.md` d'un repertoire.

    Refuse tout fichier portant encore le marqueur `A REMPLIR` : un fichier non
    fini ne doit pas finir dans Qdrant, sous peine de faire reciter des
    consignes de redaction a un recruteur.

    `README.md` est exclu : il documente le repertoire, ce n'est pas du corpus.

    Validation, chunking et controle de collision se font **avant** toute
    ecriture : sinon les fichiers alphabetiquement anterieurs au fautif seraient
    deja en base quand l'erreur tombe, malgre le tout-ou-rien annonce.
    """
    fichiers = [f for f in sorted(Path(directory).glob("*.md")) if f.name.lower() != "readme.md"]

    lots: dict[Path, list[dict]] = {}
    vues: dict[str, Path] = {}
    for fichier in fichiers:
        texte = fichier.read_text(encoding="utf-8")
        if _contient_marqueur(texte):
            raise CorpusIncompletError(
                f"{fichier} contient encore le marqueur '{MARQUEUR_A_REMPLIR}' : corpus non pret a etre ingere."
            )
        chunks = chunk_markdown(texte, source=fichier.name)
        # Le front-matter peut redefinir `source`. Deux fichiers declarant la
        # meme valeur s'effacent mutuellement : la purge du second supprime les
        # points du premier. On echoue avant d'ecrire quoi que ce soit.
        for source in {c["source"] for c in chunks}:
            if source in vues and vues[source] != fichier:
                raise CorpusIncompletError(
                    f"{fichier} et {vues[source]} declarent la meme source '{source}' : "
                    "leurs points s'effaceraient mutuellement."
                )
            vues[source] = fichier
        lots[fichier] = chunks

    for fichier, chunks in lots.items():
        qdrant_store.add_embedding(
            collection_name=collection_name,
            chunks=chunks,
            embedding_model=embedding_model,
            sources={fichier.name} | {c["source"] for c in chunks},
        )
    return fichiers


def main():
    settings = Settings()
    qdrantClient = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)
    embedding_model = EmbeddingModel(model_name=settings.embedding_model, host=settings.ollama_host, port=settings.ollama_port)
    try:
        qdrantClient.create_collection(collection_name=settings.qdrant_collection, vector_size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
        chunks = chunk_markdown("This is a sample text to be embedded.", source="sample.md")
        qdrantClient.add_embedding(collection_name=settings.qdrant_collection, chunks=chunks, embedding_model=embedding_model)
    except Exception as e:
        print(f"Error adding embedding: {e}")

if __name__ == "__main__":
    main()
