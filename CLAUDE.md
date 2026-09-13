# PortFolio-AI

## 1. Objet du projet

Portfolio interactif d'un ingénieur IA : un visiteur parle, un assistant vocal
répond sur le parcours de Mathis. Boucle visée STT → RAG → LLM → TTS, avec
clonage de sa propre voix. Tout en local, hébergé sur un homelab perso via
tunnel cloudflared (décision arrêtée, pas de cloud, pas d'API d'inférence tierce ;
la config du tunnel n'est pas versionnée).

## 2. Type d'activité

Code applicatif (Python) + infra légère (Docker Compose). Pas de sécurité
périmétrique ni de data pipeline au sens classique — le "data" ici, c'est
l'ingestion RAG.

## 3. Stack & structure

- `backend/rag/` — indexation Qdrant + embeddings via Ollama (`main.py`,
  `structure_data.py` stub). Tourne en conteneur Docker.
- `backend/stt/` — inférence Moshi/Kyutai (`model.py`), cassé (`main.py`).
  **Ne peut pas tourner en Docker** : les wheels ROCm ciblées sont
  `cp312-cp312-win_amd64` (servies par `repo.radeon.com/rocm/windows/`), donc
  Windows natif uniquement.
- `docker-compose.yml` — services `qdrant` + `rag`.
- Pas de build/test/lint configuré (pas de `pyproject.toml`, pas de CI). Chaque
  service a son propre `requirements*.txt` installé au cas par cas.

Topologie imposée : `qdrant` et `rag` en Docker, STT et TTS en process Windows
natif. Ne pas proposer de conteneuriser le STT/TTS.

Matériel cible : AMD RX 7700 XT 12 Go, Ryzen 7 7700X, 31 Go RAM, Windows 11.
Budget VRAM serré : ~2 Go/milliard de paramètres en BF16, ~0,6 Go en Q4.

## 4. Modèles retenus (ne pas re-rechercher)

- LLM : **`qwen3:8b`** via Ollama. Mesuré sur la machine cible : **6,3 Go en
  VRAM, 100 % GPU, 61 t/s** à `num_ctx` 8192. Retenu contre `gemma4:e4b`
  (9,6 Go sur disque — le « E » ne désigne pas une variante légère) et contre
  `gemma4:12b` (8,4 Go mesurés, au-delà du seuil de 7 Go fixé d'avance). Le
  choix est pris **dès la v0** pour que le LLM de la v0 soit celui de la v1 :
  changer de modèle après avoir calé le prompt système et les tests de refus
  imposerait de tout recalibrer. Projection v1 : 6,3 + 2,2 (STT) + 1,2 (TTS)
  + 0,8 ≈ 10,5 Go sur 12.
- Embedding : **Qwen3-Embedding-0.6B**, sur CPU pour épargner la VRAM.
- STT : **Qwen3-ASR-0.6B** (Apache 2.0, FLEURS-fr 4,39 %) — mais son streaming
  ne passe que par vLLM, indisponible sous Windows natif → fonctionnement tour
  par tour avec VAD. Alternative streaming réel : garder `kyutai/stt-1b-en_fr`
  (déjà en place dans `backend/stt/model.py`).
- TTS + clonage voix : **Qwen3-TTS-12Hz-0.6B-Base** (Apache 2.0, clonage dès 3s),
  support ROCm non vérifié. Plan B : **Chatterbox Multilingual V3** (500M, MIT,
  ROCm confirmé). Plan C : **Kyutai Pocket TTS** (100M, CPU).

## 5. Conventions locales

- Fichiers écrits en **UTF-8 sans BOM**. Ne jamais rediriger avec `>` ou
  `Out-File` par défaut sous PowerShell (encode en UTF-16) — utiliser
  `-Encoding utf8` explicitement ou un éditeur.
- Commits en conventional commits, doc et commentaires en français.
- `.env` non versionné, `.env.example` comme référence — mais vérifier qu'une
  variable y est bien lue par le code avant de s'y fier (cf. dette #4).

- **Étape 5, FastAPI : l'endpoint de recherche doit être `def`, jamais `async def`.**
  `EmbeddingModel.embed()` est un appel HTTP bloquant vers Ollama, forcé sur CPU
  (`num_gpu: 0`) donc lent par construction. Déclaré `def`, FastAPI l'exécute dans
  son threadpool ; déclaré `async def`, chaque recherche bloquerait la boucle
  d'événements et gèlerait **tous** les flux SSE en cours.

- **L'API tourne en process Windows natif, pas en conteneur** (`python -m uvicorn
  backend.api.main:app`). Elle importe `backend.rag.main` en cross-package alors
  que le contexte de build Docker est `./backend/rag` : la conteneuriser
  imposerait de remonter ce contexte à la racine du repo. La topologie du projet
  est donc : Qdrant en conteneur, Ollama + API + STT/TTS en natif Windows.
  Décision réversible, à retrancher à l'étape 8 quand Traefik et cloudflared
  arriveront devant.

## 6. Zones sensibles

- `backend/stt/main.py` ne compile pas : `def main():` sans corps
  (IndentationError ligne 3). Le module `backend/stt` est inutilisable tel quel.
  Réparé à l'étape 9 seulement — rien en v0 ne l'importe.
- **Ids Qdrant : dette à moitié réglée.** `id=uuid5(source#index)` a supprimé
  l'écrasement croisé d'une ingestion à l'autre, mais introduit le défaut
  symétrique : si un document raccourcit (21 chunks → 18), les points 18, 19, 20
  de la passe précédente **restent** avec leur contenu périmé et ressortiront au
  `search`. Fix prévu à l'étape 4 : `delete` filtré sur `payload.source` avant
  l'`upsert` du document.
- **Un fichier *supprimé* du corpus n'est jamais purgé.** La purge n'itère que
  sur les fichiers encore présents : après `rm parcours.md` puis réingestion,
  plus rien ne nomme cette source, ses points restent indexés indéfiniment et
  ressortent au `search`. Défaut préexistant, non introduit par la purge par
  `sources`. Le fix propre (supprimer toute source absente du répertoire) est
  une décision de conception à prendre avec l'humain.
- Aucun retrieval implémenté (pas de `search`) : le RAG est encore write-only.
  C'est l'étape 4.
- `backend/rag/structure_data.py` : code mort (la v0 part de Markdown versionné).
  **Signalé, volontairement pas supprimé** — décision à l'humain.
- `backend/rag/corpus/*.md` contient encore le marqueur `À REMPLIR`. L'ingestion
  doit refuser tout fichier qui le contient, sinon l'assistant récitera mes
  consignes de rédaction à un recruteur. Garde-fou à poser à l'étape 4.
- `_split_long_section` découpe via `texte.split()` : une section longue perd ses
  retours à la ligne, une section courte les conserve. Conséquence assumée du
  découpage mot-à-mot, pas un bug.
- Pas de `.gitattributes` alors que `core.autocrlf` est actif. `chunk_markdown`
  normalise désormais les CRLF en entrée, ce qui neutralise le risque de dérive
  des uuid5 — mais tout nouveau lecteur de fichier doit faire de même.

- **Le `watch: true` du provider file de Traefik ne fonctionne pas** sur le
  montage bind Windows/Docker : une modification de
  `deploy/traefik/config/portfolio.yml` reste sans effet tant que Traefik
  n'est pas redemarre. Verifie par mesure, pas deduit.

## 7. Definition of Done (locale)

En plus de la DoD globale :

- Toute modif touchant `docker-compose.yml` ou un `Dockerfile` : vérifier avec
  `docker compose build <service>` que le build passe réellement.
- Toute modif touchant `backend/stt/` : tester en process Windows natif, pas
  en conteneur — ça ne peut pas tourner en Docker (wheels ROCm Windows-only).
- Toute modif touchant l'ingestion Qdrant : vérifier qu'une deuxième ingestion
  n'écrase pas les points de la première (dette #5 tant qu'elle n'est pas
  corrigée ailleurs).
- Nouveau fichier texte : vérifier son encodage (`file <chemin>` ou
  équivalent) avant de committer — UTF-8 sans BOM attendu.

## 8. Backlog & workflow

Le backlog vit dans **Notion**, pas dans le repo : epic « Portfolio » sous
`Entreprise / Suivie de projet / Epic`, 11 items (front, connexion front/back,
CV, STT, TTS, LLM, RAG, Embedding, Docker, CI/CD, INFRA). Ne pas en créer un
second ici.

Il est désynchronisé du réel : Docker y est « Pas commencé » alors que le
travail est fait, et aucun item Portfolio n'est dans le sprint courant. Le
resynchroniser passe par `project-manager` — et toute écriture Notion demande
l'accord explicite de l'humain dans le tour en cours.
