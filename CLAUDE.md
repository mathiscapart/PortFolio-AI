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
- Pas de `pyproject.toml` ni de linter : chaque service a son propre
  `requirements*.txt`.
- CI GitHub Actions (`.github/workflows/ci.yml`) : tests backend sans torch,
  tests + typecheck + build du front, `docker compose build rag`, gitleaks sur
  tout l'historique, `pip-audit` et `npm audit`. Actions épinglées par SHA.
  Dependabot hebdomadaire. Le STT, le TTS et l'inférence restent hors CI (GPU).
  Invariant : `backend/api/main.py` n'importe torch et le STT qu'au démarrage
  et dans `/voice`, sinon les tests de l'API exigeraient la pile GPU.

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

## 4 bis. Spike vocal — tout est mesure, rien n'est estime

ROCm fonctionne sur la RX 7700 XT : torch 2.9.1+rocmsdk, HIP 7.2, calcul GPU
verifie. Chaine TTS -> WAV -> STT validee de bout en bout.

| Brique | VRAM mesuree | Vitesse | Note |
|---|---|---|---|
| LLM `qwen3:8b` | 6,30 Go | 61 t/s | resident, mais decharge par Ollama a expiration du keep_alive |
| STT `kyutai/stt-1b-en_fr` | 2,48 Go | 0,34x temps reel en streaming (27 ms par trame de 80 ms) | le ~1,0x du spike incluait le chargement |
| TTS Chatterbox Multilingual | 3,20 Go | genere sur ROCm | 2,7x l'estimation initiale de 1,2 Go |
| TTS Kyutai Pocket `french_24l` | **0 Go (CPU)** | **0,68x temps reel** | 336M parametres, 15 s de chargement |

Contraintes decouvertes, non devinables :

- **Qwen3-TTS est inutilisable** : son architecture `qwen3_tts` n'existe pas dans
  `transformers`, y compris en version de developpement. Ce n'est PAS un verdict
  ROCm — le GPU n'a jamais ete sollicite. Le chemin se rouvrira si Qwen publie
  le support.
- **Triton est absent des wheels ROCm Windows.** Moshi passe par `torch.compile`
  et echoue en `TritonMissing` : il faut `TORCHDYNAMO_DISABLE=1` pour retomber
  en mode eager. D'ou le facteur ~1,0x du STT.
- Le STT tourne en streaming trame par trame dans `/voice` ; le tour de parole
  reste explicite (bouton « Terminé de parler »), sans VAD.
- Chatterbox (et `perth`, qui exigeait `pkg_resources` donc `setuptools<81`) a
  ete desinstalle de `.venv-rocm` avec ses 30 dependances orphelines et `peft` :
  inutilise en production, il bloquait la mise a jour de securite de
  setuptools. Le reinstaller imposerait de nouveau `setuptools<81`.
- La liberation de VRAM entre deux briques fonctionne (12,12 Go recuperes), donc
  le chargement sequentiel reste une option viable.
- Environnements separes : `.venv-rocm` (Python 3.12, wheels ROCm) et
  `.venv-tts` (Pocket TTS, CPU). Ne jamais installer un paquet PyPI tirant
  `torch` dans `.venv-rocm` : il ecraserait le build ROCm.

## 4 ter. Boucle vocale en production — mesures

Topologie : API (`.venv-rocm`, STT GPU + RAG + LLM, port 8000) et service TTS
(`.venv-tts`, `backend/tts/server.py`, 127.0.0.1:8001, jamais expose). Les deux
se lancent par `deploy/start-natif.ps1` (logs dans `deploy/logs/`).

- **Front sur un serveur dedie** (Debian 13, toujours allume) : nginx + Traefik +
  cloudflared y tournent ; Traefik joint l'API sur le PC du GPU
  (`192.168.1.75:8000`, `API_HOST`), allume a la demande. PC eteint -> 502 sur
  `/api/health` et le front annonce l'IA hors ligne. Procedure :
  `docs/exploitation.md` section 10.

- **Pocket TTS ne doit jamais tourner dans `.venv-rocm`** : 27,95x temps reel
  sous torch ROCm contre 0,70x sous torch CPU (meme phrase). D'ou le process
  separe.
- **`num_ctx` 8192 obligatoire** (`OPTIONS_LLM`) : par defaut Ollama prend 32768,
  `qwen3:8b` monte a 9,16 Go, deborde a cote du STT et tombe a 10,5 t/s.
- **`think=False` sur `/voice`** : le raisonnement de qwen3 coutait 9,2 s de
  silence avant le premier token.
- **Passage d'identite epingle** : `/voice` ajoute toujours la section
  `parcours.md > Mon parcours en bref` aux extraits (`ajouter_identite`). Sans
  elle, une question vague ne ramenait que GitHub et Piloti, et qwen3 inventait
  « developpeur full-stack » 5 fois sur 5. Renommer cette section impose de
  mettre a jour `IDENTITE_TITRE` (verrouille par `test_pertinence.py`).
- **Pas d'endpoint texte en production** : `/chat` a ete retire de l'API (le
  front ne passe que par `/voice`). Pour interroger le RAG a l'ecrit :
  `python -m backend.tests.question_texte "question" --sources`.
- Mesure publique (`wss://<domaine>/api/voice`) : premier token 1,3 s, premier
  son 2,3 a 2,6 s apres la fin de la question.
- Une seule session vocale a la fois ; question plafonnee a 30 s, socket muet
  coupe apres 10 s. Le rate-limit Traefik ne compte que l'upgrade WebSocket.
- Voix clonee : `TTS_VOIX=backend/tts/voix/mathis.safetensors` (defaut
  `estelle`), herite par `start-natif.ps1`. Etat exporte par
  `python -m pocket_tts export-voice <wav> <safetensors> --language french_24l`.
  Poids du clonage soumis aux conditions HF de `kyutai/pocket-tts` (compte
  connecte via `hf auth login`). `backend/tts/voix/` n'est jamais versionne.
  Mesure : 0,78x temps reel, 1er son 0,28-0,39 s, identique a `estelle`.
- Nginx (front) redirige `https://…/parcours` vers `http://…/parcours/`
  (redirection absolue derriere le proxy TLS) : defaut preexistant, non corrige.

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
- `backend/rag/corpus/*.md` est le vrai parcours (sources : CV Canva et README
  GitHub), servi en production. Chaque fait doit être sourcé : aucune
  information n'est déduite. La section « Ce que je cherche » manque encore ;
  sans elle, qwen3 inventait un poste recherché, d'où la règle anti-souhaits du
  prompt `SYSTEME`. Un nom propre noyé dans un long passage remonte mal à
  l'embedding : isoler le rôle dans une sous-section courte (cas LaRuche).
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
