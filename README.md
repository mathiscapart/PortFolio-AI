# PortFolio-AI

Portfolio vocal de Mathis Capart, en ligne sur https://mathiscapart.xyz. Le
visiteur pose sa question à voix haute ; l'assistant répond avec une copie de la
voix de Mathis, uniquement à partir de son parcours.

Chaîne : reconnaissance vocale Kyutai (GPU AMD, ROCm) → recherche Qdrant →
`qwen3:8b` via Ollama → synthèse Pocket TTS. Tout tourne en local sur un homelab
Windows, exposé par Traefik et un tunnel Cloudflare, sans API d'inférence externe.

## Structure

| Dossier | Contenu |
|---|---|
| `backend/api/` | API FastAPI : `/health`, `/chat` (SSE), `/voice` (WebSocket) |
| `backend/stt/` | reconnaissance vocale Kyutai en streaming |
| `backend/tts/` | service de synthèse vocale (voix clonée dans `voix/`, non versionné) |
| `backend/rag/` | découpage, ingestion et recherche ; `corpus/` = le parcours |
| `frontend/` | Next.js en export statique : interface vocale et page parcours |
| `deploy/` | Traefik, nginx, cloudflared, script de démarrage des services natifs |
| `.github/` | CI et Dependabot |

## Démarrage rapide

```powershell
docker compose up -d qdrant
powershell -ExecutionPolicy Bypass -File deploy\start-natif.ps1
docker compose --env-file .env -f deploy/docker-compose.expose.yml up -d

# Arrêter l'API et le TTS
powershell -ExecutionPolicy Bypass -File deploy\stop-natif.ps1
```

Prérequis, déploiement, arrêt, vérifications et dépannage :
**[docs/exploitation.md](docs/exploitation.md)**.

Conventions et décisions techniques mesurées : [CLAUDE.md](CLAUDE.md).

## Tests

```powershell
python -m pytest backend/tests          # backend, sans GPU
cd frontend; npx vitest run; npx tsc --noEmit
```
