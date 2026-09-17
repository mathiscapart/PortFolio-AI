# Demarre les deux process natifs Windows du portfolio, detaches de la console :
#   - TTS (Pocket TTS, .venv-tts, torch CPU)  -> 127.0.0.1:8001, jamais expose
#   - API (STT ROCm + RAG + LLM, .venv-rocm)  -> $env:API_HOST:8000 (defaut
#     127.0.0.1). En production, Traefik tourne sur le serveur front : API_HOST=192.168.1.75,
#     et le pare-feu Windows ne doit laisser entrer que le serveur front sur le port 8000,
#     sinon le reseau local contournerait le rate-limit (docs/exploitation.md).
# Le TTS vit dans son propre venv : sous le torch ROCm il mesure 27,95x temps
# reel contre 0,70x sous torch CPU.
# Usage : powershell -ExecutionPolicy Bypass -File deploy\start-natif.ps1
$ErrorActionPreference = "Stop"
$racine = Split-Path $PSScriptRoot -Parent
$logs = Join-Path $racine "deploy\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

foreach ($port in 8000, 8001) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
}

$env:PYTHONIOENCODING = "utf-8"
# Origine du front public : l'API refuse tout WebSocket /voice venant d'ailleurs.
if (-not $env:CORS_ORIGINS) { $env:CORS_ORIGINS = "https://mathiscapart.xyz" }
# Voix clonee si son etat exporte est present (non versionne), sinon le
# defaut du catalogue. Surchargeable en definissant TTS_VOIX avant l'appel.
$voixClonee = Join-Path $racine "backend\tts\voix\mathis.safetensors"
if (-not $env:TTS_VOIX -and (Test-Path $voixClonee)) { $env:TTS_VOIX = $voixClonee }
$env:TORCHDYNAMO_DISABLE = "1"
if (-not $env:API_HOST) { $env:API_HOST = "127.0.0.1" }
# OLLAMA_HOST vaut 0.0.0.0 au niveau machine : c'est l'adresse d'ecoute du
# serveur Ollama. Heritee telle quelle, l'API tentait de se connecter a
# 0.0.0.0:11434, ce qui echoue sous Windows (/health en 503, verifie).
$env:OLLAMA_HOST = "127.0.0.1"

Start-Process -WindowStyle Hidden -WorkingDirectory $racine `
    -FilePath (Join-Path $racine ".venv-tts\Scripts\python.exe") `
    -ArgumentList "-m uvicorn backend.tts.server:app --host 127.0.0.1 --port 8001" `
    -RedirectStandardOutput "$logs\tts.out.log" -RedirectStandardError "$logs\tts.err.log"

Start-Process -WindowStyle Hidden -WorkingDirectory $racine `
    -FilePath (Join-Path $racine ".venv-rocm\Scripts\python.exe") `
    -ArgumentList "-m uvicorn backend.api.main:app --host $env:API_HOST --port 8000" `
    -RedirectStandardOutput "$logs\api.out.log" -RedirectStandardError "$logs\api.err.log"

Write-Output "TTS et API lances ; logs dans $logs"
