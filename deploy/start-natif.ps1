# Demarre les deux process natifs Windows du portfolio, detaches de la console :
#   - TTS (Pocket TTS, .venv-tts, torch CPU)  -> 127.0.0.1:8001, jamais expose
#   - API (STT ROCm + RAG + LLM, .venv-rocm)  -> 127.0.0.1:8000, joint par Traefik
#     via host.docker.internal (Docker Desktop relaie vers la boucle locale,
#     verifie) : jamais expose au reseau local, qui contournerait le rate-limit
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

Start-Process -WindowStyle Hidden -WorkingDirectory $racine `
    -FilePath (Join-Path $racine ".venv-tts\Scripts\python.exe") `
    -ArgumentList "-m uvicorn backend.tts.server:app --host 127.0.0.1 --port 8001" `
    -RedirectStandardOutput "$logs\tts.out.log" -RedirectStandardError "$logs\tts.err.log"

Start-Process -WindowStyle Hidden -WorkingDirectory $racine `
    -FilePath (Join-Path $racine ".venv-rocm\Scripts\python.exe") `
    -ArgumentList "-m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000" `
    -RedirectStandardOutput "$logs\api.out.log" -RedirectStandardError "$logs\api.err.log"

Write-Output "TTS et API lances ; logs dans $logs"
