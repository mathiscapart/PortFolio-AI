# Arrete les deux process natifs lances par start-natif.ps1 :
#   - API (127.0.0.1:8000) : le vocal et /api tombent, le site statique reste en ligne
#   - TTS (127.0.0.1:8001)
# Les conteneurs (Qdrant, nginx, Traefik, cloudflared) ne sont pas touches :
# voir docs/exploitation.md, section 4, pour couper aussi le site.
# Usage : powershell -ExecutionPolicy Bypass -File deploy\stop-natif.ps1 [-WhatIf]
[CmdletBinding(SupportsShouldProcess)]
param()

# Chargement reel du module meme sous -WhatIf (sinon il simule la creation de ses alias).
$simulation = $WhatIfPreference
$WhatIfPreference = $false
Import-Module NetTCPIP
$WhatIfPreference = $simulation

$services = @(
    @{ Port = 8000; Nom = "API" },
    @{ Port = 8001; Nom = "TTS" }
)

foreach ($service in $services) {
    $port = $service.Port
    $nom = $service.Nom
    $ecoutes = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    if ($ecoutes.Count -eq 0) {
        Write-Output "$nom (port $port) : deja arrete"
        continue
    }
    foreach ($id in ($ecoutes.OwningProcess | Sort-Object -Unique)) {
        if ($PSCmdlet.ShouldProcess("$nom (port $port, PID $id)", "Arreter")) {
            Stop-Process -Id $id -Force
            Write-Output "$nom (port $port) : arrete (PID $id)"
        }
    }
}
