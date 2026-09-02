# install.ps1 - copy the built FastGrain.aex into the shared AE plug-in folder.
# The MediaCore folder is under Program Files, so this elevates (one UAC prompt) when needed.
param(
    [string]$Configuration = "Release",
    [string]$Destination = "C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore"
)
$ErrorActionPreference = "Stop"

$src = Join-Path $PSScriptRoot "build\$Configuration\FastGrain.aex"
if (-not (Test-Path $src)) { throw "Not built yet: $src" }
$dst = Join-Path $Destination "FastGrain.aex"

try {
    Copy-Item -LiteralPath $src -Destination $dst -Force
    Write-Host "Installed (direct copy): $dst"
} catch {
    Write-Host "Direct copy denied, elevating..."
    $cmd = "Copy-Item -LiteralPath '$src' -Destination '$dst' -Force; if (`$?) { exit 0 } else { exit 1 }"
    $p = Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -PassThru -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd
    if ($p.ExitCode -ne 0) { throw "Elevated copy failed (exit $($p.ExitCode))" }
    Write-Host "Installed (elevated): $dst"
}
Write-Host "Restart After Effects to load the new build."
