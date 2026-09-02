# build.ps1 - build FastGrain.aex with MSBuild (VS 2022 / v143 toolset).
#   .\build.ps1                 Release build
#   .\build.ps1 -Configuration Debug
#   .\build.ps1 -Install        build, then copy into the AE MediaCore plug-in folder (UAC prompt)
param(
    [string]$Configuration = "Release",
    [switch]$Install,
    [switch]$Rebuild
)
$ErrorActionPreference = "Stop"

$msbuild = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
if (-not (Test-Path $msbuild)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
}
if (-not $msbuild) { throw "MSBuild not found" }

$target = if ($Rebuild) { "Rebuild" } else { "Build" }
& $msbuild "$PSScriptRoot\FastGrain.vcxproj" "/t:$target" "/p:Configuration=$Configuration" "/p:Platform=x64" /m /nologo /v:m
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$aex = Join-Path $PSScriptRoot "build\$Configuration\FastGrain.aex"
Write-Host ("Built: {0} ({1:N0} bytes)" -f $aex, (Get-Item $aex).Length)

if ($Install) {
    & "$PSScriptRoot\install.ps1" -Configuration $Configuration
}
