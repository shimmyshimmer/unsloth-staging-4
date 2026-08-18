# No sabotage: the ordinary path must name exactly what it always has.
$ErrorActionPreference = 'Stop'
$script:StudioStdoutRedirected = $true
. "$env:HELPERS_FILE"
foreach ($p in @($env:USERPROFILE, $env:LOCALAPPDATA, 'C:\', (Join-Path $env:USERPROFILE '.unsloth\studio'))) {
    Write-Output "P:$p=$(Get-StudioInstallMutexName -Path $p)"
}
