# The pre-fix installer must still fail here the way issue 9140 describes.
$ErrorActionPreference = 'Stop'
$script:StudioStdoutRedirected = $true
. "$env:RUNNER_TEMP\old-helpers.ps1"
function Add-Type { throw "(0) : error CS2001: Source file 'a.0.cs' could not be found" }
try { $null = Get-StudioInstallMutexName -Path $env:USERPROFILE; Write-Output "OLD:survived" }
catch { Write-Output "OLD:threw" }
