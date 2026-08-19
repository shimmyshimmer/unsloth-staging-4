# The fixed installer must take its lock with the compiler dead.
$ErrorActionPreference = 'Stop'
$script:StudioStdoutRedirected = $true
. "$env:RUNNER_TEMP\new-helpers.ps1"
function Add-Type { throw "(0) : error CS2001: Source file 'a.0.cs' could not be found" }
$m = Enter-StudioInstallMutex -Path $env:USERPROFILE
Write-Output "NEW:locked=$($null -ne $m)"
Exit-StudioInstallMutex -Mutex $m
