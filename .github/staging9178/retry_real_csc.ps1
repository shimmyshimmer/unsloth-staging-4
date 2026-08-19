# No stubbing here. %TEMP% points underneath a FILE, so the real CodeDom compile
# genuinely cannot write its source, and only the private-%TEMP% retry can win.
$ErrorActionPreference = 'Stop'
$script:StudioStdoutRedirected = $true
. "$env:RUNNER_TEMP\new-helpers.ps1"
$env:TMP = $env:DEAD_TEMP
$env:TEMP = $env:DEAD_TEMP
# Skip the session-wide temp fix so the retry is the only thing left.
$script:StudioTempChecked = $true
$ok = Initialize-StudioFinalPathNativeType
Write-Output "RETRY:native=$ok"
Write-Output "RETRY:type=$([bool]('UnslothStudioFinalPathV2' -as [type]))"
Write-Output "RETRY:tmp=$env:TMP"
if ($ok) { Write-Output "RETRY:resolve=$([UnslothStudioFinalPathV2]::Resolve($env:USERPROFILE))" }
