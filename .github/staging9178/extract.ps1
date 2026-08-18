# Pull the helper chains out of both revisions into dot-sourceable files.
$ErrorActionPreference = 'Stop'
function Export-Helpers {
    param([string]$From, [string[]]$Names, [string]$To)
    $src = Get-Content -Raw $From
    $parts = foreach ($n in $Names) {
        $m = [regex]::Match($src, "(?s)    function $n \{.*?\n    \}\n")
        if (-not $m.Success) { throw "helper $n not found in $From" }
        $m.Value
    }
    ($parts -join "`n") | Set-Content -Encoding UTF8 $To
}
$newNames = @('Write-StudioLine','Test-StudioDirectoryUsable','Remove-StudioStalePrivateTempDirectories',
              'New-StudioPrivateTempDirectory','Initialize-StudioTempEnvironment','Restore-StudioTempEnvironment',
              'Write-StudioFinalPathDegraded','Initialize-StudioFinalPathNativeType','Resolve-StudioLinkTarget',
              'Get-StudioLexicalPath','Resolve-StudioFinalPathInfo','Get-StudioFinalPath','Get-StudioPathHash',
              'Get-StudioInstallMutexName','Enter-StudioNamedMutex','Enter-StudioInstallMutex','Exit-StudioInstallMutex')
$oldNames = @('Write-StudioLine','Get-StudioFinalPath','Get-StudioPathHash','Get-StudioInstallMutexName',
              'Enter-StudioNamedMutex','Enter-StudioInstallMutex','Exit-StudioInstallMutex')
Export-Helpers -From ".\install.ps1" -Names $newNames -To "$env:RUNNER_TEMP\new-helpers.ps1"
Export-Helpers -From "$env:RUNNER_TEMP\old-install.ps1" -Names $oldNames -To "$env:RUNNER_TEMP\old-helpers.ps1"
[regex]::Match((Get-Content -Raw ".\install.ps1"),
    "(?s)    function Remove-StudioStalePrivateTempDirectories \{.*?\n    \}\n").Value |
    Set-Content -Encoding UTF8 "$env:RUNNER_TEMP\sweep.ps1"
Write-Host "extracted $($newNames.Count) new and $($oldNames.Count) old helpers"
