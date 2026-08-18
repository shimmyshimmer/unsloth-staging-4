# Pull the helper chains out of both revisions into dot-sourceable files.
#
# Both sources get normalised to LF first. install.ps1 is stored with LF but
# arrives CRLF on a Windows checkout, and piping `git show` through Set-Content
# rewrites the other one CRLF too, so a pattern anchored on "\n    }\n" matches
# nothing here even though it matches on Linux.
$ErrorActionPreference = 'Stop'
function Get-NormalisedSource {
    param([string]$Path)
    return ((Get-Content -Raw $Path) -replace "`r`n", "`n")
}
function Export-Helpers {
    param([string]$From, [string[]]$Names, [string]$To)
    $src = Get-NormalisedSource -Path $From
    $parts = foreach ($n in $Names) {
        $m = [regex]::Match($src, "(?s)    function $n \{.*?\n    \}\n")
        if (-not $m.Success) { throw "helper $n not found in $From" }
        $m.Value
    }
    [System.IO.File]::WriteAllText($To, ($parts -join "`n"), (New-Object System.Text.UTF8Encoding $false))
}
$newNames = @('Write-StudioLine','Test-StudioDirectoryUsable','Remove-StudioStalePrivateTempDirectories',
              'New-StudioPrivateTempDirectory','Initialize-StudioTempEnvironment','Restore-StudioTempEnvironment',
              'Write-StudioFinalPathDegraded','Initialize-StudioFinalPathNativeType','Resolve-StudioLinkTarget',
              'Get-StudioLexicalPath','Resolve-StudioFinalPathInfo','Get-StudioFinalPath','Get-StudioPathHash',
              'Get-StudioInstallMutexName','Enter-StudioNamedMutex','Enter-StudioInstallMutex','Exit-StudioInstallMutex')
$oldNames = @('Write-StudioLine','Get-StudioFinalPath','Get-StudioPathHash','Get-StudioInstallMutexName',
              'Enter-StudioNamedMutex','Enter-StudioInstallMutex','Exit-StudioInstallMutex')
# Join-Path, not string concatenation: WriteAllText below is raw .NET and does
# not fold a backslash the way a provider cmdlet would, so a hand-built path is
# only correct on Windows and quietly wrong when this is rehearsed on Linux.
$newHelpers = Join-Path $env:RUNNER_TEMP "new-helpers.ps1"
$oldHelpers = Join-Path $env:RUNNER_TEMP "old-helpers.ps1"
Export-Helpers -From ".\install.ps1" -Names $newNames -To $newHelpers
Export-Helpers -From (Join-Path $env:RUNNER_TEMP "old-install.ps1") -Names $oldNames -To $oldHelpers
Export-Helpers -From ".\install.ps1" -Names @('Remove-StudioStalePrivateTempDirectories') -To (Join-Path $env:RUNNER_TEMP "sweep.ps1")
# The pre-fix chain has to still contain the compile that issue 9140 trips over,
# otherwise the reproduction step below would be asserting against nothing.
$old = Get-NormalisedSource -Path $oldHelpers
if ($old -notmatch 'Add-Type') { throw "the pre-fix helpers contain no Add-Type; extracted the wrong revision" }
$new = Get-NormalisedSource -Path $newHelpers
if ($new -notmatch 'Initialize-StudioFinalPathNativeType') { throw "the fixed helpers are missing the native-type gate" }
Write-Host "extracted $($newNames.Count) new and $($oldNames.Count) old helpers"
