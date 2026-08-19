# Pull the helper chains out of both revisions into dot-sourceable files.
#
# Both sources get normalised to LF first. install.ps1 is stored with LF but
# arrives CRLF on a Windows checkout, and piping `git show` through Set-Content
# rewrites the other one CRLF too, so a pattern anchored on "\n    }\n" matches
# nothing here even though it matches on Linux.
#
# The chain is an explicit list rather than "every function in the file".
# Extracting all of them looks tidier and does not work: several bodies hold
# here-string templates containing a line of exactly four spaces and a brace, so
# the pattern truncates mid-template and the result stops parsing. The list is
# kept honest by the AST check at the bottom, which names what is missing instead
# of letting a later step die on "term is not recognized" somewhere unrelated.
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
              'Set-StudioPrivateTempOwner','Get-StudioPrivateTempRoots','New-StudioPrivateTempDirectory','Initialize-StudioTempEnvironment',
              'Restore-StudioTempEnvironment','Write-StudioFinalPathDegraded','Initialize-StudioFinalPathNativeType',
              'Resolve-StudioLinkTarget','Get-StudioSubstTarget','Get-StudioLexicalPath',
              'Resolve-StudioFinalPathInfo','Get-StudioFinalPath','Get-StudioPathHash',
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

# Every Studio function the extracted chain calls must also be defined in it.
# On the AST, not with a regex: install.ps1 GENERATES scripts, and a name inside
# one of those here-string templates is not a call this file makes
# (Test-StudioHealth is defined in the generated launcher and only called there).
foreach ($file in @($newHelpers, $oldHelpers)) {
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$null, [ref]$errors)
    if ($errors.Count -gt 0) {
        throw "extracted $file does not parse: line $($errors[0].Extent.StartLineNumber): $($errors[0].Message)"
    }
    $defined = @($ast.FindAll({
        param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
    }, $true) | ForEach-Object { $_.Name })
    $called = @($ast.FindAll({
        param($n) $n -is [System.Management.Automation.Language.CommandAst]
    }, $true) | ForEach-Object { $_.GetCommandName() } | Where-Object { $_ -like "*-Studio*" })
    $missing = @($called | Sort-Object -Unique | Where-Object { $defined -notcontains $_ })
    if ($missing.Count -gt 0) {
        throw "$file calls undefined functions: $($missing -join ', '). Add them to the list in extract.ps1."
    }
}
Write-Host "extracted $($newNames.Count) new and $($oldNames.Count) old helpers; every Studio call resolves"
