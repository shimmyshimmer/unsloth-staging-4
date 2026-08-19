# The parts of the identity fix whose ANSWER is platform-dependent, so Linux
# cannot decide them: whether the rewritten UNC form is rooted (Path.IsPathRooted
# says yes on Windows, no on Linux), and whether a real SUBST drive folds onto its
# physical target end to end.
$ErrorActionPreference = 'Stop'
$script:StudioStdoutRedirected = $true
. "$env:RUNNER_TEMP\new-helpers.ps1"
function Add-Type { throw "(0) : error CS2001: no compiler here" }

Write-Output "ROOTED:unc=$([System.IO.Path]::IsPathRooted('\\server\share\dir'))"
Write-Output "ROOTED:drive=$([System.IO.Path]::IsPathRooted('C:\real\target'))"
Write-Output "ROOTED:bare-unc-marker=$([System.IO.Path]::IsPathRooted('UNC\server\share'))"
# A mounted folder's target arrives as \??\Volume{GUID}\..., and the bare form is
# no more rooted than the bare UNC marker was.
Write-Output "ROOTED:volume-guid=$([System.IO.Path]::IsPathRooted('\\?\Volume{11111111-2222-3333-4444-555555555555}\dir'))"
Write-Output "ROOTED:bare-volume-marker=$([System.IO.Path]::IsPathRooted('Volume{11111111-2222-3333-4444-555555555555}\dir'))"
# And what the final normalization would do to each: GetPathRoot has to stay
# non-empty for the extended form, or the relaxed process comparison is disabled
# and the two spellings of one volume hash differently.
Write-Output "ROOT:extended=[$([System.IO.Path]::GetPathRoot('\\?\Volume{11111111-2222-3333-4444-555555555555}\dir'))]"
Write-Output "ROOT:stripped=[$([System.IO.Path]::GetPathRoot('Volume{11111111-2222-3333-4444-555555555555}\dir'))]"

# A real SUBST drive, resolved through the real helper (no injected map).
$target = Join-Path $env:RUNNER_TEMP 'substroot'
$venv = Join-Path $target 'venv'
New-Item -ItemType Directory -Path $venv -Force | Out-Null
cmd /c subst "Y:" "$target" | Out-Host
try {
    $script:StudioSubstMap = $null
    Write-Output "REALSUBST:mapped=$(Get-StudioSubstTarget -Path 'Y:\venv')"
    $viaAlias = Get-StudioFinalPath -Path "Y:\venv"
    $viaReal = Get-StudioFinalPath -Path $venv
    Write-Output "REALSUBST:alias=$viaAlias"
    Write-Output "REALSUBST:real=$viaReal"
    Write-Output "REALSUBST:same=$([string]::Equals($viaAlias, $viaReal, [System.StringComparison]::OrdinalIgnoreCase))"
} finally {
    cmd /c subst "Y:" /d | Out-Host
}

# 8.3 short names, end to end through the real resolver. Measured separately:
# on the system volume 8dot3 creation is ON, and of the compiler-free APIs only
# GetFullPath and (Get-Item).FullName expand a short name -- Resolve-Path,
# FileSystemObject.Path and cmd's %~f all hand the short form straight back.
# Get-StudioLexicalPath already starts with GetFullPath, so the question is
# whether the two spellings reach the same identity without any further work.
$snRoot = Join-Path $env:SystemDrive 'sn2'
$snLong = Join-Path $snRoot 'Program Files Like This'
New-Item -ItemType Directory -Path (Join-Path $snLong 'venv') -Force | Out-Null
$fso = New-Object -ComObject Scripting.FileSystemObject
$snShort = $fso.GetFolder($snLong).ShortPath
Write-Output "SHORTID:short=$snShort"
Write-Output "SHORTID:long=$snLong"
if ($snShort -ne $snLong) {
    $viaShort = Get-StudioFinalPath -Path (Join-Path $snShort 'venv')
    $viaLong = Get-StudioFinalPath -Path (Join-Path $snLong 'venv')
    Write-Output "SHORTID:viaShort=$viaShort"
    Write-Output "SHORTID:viaLong=$viaLong"
    Write-Output "SHORTID:same=$([string]::Equals($viaShort, $viaLong, [System.StringComparison]::OrdinalIgnoreCase))"
} else {
    Write-Output "SHORTID:same=n/a (no short name on this volume)"
}
