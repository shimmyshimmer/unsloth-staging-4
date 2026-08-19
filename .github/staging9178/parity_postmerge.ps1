# Old vs new, on a HEALTHY Windows PowerShell host: the compiler works and TEMP
# is fine, which is the state every ordinary user is in. The claim under test is
# that the change is invisible there, so the install mutex name must come out
# byte-identical from both revisions for every path shape worth trying. A single
# divergence would mean two installers could enter one directory.
#
# Measured here rather than on Linux because the PRE-change resolver is a
# P/Invoke to GetFinalPathNameByHandle: off Windows it throws for every input,
# and comparing against that would only re-measure the absence of Win32.
$ErrorActionPreference = 'Stop'

$names = @('Write-StudioLine','Test-StudioDirectoryUsable','Remove-StudioStalePrivateTempDirectories',
           'Get-StudioPrivateTempRoots','New-StudioPrivateTempDirectory','Set-StudioPrivateTempOwner',
           'Initialize-StudioTempEnvironment','Restore-StudioTempEnvironment','Write-StudioFinalPathDegraded',
           'Initialize-StudioFinalPathNativeType','Resolve-StudioLinkTarget','Get-StudioSubstTarget',
           'Get-StudioLexicalPath','Resolve-StudioFinalPathInfo','Get-StudioFinalPath','Get-StudioPathHash',
           'Get-StudioInstallMutexName','Test-StudioPathEqual','Enter-StudioNamedMutex',
           'Enter-StudioInstallMutex','Exit-StudioInstallMutex')

function Get-Names {
    param([string]$Source, [string[]]$Paths, [string]$TypeSuffix)
    $src = Get-Content -Raw -LiteralPath $Source
    # The two revisions define the same C# type name, and a session can only hold
    # one. Each side therefore runs in its OWN child process.
    $body = @()
    $body += '$script:StudioStdoutRedirected = $true'
    foreach ($n in $names) {
        $m = [regex]::Match($src, "    function $n \{.*?\n    \}\n", 'Singleline')
        if ($m.Success) { $body += $m.Value }
    }
    $body += 'foreach ($p in $args) {'
    $body += '    $r = try { Get-StudioInstallMutexName -Path $p } catch { "THREW:" + $_.Exception.Message }'
    $body += '    Write-Output ("{0}`t{1}" -f $p, $r)'
    $body += '}'
    $file = Join-Path $env:RUNNER_TEMP ("parity-" + $TypeSuffix + ".ps1")
    Set-Content -LiteralPath $file -Value ($body -join "`n") -Encoding UTF8
    $out = & powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $file @Paths
    $table = @{}
    foreach ($line in @($out)) {
        $parts = ([string]$line) -split "`t", 2
        if ($parts.Count -eq 2) { $table[$parts[0]] = $parts[1] }
    }
    return $table
}

# Real directories where it matters, so the resolver has something to resolve.
$base = Join-Path $env:RUNNER_TEMP 'parity'
$deep = Join-Path $base ('d' * 60)
New-Item -ItemType Directory -Path (Join-Path $base 'a b c\studio') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $base 'Studio') -Force | Out-Null
New-Item -ItemType Directory -Path $deep -Force | Out-Null
# A junction and its target: the two spellings of one directory.
$linkTarget = Join-Path $base 'realtarget'
New-Item -ItemType Directory -Path $linkTarget -Force | Out-Null
cmd /c mklink /J (Join-Path $base 'link') $linkTarget | Out-Host

$paths = @(
    (Join-Path $base 'Studio'),
    (Join-Path $base 'Studio\'),
    (Join-Path $base 'studio'),
    (Join-Path $base 'STUDIO'),
    (Join-Path $base 'a b c\studio'),
    (Join-Path $base 'a b c\..\a b c\studio'),
    (Join-Path $base 'missing\deeper\still'),
    $deep,
    (Join-Path $base 'link'),
    $linkTarget,
    $env:SystemDrive + '\',
    $env:RUNNER_TEMP,
    (Join-Path $env:RUNNER_TEMP 'studio-unicode-eu'),
    '\\?\' + (Join-Path $base 'Studio')
)

$before = Get-Names -Source "$env:RUNNER_TEMP\old-install.ps1" -Paths $paths -TypeSuffix 'old'
$after = Get-Names -Source "$env:RUNNER_TEMP\new-install.ps1" -Paths $paths -TypeSuffix 'new'

$mismatch = 0
foreach ($p in $paths) {
    $b = $before[$p]
    $a = $after[$p]
    if ($b -eq $a) {
        Write-Output ("PARITY:ok`t{0}`t{1}" -f $p, $a)
    } else {
        $mismatch++
        Write-Output ("PARITY:MISMATCH`t{0}`n   before={1}`n   after ={2}" -f $p, $b, $a)
    }
}
Write-Output ("PARITY:mismatches={0} of {1}" -f $mismatch, $paths.Count)
if ($mismatch -gt 0) { throw "the change is not invisible on a healthy host" }
