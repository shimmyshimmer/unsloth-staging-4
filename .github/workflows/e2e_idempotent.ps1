# End-to-end: does the $ProgressPreference change alter WHAT gets installed, and is a
# second install a no-op?
#
#   run0  UNPATCHED scripts        -> manifest0
#   run1  PATCHED scripts (fresh)  -> manifest1     manifest0 == manifest1  => output identical
#   run2  PATCHED scripts (again)  -> manifest2     manifest1 == manifest2  => idempotent
#
# Speed is NOT the subject here: a hosted runner already has Python and the VC++ runtime, so
# both slow downloads short-circuit. Repo A measures the speed; this measures safety.
$ErrorActionPreference = 'Continue'
$repo = $env:GITHUB_WORKSPACE
$home_ = Join-Path $repo '.studio-home'
$fail = 0

# Volatile by nature: logs, caches, compiled bytecode, anything carrying a timestamp, pid or
# random venv path. Excluding them is what makes two installs comparable at all; everything
# else must match exactly.
$volatile = @(
    '\.log$', '\.pyc$', '__pycache__', '[\\/]logs?[\\/]', '[\\/]\.cache[\\/]',
    '[\\/]tmp[\\/]', '[\\/]temp[\\/]', '\.lock$', '\.pid$', 'RECORD$', 'INSTALLER$',
    'direct_url\.json$', '\.dist-info[\\/]', 'pip-selfcheck', '\.tmp$'
)

function Get-Manifest($root, $outFile) {
    if (-not (Test-Path $root)) { "MANIFEST`t$outFile`tMISSING ROOT $root"; return 0 }
    $rows = @()
    Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.FullName.Substring($root.Length).TrimStart('\', '/')
        foreach ($v in $volatile) { if ($rel -match $v) { return } }
        try { $h = (Get-FileHash $_.FullName -Algorithm SHA256).Hash } catch { $h = 'UNREADABLE' }
        $rows += "$rel`t$h"
    }
    $rows = $rows | Sort-Object
    $rows | Set-Content -LiteralPath $outFile -Encoding UTF8
    "MANIFEST`t$outFile`t$($rows.Count) files"
    return $rows.Count
}

function Invoke-Install($tag) {
    $env:UNSLOTH_STUDIO_HOME   = $home_
    $env:UNSLOTH_NO_TORCH      = '1'
    $env:UNSLOTH_SKIP_AUTOSTART = '1'
    $env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'install.ps1') --local --no-torch 2>&1 |
        Tee-Object -FilePath (Join-Path $repo "install_$tag.log") | Out-Null
    $code = $LASTEXITCODE
    $sw.Stop()
    "INSTALL`t$tag`texit=$code`t$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
    return $code
}

function Compare-Manifest($a, $b, $label) {
    if (-not (Test-Path $a) -or -not (Test-Path $b)) { "COMPARE`t$label`tSKIP (missing manifest)"; return 1 }
    $d = Compare-Object (Get-Content $a) (Get-Content $b)
    if (-not $d) { "COMPARE`t$label`tPASS`tmanifests identical"; return 0 }
    "COMPARE`t$label`tFAIL`t$($d.Count) differing entries:"
    $d | Select-Object -First 40 | ForEach-Object { "   $($_.SideIndicator) $($_.InputObject)" }
    return 1
}

# ---- run0: UNPATCHED (restore the two scripts from upstream main) ----
git -C $repo show origin/main:install.ps1       | Set-Content -LiteralPath (Join-Path $repo 'install.ps1') -Encoding UTF8
git -C $repo show origin/main:studio/setup.ps1  | Set-Content -LiteralPath (Join-Path $repo 'studio/setup.ps1') -Encoding UTF8
"PATCH`trun0 uses UNPATCHED install.ps1 + setup.ps1 (from origin/main)"
Remove-Item -Recurse -Force $home_ -ErrorAction SilentlyContinue
$null = Invoke-Install 'run0_unpatched'
$null = Get-Manifest $home_ (Join-Path $repo 'manifest0.txt')

# ---- restore the patched scripts ----
git -C $repo checkout -- install.ps1 studio/setup.ps1
"PATCH`trun1/run2 use PATCHED install.ps1 + setup.ps1"
if ((Select-String -Path (Join-Path $repo 'install.ps1') -Pattern "ProgressPreference" -Quiet) -and
    (Select-String -Path (Join-Path $repo 'studio/setup.ps1') -Pattern "ProgressPreference" -Quiet)) {
    "PATCH`tconfirmed present in both scripts"
} else { "PATCH`tFAIL patched scripts do not contain ProgressPreference"; $fail = 1 }

# ---- run1: PATCHED, fresh ----
Remove-Item -Recurse -Force $home_ -ErrorAction SilentlyContinue
$null = Invoke-Install 'run1_patched_fresh'
$null = Get-Manifest $home_ (Join-Path $repo 'manifest1.txt')

# ---- run2: PATCHED again over the top -> idempotency ----
$null = Invoke-Install 'run2_patched_rerun'
$null = Get-Manifest $home_ (Join-Path $repo 'manifest2.txt')

$fail += Compare-Manifest (Join-Path $repo 'manifest0.txt') (Join-Path $repo 'manifest1.txt') 'unpatched-vs-patched(OUTPUT IDENTICAL)'
$fail += Compare-Manifest (Join-Path $repo 'manifest1.txt') (Join-Path $repo 'manifest2.txt') 'patched-run1-vs-run2(IDEMPOTENT)'

if ($fail -ne 0) { exit 1 }
"ALL CHECKS PASSED"
