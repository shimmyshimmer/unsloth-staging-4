# CONTROL: two FRESH installs with the SAME (unpatched) scripts.
# If the same files differ here as differ across the patch, those files are per-install
# nondeterminism (install ids, timestamps) and the patch changed nothing. If they come out
# identical here, then the patch IS the cause and the change is not safe.
$ErrorActionPreference = 'Continue'
$repo  = $env:GITHUB_WORKSPACE
$home_ = Join-Path $repo '.studio-home'

$volatile = @('\.log$', '\.pyc$', '__pycache__', '[\\/]logs?[\\/]', '[\\/]\.cache[\\/]',
              '[\\/]tmp[\\/]', '[\\/]temp[\\/]', '\.lock$', '\.pid$', 'RECORD$', 'INSTALLER$',
              'direct_url\.json$', '\.dist-info[\\/]', 'pip-selfcheck', '\.tmp$')

function Get-Manifest($root, $outFile) {
    $rows = @()
    Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.FullName.Substring($root.Length).TrimStart('\','/')
        foreach ($v in $volatile) { if ($rel -match $v) { return } }
        try { $h = (Get-FileHash $_.FullName -Algorithm SHA256).Hash } catch { $h = 'UNREADABLE' }
        $rows += "$rel`t$h"
    }
    ($rows | Sort-Object) | Set-Content -LiteralPath $outFile -Encoding UTF8
    Write-Host "MANIFEST`t$outFile`t$($rows.Count) files"
}

function Invoke-Install($tag) {
    $env:UNSLOTH_STUDIO_HOME = $home_
    $env:UNSLOTH_NO_TORCH = '1'; $env:UNSLOTH_SKIP_AUTOSTART = '1'
    $env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'install.ps1') --local --no-torch 2>&1 |
        Out-File -FilePath (Join-Path $repo "ctl_$tag.log")
    Write-Host "INSTALL`t$tag`texit=$LASTEXITCODE"
}

# Both runs use the UNPATCHED scripts -- the only variable is "a second fresh install".
git -C $repo show origin/main:install.ps1      | Set-Content -LiteralPath (Join-Path $repo 'install.ps1') -Encoding UTF8
git -C $repo show origin/main:studio/setup.ps1 | Set-Content -LiteralPath (Join-Path $repo 'studio/setup.ps1') -Encoding UTF8
Write-Host "CONTROL`tboth runs use UNPATCHED scripts"

Remove-Item -Recurse -Force $home_ -ErrorAction SilentlyContinue
Invoke-Install 'ctlA'; Get-Manifest $home_ (Join-Path $repo 'ctlA.txt')
Remove-Item -Recurse -Force $home_ -ErrorAction SilentlyContinue
Invoke-Install 'ctlB'; Get-Manifest $home_ (Join-Path $repo 'ctlB.txt')

$d = Compare-Object (Get-Content (Join-Path $repo 'ctlA.txt')) (Get-Content (Join-Path $repo 'ctlB.txt'))
if (-not $d) { Write-Host "CONTROL_RESULT`tIDENTICAL -- two unpatched installs match byte for byte" }
else {
    Write-Host "CONTROL_RESULT`t$($d.Count) differing lines between two UNPATCHED installs:"
    $d | ForEach-Object { Write-Host "   $($_.SideIndicator) $($_.InputObject)" }
}
