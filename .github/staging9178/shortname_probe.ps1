# Which API expands an 8.3 short name to its long form WITHOUT a compiler?
# The hosts this fix serves are exactly the ones where Add-Type does not work, so
# GetLongPathName via P/Invoke is not available there.
$ErrorActionPreference = 'Continue'
# 8dot3 name creation is per-VOLUME: enabled by default on the system volume,
# disabled by default on other volumes since Windows 8. RUNNER_TEMP is on D:
# here, sotest both or the answer is only about the wrong disk.
$root = Join-Path $env:PROBE_ROOT 'shortname'
$long = Join-Path $root 'Program Files Like This'
New-Item -ItemType Directory -Path (Join-Path $long 'venv') -Force | Out-Null
Write-Output "SHORT:volume=[$([System.IO.Path]::GetPathRoot($long))]"

# Ask the filesystem for the short name it actually assigned, if any.
$short = $null
try {
    $fso = New-Object -ComObject Scripting.FileSystemObject
    $short = $fso.GetFolder($long).ShortPath
} catch { Write-Output "SHORT:fso-shortpath-threw=$($_.Exception.GetType().Name)" }
Write-Output "SHORT:short=[$short]"
if ([string]::IsNullOrWhiteSpace($short) -or $short -eq $long) {
    Write-Output "SHORT:8dot3-disabled=True"
    return
}

Write-Output "SHORT:getfullpath=[$([System.IO.Path]::GetFullPath($short))]"
try { Write-Output "SHORT:resolvepath=[$((Resolve-Path -LiteralPath $short).ProviderPath)]" } catch {}
try { Write-Output "SHORT:getitem-fullname=[$((Get-Item -LiteralPath $short -Force).FullName)]" } catch {}
try {
    $fso2 = New-Object -ComObject Scripting.FileSystemObject
    Write-Output "SHORT:fso-path=[$($fso2.GetFolder($short).Path)]"
} catch { Write-Output "SHORT:fso-path-threw=$($_.Exception.GetType().Name)" }
try {
    $expanded = & cmd /c "for %I in (`"$short`") do @echo %~fI"
    Write-Output "SHORT:cmd-tilde-f=[$expanded]"
} catch {}
