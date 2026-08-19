# Does New-Item -Path treat [] in a path as a wildcard on Windows PowerShell 5.1?
# Test-StudioDirectoryUsable creates the candidate before probing it, so if this
# fails on a profile like C:\Users\a[b]\AppData\Local every private-temp candidate
# fails and the installer stays on the broken %TEMP% it was trying to escape.
$ErrorActionPreference = 'Continue'
$base = Join-Path $env:RUNNER_TEMP ("wild-" + [guid]::NewGuid().ToString('N').Substring(0, 6))
$bracket = Join-Path $base "Local[1]"
$nested = Join-Path $bracket "Unsloth Studio\temp\ust-1-abc"
try {
    New-Item -ItemType Directory -Path $nested -Force -ErrorAction Stop | Out-Null
    Write-Output "WILD:newitem=ok"
} catch {
    Write-Output "WILD:newitem=threw:$($_.Exception.GetType().Name)"
}
Write-Output "WILD:newitem-exists=$(Test-Path -LiteralPath $nested -PathType Container)"
$nested2 = Join-Path $bracket "Unsloth Studio\temp\ust-2-abc"
try {
    [System.IO.Directory]::CreateDirectory($nested2) | Out-Null
    Write-Output "WILD:createdirectory=ok"
} catch {
    Write-Output "WILD:createdirectory=threw:$($_.Exception.GetType().Name)"
}
Write-Output "WILD:createdirectory-exists=$(Test-Path -LiteralPath $nested2 -PathType Container)"
