# Does Remove-Item -Recurse walk INTO a junction nested below the directory it is
# deleting, on Windows PowerShell 5.1? The sweep only checks the top-level ust-*
# entry for the reparse attribute, so a junction left inside one by any other tool
# would be reached by the recursive delete.
$ErrorActionPreference = 'Continue'
$stale = $env:NESTED_STALE
$target = $env:NESTED_TARGET
try {
    Remove-Item -LiteralPath $stale -Recurse -Force -ErrorAction SilentlyContinue
} catch {
    Write-Output "NESTED:threw:$($_.Exception.GetType().Name)"
}
Write-Output "NESTED:stale-gone=$(-not (Test-Path -LiteralPath $stale))"
Write-Output "NESTED:target-intact=$(Test-Path -LiteralPath (Join-Path $target 'keepme.txt'))"
