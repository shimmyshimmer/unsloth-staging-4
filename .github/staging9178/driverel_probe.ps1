# A symlink can store a DRIVE-LESS ROOTED target such as "\real". IsPathRooted
# says true, so nothing anchors it to the link, and GetFullPath then resolves it
# against whatever drive the PROCESS is sitting on. Windows itself resolves such
# a target on the LINK's own volume, so the two answers differ whenever the
# process is on another drive -- which is the whole point of measuring it here
# rather than on Linux, where a backslash is not even a separator.
$ErrorActionPreference = 'Stop'
$script:StudioStdoutRedirected = $true
. "$env:RUNNER_TEMP\new-helpers.ps1"
function Add-Type { throw "(0) : error CS2001: no compiler here" }

$linkDrive = (Split-Path -Qualifier $env:RUNNER_TEMP)          # e.g. "D:"
$otherDrive = if ($linkDrive -eq $env:SystemDrive) { $null } else { $env:SystemDrive }
Write-Output "DRIVEREL:linkDrive=$linkDrive"
Write-Output "DRIVEREL:otherDrive=$otherDrive"

$realName = 'stgt9178'
$real = Join-Path "$linkDrive\" $realName
New-Item -ItemType Directory -Path $real -Force | Out-Null
$link = Join-Path $env:RUNNER_TEMP 'driverel-link'
if (Test-Path -LiteralPath $link) { [System.IO.Directory]::Delete($link, $false) }

# mklink /D with a drive-less target stores the target verbatim; that is the
# shape under test, and there is no supported API to create it any other way.
cmd /c mklink /D "$link" "\$realName" | Out-Host
if (-not (Test-Path -LiteralPath $link)) {
    Write-Output "DRIVEREL:created=False"
    return
}
Write-Output "DRIVEREL:created=True"
Write-Output "DRIVEREL:raw=$(@((Get-Item -LiteralPath $link -Force).Target) | Select-Object -First 1)"

$previousCwd = [System.IO.Directory]::GetCurrentDirectory()
try {
    if ($otherDrive) {
        # Set-Location does NOT move the process working directory, and it is the
        # process one that GetFullPath reads.
        [System.IO.Directory]::SetCurrentDirectory("$otherDrive\")
    }
    Write-Output "DRIVEREL:processCwd=$([System.IO.Directory]::GetCurrentDirectory())"
    Write-Output "DRIVEREL:resolved=$(Resolve-StudioLinkTarget -Path $link)"
    Write-Output "DRIVEREL:expected=$real"
} finally {
    [System.IO.Directory]::SetCurrentDirectory($previousCwd)
    try { [System.IO.Directory]::Delete($link, $false) } catch {}
}
