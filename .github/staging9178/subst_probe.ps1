# Which API reveals a SUBST drive's target on Windows PowerShell 5.1, WITHOUT
# Add-Type? The whole point is hosts where the compiler is unavailable, so
# QueryDosDevice via P/Invoke is not an option there.
$ErrorActionPreference = 'Continue'
$target = $env:SUBST_TARGET
$drive = $env:SUBST_DRIVE            # e.g. "X:"
cmd /c subst "$drive" "$target" | Out-Host
try {
    Write-Output "SUBST:exists=$(Test-Path -LiteralPath ($drive + '\'))"

    $psd = Get-PSDrive -Name $drive.Substring(0, 1) -ErrorAction SilentlyContinue
    Write-Output "SUBST:psdrive-displayroot=[$($psd.DisplayRoot)]"
    Write-Output "SUBST:psdrive-root=[$($psd.Root)]"

    $item = Get-Item -LiteralPath ($drive + '\') -Force -ErrorAction SilentlyContinue
    Write-Output "SUBST:getitem-target=[$($item.Target)]"
    Write-Output "SUBST:getitem-attrs=[$($item.Attributes)]"

    $listing = @(cmd /c subst)
    Write-Output "SUBST:subst-listing=[$($listing -join ' | ')]"

    $wmi = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='$drive'" -ErrorAction SilentlyContinue
    Write-Output "SUBST:wmi-providername=[$($wmi.ProviderName)]"
    Write-Output "SUBST:wmi-drivetype=[$($wmi.DriveType)]"

    $full = [System.IO.Path]::GetFullPath(($drive + '\venv\Scripts\python.exe'))
    Write-Output "SUBST:getfullpath=[$full]"
    $rp = $null
    try { $rp = (Resolve-Path -LiteralPath ($drive + '\') -ErrorAction Stop).ProviderPath } catch {}
    Write-Output "SUBST:resolvepath=[$rp]"
} finally {
    cmd /c subst "$drive" /d | Out-Host
}
