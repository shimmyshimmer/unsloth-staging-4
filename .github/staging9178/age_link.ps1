# Age a reparse point ITSELF.
#
# `(Get-Item $link).LastWriteTime = ...` opens the junction without
# FILE_FLAG_OPEN_REPARSE_POINT, so the write follows the link and lands on the
# target while the link keeps today's timestamp. The sweep reads the link's own
# timestamp, so a junction aged that way still looks fresh and is skipped at the
# cutoff before the reparse-point guard is ever reached. Open the link itself and
# set the time on that handle.
$ErrorActionPreference = 'Stop'
$source = @'
using System;
using System.Runtime.InteropServices;
public static class UnslothProbeLinkTime {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr CreateFileW(string name, uint access, uint share, IntPtr sec,
                                     uint disposition, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetFileTime(IntPtr h, IntPtr creation, IntPtr access, ref long write);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool CloseHandle(IntPtr h);
    public static void AgeDays(string path, int days) {
        const uint FILE_WRITE_ATTRIBUTES = 0x100;
        const uint SHARE_ALL = 7;
        const uint OPEN_EXISTING = 3;
        const uint BACKUP_SEMANTICS = 0x02000000;
        const uint OPEN_REPARSE_POINT = 0x00200000;
        IntPtr h = CreateFileW(path, FILE_WRITE_ATTRIBUTES, SHARE_ALL, IntPtr.Zero, OPEN_EXISTING,
                               BACKUP_SEMANTICS | OPEN_REPARSE_POINT, IntPtr.Zero);
        if (h == new IntPtr(-1)) throw new Exception("open failed: " + Marshal.GetLastWin32Error());
        try {
            long when = DateTime.Now.AddDays(-days).ToFileTime();
            if (!SetFileTime(h, IntPtr.Zero, IntPtr.Zero, ref when))
                throw new Exception("SetFileTime failed: " + Marshal.GetLastWin32Error());
        } finally { CloseHandle(h); }
    }
}
'@
if (-not ('UnslothProbeLinkTime' -as [type])) { Add-Type -TypeDefinition $source }
[UnslothProbeLinkTime]::AgeDays($env:AGE_LINK_PATH, 3)
Write-Host "aged the reparse point at $env:AGE_LINK_PATH"
