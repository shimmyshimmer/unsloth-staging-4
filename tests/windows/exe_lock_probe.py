# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Diagnose which process holds unsloth.exe when `unsloth studio update` cannot
replace it (issue #7697).

Prints the process chain and then tries the same os.replace the updater does, so
the failure can be attributed instead of guessed at. Read-only apart from the
rename it attempts and immediately undoes.
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path


def _proc_chain():
    """(pid, name, exe) for this process and its ancestors, via CreateToolhelp32Snapshot."""
    out = []
    try:
        import ctypes.wintypes as w

        TH32CS_SNAPPROCESS = 0x00000002

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", w.DWORD), ("cntUsage", w.DWORD), ("th32ProcessID", w.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", w.DWORD), ("cntThreads", w.DWORD),
                ("th32ParentProcessID", w.DWORD), ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", w.DWORD), ("szExeFile", ctypes.c_char * 260),
            ]

        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        entry = PROCESSENTRY32(); entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        table = {}
        if k32.Process32First(snap, ctypes.byref(entry)):
            while True:
                table[entry.th32ProcessID] = (
                    entry.szExeFile.decode(errors="replace"), entry.th32ParentProcessID
                )
                if not k32.Process32Next(snap, ctypes.byref(entry)):
                    break
        k32.CloseHandle(snap)
        pid = os.getpid()
        for _ in range(8):
            if pid not in table:
                break
            name, parent = table[pid]
            out.append({"pid": pid, "name": name})
            pid = parent
    except Exception as e:  # diagnostics must never be the thing that fails
        out.append({"error": repr(e)})
    return out


def _try_replace(exe: Path):
    """The updater's own move, reported rather than swallowed."""
    if not exe.is_file():
        return {"path": str(exe), "exists": False}
    stale = exe.with_suffix(".exe.probe")
    try:
        os.replace(exe, stale)
    except OSError as e:
        return {"path": str(exe), "exists": True, "renamed": False,
                "winerror": getattr(e, "winerror", None), "err": str(e)}
    try:
        os.replace(stale, exe)          # put it straight back
    except OSError as e:
        return {"path": str(exe), "exists": True, "renamed": True,
                "restore_failed": str(e)}
    return {"path": str(exe), "exists": True, "renamed": True}


def main():
    home = Path(os.path.expanduser("~")) / ".unsloth" / "studio"
    report = {
        "sys_executable": sys.executable,
        "argv0": sys.argv[0],
        "pid": os.getpid(),
        "process_chain": _proc_chain(),
        "scripts_exe": _try_replace(home / "unsloth_studio" / "Scripts" / "unsloth.exe"),
        "shim_exe": _try_replace(home / "bin" / "unsloth.exe"),
    }
    print(json.dumps(report, indent = 2))


if __name__ == "__main__":
    main()
