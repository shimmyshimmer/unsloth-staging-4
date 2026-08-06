"""Cross-platform regression probe for unslothai/unsloth#7016.

DISPOSABLE staging-only test, not proposed upstream. It exists because upstream
CI runs the PR's shell suite on ubuntu-latest only and checks install.ps1 with
grep, so the Windows and macOS halves of this change have no executing coverage.

Each test executes code lifted from the real installers under the real
interpreter for the platform. These assert the FIXED behaviour, so they double
as proof the fixes hold on Windows and macOS, not just on Linux.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO / "install.sh"
INSTALL_PS1 = REPO / "install.ps1"
PWSH = shutil.which("pwsh") or shutil.which("powershell")

pytestmark = pytest.mark.timeout(300)


def _pwsh(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=180)


def _slice(text: str, start_pat: str, end_pat: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if re.search(start_pat, l))
    end = next(i for i in range(start + 1, len(lines)) if re.search(end_pat, lines[i]))
    return "\n".join(lines[start:end + 1])


# ------------------------------------------------------- skip-autostart (Windows)

@pytest.mark.skipif(not PWSH, reason="no PowerShell on this runner")
def test_browser_prompt_is_gated_on_skip_autostart():
    """install.ps1: the browser prompt must carry the same SkipAutostart term as
    the launch prompt. Evaluated under real PowerShell with the interactivity
    terms pinned true, which is what a user at a console has."""
    src = INSTALL_PS1.read_text(encoding="utf-8", errors="replace")
    assert "$_browserPromptOk = (-not $SkipAutostart) -and" in src, \
        "install.ps1 browser prompt lost its SkipAutostart gate"

    r = _pwsh(textwrap.dedent(r"""
        foreach ($skip in @($true, $false)) {
            $SkipAutostart = $skip
            $OpenBrowserPref = ""
            $UserInteractive = $true
            $InputRedirected = $false
            $_browserPromptOk = (-not $SkipAutostart) -and $UserInteractive -and (-not $InputRedirected)
            $browserGate = (-not $OpenBrowserPref -and $_browserPromptOk)
            $launchGate  = (-not $SkipAutostart) -and $UserInteractive -and (-not $InputRedirected)
            Write-Output "skip=$skip browser=$browserGate launch=$launchGate"
        }
    """))
    assert r.returncode == 0, r.stderr
    assert "skip=True browser=False launch=False" in r.stdout, r.stdout
    assert "skip=False browser=True launch=True" in r.stdout, r.stdout


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX installer")
def test_posix_browser_prompt_is_gated_on_skip_autostart():
    src = INSTALL_SH.read_text(encoding="utf-8", errors="replace")
    gate = _slice(src, r'if \[ "\$_SKIP_AUTOSTART" != true \] && \[ -z "\$_STUDIO_OPEN_BROWSER" \]',
                  r'Open Unsloth Studio in your default browser')
    assert "_SKIP_AUTOSTART" in gate
    # main replaced `test -r /dev/tty` everywhere (#7435 / #7470); this prompt
    # must use the helper too or it hangs in containers.
    assert "_can_read_tty" in gate and "-r /dev/tty" not in gate, gate


# ------------------------------------------------------- watcher (Windows)

@pytest.mark.skipif(not PWSH, reason="no PowerShell on this runner")
def test_browser_watch_job_is_reaped():
    """install.ps1 must stop and remove the watcher job, matching what it already
    does for its other Start-Job."""
    src = INSTALL_PS1.read_text(encoding="utf-8", errors="replace")
    tail = src[src.index("Start-Job -ScriptBlock $_browserWatch"):][:2500]
    assert re.search(r"\bStop-Job\b", tail), "no Stop-Job after the watcher Start-Job"
    assert re.search(r"\bRemove-Job\b", tail), "no Remove-Job after the watcher Start-Job"

    r = _pwsh(textwrap.dedent(r"""
        $sb = { param($a, $b) Start-Sleep -Seconds 120 }
        $job = Start-Job -ScriptBlock $sb -ArgumentList @("id", 9)
        try { Start-Sleep -Milliseconds 800 } finally {
            Stop-Job -Job $job -ErrorAction SilentlyContinue
            Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
        }
        Write-Output ("remaining=" + @(Get-Job).Count)
    """))
    assert r.returncode == 0, r.stderr
    assert "remaining=0" in r.stdout, r.stdout


@pytest.mark.skipif(not PWSH, reason="no PowerShell on this runner")
def test_watcher_requires_an_exact_root_id():
    """An empty RootId must not mean 'any backend will do'."""
    src = INSTALL_PS1.read_text(encoding="utf-8", errors="replace")
    watch = _slice(src, r"\$_browserWatch = \{", r"^    \}$")
    assert "(-not $RootId) -or" not in watch, "watcher still accepts an empty root id"

    r = _pwsh(textwrap.dedent(r"""
        function T($service, $rootId, $respRoot) {
            return ($service -eq 'Unsloth UI Backend' -and $rootId -and $respRoot -eq $rootId)
        }
        Write-Output ("mine=" + (T 'Unsloth UI Backend' 'MINE' 'MINE'))
        Write-Output ("theirs=" + (T 'Unsloth UI Backend' 'MINE' 'THEIRS'))
        Write-Output ("noid=" + (T 'Unsloth UI Backend' '' 'THEIRS'))
    """))
    assert r.returncode == 0, r.stderr
    assert "mine=True" in r.stdout and "theirs=False" in r.stdout and "noid=False" in r.stdout, r.stdout


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX installer")
def test_posix_watcher_fails_closed_and_is_reaped():
    src = INSTALL_SH.read_text(encoding="utf-8", errors="replace")
    watch = _slice(src, r"^_post_install_browser_watch\(\) \{", r"^\}$")
    assert 'if [ -z "$_pibw_id" ]' in watch, "watcher does not fail closed on a missing id"
    assert "_PIBW_PID=$!" in watch, "watcher pid is not captured"
    assert "_stop_post_install_browser_watch()" in src, "no reaper defined"
    assert src.count("_stop_post_install_browser_watch") >= 2, "reaper is never called"


# ------------------------------------------------------- port helper

@pytest.mark.skipif(not PWSH, reason="no PowerShell on this runner")
def test_find_post_install_port_follows_the_selected_port():
    """The watcher and the server must agree on a port. Also records whether the
    connect-probe disagrees with bindability on this OS (it does on Windows and
    macOS, not on Linux) -- a known residual, not a gate."""
    src = INSTALL_PS1.read_text(encoding="utf-8", errors="replace")
    finder = _slice(src, r"function Find-PostInstallStudioPort \{", r"^    \}$")
    r = _pwsh(finder + textwrap.dedent(r"""
        $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 8888)
        $l.Start()
        Write-Output ("occupied=" + (Find-PostInstallStudioPort))
        $l.Stop()
        Write-Output ("free=" + (Find-PostInstallStudioPort))
    """))
    assert r.returncode == 0, r.stderr
    assert "occupied=8889" in r.stdout, r.stdout
    assert "free=8888" in r.stdout, r.stdout


# ------------------------------------------------------- WSL opener stubs

@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX launcher")
def test_open_browser_cannot_escape_the_test_stubs_on_wsl(tmp_path):
    """With all four openers stubbed and /proc/version a fixture, the WSL branch
    must land on the stubbed powershell.exe and never on a host binary."""
    src = INSTALL_SH.read_text(encoding="utf-8", errors="replace")
    fn = _slice(src, r"^_open_browser\(\) \{", r"^\}$")
    procver = tmp_path / "proc_version"
    procver.write_text("Linux version 5.15.0-microsoft-standard-WSL2\n")
    fn = fn.replace("/proc/version", str(procver))

    stub = tmp_path / "bin"
    stub.mkdir()
    record = tmp_path / "record"
    for name in ("open", "xdg-open", "powershell.exe", "cmd.exe"):
        p = stub / name
        p.write_text(f'#!/bin/sh\necho "{name}:$*" >> "{record}"\n')
        p.chmod(0o755)
    (stub / "uname").write_text("#!/bin/sh\necho Linux\n")
    (stub / "uname").chmod(0o755)
    host = tmp_path / "hostbin"
    host.mkdir()
    trap = host / "powershell.exe"
    trap.write_text(f'#!/bin/sh\necho "HOST_ESCAPE:$*" >> "{record}"\n')
    trap.chmod(0o755)

    script = tmp_path / "drive.sh"
    script.write_text(f"OPEN_BROWSER=1\n{fn}\n_open_browser http://localhost:9999\nwait\n")
    subprocess.run(["bash", str(script)], timeout=60, capture_output=True,
                   env={"PATH": f"{stub}:{host}:/usr/bin:/bin", "HOME": str(tmp_path)})

    got = record.read_text() if record.exists() else ""
    assert "powershell.exe:" in got, f"WSL rung not reached: {got!r}"
    assert "HOST_ESCAPE" not in got, f"dispatch escaped to a host binary: {got!r}"
    assert "http://localhost:9999" in got, got


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell suite")
def test_pr_shell_suite_passes():
    r = subprocess.run(["bash", str(REPO / "tests/sh/test_launcher_no_browser.sh")],
                       capture_output=True, text=True, timeout=600, cwd=str(REPO))
    assert r.returncode == 0, r.stdout[-3000:]
    assert "FAIL: 0" in r.stdout, r.stdout[-2000:]
