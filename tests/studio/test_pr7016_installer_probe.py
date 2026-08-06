"""Cross-platform probe for unslothai/unsloth#7016 (launcher --no-browser).

DISPOSABLE staging-only test. Not proposed upstream: it exists to give the
Windows and macOS halves of PR 7016 real execution, because upstream CI runs the
PR's own shell test on ubuntu-latest only and checks install.ps1 with grep.

Each test evaluates code lifted out of the real installers under the real
interpreter for the platform, rather than asserting on source text.
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


def _pwsh(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, env=e, timeout=180,
    )


def _slice(text: str, start_pat: str, end_pat: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if re.search(start_pat, l))
    end = next(i for i in range(start + 1, len(lines)) if re.search(end_pat, lines[i]))
    return "\n".join(lines[start:end + 1])


# ---------------------------------------------------------------- finding A (Windows)

@pytest.mark.skipif(not PWSH, reason="no PowerShell on this runner")
def test_a_browser_prompt_gate_ignores_skip_autostart():
    """install.ps1:2636 omits the `-not $SkipAutostart` term that :2690 has.

    Evaluates both gate expressions under real PowerShell with the two
    interactivity terms pinned true, which is what a user at a console has.
    """
    src = INSTALL_PS1.read_text(encoding="utf-8", errors="replace")
    assert "$_browserPromptOk" in src, "install.ps1 no longer defines $_browserPromptOk"

    script = textwrap.dedent(r"""
        $SkipAutostart = $true
        $OpenBrowserPref = ""
        # Pin the two interactivity terms true (a real console), isolating the
        # only difference between the two gates: the $SkipAutostart term.
        $UserInteractive = $true
        $InputRedirected = $false

        $_browserPromptOk = $UserInteractive -and (-not $InputRedirected)
        $browserGate = (-not $OpenBrowserPref -and $_browserPromptOk)
        $launchGate  = (-not $SkipAutostart) -and $UserInteractive -and (-not $InputRedirected)

        Write-Output "browserGate=$browserGate"
        Write-Output "launchGate=$launchGate"
    """)
    r = _pwsh(script)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "launchGate=False" in out, f"documented gate broke: {out}"
    # The bug: with SkipAutostart set, the launch prompt is suppressed but the
    # new browser prompt is not.
    assert "browserGate=True" in out, f"finding A not reproduced on this runner: {out}"


# ---------------------------------------------------------------- finding E (Windows)

@pytest.mark.skipif(not PWSH, reason="no PowerShell on this runner")
@pytest.mark.parametrize(
    "variant",
    ["missing", "bom", "crlf_reindented", "pre_pr_launcher"],
)
def test_e_preference_readback_silently_resets_to_on(tmp_path, variant):
    """install.ps1:666-675 is the ONLY store of the Windows no-browser choice.

    Every parse failure falls through to '1' (browser on). Unix persists the
    same choice in studio.conf, which does not have these failure modes.
    """
    launcher = tmp_path / "launch-studio.ps1"
    good = "$openBrowserDefault = '0'\n"
    if variant == "missing":
        pass  # no file at all
    elif variant == "bom":
        launcher.write_bytes(b"\xef\xbb\xbf" + good.encode())
    elif variant == "crlf_reindented":
        launcher.write_bytes(b"    $openBrowserDefault = '0'\r\n")
    elif variant == "pre_pr_launcher":
        launcher.write_text("# launcher generated before PR 7016\n$timeoutSec = 60\n")

    script = textwrap.dedent(rf"""
        $OpenBrowserPref = ""
        $launcherPs1 = '{launcher.as_posix()}'
        # verbatim from install.ps1:666-675
        $_openBrowser = $OpenBrowserPref
        if (-not $_openBrowser -and (Test-Path -LiteralPath $launcherPs1)) {{
            try {{
                $_prevLauncher = [System.IO.File]::ReadAllText($launcherPs1)
                if ($_prevLauncher -match "(?m)^\$openBrowserDefault = '([01])'") {{
                    $_openBrowser = $Matches[1]
                }}
            }} catch {{}}
        }}
        if ($_openBrowser -ne '0') {{ $_openBrowser = '1' }}
        Write-Output "resolved=$_openBrowser"
    """)
    r = _pwsh(script)
    assert r.returncode == 0, r.stderr
    resolved = re.search(r"resolved=(\d)", r.stdout).group(1)
    if variant == "bom":
        # A BOM does not shift the first line for .NET ReadAllText, so this one
        # should survive. Recorded either way.
        print(f"[{variant}] resolved={resolved}")
    else:
        assert resolved == "1", f"[{variant}] expected silent reset to on, got {resolved}"
        print(f"[{variant}] saved '0' silently became '{resolved}' (browser ON)")


# ---------------------------------------------------------------- finding C (Windows)

@pytest.mark.skipif(not PWSH, reason="no PowerShell on this runner")
def test_c_browser_watch_job_outlives_the_install():
    """install.ps1:2756 does `$null = Start-Job`; nothing ever stops or removes it.

    Start the real watcher scriptblock against a port nothing serves, then walk
    away as the installer does. The job is still running.
    """
    src = INSTALL_PS1.read_text(encoding="utf-8", errors="replace")
    assert "$_browserWatch = {" in src
    # Scope to the watcher's own call site: install.ps1 has another Start-Job
    # elsewhere that IS managed (Wait-Job / Stop-Job / Remove-Job), which is the
    # convention this one departs from.
    watch_call = _slice(src, r"Start-Job -ScriptBlock \$_browserWatch", r"^\s*\}\s*$")
    tail = src[src.index("Start-Job -ScriptBlock $_browserWatch"):]
    assert not re.search(r"\b(Stop-Job|Remove-Job|Wait-Job)\b.*_browserWatch", src), \
        "install.ps1 now cleans up the browser watcher; finding C is fixed"
    assert "$null = Start-Job" in watch_call, \
        f"watcher no longer discards the job handle: {watch_call}"
    assert not re.search(r"\b(Stop-Job|Remove-Job|Wait-Job)\b", tail[:2000]), \
        "a cleanup call now follows the watcher; finding C is fixed"

    watch = _slice(src, r"\$_browserWatch = \{", r"^    \}$")
    script = watch + textwrap.dedent(r"""
        # Port 9 (discard) never answers /api/health, so the watcher polls its
        # full 120s deadline exactly as it would after a failed server start.
        $null = Start-Job -ScriptBlock $_browserWatch -ArgumentList @("some-root-id", 9)
        Start-Sleep -Seconds 3
        # The installer returns here. Nothing stopped the job.
        $running = @(Get-Job | Where-Object { $_.State -eq 'Running' }).Count
        Write-Output "running_jobs=$running"
    """)
    r = _pwsh(script)
    assert r.returncode == 0, r.stderr
    n = int(re.search(r"running_jobs=(\d+)", r.stdout).group(1))
    assert n >= 1, f"finding C not reproduced: {r.stdout}"
    print(f"watcher job still running after the installer moved on: {n}")


# ---------------------------------------------------------------- finding D (Windows)

@pytest.mark.skipif(not PWSH, reason="no PowerShell on this runner")
def test_d_port_probe_calls_a_bound_socket_free():
    """install.ps1:2695 probes with connect(); a bound-not-listening socket reads free.

    The launcher's own Find-FreeLaunchPort (install.ps1:777-799) binds instead,
    so the two disagree about the same port.
    """
    src = INSTALL_PS1.read_text(encoding="utf-8", errors="replace")
    finder = _slice(src, r"function Find-PostInstallStudioPort \{", r"^    \}$")
    script = finder + textwrap.dedent(r"""
        # Bind 8888 WITHOUT calling Start()/Listen(): connect() fails, bind() would too.
        $sock = [System.Net.Sockets.Socket]::new(
            [System.Net.Sockets.AddressFamily]::InterNetwork,
            [System.Net.Sockets.SocketType]::Stream,
            [System.Net.Sockets.ProtocolType]::Tcp)
        $sock.Bind([System.Net.IPEndPoint]::new([System.Net.IPAddress]::Loopback, 8888))

        $picked = Find-PostInstallStudioPort
        Write-Output "picked=$picked"

        # Now show the port is not actually bindable.
        $bindable = $true
        try {
            $probe = [System.Net.Sockets.Socket]::new(
                [System.Net.Sockets.AddressFamily]::InterNetwork,
                [System.Net.Sockets.SocketType]::Stream,
                [System.Net.Sockets.ProtocolType]::Tcp)
            $probe.Bind([System.Net.IPEndPoint]::new([System.Net.IPAddress]::Loopback, $picked))
            $probe.Dispose()
        } catch { $bindable = $false }
        Write-Output "bindable=$bindable"
        $sock.Dispose()
    """)
    r = _pwsh(script)
    assert r.returncode == 0, r.stderr
    picked = int(re.search(r"picked=(\d+)", r.stdout).group(1))
    bindable = "bindable=True" in r.stdout
    print(f"Find-PostInstallStudioPort picked {picked}; actually bindable: {bindable}")
    if picked == 8888 and not bindable:
        print("finding D reproduced: the helper handed the server a port it cannot bind")
    else:
        pytest.skip(f"host declined the bind-without-listen setup (picked={picked})")


# ---------------------------------------------------------------- finding A (POSIX)

@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX installer")
def test_a_posix_browser_prompt_gate_ignores_skip_autostart():
    """install.sh:3221 gates on [ -t 1 ] && [ -r /dev/tty ] but not _SKIP_AUTOSTART."""
    src = INSTALL_SH.read_text(encoding="utf-8", errors="replace")
    prompt_gate = _slice(src, r'if \[ -z "\$_STUDIO_OPEN_BROWSER" \]', r'Open Unsloth Studio in your default browser')
    launch_gate = next(l for l in src.splitlines()
                       if '"$_SKIP_AUTOSTART" != true' in l and "-t 1" in l)
    assert "_SKIP_AUTOSTART" in launch_gate
    assert "_SKIP_AUTOSTART" not in prompt_gate, "finding A is fixed"
    print("browser prompt gate:", prompt_gate.splitlines()[0].strip())
    print("launch prompt gate: ", launch_gate.strip())


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX installer")
def test_a2_posix_prompt_uses_stale_tty_test(tmp_path):
    """After a rebase onto main, install.sh:3221 should use _can_read_tty().

    main added that helper precisely because `test -r /dev/tty` passes in
    containers where open() then fails with ENXIO, leaving a dangling question
    in the log. The new prompt still uses the old test.
    """
    src = INSTALL_SH.read_text(encoding="utf-8", errors="replace")
    if "_can_read_tty()" not in src:
        pytest.skip("branch predates the _can_read_tty helper (pre-rebase)")
    prompt_gate = _slice(src, r'if \[ -z "\$_STUDIO_OPEN_BROWSER" \]', r'Open Unsloth Studio in your default browser')
    assert "-r /dev/tty" in prompt_gate and "_can_read_tty" not in prompt_gate, \
        "the new prompt now uses _can_read_tty; this finding is fixed"
    print("new prompt still uses the pre-#7435 tty test:", prompt_gate.splitlines()[0].strip())


# ---------------------------------------------------------------- finding B (POSIX)

@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX launcher")
def test_b_open_browser_escapes_the_test_stubs_on_wsl(tmp_path):
    """tests/sh/test_launcher_no_browser.sh:121 stubs only open + xdg-open.

    Force the WSL branch (fixture /proc/version + uname stub) with exactly the
    PR's two stubs on PATH, and show dispatch reaches a real powershell.exe.
    """
    src = INSTALL_SH.read_text(encoding="utf-8", errors="replace")
    fn = _slice(src, r"^_open_browser\(\) \{", r"^\}$")

    procver = tmp_path / "proc_version"
    procver.write_text("Linux version 5.15.0-microsoft-standard-WSL2\n")
    fn = fn.replace("/proc/version", str(procver))

    stubdir = tmp_path / "bin"
    stubdir.mkdir()
    record = tmp_path / "record"
    # The PR's two stubs, plus a fake host powershell.exe standing in for the
    # real one WSL puts on PATH. If dispatch were correctly stubbed it would
    # never be reached.
    for name in ("open", "xdg-open"):
        p = stubdir / name
        p.write_text(f'#!/bin/sh\necho "STUB_{name}:$1" >> "{record}"\n')
        p.chmod(0o755)
    hostdir = tmp_path / "hostbin"
    hostdir.mkdir()
    ps = hostdir / "powershell.exe"
    ps.write_text(f'#!/bin/sh\necho "HOST_BROWSER_OPENED:$*" >> "{record}"\n')
    ps.chmod(0o755)
    un = stubdir / "uname"
    un.write_text('#!/bin/sh\necho Linux\n')
    un.chmod(0o755)

    script = tmp_path / "drive.sh"
    script.write_text(f"OPEN_BROWSER=1\n{fn}\n_open_browser http://localhost:9999\nwait\n")
    env = dict(os.environ, PATH=f"{stubdir}:{hostdir}:{os.environ['PATH']}")
    subprocess.run(["bash", str(script)], env=env, capture_output=True, timeout=60)

    got = record.read_text() if record.exists() else ""
    assert "HOST_BROWSER_OPENED" in got, f"finding B not reproduced; recorded: {got!r}"
    assert "STUB_" not in got, f"a stub was reached after all: {got!r}"
    print("WSL dispatch escaped the test's stubs and reached the host opener:", got.strip())
