"""End-to-end proof of unslothai/unsloth#9167 on a REAL Windows host.

The PR's own tests only assert on the argv that _resolved_launch_command returns;
none of them spawn anything, so none of them demonstrate the reported failure or
that the fix clears it. This does, by actually calling subprocess:

  A. defect     spawn the extensionless npm POSIX shim  -> expect WinError 193
  B. fix        spawn _resolved_launch_command's argv    -> expect it to RUN
  C. npm shim   same, through a real cmd-shim + node     -> expect node to run it
  D. which()    report what shutil.which does on THIS interpreter, which says
                whether this CPython carries the gh-127001 regression at all

Exits non-zero if the defect does not reproduce or the fix does not work, so a
green job means something rather than merely "the asserts we wrote passed".
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.getcwd())
from unsloth_cli.commands.start import (  # noqa: E402
    _prefer_windows_cmd_sibling,
    _resolved_launch_command,
)

MARKER = "UNSLOTH_SHIM_RAN_OK"
WINERROR_193 = 193

NPM_POSIX_SHIM = '#!/bin/sh\nbasedir=$(dirname "$0")\nexec node "$basedir/index.js" "$@"\n'

results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush = True)


def run(argv):
    return subprocess.run(argv, capture_output = True, text = True, timeout = 120)


def scenario_a_defect_reproduces(tmp):
    """The bug itself: CreateProcess cannot load a shebang script."""
    shim = tmp / "fakeagent"
    shim.write_text(NPM_POSIX_SHIM, encoding = "utf-8")
    (tmp / "fakeagent.cmd").write_text(f"@ECHO off\r\n@ECHO {MARKER}\r\n", encoding = "utf-8")
    try:
        completed = run([str(shim)])
    except OSError as exc:
        ok = getattr(exc, "winerror", None) == WINERROR_193
        record("A. defect reproduces (WinError 193)", ok, f"{type(exc).__name__}: {exc}")
        return
    record(
        "A. defect reproduces (WinError 193)",
        False,
        f"expected WinError 193, but it ran: rc={completed.returncode} out={completed.stdout!r}",
    )


def scenario_b_fix_launches(tmp):
    """The fix: the resolved argv must actually start and produce output."""
    shim = tmp / "fakeagent"
    shim.write_text(NPM_POSIX_SHIM, encoding = "utf-8")
    (tmp / "fakeagent.cmd").write_text(f"@ECHO off\r\n@ECHO {MARKER}\r\n", encoding = "utf-8")
    argv = _resolved_launch_command(str(shim), [])
    completed = run(argv)
    ok = completed.returncode == 0 and MARKER in completed.stdout
    record(
        "B. fix launches the sibling",
        ok,
        f"argv[0]={Path(argv[0]).name} rc={completed.returncode} out={completed.stdout.strip()!r}",
    )


def scenario_c_npm_shim_through_node(tmp):
    """A real cmd-shim should resolve to node.exe + the script, bypassing cmd.exe."""
    node = shutil.which("node")
    if node is None:
        record("C. npm shim runs via node", True, "skipped: no node on PATH")
        return
    package = tmp / "node_modules" / "fakeagent"
    package.mkdir(parents = True)
    (package / "index.js").write_text(f'console.log("{MARKER}");\n', encoding = "utf-8")
    shim = tmp / "npmagent"
    shim.write_text(NPM_POSIX_SHIM, encoding = "utf-8")
    # Byte-exact cmd-shim v7 template, matching test_start.py's _npm_node_cmd_shim.
    # The blank \r\n separators matter: without them _NPM_NODE_CMD_SHIMS does not
    # match and the parser silently falls through to spawning the .cmd itself.
    (tmp / "npmagent.cmd").write_text(
        "@ECHO off\r\n"
        "GOTO start\r\n"
        ":find_dp0\r\n"
        "SET dp0=%~dp0\r\n"
        "EXIT /b\r\n"
        ":start\r\n"
        "SETLOCAL\r\n"
        "CALL :find_dp0\r\n"
        "\r\n"
        'IF EXIST "%dp0%\\node.exe" (\r\n'
        '  SET "_prog=%dp0%\\node.exe"\r\n'
        ") ELSE (\r\n"
        '  SET "_prog=node"\r\n'
        "  SET PATHEXT=%PATHEXT:;.JS;=;%\r\n"
        ")\r\n"
        "\r\n"
        "endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "
        '"%_prog%"  "%dp0%\\node_modules\\fakeagent\\index.js" %*\r\n',
        encoding = "utf-8",
    )
    argv = _resolved_launch_command(str(shim), [])
    completed = run(argv)
    # The point of this scenario is the cmd.exe BYPASS, so a .cmd argv[0] is a
    # failure even if it happens to run: that is the fallback, not the parser.
    bypassed = Path(argv[0]).suffix.lower() not in {".cmd", ".bat"}
    ok = bypassed and completed.returncode == 0 and MARKER in completed.stdout
    record(
        "C. npm shim resolves to node, bypassing cmd.exe",
        ok,
        f"argv[0]={Path(argv[0]).name} bypassed_cmd={bypassed} "
        f"rc={completed.returncode} out={completed.stdout.strip()!r}",
    )


def scenario_d_which_behaviour(tmp):
    """Informational: does THIS interpreter carry the gh-127001 regression?"""
    directory = tmp / "whichdir"
    directory.mkdir()
    (directory / "probeagent").write_text(NPM_POSIX_SHIM, encoding = "utf-8")
    (directory / "probeagent.cmd").write_text("@ECHO off\r\n", encoding = "utf-8")
    saved = os.environ["PATH"]
    os.environ["PATH"] = str(directory) + os.pathsep + saved
    try:
        resolved = shutil.which("probeagent")
    finally:
        os.environ["PATH"] = saved
    suffix = Path(resolved).suffix if resolved else None
    rescued = _prefer_windows_cmd_sibling(resolved)
    regressed = suffix == ""
    record(
        "D. which() on this interpreter",
        True,
        f"python={'.'.join(map(str, sys.version_info[:3]))} which->{suffix or 'NO SUFFIX'} "
        f"({'carries gh-127001' if regressed else 'gh-127001 fixed'}), "
        f"after rescue->{Path(rescued).suffix if rescued else None}",
    )


def main():
    if os.name != "nt":
        print("This check is Windows-only.")
        return 0
    print(f"=== #9167 end-to-end on {sys.platform}, python "
          f"{'.'.join(map(str, sys.version_info[:3]))} ===", flush = True)
    for scenario in (
        scenario_a_defect_reproduces,
        scenario_b_fix_launches,
        scenario_c_npm_shim_through_node,
        scenario_d_which_behaviour,
    ):
        tmp = Path(tempfile.mkdtemp(prefix = "e2e_"))
        try:
            scenario(tmp)
        except Exception as exc:  # noqa: BLE001
            record(scenario.__name__, False, f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors = True)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} PASS", flush = True)
    if failed:
        print("FAILED: " + ", ".join(failed), flush = True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
