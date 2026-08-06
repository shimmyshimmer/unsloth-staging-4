"""Empirical cmd.exe grammar oracle. Runs on a real windows-latest runner.

For each candidate command string, substitute a harmless marker program for the
blocked name and ask cmd.exe itself whether the marker was launched. The marker
is `mk.cmd` on PATH; it writes a sentinel file and exits. If the sentinel exists
afterwards, cmd resolved that spelling to the marker and launched it.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.environ.get("ORACLE_HOME", "oracle")).resolve()
HOME.mkdir(parents=True, exist_ok=True)
SENTINEL = HOME / "hit.txt"

# The marker stands in for the blocked program. `start` resolves `mk` through
# PATHEXT to mk.cmd exactly as it would resolve `powershell` to powershell.exe.
(HOME / "mk.cmd").write_text('@echo hit> "%ORACLE_HOME%\\hit.txt"\r\n', encoding="ascii")
# A second spelling so a case can distinguish "launched mk" from "launched mk2".
(HOME / "mk2.cmd").write_text('@echo hit2> "%ORACLE_HOME%\\hit.txt"\r\n', encoding="ascii")

CASES = [
    # baseline: these must launch, or the harness itself is broken
    ("baseline-plain", 'start "" mk'),
    ("baseline-notitle", "start mk"),
    ("baseline-direct", "mk"),
    # baseline: these must NOT launch
    ("baseline-echo", 'echo start "" mk'),
    ("baseline-rem", 'rem start "" mk'),
    # claim: cmd's ,;= argument delimiters apply to a start target
    ("delim-comma", 'start "" mk,junk'),
    ("delim-semi", 'start "" mk;junk'),
    ("delim-equals", 'start "" mk=junk'),
    ("delim-comma-space", 'start "" mk, -c ls'),
    ("delim-semi-space", 'start "" mk; -c ls'),
    # claim: a trailing caret / unpaired bang is dropped from the name
    ("caret-trailing", 'start "" mk^'),
    ("bang-trailing", 'start "" mk!'),
    # claim: quotes/carets/trailing dots inside the nested shell name still resolve
    ("nested-quoted", 'c"m"d /c start "" mk'),
    ("nested-caret", "c^md /c start \"\" mk"),
    ("nested-dot", 'cmd.exe. /c start "" mk'),
    # claim: comma/semicolon can replace whitespace around cmd /c
    ("glued-comma", 'cmd.exe,/c,start "" mk'),
    ("glued-semi", "cmd;/c;start \"\" mk"),
    ("glued-lead", ',;,cmd.exe,;,/c,;,start "" mk'),
    # claim: file:// resolves to a local executable through start
    ("url-file", 'start "" file:///%ORACLE_HOME_FWD%/mk.cmd'),
    # claim: CALL at the command prompt still dispatches
    ("call-prompt", 'call start "" mk'),
    # claim: forfiles /C runs a command
    ("forfiles", 'forfiles /p . /m *.* /c "cmd /c start \\"\\" %ORACLE_HOME%\\mk.cmd" '),
    # control: an expansion contributing characters to the name
    ("expand-name", 'set N=k& start "" m%N%'),
    ("expand-delayed", 'start "" m!N!'),
]


def run_case(raw):
    if SENTINEL.exists():
        SENTINEL.unlink()
    text = raw.replace("%ORACLE_HOME_FWD%", str(HOME).replace("\\", "/"))
    text = text.replace("%ORACLE_HOME%", str(HOME))
    env = dict(os.environ)
    env["ORACLE_HOME"] = str(HOME)
    env["PATH"] = str(HOME) + os.pathsep + env["PATH"]
    env["N"] = "k"
    # Pass the command line to CreateProcess verbatim so cmd sees exactly these
    # bytes; subprocess list-quoting would rewrite the punctuation under test.
    line = 'cmd.exe /c ' + text
    # DEVNULL, not a pipe: `start` detaches a child that inherits the handles, so a
    # pipe is not closed until that child exits and a `/k` case would deadlock the
    # read even after the timeout kills cmd itself.
    proc = subprocess.Popen(line, env=env, cwd=str(HOME),
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        rc, err = proc.wait(timeout=20), ""
    except subprocess.TimeoutExpired:
        proc.kill()
        rc, err = None, "TIMEOUT"
    # `start` returns immediately; give the detached marker time to write.
    for _ in range(30):
        if SENTINEL.exists():
            break
        time.sleep(0.2)
    launched = SENTINEL.exists()
    if launched:
        SENTINEL.unlink()
    return {"launched": launched, "rc": rc, "stderr": err.strip()}


results = {}
for name, raw in CASES:
    results[name] = {"command": raw, **run_case(raw)}
    r = results[name]
    print(f"{'LAUNCH' if r['launched'] else '  no  '}  rc={str(r['rc']):<5} {name:<18} | {raw}", flush=True)

Path("oracle_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

# The harness is only trustworthy if the baselines behave.
ok = (results["baseline-plain"]["launched"] and results["baseline-direct"]["launched"]
      and not results["baseline-echo"]["launched"] and not results["baseline-rem"]["launched"])
print("\nharness baselines sane:", ok)
sys.exit(0 if ok else 1)
