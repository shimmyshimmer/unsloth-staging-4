"""Round 2 of the cmd.exe grammar oracle: the claims round 1 did not cover.

Each case gets its OWN sentinel filename. In round 1 a detached child from the
previous case could still write after the next case had cleared the shared file,
which is the only way `expand-delayed` could have reported a launch.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.environ.get("ORACLE_HOME", "oracle")).resolve()
HOME.mkdir(parents=True, exist_ok=True)

# One marker per case: mk_<n>.cmd writes hit_<n>.txt. `%~n0` is the script's own
# base name, so a single template serves every marker.
TEMPLATE = '@echo hit> "%ORACLE_HOME%\\hit_%~n0.txt"\r\n'

CASES = [
    ("baseline-plain", 'start "" {mk}'),
    ("baseline-echo", 'echo start "" {mk}'),
    # attached / multi switches on start
    ("start-d-attached", 'start "" /d{home} {mk}'),
    ("start-d-quoted", 'start "" /d"{home}" {mk}'),
    ("start-multi-switch", 'start /low/i "" {mk}'),
    ("start-b", 'start /b "" {mk}'),
    # attached payload on cmd
    ("cmd-c-attached", 'cmd /cstart "" {mk}'),
    ("cmd-k-attached", 'cmd /kstart "" {mk}'),
    ("cmd-c-attached-direct", "cmd /c{mk}"),
    # nested cmd reached through a full path
    ("nested-fullpath", 'start "" "C:\\Windows\\System32\\cmd.exe" /c start "" {mk}'),
    ("nested-fullpath-bare", 'C:\\Windows\\System32\\cmd.exe /c start "" {mk}'),
    # trailing dot / space on the start target itself
    ("target-trailing-dot", 'start "" {mk}.'),
    ("target-trailing-dots", 'start "" {mk}..'),
    # delayed expansion actually enabled
    ("bang-delayed-on", 'cmd /v:on /c start "" {mk}!'),
    ("bang-name-delayed-on", 'cmd /v:on /c "set N=k&start "" m!N!"'),
    # an expansion whose value carries a command separator
    ("expand-injects-sep", 'start "" %SEPVAR% {mk}'),
    # a variable name containing a space
    ("space-var-name", 'call start "" m%my var%'),
    # comma/semicolon between start and its title, and around the target
    ("start-comma-title", 'start,"",{mk}'),
    ("start-semi-title", 'start;"";{mk}'),
]


def run_case(name, index):
    marker = f"mk{index}"
    (HOME / f"{marker}.cmd").write_text(TEMPLATE, encoding="ascii")
    sentinel = HOME / f"hit_{marker}.txt"
    if sentinel.exists():
        sentinel.unlink()
    text = CASES[index][1].format(mk=marker, home=str(HOME))
    env = dict(os.environ)
    env["ORACLE_HOME"] = str(HOME)
    env["PATH"] = str(HOME) + os.pathsep + env["PATH"]
    env["SEPVAR"] = "&"
    env["my var"] = "k"
    proc = subprocess.Popen("cmd.exe /c " + text, env=env, cwd=str(HOME),
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        rc = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = None
    for _ in range(25):
        if sentinel.exists():
            break
        time.sleep(0.2)
    return {"launched": sentinel.exists(), "rc": rc, "command": text}


results = {}
for i, (name, _) in enumerate(CASES):
    r = run_case(name, i)
    results[name] = r
    print(f"{'LAUNCH' if r['launched'] else '  no  '}  rc={str(r['rc']):<5} {name:<22} | {r['command']}", flush=True)

Path("oracle2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
ok = results["baseline-plain"]["launched"] and not results["baseline-echo"]["launched"]
print("\nharness baselines sane:", ok)
sys.exit(0 if ok else 1)
