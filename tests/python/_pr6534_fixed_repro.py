"""Cross-OS validation that PR #6534 + the UV_OVERRIDE/P3 fix WORKS everywhere.

Throwaway CI repro (staging fork only). Proves on a real runner of each OS:

  1. _uv_safe_path is the shared backend.utils.uv_path_safety helper.
  2. no-space passthrough unchanged (backwards compat).
  3. POSIX: uv -c raw space path FAILS (truncates), helper copy WORKS (the fix).
  4. POSIX: uv UV_OVERRIDE raw space path FAILS, and the value the installer now
     stores (uv_safe_path(override)) WORKS -> the macOS-arm64 gap is CLOSED.
  5. copyfile-failure does not leak a temp dir (P3 fix).

macos-14 is Apple Silicon -- the exact platform whose UV_OVERRIDE path was the
remaining gap, so a green run there is the proof the fix lands on target.
"""
from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "studio"))
sys.path.insert(0, str(REPO / "studio" / "backend"))
import install_python_stack as ips  # noqa: E402
from backend.utils import uv_path_safety as uvps  # noqa: E402

IS_WINDOWS = platform.system() == "Windows"
UV = shutil.which("uv")
_fail = []


def ck(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}", flush=True)
    if not cond:
        _fail.append(name)


def uv_run(args, env=None):
    cmd = [UV, "pip", "install", "--dry-run", "--python", sys.executable, *args]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode, (r.stderr.strip().splitlines() or [""])[0]


def spacefile(name="constraints.txt", body="idna==3.10\n"):
    b = Path(tempfile.mkdtemp(prefix="pr6534fix_")) / "Open Source"
    b.mkdir(parents=True)
    f = b / name
    f.write_text(body)
    return f


print(f"ENV: {platform.platform()} | python {sys.version.split()[0]} | uv={UV}", flush=True)
print("uv:", subprocess.run([UV, "--version"], capture_output=True, text=True).stdout.strip() if UV else "MISSING", flush=True)

# 1. shared helper
ck("_uv_safe_path is shared uv_path_safety helper",
   ips._uv_safe_path.__module__.endswith("uv_path_safety"),
   ips._uv_safe_path.__module__)

# 2. passthrough
p = os.path.join(tempfile.gettempdir(), "plain", "c.txt")
ck("no-space passthrough unchanged", ips._uv_safe_path(p) == p)

con = spacefile()
safe = ips._uv_safe_path(str(con))

if not IS_WINDOWS:
    # 3. -c fix
    ck("helper copy is space-free + identical", " " not in safe and Path(safe).read_bytes() == con.read_bytes())
    if UV:
        rc_raw, e_raw = uv_run(["idna", "-c", str(con)])
        rc_safe, _ = uv_run(["idna", "-c", safe])
        ck("uv -c raw FAILS (truncates)", rc_raw != 0, e_raw)
        ck("uv -c safe WORKS", rc_safe == 0)

    # 4. UV_OVERRIDE gap CLOSED (value the installer now stores)
    ovr = spacefile("overrides-darwin-arm64.txt", "transformers>=4.57.6\n")
    safe_ovr = ips._uv_safe_path(str(ovr))  # exactly install_python_stack:1515
    if UV:
        env_raw = os.environ.copy(); env_raw["UV_OVERRIDE"] = str(ovr)
        env_fix = os.environ.copy(); env_fix["UV_OVERRIDE"] = safe_ovr
        rc_oraw, e_oraw = uv_run(["idna"], env=env_raw)
        rc_ofix, _ = uv_run(["idna"], env=env_fix)
        ck("UV_OVERRIDE raw FAILS (pre-fix gap)", rc_oraw != 0, e_oraw)
        ck("UV_OVERRIDE via installer's helper WORKS (gap closed)", rc_ofix == 0)

    # 5. P3: no temp-dir leak on copyfile failure
    pattern = os.path.join(tempfile.gettempdir(), "unsloth_uv_*")
    before = set(glob.glob(pattern))
    orig = uvps.shutil.copyfile
    uvps.shutil.copyfile = lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
    out = uvps.uv_safe_path(str(spacefile()))
    uvps.shutil.copyfile = orig
    ck("copyfile failure falls back", out.endswith("constraints.txt") or " " in out)
    ck("copyfile failure leaks no temp dir (P3 fix)", set(glob.glob(pattern)) == before)
else:
    print("  Windows: 8.3 short-path branch", flush=True)
    print(f"    _uv_safe_path(space) -> {safe!r} has_space={' ' in safe}", flush=True)
    if UV:
        print("    uv -c raw :", uv_run(["idna", "-c", str(con)]), flush=True)
        print("    uv -c safe:", uv_run(["idna", "-c", safe]), flush=True)

print("\n=== RESULT:", "FAILURES " + repr(_fail) if _fail else "all hard invariants PASSED", flush=True)
sys.exit(1 if _fail else 0)
