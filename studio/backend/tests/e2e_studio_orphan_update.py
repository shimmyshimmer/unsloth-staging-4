"""End-to-end reproduction of the Discord report against a REAL Studio install.

Everything runs under a dedicated UNSLOTH_STUDIO_HOME so nothing touches the
machine's own install.

Sequence:
  1. launch the installed Studio
  2. load a small GGUF so the server owns a real child (llama-server)
  3. run a sandboxed tool call the way the chat loop does, and check whether the
     files it produced can be retrieved over the API
  4. hard-kill ONLY the top-level Studio process (what closing the console
     window / "End Task" does -- the graceful path never runs)
  5. list processes still alive under the studio home
  6. run the real `unsloth studio update` and report what it says

Env: STUDIO_HOME, STUDIO_BIN, STUDIO_PORT, STUDIO_PASSWORD, STUDIO_REPO
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

HOME = Path(os.environ["STUDIO_HOME"]).resolve()
BIN = os.environ["STUDIO_BIN"]
PORT = int(os.environ.get("STUDIO_PORT", "8899"))
PASSWORD = os.environ.get("STUDIO_PASSWORD", "reproPass123")
REPO = os.environ.get("STUDIO_REPO", "")
BASE = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("STUDIO_MODEL", "unsloth/Qwen3-0.6B-GGUF")
IS_WIN = sys.platform == "win32"

LOG = HOME.parent / "e2e_server.log"
results: dict = {"platform": sys.platform, "studio_home": str(HOME)}


def say(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def under_home(proc: psutil.Process) -> bool:
    try:
        exe = proc.exe()
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return False
    if not exe:
        return False
    try:
        return str(HOME).lower() in str(Path(exe).resolve()).lower()
    except OSError:
        return False


def studio_processes() -> list[dict]:
    out = []
    for proc in psutil.process_iter(["pid", "name"]):
        if under_home(proc):
            try:
                out.append({"pid": proc.pid, "name": proc.name(), "exe": proc.exe()})
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
    return out


# ---------------------------------------------------------------- 1. launch
say("launching Studio")
env = dict(os.environ)
env["UNSLOTH_STUDIO_HOME"] = str(HOME)
env["UNSLOTH_STUDIO_PASSWORD"] = PASSWORD
# The Windows launcher starts Studio with WorkingDirectory = %USERPROFILE%
# (install.ps1), so mirror that: it is what decides where unsloth_compiled_cache
# is created.
launch_cwd = Path.home()
with open(LOG, "wb") as log:
    server = subprocess.Popen(
        [BIN, "studio", "-H", "127.0.0.1", "-p", str(PORT)],
        env=env, cwd=str(launch_cwd), stdout=log, stderr=subprocess.STDOUT,
    )
print(f"server pid {server.pid}, cwd {launch_cwd}")

deadline = time.time() + 900
token = None
while time.time() < deadline:
    if server.poll() is not None:
        print(LOG.read_text(errors="replace")[-4000:])
        raise SystemExit(f"server exited early with {server.returncode}")
    try:
        r = httpx.post(f"{BASE}/api/auth/login",
                       json={"username": "unsloth", "password": PASSWORD}, timeout=10)
        if r.status_code == 200:
            token = r.json()["access_token"]
            break
    except Exception:
        pass
    time.sleep(5)
if not token:
    print(LOG.read_text(errors="replace")[-4000:])
    raise SystemExit("server never became ready")
H = {"Authorization": f"Bearer {token}"}
say(f"Studio up after {int(900 - (deadline - time.time()))}s")

# ---------------------------------------------------------------- 2. model
say(f"loading {MODEL}")
with httpx.Client(timeout=1800) as c:
    r = c.post(f"{BASE}/api/inference/load", headers=H,
               json={"model_path": MODEL, "gguf_variant": "Q4_K_M", "max_seq_length": 4096})
    print("load ->", r.status_code)
    results["model_loaded"] = r.status_code == 200

children = [p for p in studio_processes() if p["pid"] != server.pid]
print("children under studio home:", json.dumps(children, indent=2))
results["children_after_load"] = children

# ---------------------------------------------------------------- 3. files
say("sandbox files: can the user get them back?")
sys.path.insert(0, str(Path(REPO) / "studio" / "backend"))
session = "__LOCALID_e2eRepr"
try:
    from core.inference.tools import get_sandbox_workdir

    workdir = Path(get_sandbox_workdir(session))
except Exception as exc:  # backend deps missing in this interpreter
    workdir = Path.home() / "studio_sandbox" / session
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"(computed path directly: {exc})")
print("workdir:", workdir)
results["sandbox_workdir"] = str(workdir)
results["sandbox_under_studio_home"] = str(HOME).lower() in str(workdir).lower()

(workdir / "results.csv").write_text("a,b\n1,2\n")
(workdir / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
fetched = {}
with httpx.Client(timeout=30) as c:
    for name in ("chart.png", "results.csv"):
        r = c.get(f"{BASE}/api/inference/sandbox/{session}/{name}", headers=H)
        fetched[name] = r.status_code
    listing = c.get(f"{BASE}/api/inference/sandbox/{session}", headers=H).status_code
print("fetch:", fetched, "listing route:", listing)
results["sandbox_fetch"] = fetched
results["sandbox_listing_status"] = listing

# ---------------------------------------------------------------- 4. hard kill
say("hard-killing ONLY the top-level Studio process (console close / End Task)")
before = studio_processes()
print("before:", json.dumps(before, indent=2))
if IS_WIN:
    subprocess.run(["taskkill", "/PID", str(server.pid), "/F"], capture_output=True)
else:
    os.kill(server.pid, 9)
try:
    server.wait(timeout=30)
except subprocess.TimeoutExpired:
    pass
time.sleep(10)

survivors = studio_processes()
print("survivors:", json.dumps(survivors, indent=2))
results["survivors"] = survivors

# ---------------------------------------------------------------- 5. update
say("running the real `unsloth studio update`")
upd_env = dict(os.environ)
upd_env["UNSLOTH_STUDIO_HOME"] = str(HOME)
if REPO:
    upd_env["STUDIO_LOCAL_REPO"] = REPO
cmd = [BIN, "studio", "update", "--local", "--no-verify"] if REPO else [BIN, "studio", "update"]
proc = subprocess.run(cmd, env=upd_env, cwd=str(launch_cwd), capture_output=True,
                      text=True, timeout=3600)
tail = (proc.stdout or "")[-3000:] + "\n--- stderr ---\n" + (proc.stderr or "")[-3000:]
print(f"update exit={proc.returncode}\n{tail}")
results["update_exit"] = proc.returncode
results["update_blocked"] = "in use by" in (proc.stdout or "") + (proc.stderr or "")

# ---------------------------------------------------------------- 6. cwd cache
cache = launch_cwd / "unsloth_compiled_cache"
results["compiled_cache_in_launch_cwd"] = cache.is_dir()
print(f"\n{cache} exists: {cache.is_dir()}")

# ---------------------------------------------------------------- cleanup
for p in studio_processes():
    try:
        psutil.Process(p["pid"]).kill()
    except Exception:
        pass

say("RESULTS")
print(json.dumps(results, indent=2))
Path(os.environ.get("E2E_RESULTS", "e2e_results.json")).write_text(json.dumps(results, indent=2))
shutil.copy(LOG, Path.cwd() / "e2e_server.log") if LOG.is_file() else None
