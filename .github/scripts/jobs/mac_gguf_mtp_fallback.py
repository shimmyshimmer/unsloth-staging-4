"""Apple Silicon: Gemma 4 GGUF + MTP drafter on llama.cpp Metal, and Studio's MTP crash fallback.

Phase A (root cause, unslothai/unsloth#13253): raw llama-server, all layers on Metal, Gemma 4
E2B + its MTP drafter, per llama.cpp prebuilt (--builds) and spec mode (MTP off / draft-mtp n=2 /
ngram-mod,draft-mtp n=3). Records whether the server survives chat requests. Observation only:
paravirtual Metal on a hosted runner is not an M2, so a survive here does not clear real hardware.

Phase B (the fix): Studio's real LlamaCppBackend from two trees (--base-src, --head-src), CPU
llama-server via LLAMA_SERVER_PATH. A crash injector SIGABRTs a healthy draft-mtp server once it
starts a task (the reported "passes load, dies on every decode"). Checks: base reproduces the
every-request failure loop, head recovers to an MTP-free server, head keeps MTP when nothing
crashes.

Exit 0 all checks pass, 1 otherwise, 3 skipped (not Apple Silicon). Last line JOB_RESULT {...}.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import textwrap
import threading
import time
import urllib.request
from pathlib import Path

REPO = "unsloth/gemma-4-E2B-it-GGUF"
MODEL = "gemma-4-E2B-it-UD-Q4_K_XL.gguf"
DRAFTER = "mtp-gemma-4-E2B-it.gguf"
PROMPTS = [
    "Hi",
    "Write two lines about the sea.",
    "Why is the sky blue? One sentence.",
    "Name three prime numbers.",
]
SPEC_MODES = {
    "off": [],
    "mtp_n2": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    "ngram_mtp_n3": ["--spec-type", "ngram-mod,draft-mtp", "--spec-draft-n-max", "3"],
}


def log(*a):
    print(*a, flush = True)


def sh(cmd, **kw):
    log("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check = True, **kw)


def fetch_build(tag: str, root: Path) -> Path:
    dst = root / tag
    hits = list(dst.rglob("llama-server")) if dst.exists() else []
    if hits:
        return hits[0]
    dst.mkdir(parents = True, exist_ok = True)
    url = (
        f"https://github.com/unslothai/llama.cpp/releases/download/{tag}/"
        f"llama-{tag}-bin-macos-arm64.tar.gz"
    )
    tgz = dst / "b.tgz"
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(dst)
    server = next(dst.rglob("llama-server"))
    server.chmod(0o755)
    return server


def free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def raw_server_run(server: Path, model: str, drafter: str, mode: str, logdir: Path) -> dict:
    import httpx

    port = free_port()
    cmd = [
        str(server),
        "-m",
        model,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "-c",
        "8192",
        "-ngl",
        "99",
        "--jinja",
        "--reasoning",
        "off",
    ]
    if mode != "off":
        cmd += ["--model-draft", drafter, "-ngld", "99", *SPEC_MODES[mode]]
    logf = logdir / f"raw_{server.parent.name}_{mode}.log"
    p = subprocess.Popen(
        cmd,
        stdout = open(logf, "w"),
        stderr = subprocess.STDOUT,
        cwd = server.parent,
        start_new_session = True,
    )
    out = {"mode": mode, "build": server.parent.name, "requests": [], "healthy": False}
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 300 and p.poll() is None:
        try:
            if httpx.get(url + "/health", timeout = 2).status_code == 200:
                out["healthy"] = True
                break
        except Exception:
            pass
        time.sleep(1)
    out["load_s"] = round(time.time() - t0, 1)
    for pr in PROMPTS if out["healthy"] else []:
        if p.poll() is not None:
            out["requests"].append("SERVER_DEAD")
            continue
        try:
            r = httpx.post(
                url + "/v1/chat/completions",
                timeout = 300,
                json = {
                    "messages": [{"role": "user", "content": pr}],
                    "max_tokens": 48,
                    "temperature": 0.0,
                },
            )
            out["requests"].append(r.status_code)
        except Exception as e:
            time.sleep(1)
            out["requests"].append(f"EXC {type(e).__name__}")
    out["exit_code"] = p.poll()
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout = 20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    text = logf.read_text(errors = "replace")
    out["metal_device"] = next(
        (
            ln.strip()
            for ln in text.splitlines()
            if "ggml_metal_device_init: GPU name" in ln or "using device Metal" in ln
        ),
        None,
    )
    out["offloaded"] = next(
        (ln.strip()[-60:] for ln in text.splitlines() if "offloaded" in ln), None
    )
    out["draft_acceptance"] = [
        ln.split("draft acceptance")[-1].strip()[:60]
        for ln in text.splitlines()
        if "draft acceptance" in ln
    ][-2:]
    out["survived"] = (
        out["healthy"] and out["exit_code"] is None and all(c == 200 for c in out["requests"])
    )
    if not out["survived"]:
        out["log_tail"] = text.splitlines()[-25:]
    return out


# Runs in a child per arm: a fresh interpreter imports exactly one Studio tree.
STUDIO_ARM = textwrap.dedent(r"""
import json, os, signal, sys, threading, time
src, model, drafter, inject = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
sys.path.insert(0, src)
import psutil
from core.inference.llama_cpp import LlamaCppBackend, GgufLoadIntent
b = LlamaCppBackend()
ok = b.load_model(GgufLoadIntent(model_identifier = "unsloth/gemma-4-E2B-it-GGUF",
                                 gguf_path = model, mtp_draft_path = drafter, n_ctx = 8192,
                                 speculative_type = "auto"))
def argv(p):
    try:
        return psutil.Process(p.pid).cmdline()
    except Exception:
        return []
def mtp_live():
    p = b._process
    return p is not None and p.poll() is None and any("draft-mtp" in a for a in argv(p))
res = {"load_ok": bool(ok), "mtp_at_load": mtp_live(),
       "fallback_armed": bool(getattr(b, "_mtp_runtime_fallback_active", False)),
       "requests": [], "aborts": 0}
stop = threading.Event()
killed = set()
def injector():
    while inject and not stop.is_set():
        p = b._process
        if (p is not None and p.poll() is None and p.pid not in killed and mtp_live()
                and any("processing task" in str(l) for l in list(b._stdout_lines))):
            time.sleep(0.3)
            if p.poll() is None:
                os.kill(p.pid, signal.SIGABRT); killed.add(p.pid); res["aborts"] += 1
        time.sleep(0.05)
threading.Thread(target = injector, daemon = True).start()
for i in range(6):
    try:
        out = ""
        for ch in b.generate_chat_completion([{"role": "user", "content": f"Count to 10. ({i})"}],
                                             max_tokens = 32, temperature = 0.0,
                                             enable_thinking = False):
            if isinstance(ch, str):
                out = ch
        res["requests"].append("OK")
    except Exception as e:
        res["requests"].append(f"FAIL {type(e).__name__}: {str(e)[:80]}")
    deadline = time.time() + 180
    while time.time() < deadline and (getattr(b, "_mtp_runtime_fallback_in_progress", False)
                                       or not b.is_loaded):
        time.sleep(0.5)
    time.sleep(1)
stop.set()
res["mtp_at_end"] = mtp_live()
res["server_alive_at_end"] = b._process is not None and b._process.poll() is None
res["fallback_reason"] = b._spec_fallback_reason
res["requested_mode"] = b._requested_spec_mode
print("ARM_RESULT " + json.dumps(res), flush = True)
b.unload_model()
""")


def studio_arm(
    src: str,
    model: str,
    drafter: str,
    inject: bool,
    server: Path,
    home: Path,
    logdir: Path,
    label: str,
) -> dict:
    home.mkdir(parents = True, exist_ok = True)
    script = logdir / "studio_arm.py"
    script.write_text(STUDIO_ARM)
    env = dict(
        os.environ,
        LLAMA_SERVER_PATH = str(server),
        UNSLOTH_STUDIO_HOME = str(home),
        PYTHONUNBUFFERED = "1",
    )
    logf = logdir / f"studio_{label}.log"
    t0 = time.time()
    with open(logf, "w") as fh:
        rc = subprocess.run(
            [sys.executable, str(script), src, model, drafter, "1" if inject else "0"],
            stdout = fh,
            stderr = subprocess.STDOUT,
            env = env,
            timeout = 1800,
        ).returncode
    text = logf.read_text(errors = "replace")
    line = next((ln for ln in reversed(text.splitlines()) if ln.startswith("ARM_RESULT ")), None)
    res = (
        json.loads(line[len("ARM_RESULT ") :])
        if line
        else {"error": f"rc={rc}", "tail": text.splitlines()[-30:]}
    )
    res["wall_s"] = round(time.time() - t0, 1)
    res["log_lines_of_interest"] = [
        ln[:220]
        for ln in text.splitlines()
        if any(
            k in ln
            for k in (
                "respawn",
                "mid-generation",
                "Reloaded without",
                "exited (code",
                "Metal",
                "paravirtual",
            )
        )
    ][:20]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", default = "b11160-mix-a6922cc,b11541-mix-2500203")
    ap.add_argument("--base-sha", required = True, help = "unslothai/unsloth commit for the base arm")
    ap.add_argument("--head-src", default = "studio/backend")
    ap.add_argument("--skip-raw", action = "store_true")
    ap.add_argument("--out", default = "mac_gguf_mtp_fallback.json")
    a = ap.parse_args()
    result = {
        "job": "mac_gguf_mtp_fallback",
        "model": f"{REPO}/{MODEL}",
        "backend": "llama.cpp",
        "device": f"{platform.system()} {platform.machine()}",
        "config": vars(a),
        "observations": {},
        "checks": {},
        "passed": False,
        "error": None,
        "skipped": False,
        "finished": False,
    }

    def dump():
        Path(a.out).write_text(json.dumps(result, indent = 1))

    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        result.update(skipped = True, error = "needs Apple Silicon")
        dump()
        print("JOB_RESULT " + json.dumps({"passed": False, "skipped": True}))
        sys.exit(3)
    try:
        sh(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "httpx",
                "structlog",
                "rich",
                "click",
                "psutil",
                "huggingface_hub",
            ]
        )
        from huggingface_hub import hf_hub_download

        work = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "mtp13253"
        logdir = Path("mac_gguf_mtp_logs").resolve()
        logdir.mkdir(exist_ok = True)
        model = hf_hub_download(REPO, MODEL, local_dir = work / "models")
        drafter = hf_hub_download(REPO, DRAFTER, local_dir = work / "models")
        servers = {t: fetch_build(t, work / "bin") for t in a.builds.split(",")}
        sh(["sysctl", "-n", "hw.memsize", "machdep.cpu.brand_string"])
        sh(["system_profiler", "SPDisplaysDataType"])

        if not a.skip_raw:
            raw = []
            for tag, server in servers.items():
                for mode in SPEC_MODES:
                    r = raw_server_run(server, model, drafter, mode, logdir)
                    log("RAW", json.dumps({k: v for k, v in r.items() if k != "log_tail"}))
                    raw.append(r)
                    result["observations"]["metal_raw"] = raw
                    dump()
            ctl = [r for r in raw if r["mode"] == "off"]
            result["checks"]["metal_control_serves"] = {
                "ok": all(r["survived"] for r in ctl),
                "detail": "MTP-off Metal runs must serve, else the Metal probe proves nothing",
            }

        # Base tree: the commit the PR branches from, as a tarball (shallow checkout here).
        base_root = work / "base_src"
        if not base_root.exists():
            tgz = work / "base.tgz"
            urllib.request.urlretrieve(
                f"https://codeload.github.com/unslothai/unsloth/tar.gz/{a.base_sha}", tgz
            )
            with tarfile.open(tgz) as tf:
                tf.extractall(base_root)
        base_src = str(next(base_root.glob("*/studio/backend")))
        head_src = str(Path(a.head_src).resolve())
        server = servers[a.builds.split(",")[-1]]
        arms = {}
        for label, src, inject in (
            ("base_crash", base_src, True),
            ("head_crash", head_src, True),
            ("head_healthy", head_src, False),
        ):
            arms[label] = studio_arm(
                src, model, drafter, inject, server, work / f"home_{label}", logdir, label
            )
            log("ARM", label, json.dumps(arms[label]))
            result["observations"]["studio"] = arms
            dump()
        bc, hc, hh = arms["base_crash"], arms["head_crash"], arms["head_healthy"]
        n_ok = lambda r: sum(x == "OK" for x in r.get("requests", []))
        result["checks"]["base_reproduces_failure_loop"] = {
            "ok": bool(bc.get("mtp_at_load")) and n_ok(bc) <= 1 and bc.get("aborts", 0) >= 3,
            "detail": f"base requests={bc.get('requests')} aborts={bc.get('aborts')}",
        }
        result["checks"]["head_recovers_without_mtp"] = {
            "ok": bool(hc.get("mtp_at_load"))
            and n_ok(hc) >= 4
            and hc.get("aborts") == 1
            and not hc.get("mtp_at_end")
            and hc.get("server_alive_at_end")
            and hc.get("fallback_reason") == "runtime_error",
            "detail": f"head requests={hc.get('requests')} aborts={hc.get('aborts')} "
            f"mtp_at_end={hc.get('mtp_at_end')} reason={hc.get('fallback_reason')} "
            f"mode={hc.get('requested_mode')}",
        }
        result["checks"]["head_keeps_mtp_when_healthy"] = {
            "ok": n_ok(hh) == 6 and bool(hh.get("mtp_at_load")) and bool(hh.get("mtp_at_end")),
            "detail": f"healthy requests={hh.get('requests')} mtp_at_end={hh.get('mtp_at_end')}",
        }
    except Exception as e:  # noqa: BLE001
        import traceback
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    result["passed"] = (
        result["error"] is None
        and bool(result["checks"])
        and all(c["ok"] for c in result["checks"].values())
    )
    result["finished"] = True
    dump()
    print(
        "JOB_RESULT "
        + json.dumps(
            {
                "passed": result["passed"],
                "checks": {k: v["ok"] for k, v in result["checks"].items()},
                "error": result["error"],
            }
        )
    )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
