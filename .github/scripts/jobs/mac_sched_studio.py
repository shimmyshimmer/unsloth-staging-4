"""Apple Silicon Studio check for scheduled tasks (unslothai/unsloth#13871): a Run now answers on a loaded MLX model.

Run from the staged checkout: installs Studio from it (`install.sh --local`, a throwaway home), starts it,
creates a daily task, presses Run now with nothing loaded (expects no_model), loads an MLX model and
presses Run now again (expects a saved chat naming that model, with a non-empty answer). Writes one
metrics JSON (`--out`) and a last `JOB_RESULT {...}` line; exit 0 only if every check passes.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--out", default = "outputs/jobs/mac_sched_studio.json")
    ap.add_argument("--tiny", action = "store_true", help = "accepted for the jobs contract; no effect")
    args = ap.parse_args()

    repo = Path.cwd()
    home = Path(tempfile.mkdtemp(prefix = "sched_studio_"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    result = {
        "job": "mac_sched_studio",
        "model": args.model,
        "backend": "mlx",
        "checks": {},
        "steps": {},
        "skipped": False,
        "error": None,
        "finished": False,
    }

    def call(
        method,
        path,
        payload = None,
        token = None,
        timeout = 900,
    ):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            base + path,
            method = method,
            headers = headers,
            data = None if payload is None else json.dumps(payload).encode(),
        )
        try:
            with urllib.request.urlopen(req, timeout = timeout) as r:
                body = r.read()
                return r.status, (json.loads(body) if body else None)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                return exc.code, json.loads(body)
            except ValueError:
                return exc.code, body.decode(errors = "replace")[:300]

    def check(
        name,
        ok,
        detail = "",
    ):
        result["checks"][name] = {"ok": bool(ok), "detail": detail}

    env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home), "UNSLOTH_SKIP_AUTOSTART": "1"}
    env.pop("UNSLOTH_IS_PRESENT", None)
    t0 = time.time()
    rc = subprocess.run(["bash", "install.sh", "--local"], cwd = repo, env = env).returncode
    result["steps"]["install_s"] = round(time.time() - t0, 1)
    check("studio_installed", rc == 0, f"install.sh rc={rc}")
    proc = None
    log_path = home / "studio.log"
    try:
        if rc != 0:
            return 1
        log = open(log_path, "w")
        proc = subprocess.Popen(
            [str(home / "bin" / "unsloth"), "studio", "-H", "127.0.0.1", "-p", str(port)],
            env = env,
            stdout = log,
            stderr = subprocess.STDOUT,
            start_new_session = True,
        )
        up = False
        for _ in range(200):
            try:
                if call("GET", "/api/health", timeout = 5)[0] == 200:
                    up = True
                    break
            except Exception:
                pass
            time.sleep(3)
        check("studio_up", up)
        if not up:
            return 1
        boot = (home / "auth" / ".bootstrap_password").read_text().strip()
        _, tok = call("POST", "/api/auth/login", {"username": "unsloth", "password": boot})
        _, tok = call(
            "POST",
            "/api/auth/change-password",
            {"current_password": boot, "new_password": "mac-sched-13871"},
            tok["access_token"],
        )
        token = tok["access_token"]
        code, task = call(
            "POST",
            "/api/scheduled-tasks",
            {
                "title": "Haiku",
                "prompt": "Write a three line haiku about the sea. Reply with the haiku only.",
                "schedule": {"kind": "daily", "time": "09:00"},
                "timezone": "UTC",
            },
            token,
        )
        check("task_created", code == 201, f"HTTP {code}")
        if code != 201:
            return 1

        def run_now():
            c, run = call("POST", f"/api/scheduled-tasks/{task['id']}/run", {}, token)
            if c != 202:
                return {"run_now_status": c}
            end = time.time() + 900
            while time.time() < end:
                _, t = call("GET", f"/api/scheduled-tasks/{task['id']}", None, token)
                last = (t or {}).get("lastRun") or {}
                if last.get("id") == run["id"] and last.get("status") in ("succeeded", "failed"):
                    out = {
                        "status": last["status"],
                        "error_code": last.get("errorCode"),
                        "error": last.get("error"),
                    }
                    if last.get("threadId"):
                        thread = (
                            call("GET", f"/api/chat/threads/{last['threadId']}", None, token)[1]
                            or {}
                        )
                        msgs = (
                            call(
                                "GET", f"/api/chat/threads/{last['threadId']}/messages", None, token
                            )[1]
                            or {}
                        ).get("messages") or []
                        out["thread_model"] = thread.get("modelId")
                        out["answer"] = " ".join(
                            json.dumps(m.get("content"))
                            for m in msgs
                            if m.get("role") == "assistant"
                        )[:400]
                    return out
                time.sleep(1)
            return {"status": "never_settled"}

        unloaded = run_now()
        result["steps"]["run_unloaded"] = unloaded
        check(
            "no_model_reason",
            (unloaded.get("status"), unloaded.get("error_code")) == ("failed", "no_model"),
            json.dumps(unloaded),
        )
        t1 = time.time()
        code, body = call(
            "POST",
            "/api/inference/load",
            {"model_path": args.model, "max_seq_length": 4096},
            token,
            timeout = 1800,
        )
        result["steps"]["load"] = {
            "status": code,
            "s": round(time.time() - t1, 1),
            "is_mlx": (body or {}).get("is_mlx") if isinstance(body, dict) else None,
            "detail": None if code == 200 else body,
        }
        check(
            "mlx_model_loaded",
            code == 200 and isinstance(body, dict) and body.get("is_mlx") is True,
            json.dumps(result["steps"]["load"])[:300],
        )
        t2 = time.time()
        loaded = run_now()
        loaded["s"] = round(time.time() - t2, 1)
        result["steps"]["run_loaded"] = loaded
        check(
            "answers_on_mlx_model",
            loaded.get("status") == "succeeded"
            and loaded.get("thread_model") == args.model
            and len((loaded.get("answer") or "").strip('" ')) > 0,
            json.dumps(loaded)[:400],
        )
        result["finished"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, 15)
            except Exception:
                pass
        result["passed"] = (
            bool(result["checks"])
            and result["error"] is None
            and all(c["ok"] for c in result["checks"].values())
        )
        if not result["passed"] and log_path.exists():
            print(log_path.read_text(errors = "replace")[-6000:], flush = True)
        Path(args.out).parent.mkdir(parents = True, exist_ok = True)
        Path(args.out).write_text(json.dumps(result, indent = 2))
        print("JOB_RESULT " + json.dumps(result), flush = True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
