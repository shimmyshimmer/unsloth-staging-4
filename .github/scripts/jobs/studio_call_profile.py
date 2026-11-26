"""Where a sandboxed Studio tool call spends its time, per step, on Windows (MXC), Linux (bwrap) and macOS (Seatbelt).

plans/win_studio_call_speed.md, Phase 1. One Studio worker runs `execute_tool` in required mode with every function
of the sandbox modules (and a curated list in tools.py) wrapped by a wall-clock span timer (inclusive and exclusive
ms, calls per tool call); then a cProfile pass; then host-side reference launches of Studio's real child (its argv
and environment, with and without the sitecustomize shim) and `-X importtime` of that child.

Usage: python .github/scripts/jobs/studio_call_profile.py --out job_studio_call_profile.json [--calls 40]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sandbox_bench_common as sbc  # noqa: E402

_PROFILE_WORKER = r"""
import cProfile, inspect, io, json, os, pstats, statistics, subprocess, sys, threading, time
sys.path.insert(0, os.environ["FB_BACKEND"])
import logging
logging.disable(logging.WARNING)
from core.inference import tools, os_sandbox

_local = threading.local()
_spans = {}          # name -> [inclusive_ns, exclusive_ns, calls] for the current tool call
_spans_lock = threading.Lock()
_popen_names = []
_capture = {}

def _wrap(owner, name, label):
    fn = getattr(owner, name)
    if getattr(fn, "_fb_wrapped", False):
        return
    def wrapper(*a, **k):
        stack = getattr(_local, "stack", None)
        if stack is None:
            stack = _local.stack = []
        stack.append(0)
        t = time.perf_counter_ns()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter_ns() - t
            child = stack.pop()
            if stack:
                stack[-1] += dt
            with _spans_lock:
                s = _spans.setdefault(label, [0, 0, 0])
                s[0] += dt; s[1] += dt - child; s[2] += 1
    wrapper._fb_wrapped = True
    wrapper.__wrapped__ = fn
    setattr(owner, name, wrapper)

wrapped, missing = [], []
mods = ["os_sandbox"]
if sys.platform == "win32":
    mods += ["mxc_runtime", "mxc_policy", "mxc_adapter", "mxc_probe", "mxc_read_grants", "mxc_drive_alias",
             "sandbox_windows_mxc"]
elif sys.platform == "darwin":
    mods += ["sandbox_macos"]
else:
    mods += ["sandbox_linux", "sandbox_landlock", "sandbox_seccomp"]
import importlib
for m in mods:
    try:
        mod = importlib.import_module("core.inference." + m)
    except Exception as exc:
        missing.append(f"{m}: {exc}")
        continue
    for name, obj in list(vars(mod).items()):
        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
            _wrap(mod, name, f"{m}.{name}")
            wrapped.append(f"{m}.{name}")
TOOLS = ["_python_exec", "_bash_exec", "_check_code_safety", "_tool_workdir_for_guard", "_get_workdir",
         "_account_confinement", "_call_started", "_call_finished", "_snapshot_and_history_begin", "_build_safe_env",
         "_refresh_sandbox_memory_limit", "_harden_parent_against_proc_env_leak", "_prepare_tool_launch",
         "_is_user_project_folder", "_apply_prepared_launch", "_capture_process_group", "_cap_tool_memory",
         "_adopt_tool_pid", "_forget_tool_pid", "_drain_process_output", "_wait_process", "_truncate",
         "_created_file_sentinels", "_note_tool_execution", "_forget_sandbox_capability_if_the_backend_failed",
         "_profile_for_request", "_terminal_profile", "_python_source_encoding", "_references_studio_credential_here",
         "_python_builds_a_credential_path", "_needs_a_workdir", "_spill_scope", "_requested_execution_mode",
         "_reaches_host_paths", "_defuse_sentinels", "_missing_path_hint", "_memory_exceeded_note",
         "_completion_reason", "_check_command_safety", "_terminal_is_potentially_unsafe", "_terminal_is_high_risk"]
for name in TOOLS:
    if hasattr(tools, name) and inspect.isfunction(getattr(tools, name)):
        _wrap(tools, name, f"tools.{name}")
        wrapped.append(f"tools.{name}")
    else:
        missing.append(f"tools.{name}")
_orig_build_env = tools._build_safe_env
def _capture_env(workdir, *a, **k):
    env = _orig_build_env(workdir, *a, **k)
    _capture["env"], _capture["workdir"] = dict(env), workdir
    return env
tools._build_safe_env = _capture_env
_Popen_init = subprocess.Popen.__init__
def _popen_init(self, args, *a, **k):
    t = time.perf_counter_ns()
    try:
        return _Popen_init(self, args, *a, **k)
    finally:
        argv0 = args[0] if isinstance(args, (list, tuple)) and args else str(args).split(" ")[0]
        _popen_names.append((os.path.basename(str(argv0)), round((time.perf_counter_ns() - t) / 1e6, 3)))
subprocess.Popen.__init__ = _popen_init
pin = None
try:
    from core.inference import mxc_runtime
    pin = mxc_runtime.selected_pin().version if sys.platform == "win32" else None
except Exception:
    pass
print("READY " + json.dumps({"pin": pin, "wrapped": len(wrapped), "missing": missing}), flush = True)

def one(tool, i):
    args = {"code": f"print('FB-OK')\n# {i}"} if tool == "python" else {"command": "echo FB-OK"}
    with _spans_lock:
        _spans.clear()
    del _popen_names[:]
    t = time.perf_counter_ns()
    text = tools.execute_tool(tool, args, session_id = None, tool_execution_mode = "required")
    ms = (time.perf_counter_ns() - t) / 1e6
    rec = getattr(tools, "_last_tool_execution_record", None)
    with _spans_lock:
        spans = {k: [round(v[0] / 1e6, 3), round(v[1] / 1e6, 3), v[2]] for k, v in _spans.items()}
    return {"ms": ms, "ok": "FB-OK" in text, "backend": getattr(rec, "backend", None), "spans": spans,
            "popen": list(_popen_names), "text": text[:200] if "FB-OK" not in text else ""}

i = 0
for line in sys.stdin:
    cmd = line.split()
    if not cmd:
        continue
    if cmd[0] == "call":
        i += 1
        print("RES " + json.dumps(one(cmd[1], i)), flush = True)
    elif cmd[0] == "cprofile":
        tool, n = cmd[1], int(cmd[2])
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(n):
            i += 1
            one(tool, i)
        pr.disable()
        out = {}
        for key in ("cumulative", "tottime"):
            s = io.StringIO()
            st = pstats.Stats(pr, stream = s)
            st.strip_dirs().sort_stats(key).print_stats(45)
            out[key] = s.getvalue()[-9000:]
        print("RES " + json.dumps(out), flush = True)
    elif cmd[0] == "capture":
        print("RES " + json.dumps({"env": _capture.get("env"), "workdir": _capture.get("workdir")}), flush = True)
"""


class ProfileWorker:
    def __init__(self, env, work):
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _PROFILE_WORKER],
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            env = env,
            cwd = str(work),
            bufsize = 1,
        )
        line = self.proc.stdout.readline()
        self.ready = json.loads(line[6:]) if line.startswith("READY ") else {"error": line[-300:]}

    def ask(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            if line.startswith("RES "):
                return json.loads(line[4:])

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def aggregate(calls):
    good = [c for c in calls if c and c["ok"]]
    names = sorted({k for c in good for k in c["spans"]})
    rows = {}
    for n in names:
        inc = [c["spans"].get(n, [0, 0, 0])[0] for c in good]
        exc = [c["spans"].get(n, [0, 0, 0])[1] for c in good]
        cnt = [c["spans"].get(n, [0, 0, 0])[2] for c in good]
        rows[n] = {
            "incl_ms": round(statistics.median(inc), 3),
            "excl_ms": round(statistics.median(exc), 3),
            "excl_p90": round(sorted(exc)[(9 * len(exc)) // 10], 3),
            "calls": statistics.median(cnt),
        }
    total = [c["ms"] for c in good]
    popen = {}
    for c in good:
        for name, ms in c["popen"]:
            popen.setdefault(name, []).append(ms)
    return {
        "n": len(good),
        "failed": len(calls) - len(good),
        "total_ms": round(statistics.median(total), 2) if total else None,
        "total_p90": round(sorted(total)[(9 * len(total)) // 10], 2) if total else None,
        "backends": sorted({str(c["backend"]) for c in good}),
        "popen_per_call": {
            k: {
                "n_per_call": round(len(v) / max(len(good), 1), 2),
                "ms_median": round(statistics.median(v), 3),
            }
            for k, v in popen.items()
        },
        "top_exclusive": dict(sorted(rows.items(), key = lambda kv: -kv[1]["excl_ms"])[:40]),
        "fail_text": [c["text"] for c in calls if c and not c["ok"]][:3],
    }


def host_reference(capture, work, rounds):
    """Studio's real child, run on the host: what the shim and Studio's env cost before any sandbox."""
    env = capture.get("env") or {}
    workdir = capture.get("workdir") or str(work)
    script = Path(workdir) / "fb_profile_ref.py"
    script.write_text("print('FB-OK')\n", encoding = "utf-8")
    noshim = dict(env)
    pp = [
        p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p and "sandbox_site" not in p
    ]
    if pp:
        noshim["PYTHONPATH"] = os.pathsep.join(pp)
    else:
        noshim.pop("PYTHONPATH", None)
    base = {
        k: v
        for k, v in os.environ.items()
        if k.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME"}
    }
    arms = {
        "python_S": ([sys.executable, "-S", "-c", "print('FB-OK')"], {"env": base}),
        "python_plain": ([sys.executable, "-u", str(script)], {"env": base, "cwd": workdir}),
        "studio_child_noshim": (
            [sys.executable, "-u", str(script)],
            {"env": noshim, "cwd": workdir},
        ),
        "studio_child": ([sys.executable, "-u", str(script)], {"env": env, "cwd": workdir}),
    }
    records = []

    def call(a):
        argv, kw = a
        t = time.perf_counter()
        r = subprocess.run(argv, capture_output = True, text = True, errors = "replace", timeout = 60, **kw)
        return (time.perf_counter() - t) * 1000, r.returncode == 0 and "FB-OK" in r.stdout

    out = sbc.run_rounds(arms, call, 10, rounds, 3407, records, "host_ref")
    out["medians"] = {
        a: round(
            statistics.median([r["ms"] for r in records if r["arm"] == a and r["ok"]] or [0]), 2
        )
        for a in arms
    }
    out["ok"] = {a: sum(r["ok"] for r in records if r["arm"] == a) for a in arms}
    r = subprocess.run(
        [sys.executable, "-X", "importtime", "-u", str(script)],
        capture_output = True,
        text = True,
        errors = "replace",
        env = env,
        cwd = workdir,
        timeout = 60,
    )
    rows = []
    for line in r.stderr.splitlines():
        if line.startswith("import time:") and "|" in line:
            parts = [p.strip() for p in line[len("import time:") :].split("|")]
            if parts[0].isdigit():
                rows.append((int(parts[0]), int(parts[1]), parts[2]))
    out["importtime_top_self_us"] = sorted(rows, reverse = True)[:30]
    out["importtime_total_cumulative_us"] = sum(x[0] for x in rows)
    out["env_keys"] = sorted(env)
    out["pythonpath"] = env.get("PYTHONPATH")
    with __import__("contextlib").suppress(OSError):
        script.unlink()
    return out


def windows_setup(work):
    import mxc_bench as mb
    from mxc_fair_bench import INSTALLER

    server = sys.getwindowsversion().product_type != 1
    home = work / "home"
    env = {"UNSLOTH_STUDIO_HOME": str(home)}
    if server:
        env.update(UNSLOTH_MXC_ALLOW_DACL_FALLBACK = "1", UNSLOTH_MXC_PERSISTENT_READ_GRANTS = "1")
    inst = home / "mxc-runtime" / f"windows-{'arm64' if mb._arch() == 'arm64' else 'x86_64'}"
    full = {**os.environ, **env}
    a = subprocess.run(
        [sys.executable, str(INSTALLER), "--install-dir", str(inst)],
        capture_output = True,
        text = True,
        env = full,
        timeout = 900,
    )
    rec = {
        "server": server,
        "install_rc": a.returncode,
        "install_tail": (a.stdout + a.stderr)[-400:],
    }
    if server:
        b = subprocess.run(
            [sys.executable, str(INSTALLER), "--prepare-host", "--install-dir", str(inst)],
            capture_output = True,
            text = True,
            env = full,
            timeout = 1800,
        )
        rec["prep_rc"] = b.returncode
    return env, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default = "job_studio_call_profile.json")
    p.add_argument("--calls", type = int, default = 40)
    p.add_argument("--warmup", type = int, default = 10)
    p.add_argument("--cprofile-calls", type = int, default = 15)
    p.add_argument("--ref-rounds", type = int, default = 30)
    p.add_argument("--tiny", action = "store_true")
    p.add_argument(
        "--defender-toggle", action = "store_true"
    )  # accepted for a shared --job line; unused
    args = p.parse_args()
    if args.tiny:
        args.calls, args.warmup, args.cprofile_calls, args.ref_rounds = 5, 2, 3, 3
    out_path = Path(args.out).resolve()
    result = {"job": "studio_call_profile", "passed": False, "sections": {}}

    def flush():
        out_path.write_text(json.dumps(result, indent = 1, default = str))

    def section(name, fn):
        t = time.perf_counter()
        try:
            result["sections"][name] = fn()
        except Exception:  # noqa: BLE001
            result["sections"][name] = {"error": traceback.format_exc()[-3000:]}
        if isinstance(result["sections"][name], dict):
            result["sections"][name]["_section_s"] = round(time.perf_counter() - t, 1)
        flush()
        print(
            f"=== {name} ===\n{json.dumps(result['sections'][name], default = str)[:6000]}",
            flush = True,
        )
        return result["sections"][name]

    worker = None
    try:
        sbc.ensure_psutil()
        work = Path(tempfile.mkdtemp(prefix = "sbxprof-"))
        extra_env = {}
        if sys.platform == "win32":
            import mxc_fair_bench as mfb
            section("fingerprint", mfb.fingerprint)
        else:
            if sys.platform.startswith("linux"):
                import sandbox_fair_bench as sfb
                section("ensure_bwrap", sfb.ensure_bwrap)
            section("fingerprint", sbc.posix_fingerprint)
        section("settle", sbc.settle)
        section("calibration", sbc.calibrate)
        import mxc_bench as mb

        section("studio_deps", mb._ensure_studio_deps)
        if sys.platform == "win32":
            setup = section("windows_setup", lambda: dict(zip(("env", "rec"), windows_setup(work))))
            extra_env = setup.get("env") or {}
        else:
            extra_env = {"UNSLOTH_STUDIO_HOME": str(work / "home")}
        env = {
            **os.environ,
            "FB_BACKEND": str(sbc.BACKEND),
            "PYTHONIOENCODING": "utf-8",
            **extra_env,
        }
        worker = ProfileWorker(env, work)
        result["sections"]["worker_ready"] = worker.ready
        if "error" in worker.ready:
            raise RuntimeError(f"profile worker failed: {worker.ready}")
        calls = {}
        for tool in ("python", "terminal"):

            def run(tool = tool):
                for _ in range(args.warmup):
                    worker.ask(f"call {tool}")
                cs = [worker.ask(f"call {tool}") for _ in range(args.calls)]
                calls[tool] = cs
                return aggregate(cs)

            section(f"spans_{tool}", run)
        result["raw_calls"] = calls
        for tool in ("python", "terminal"):
            section(
                f"cprofile_{tool}",
                lambda tool = tool: worker.ask(f"cprofile {tool} {args.cprofile_calls}"),
            )
        cap = worker.ask("capture")
        section("host_reference", lambda: host_reference(cap or {}, work, args.ref_rounds))
        sp = result["sections"]
        result["passed"] = all(
            (sp.get(f"spans_{t}") or {}).get("n", 0) >= max(1, args.calls // 2)
            for t in ("python", "terminal")
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        if worker is not None:
            worker.close()
        flush()
        print(
            "JOB_RESULT "
            + json.dumps(
                {
                    "passed": result["passed"],
                    "totals": {
                        t: (result["sections"].get(f"spans_{t}") or {}).get("total_ms")
                        for t in ("python", "terminal")
                    },
                }
            ),
            flush = True,
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
