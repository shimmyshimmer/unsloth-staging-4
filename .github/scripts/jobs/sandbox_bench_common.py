"""Shared protocol for the sandbox latency benches (plans/sandbox_bench_and_hash.md).

Same on every OS: fingerprint, settle until idle, CPU calibration, balanced interleaved rounds with warmup to a
steady state, an A/A slot, one record per call (output-checked), Studio `execute_tool` workers, summaries.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path.cwd()
BACKEND = REPO / "studio" / "backend"
STUDIO_BACKENDS = {"bubblewrap", "macos-seatbelt", "mxc-processcontainer"}


def ensure_psutil():
    try:
        import psutil  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "psutil"],
            capture_output = True,
            timeout = 600,
        )


def _run(argv, timeout = 60):
    try:
        r = subprocess.run(argv, capture_output = True, text = True, errors = "replace", timeout = timeout)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def posix_fingerprint():
    import psutil

    info = {
        "os": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "logical": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "env": {
            k: os.environ.get(k)
            for k in ("ImageOS", "ImageVersion", "RUNNER_NAME", "RUNNER_ARCH", "GITHUB_RUN_ID")
        },
    }
    if sys.platform == "darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["cores"] = _run(["sysctl", "-n", "hw.physicalcpu"])
        info["os_version"] = _run(["sw_vers", "-productVersion"])
    else:
        lscpu = _run(["lscpu"]) or ""
        fields = dict(line.split(":", 1) for line in lscpu.splitlines() if ":" in line)
        info["cpu"] = (fields.get("Model name") or "").strip() or None
        info["cores"] = (fields.get("Core(s) per socket") or "").strip() or None
        info["threads_per_core"] = (fields.get("Thread(s) per core") or "").strip() or None
        with contextlib.suppress(Exception):
            info["os_release"] = (
                dict(
                    line.split("=", 1)
                    for line in Path("/etc/os-release").read_text().splitlines()
                    if "=" in line
                )
                .get("PRETTY_NAME", "")
                .strip('"')
            )
    procs = sorted(
        psutil.process_iter(["name", "cpu_times"]),
        key = lambda p: -sum((p.info["cpu_times"] or (0, 0))[:2]),
    )
    info["top_cpu"] = [
        {"Name": p.info["name"], "cpu_s": round(sum((p.info["cpu_times"] or (0, 0))[:2]), 1)}
        for p in procs[:10]
    ]
    return info


def settle(
    max_s = 300,
    idle_pct = 10.0,
    hold_s = 20,
):
    import psutil

    trace, below, t0 = [], 0, time.monotonic()
    while time.monotonic() - t0 < max_s:
        pct = psutil.cpu_percent(interval = 1.0)
        trace.append(round(pct, 1))
        below = below + 1 if pct < idle_pct else 0
        if below >= hold_s:
            break
    return {
        "settled": below >= hold_s,
        "waited_s": round(time.monotonic() - t0, 1),
        "trace": trace[-60:],
        "median_pct": statistics.median(trace) if trace else None,
    }


def calibrate(reps = 7):
    blob = os.urandom(64 * 1024 * 1024)
    sha, loop = [], []
    for _ in range(reps):
        t = time.perf_counter()
        hashlib.sha256(blob).hexdigest()
        sha.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        acc = 0
        for i in range(2_000_000):
            acc = (acc * 31 + i) & 0xFFFFFFFF
        loop.append((time.perf_counter() - t) * 1000)
    return {
        "sha256_64mb_ms": round(statistics.median(sha), 2),
        "py_loop_2m_ms": round(statistics.median(loop), 2),
    }


def _cpu_now():
    import psutil
    return psutil.cpu_percent(interval = 0.2)


def _call(argv, kw):
    t = time.perf_counter()
    try:
        r = subprocess.run(
            argv,
            capture_output = True,
            text = True,
            errors = "replace",
            stdin = subprocess.DEVNULL,
            **{k: v for k, v in kw.items() if k != "stdin"},
        )
    except subprocess.TimeoutExpired:
        return (time.perf_counter() - t) * 1000, False
    return (time.perf_counter() - t) * 1000, r.returncode == 0 and r.stdout.strip().endswith("1")


def steady(
    xs,
    window = 5,
    drift = 0.03,
    cv = 0.08,
):
    if len(xs) < 2 * window:
        return False
    a, b = xs[-2 * window : -window], xs[-window:]
    ma, mbm = statistics.median(a), statistics.median(b)
    return abs(mbm - ma) / ma <= drift and statistics.pstdev(b) / statistics.fmean(b) <= cv


def run_rounds(arms, call, warm_max, rounds, seed, records, phase):
    """Warm all arms in rotation until each is steady (two consecutive checks), then timed rounds."""
    names = list(arms)
    rng = random.Random(seed)
    rng.shuffle(names)
    warm = {a: [] for a in names}
    ok_checks = {a: 0 for a in names}
    w = 0
    for w in range(warm_max):
        for a in names[w % len(names) :] + names[: w % len(names)]:
            ms, ok = call(arms[a])
            if ok:
                warm[a].append(ms)
        for a in names:
            ok_checks[a] = ok_checks[a] + 1 if steady(warm[a]) else 0
        if w >= 9 and all(c >= 2 for c in ok_checks.values()):
            break
    state = {"warm_rounds": w + 1, "steady": {a: ok_checks[a] >= 2 for a in names}}
    for r in range(rounds):
        order = names[r % len(names) :] + names[: r % len(names)]
        if r % 2:
            order.reverse()
        load = _cpu_now()
        for pos, a in enumerate(order):
            ms, ok = call(arms[a])
            records.append(
                {
                    "phase": phase,
                    "arm": a,
                    "round": r,
                    "pos": pos,
                    "ms": round(ms, 2),
                    "ok": ok,
                    "load_pct": load,
                }
            )
    return state


_WORKER = r"""
import json, os, sys, time
sys.path.insert(0, os.environ["FB_BACKEND"])
import logging
logging.disable(logging.WARNING)
from core.inference import tools
pin = None
try:
    from core.inference import mxc_runtime
    pin = mxc_runtime.selected_pin().version if sys.platform == "win32" else None
except Exception:
    pass
print("READY " + json.dumps({"pin": pin}), flush = True)
i = 0
for line in sys.stdin:
    tool = line.strip()
    if not tool:
        continue
    i += 1
    args = {"code": f"print('FB-OK')\n# {i}"} if tool == "python" else {"command": "echo FB-OK"}
    t = time.perf_counter()
    text = tools.execute_tool(tool, args, session_id = None, tool_execution_mode = "required")
    ms = (time.perf_counter() - t) * 1000
    rec = getattr(tools, "_last_tool_execution_record", None)
    print("RES " + json.dumps({"ms": ms, "ok": "FB-OK" in text, "backend": getattr(rec, "backend", None)}), flush = True)
"""


class Worker:
    def __init__(self, name, env, work):
        self.name = name
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER],
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            env = env,
            cwd = str(work),
            bufsize = 1,
        )
        line = self.proc.stdout.readline()
        self.ready = json.loads(line[6:]) if line.startswith("READY ") else {"error": line[-300:]}
        self.backends = set()

    def call(self, tool):
        self.proc.stdin.write(tool + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None, False, None
            if line.startswith("RES "):
                d = json.loads(line[4:])
                self.backends.add(d.get("backend"))
                return d["ms"], d["ok"] and d.get("backend") in STUDIO_BACKENDS, d.get("backend")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def studio_rounds(
    workers,
    warm_max,
    rounds,
    seed,
    records,
    prefix = "studio",
):
    """Interleave already-started workers per tool (python, terminal); same scheduler as the bare arms."""
    out = {}
    for tool in ("python", "terminal"):
        arms = {f"studio|{name}|{tool}": name for name in workers}

        def call(name, tool = tool):
            ms, ok, _backend = workers[name].call(tool)
            return (ms or 0.0), ok

        out[tool] = run_rounds(arms, call, warm_max, rounds, seed, records, f"{prefix}_{tool}")
    out["backends"] = {n: sorted(map(str, w.backends)) for n, w in workers.items()}
    return out


def summarize(records):
    by = {}
    for r in records:
        if r["ok"]:
            by.setdefault((r["phase"], r["arm"]), []).append(r)
    summ = {}
    for (phase, arm), rows in sorted(by.items()):
        ms = [x["ms"] for x in rows]
        summ[f"{phase}|{arm}"] = {
            "n": len(ms),
            "median_ms": round(statistics.median(ms), 2),
            "p10": round(sorted(ms)[len(ms) // 10], 2),
            "p90": round(sorted(ms)[(9 * len(ms)) // 10], 2),
        }
    return summ
