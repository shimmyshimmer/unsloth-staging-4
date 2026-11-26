"""One sandbox latency bench for every OS (plans/sandbox_bench_and_hash.md).

Windows: MXC (jobs/mxc_fair_bench.py). Linux: bubblewrap. macOS: Seatbelt. Same protocol and workloads everywhere:
host spawn (`sh -c 'echo 1'` / `cmd /d /c echo 1`, `python -S -c print(1)`), the bare sandbox around the same
commands, and Studio `execute_tool` (Python and Terminal, required mode) through two workers (A/A), interleaved
per round after warmup to a steady state, on a settled VM with a CPU calibration and a fingerprint.

Usage: python .github/scripts/jobs/sandbox_fair_bench.py --out job_sandbox_fair.json [--rounds 80]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sandbox_bench_common as sbc  # noqa: E402

_ENSURE_BWRAP = (
    "command -v bwrap >/dev/null || { apt-get install -y --no-install-recommends bubblewrap || "
    "{ apt-get update && apt-get install -y --no-install-recommends bubblewrap; }; }; "
    'if [ "$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)" = 1 ]; then '
    "apt-get install -y --no-install-recommends apparmor-profiles || true; "
    "[ -f /etc/apparmor.d/bwrap-userns-restrict ] || install -m 644 "
    "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict /etc/apparmor.d/ || true; "
    "apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict || true; fi"
)


def ensure_bwrap():
    pre = ["sudo", "-n"] if shutil.which("sudo") and os.geteuid() != 0 else []
    r = subprocess.run(
        pre + ["sh", "-c", _ENSURE_BWRAP], capture_output = True, text = True, timeout = 900
    )
    version = (
        subprocess.run(["bwrap", "--version"], capture_output = True, text = True).stdout.strip()
        if shutil.which("bwrap")
        else None
    )
    return {"rc": r.returncode, "version": version, "tail": (r.stdout + r.stderr)[-500:]}


def posix_arms():
    sh = "/bin/sh"
    py = [sys.executable, "-S", "-c", "print(1)"]
    host = {"host|sh_echo": ([sh, "-c", "echo 1"], {}), "host|python_pass": (py, {})}
    if sys.platform == "darwin":
        # Minimal profile: the cost of entering Seatbelt itself; Studio's generated profile is in the Studio arms.
        wrap = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)"]
        name = "seatbelt"
    else:
        wrap = [
            "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
        ]
        name = "bwrap"
    arms = dict(host)
    arms[f"{name}|sh_echo"] = (wrap + [sh, "-c", "echo 1"], {})
    arms[f"{name}|python_pass"] = (wrap + py, {})
    arms[f"{name}'|python_pass"] = (wrap + py, {})  # A/A slot
    return arms


def posix_main(args, result, section, flush):
    sbc.ensure_psutil()
    result["tier_mode"] = "macos-seatbelt" if sys.platform == "darwin" else "bubblewrap"
    if sys.platform.startswith("linux"):
        section("ensure_bwrap", ensure_bwrap)
    section("fingerprint_start", sbc.posix_fingerprint)
    section("settle", sbc.settle)
    section("calibration_start", sbc.calibrate)
    import mxc_bench as mb

    section("studio_deps", mb._ensure_studio_deps)
    arms = posix_arms()
    section(
        "bare",
        lambda: sbc.run_rounds(
            arms, lambda a: sbc._call(*a), 30, args.rounds, args.seed, result["records"], "bare"
        )
        | {"arms": sorted(arms)},
    )
    section("calibration_mid", sbc.calibrate)
    work = Path(tempfile.mkdtemp(prefix = "sbxfair-"))

    def studio():
        env = {**os.environ, "FB_BACKEND": str(sbc.BACKEND), "PYTHONIOENCODING": "utf-8"}
        workers = {}
        # Creation order flips with the seed's parity: a worker-order bias then shows up as an A/A sign flip.
        order = ("studio", "studio'") if args.seed % 2 else ("studio'", "studio")
        for name in order:
            w = sbc.Worker(
                name, {**env, "UNSLOTH_STUDIO_HOME": str(work / f"home_{len(workers)}")}, work
            )
            if "error" not in w.ready:
                workers[name] = w
        try:
            return {
                "ready": {n: w.ready for n, w in workers.items()},
                **sbc.studio_rounds(workers, 30, args.studio_rounds, args.seed, result["records"]),
            }
        finally:
            for w in workers.values():
                w.close()

    section("studio", studio)
    section("calibration_end", sbc.calibrate)
    section("fingerprint_end", sbc.posix_fingerprint)


def main():
    if sys.platform == "win32":
        import mxc_fair_bench
        return mxc_fair_bench.main()
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--out", default = "job_sandbox_fair.json")
    p.add_argument("--rounds", type = int, default = 80)
    p.add_argument("--studio-rounds", type = int, default = 40)
    p.add_argument("--seed", type = int, default = 3407)
    p.add_argument("--defender-toggle", action = "store_true")  # Windows only; accepted everywhere
    args = p.parse_args()
    out_path = Path(args.out).resolve()
    result = {"job": "sandbox_fair_bench", "passed": False, "sections": {}, "records": []}

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
            f"=== {name} ===\n{json.dumps(result['sections'][name], default = str)[:4000]}",
            flush = True,
        )

    try:
        posix_main(args, result, section, flush)
        result["summary"] = sbc.summarize(result["records"])
        total = len(result["records"])
        bad = sum(not r["ok"] for r in result["records"])
        result["ok_rate"] = round(1 - bad / total, 4) if total else 0
        studio_ok = any(k.startswith("studio_") for k in result["summary"])
        result["passed"] = bool(total) and result["ok_rate"] >= 0.95 and studio_ok
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        flush()
        print(
            "JOB_RESULT "
            + json.dumps(
                {"passed": result["passed"], "summary": result.get("summary")}, default = str
            )[:20000],
            flush = True,
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
