"""Fair MXC latency bench on one GitHub Windows runner (staging CI job; plans/mxc_fair_bench.md).

Every comparison is made on this VM, in a measured steady state:
  1. fingerprint (CPU model, cores, RAM, OS build/UBR, image, power plan, Defender, top CPU processes)
  2. settle: wait for the VM to go idle (post-boot Defender / update / ngen work)
  3. CPU calibration at start, middle and end (sha256 + pure-Python loop), to expose drift and give a
     secondary normalisation factor
  4. bare `wxc-exec` for MXC 0.8.0 / 0.9.0 / 1.0.1 / 1.1.0 with Studio's policy shape, interleaved per round
     (rotated, alternately reversed) with host spawn arms and an A/A slot; warmup until steady state
  5. Studio `tools.execute_tool` (required mode) through long-lived workers pinned to 1.1.0 / 1.0.1 plus an
     A/A 1.0.1 worker, interleaved per round
Each call is output-checked and recorded; summaries carry medians and paired per-round effects.

Usage: python .github/scripts/jobs/mxc_fair_bench.py --tiny --out job_mxc_fair.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mxc_bench as mb  # noqa: E402
from sandbox_bench_common import (  # noqa: E402
    Worker,
    _call,
    calibrate,
    run_rounds,
    settle,
    studio_rounds,
    summarize,
)

REPO = Path.cwd()
BACKEND = REPO / "studio" / "backend"
INSTALLER = REPO / "studio" / "install_mxc_prebuilt.py"
VERSIONS = ("0.8.0", "0.9.0", "1.0.1", "1.1.0")
SHAPES = (None, "0.9.0-alpha", "1.0.0")  # legacy first, then each schema the parser may demand


def _ps(script, timeout = 120):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        errors = "replace",
        timeout = timeout,
    )
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {"raw": r.stdout.strip()[-1500:], "err": r.stderr.strip()[-500:]}


def fingerprint():
    info = _ps(
        "$c=Get-CimInstance Win32_Processor|Select -First 1 Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed;"
        "$o=Get-CimInstance Win32_OperatingSystem|Select Caption,Version,BuildNumber,ProductType,TotalVisibleMemorySize,FreePhysicalMemory,LastBootUpTime;"
        "$cs=Get-CimInstance Win32_ComputerSystem|Select Model,HypervisorPresent,NumberOfLogicalProcessors;"
        "$ubr=(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion').UBR;"
        "$pw=(powercfg /getactivescheme) -join ' ';"
        "$d=$null; try { $d=Get-MpComputerStatus|Select AMServiceEnabled,RealTimeProtectionEnabled,AntivirusEnabled } catch {};"
        "$top=Get-Process|Sort-Object CPU -Descending|Select -First 10 Name,@{n='cpu_s';e={[math]::Round($_.CPU,1)}};"
        "[pscustomobject]@{cpu=$c;os=$o;system=$cs;ubr=$ubr;power=$pw;defender=$d;top_cpu=$top}|ConvertTo-Json -Depth 4 -Compress"
    )
    info["env"] = {
        k: os.environ.get(k)
        for k in ("ImageOS", "ImageVersion", "RUNNER_NAME", "RUNNER_ARCH", "GITHUB_RUN_ID")
    }
    info["python"] = {
        "version": sys.version.split()[0],
        "machine": platform.machine(),
        "arch": mb._arch(),
    }
    return info


# ---------------------------------------------------------------- bare arms


def qualify(versions_root, work, state, dacl, workloads):
    """Per version: probe tier and the first policy shape that runs each workload with the right output."""
    base = mb._raw_configs(work, dacl, port = 9)
    out = {}
    for v in VERSIONS:
        d = versions_root / v
        exe = d / "wxc-exec.exe"
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
        cell = {
            "exe_sha256": (meta.get("wxc-exec.exe") or {}).get("sha256"),
            "probe": mb._probe(d, state, n = 1),
        }
        tier = (
            cell["probe"].get("stdout", {}).get("tier")
            if isinstance(cell["probe"].get("stdout"), dict)
            else None
        )
        cell["tier"] = tier
        cell["workloads"] = {}
        for wl in workloads:
            for shape in SHAPES:
                cfg = copy.deepcopy(base[wl]) if shape is None else mb._migrate(base[wl], shape)
                argv, kw = mb._run_cfg(exe, cfg, state)
                try:
                    r = subprocess.run(argv, capture_output = True, text = True, errors = "replace", **kw)
                except subprocess.TimeoutExpired:
                    continue
                if r.returncode == 0 and r.stdout.strip().endswith("1"):
                    cell["workloads"][wl] = {"shape": shape or "legacy", "argv": argv, "kw": kw}
                    break
                cell.setdefault("errors", []).append(
                    {
                        "wl": wl,
                        "shape": shape,
                        "rc": r.returncode,
                        "err": (r.stderr or r.stdout)[-300:],
                    }
                )
        out[v] = cell
    return out


def _host_arms(workloads):
    system32 = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")
    arms = {}
    if "cmd_echo" in workloads:
        arms["host|cmd_echo"] = ([os.path.join(system32, "cmd.exe"), "/d", "/c", "echo 1"], {})
    if "python_pass" in workloads:
        arms["host|python_pass"] = ([sys.executable, "-S", "-c", "print(1)"], {})
    return arms


# ---------------------------------------------------------------- Studio arms


def defender_state():
    d = _ps(
        "Get-MpComputerStatus|Select RealTimeProtectionEnabled,IsTamperProtected|ConvertTo-Json -Compress"
    )
    return d if isinstance(d, dict) else {}


def set_realtime(enabled):
    """Flip Defender real-time protection; returns the state read back (Tamper Protection may refuse)."""
    flag = "$false" if enabled else "$true"
    r = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"Set-MpPreference -DisableRealtimeMonitoring {flag}",
        ],
        capture_output = True,
        text = True,
        errors = "replace",
        timeout = 120,
    )
    time.sleep(5)
    st = defender_state()
    st["set_rc"] = r.returncode
    st["set_err"] = r.stderr.strip()[-300:]
    return st


def studio_phase(
    work,
    pins,
    dacl,
    warm_max,
    rounds,
    seed,
    records,
    prefix = "studio",
):
    out = {"homes": {}}
    base_env = {**os.environ, "FB_BACKEND": str(BACKEND), "PYTHONIOENCODING": "utf-8"}
    if dacl:
        base_env.update(UNSLOTH_MXC_ALLOW_DACL_FALLBACK = "1", UNSLOTH_MXC_PERSISTENT_READ_GRANTS = "1")
    workers = {}
    for name, (pin, extra) in pins.items():
        home = work / f"home_{name.replace('~', '_')}"
        env = {**base_env, "UNSLOTH_STUDIO_HOME": str(home), "UNSLOTH_MXC_PIN": pin, **extra}
        inst_dir = (
            home / "mxc-runtime" / f"windows-{'arm64' if mb._arch() == 'arm64' else 'x86_64'}"
        )
        r = subprocess.run(
            [sys.executable, str(INSTALLER), "--install-dir", str(inst_dir)],
            capture_output = True,
            text = True,
            env = env,
            timeout = 900,
        )
        rec = {"install_rc": r.returncode, "install_out": (r.stdout + r.stderr)[-400:]}
        if dacl:
            p = subprocess.run(
                [sys.executable, str(INSTALLER), "--prepare-host", "--install-dir", str(inst_dir)],
                capture_output = True,
                text = True,
                env = env,
                timeout = 1800,
            )
            rec["prep_rc"] = p.returncode
        exe = inst_dir / "wxc-exec.exe"
        if exe.is_file():
            # Studio hashes this file on every launch (acquire_runtime); time exactly that cost here.
            sys.path.insert(0, str(BACKEND))
            from core.inference import mxc_runtime as _rt

            def old_hashlib(path):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            def timed(fn):
                xs, value = [], None
                for _ in range(15):
                    t = time.perf_counter()
                    value = fn(exe)
                    xs.append((time.perf_counter() - t) * 1000)
                return round(statistics.median(xs), 2), value

            studio_ms, studio_digest = timed(_rt._sha256_file)
            old_ms, old_digest = timed(old_hashlib)
            rec["exe_bytes"] = exe.stat().st_size
            rec["exe_sha256_ms"] = studio_ms  # what this Studio tree does per launch
            rec["exe_hashlib_ms"] = old_ms
            rec["native_hash"] = getattr(_rt, "_native_sha256", lambda: None)() is not None
            rec["digests_equal"] = studio_digest == old_digest
        w = Worker(name, env, work)
        rec["ready"] = w.ready
        out["homes"][name] = rec
        if w.ready.get("pin") == pin:
            workers[name] = w
        else:
            w.close()
    try:
        out.update(studio_rounds(workers, warm_max, rounds, seed, records, prefix))
    finally:
        for w in workers.values():
            w.close()
    return out


# ---------------------------------------------------------------- summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--out", default = "job_mxc_fair.json")
    p.add_argument("--rounds", type = int, default = 40)
    p.add_argument("--studio-rounds", type = int, default = 30)
    p.add_argument("--seed", type = int, default = 3407)
    p.add_argument("--defender-toggle", action = "store_true")
    args = p.parse_args()
    out_path = Path(args.out).resolve()
    result = {"job": "mxc_fair_bench", "passed": False, "sections": {}, "records": []}

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
        if sys.platform != "win32":
            result["skipped"] = "Windows only"
            result["passed"] = True
            return 0
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "psutil"],
            capture_output = True,
            timeout = 600,
        )
        work = Path(tempfile.mkdtemp(prefix = "mxcfair-"))
        vroot, state = work / "versions", work / "dacl-state"
        state.mkdir(parents = True)
        section("fingerprint_start", fingerprint)
        section("settle", settle)
        section("calibration_start", calibrate)
        section(
            "fetch",
            lambda: {
                v: json.loads((mb._fetch_version(v, vroot) / "meta.json").read_text())
                for v in VERSIONS
            },
        )

        server = sys.getwindowsversion().product_type != 1
        result["tier_mode"] = "appcontainer-dacl" if server else "base-container"
        section("studio_deps", mb._ensure_studio_deps)

        def native_hash_tests():
            # The real-CNG tests, here because the ARM64 legs' pytest step can die in the harness install.
            t = BACKEND / "tests" / "test_mxc_native_hash.py"
            if not t.exists():
                return {"skipped": "no test file in this tree"}
            r = subprocess.run(
                [sys.executable, "-m", "pytest", str(t), "-q", "-p", "no:cacheprovider", "-rs"],
                capture_output = True,
                text = True,
                cwd = str(BACKEND),
                timeout = 900,
                env = {**os.environ, "PYTHONPATH": str(BACKEND)},
            )
            return {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}

        section("native_hash_tests", native_hash_tests)
        if server:
            # Host prep once for the machine (1.0.1's wxc-host-prep through the installer).
            def prep():
                home = work / "home_prep"
                env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home), "UNSLOTH_MXC_PIN": "1.0.1"}
                d = home / "mxc-runtime" / "windows-x86_64"
                a = subprocess.run(
                    [sys.executable, str(INSTALLER), "--install-dir", str(d)],
                    capture_output = True,
                    text = True,
                    env = env,
                    timeout = 900,
                )
                b = subprocess.run(
                    [sys.executable, str(INSTALLER), "--prepare-host", "--install-dir", str(d)],
                    capture_output = True,
                    text = True,
                    env = env,
                    timeout = 1800,
                )
                return {
                    "install_rc": a.returncode,
                    "prep_rc": b.returncode,
                    "out": (b.stdout + b.stderr)[-500:],
                }

            section("host_prep", prep)
        # Bare Python on the DACL tier re-ACLs the whole Python install per launch (14-27 s): shell only there.
        workloads = ("cmd_echo",) if server else ("cmd_echo", "python_pass")

        def do_qualify():
            _qualified.update(qualify(vroot, work, state, server, workloads))
            return {
                v: {
                    **{k: x for k, x in c.items() if k != "workloads"},
                    "workloads": {wl: w["shape"] for wl, w in c["workloads"].items()},
                }
                for v, c in _qualified.items()
            }

        section("qualify", do_qualify)
        arms = dict(_host_arms(workloads))
        for v, c in _qualified.items():
            for wl, w in c["workloads"].items():
                arms[f"{v}|{wl}"] = (w["argv"], w["kw"])
                if v == "1.0.1":
                    arms[f"{v}'|{wl}"] = (w["argv"], w["kw"])  # A/A slot: same binary, same policy
        warm_max = 30
        section(
            "bare",
            lambda: run_rounds(
                arms,
                lambda a: _call(*a),
                warm_max,
                args.rounds,
                args.seed,
                result["records"],
                "bare",
            )
            | {"arms": sorted(arms)},
        )
        section("calibration_mid", calibrate)
        # "~hashlib" workers turn the CNG hash off (UNSLOTH_MXC_NATIVE_HASH=0): the old per-launch cost, same VM.
        off = {"UNSLOTH_MXC_NATIVE_HASH": "0"}
        pins = {"1.0.1": ("1.0.1", {}), "1.0.1'": ("1.0.1", {}), "1.0.1~hashlib": ("1.0.1", off)}
        if not server:
            pins["1.1.0"] = ("1.1.0", {})
            pins["1.1.0~hashlib"] = ("1.1.0", off)
        if sys.getwindowsversion().build >= 26100:
            section(
                "studio",
                lambda: studio_phase(
                    work, pins, server, warm_max, args.studio_rounds, args.seed, result["records"]
                ),
            )
        else:
            result["sections"]["studio"] = {"skipped": "Studio needs Windows build 26100 for MXC"}
        if args.defender_toggle:
            # Same VM, Defender real-time protection flipped from the image's default, then restored.
            original = defender_state().get("RealTimeProtectionEnabled")
            toggled = set_realtime(not original)
            result["sections"]["defender_toggle"] = {"original": original, "toggled": toggled}
            flush()
            if toggled.get("RealTimeProtectionEnabled") is (not original):
                tag = "rtp_on" if not original else "rtp_off"
                section(
                    f"bare_{tag}",
                    lambda: run_rounds(
                        arms,
                        lambda a: _call(*a),
                        warm_max,
                        args.rounds,
                        args.seed,
                        result["records"],
                        f"bare_{tag}",
                    ),
                )
                if sys.getwindowsversion().build >= 26100:
                    section(
                        f"studio_{tag}",
                        lambda: studio_phase(
                            work,
                            pins,
                            server,
                            warm_max,
                            args.studio_rounds,
                            args.seed,
                            result["records"],
                            prefix = f"studio_{tag}",
                        ),
                    )
            result["sections"]["defender_toggle"]["restored"] = set_realtime(bool(original))
            flush()
            # Shipped state again: shipped -> toggled -> shipped, so time drift shows as shipped-vs-shipped.
            section(
                "bare_after",
                lambda: run_rounds(
                    arms,
                    lambda a: _call(*a),
                    warm_max,
                    args.rounds,
                    args.seed,
                    result["records"],
                    "bare_after",
                ),
            )
        section("calibration_end", calibrate)
        section("fingerprint_end", fingerprint)
        result["summary"] = summarize(result["records"])
        total = len(result["records"])
        bad = sum(not r["ok"] for r in result["records"])
        result["ok_rate"] = round(1 - bad / total, 4) if total else 0
        result["passed"] = bool(total) and result["ok_rate"] >= 0.95
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


_qualified: dict = {}

if __name__ == "__main__":
    raise SystemExit(main())
