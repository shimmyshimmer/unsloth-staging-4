#!/usr/bin/env python3
"""Download a few real GGUF files anonymously, under Xet (stock vs capped) and HTTP.

Runs on a GitHub runner: ~16GB RAM, 4 cores, ~14GB free disk, no HF token. That is a far better
model of a normal user's machine than the host this change was developed on, and it is the case
where hf_xet's stock 8GB reconstruction buffer is most likely to hurt.

Single files rather than whole repos, and the cache is deleted after every cell, because the disk
budget is smaller than any one of these repos in full.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

CACHE = Path("hf-cache").resolve()
LOGS = Path("xetlogs").resolve()
RESULTS = Path("xet-download-results.jsonl").resolve()

# (repo, filename, approx GB) -- kept small enough that several fit in a runner's disk budget one
# at a time, and varied enough to cover a plain GGUF and a quant subfolder layout.
CELLS = [
    ("unsloth/Qwen3.5-2B-MTP-GGUF", "Qwen3.5-2B-IQ4_XS.gguf", 1.25),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_0.gguf", 2.58),
    ("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-UD-Q4_K_XL.gguf", 3.18),
]

CHILD = r"""
import json, sys, time
from huggingface_hub import hf_hub_download
repo, filename, cache = json.loads(sys.argv[1])
t0 = time.time()
try:
    p = hf_hub_download(repo_id=repo, filename=filename, cache_dir=cache)
    print("RESULT " + json.dumps({"ok": True, "seconds": time.time() - t0}))
except BaseException as e:
    print("RESULT " + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}))
"""


def build_env(transport: str, profile: str, log_dir: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_XET_")}
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    log_dir.mkdir(parents = True, exist_ok = True)
    env["HF_XET_LOG_DEST"] = os.path.join(str(log_dir), "")
    env["HF_XET_LOG_FORMAT"] = "json"
    if transport == "http":
        env["HF_HUB_DISABLE_XET"] = "1"
        return env
    env["HF_HUB_DISABLE_XET"] = "0"
    if profile == "stock":
        env["HF_XET_HIGH_PERFORMANCE"] = "1"   # what Unsloth used to set at import
        return env
    from unsloth_zoo.hf_xet_tuning import xet_env_overrides

    env.update(xet_env_overrides())
    return env


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _d, files in os.walk(path, onerror = lambda e: None):
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
                total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
            except OSError:
                pass
    return total


def run_cell(repo: str, filename: str, transport: str, profile: str) -> dict:
    shutil.rmtree(CACHE, ignore_errors = True)
    CACHE.mkdir(parents = True, exist_ok = True)
    log_dir = LOGS / f"{repo.replace('/', '__')}_{transport}_{profile}"
    env = build_env(transport, profile, log_dir)
    payload = json.dumps([repo, filename, str(CACHE)])

    print(f"--- {repo}/{filename} {transport}/{profile}", flush = True)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-u", "-c", CHILD, payload], env = env,
                            stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
    peak = 0
    try:
        root = psutil.Process(proc.pid)
    except psutil.Error:
        root = None
    while proc.poll() is None:
        if root is not None:
            rss = 0
            try:
                for p in [root] + root.children(recursive = True):
                    try:
                        rss += p.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            peak = max(peak, rss)
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    wall = time.time() - t0

    result = {"ok": False, "error": "no result line"}
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    downloaded = dir_bytes(CACHE)

    try:
        from unsloth_zoo.hf_xet_tuning import scan_xet_log

        xet_errors = scan_xet_log(log_dir, max_messages = 3)
    except Exception:
        xet_errors = []

    row = {
        "repo": repo, "filename": filename, "transport": transport, "profile": profile,
        "ok": bool(result.get("ok")), "error": result.get("error"),
        "wall_seconds": round(wall, 2), "bytes": downloaded,
        "peak_rss_gb": round(peak / 1e9, 3),
        "throughput_mbps": round(downloaded * 8 / 1e6 / wall, 1) if wall > 0 else 0,
        "xet_log_errors": xet_errors,
        "stderr_tail": (stderr or "")[-800:],
    }
    print(f"    ok={row['ok']} peak_rss={row['peak_rss_gb']}GB wall={row['wall_seconds']}s "
          f"tput={row['throughput_mbps']}Mbps xet_errors={len(xet_errors)}", flush = True)
    shutil.rmtree(CACHE, ignore_errors = True)
    return row


def main() -> int:
    RESULTS.unlink(missing_ok = True)
    failures = 0
    for repo, filename, approx_gb in CELLS:
        free_gb = shutil.disk_usage(".").free / 1e9
        if free_gb < approx_gb * 1.5:
            print(f"skipping {repo}: {free_gb:.1f}GB free, need ~{approx_gb * 1.5:.1f}GB",
                  flush = True)
            continue
        for transport, profile in (("xet", "stock"), ("xet", "tuned"), ("http", "tuned")):
            row = run_cell(repo, filename, transport, profile)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            if not row["ok"]:
                failures += 1

    print(f"\n{failures} failed cells", flush = True)
    # A failed cell is a finding, not a broken job: an anonymous rate limit or a transport error is
    # exactly what this job exists to surface, and the artifact records it either way.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
