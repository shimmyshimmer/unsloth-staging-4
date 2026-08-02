#!/usr/bin/env python3
"""Download real model files anonymously, under Xet (stock vs capped) and HTTP, and measure.

Runs on a GitHub runner: ~16GB RAM, 4 cores, ~14GB free disk, no HF token. That is a far better
model of a normal user's machine than the host this change was developed on, and it is the case
where hf_xet's stock 8GB reconstruction buffer is most likely to hurt.

Single files rather than whole repos, and the cache is deleted after every cell, because the disk
budget is smaller than any one of these repos in full.

Two of the requested repos ship one monolithic 17-22GB file, which no runner can hold. Those cells
run TRUNCATED: the download is stopped after a byte/time budget and the row is marked
`truncated: true`. Peak RSS is still valid -- hf_xet allocates its reconstruction buffers up front,
so the memory ceiling is reached in the first seconds, long before the byte budget -- and so is
throughput. Only "did the whole file arrive" is unanswered, and the smaller cells answer that.

    python xet_download_matrix.py                       # every cell
    python xet_download_matrix.py --repo unsloth/...    # one repo (the CI job matrix does this)
"""

from __future__ import annotations

import argparse
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

# (repo, filename, approx GB, truncate_gb or None)
# Covers every repo in the request: plain GGUF, a quant-subfolder layout, a multi-shard safetensors
# shard, an mmproj sidecar, and the two monolithic MoE quants (truncated).
CELLS = [
    ("unsloth/Qwen3.5-2B-MTP-GGUF", "Qwen3.5-2B-IQ4_XS.gguf", 1.25, None),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_0.gguf", 2.58, None),
    ("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-UD-Q4_K_XL.gguf", 3.18, None),
    ("unsloth/Qwen3-30B-A3B-Instruct-2507", "model-00003-of-00016.safetensors", 4.00, None),
    ("unsloth/gemma-4-31B-it-GGUF", "mmproj-BF16.gguf", 1.20, None),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf", 17.0, 4.0),
    ("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", 22.4, 4.0),
]

TRUNCATE_SECONDS = 180.0

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


def build_env(transport: str, profile: str, log_dir: Path, *, token: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_XET_")}
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    if token:
        env["HF_TOKEN"] = token
        env.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
    else:
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        for k in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            env.pop(k, None)
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
                # st_blocks, not st_size: Xet writes a sparse file out of order, so st_size jumps to
                # the full length immediately and would report a 17GB "download" in one second.
                total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
            except OSError:
                pass
    return total


def run_cell(repo: str, filename: str, transport: str, profile: str,
             truncate_gb: float | None, token: str | None) -> dict:
    shutil.rmtree(CACHE, ignore_errors = True)
    CACHE.mkdir(parents = True, exist_ok = True)
    auth = "token" if token else "anon"
    log_dir = LOGS / f"{repo.replace('/', '__')}_{transport}_{profile}_{auth}"
    env = build_env(transport, profile, log_dir, token = token)
    payload = json.dumps([repo, filename, str(CACHE)])

    print(f"--- {repo}/{filename} {transport}/{profile} {auth}"
          f"{f' [truncate {truncate_gb}GB]' if truncate_gb else ''}", flush = True)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-u", "-c", CHILD, payload], env = env,
                            stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True,
                            start_new_session = True)
    peak = 0
    truncated = False
    last_probe = 0.0
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
        now = time.time()
        if truncate_gb and now - last_probe > 3:
            last_probe = now
            if dir_bytes(CACHE) >= truncate_gb * 1e9 or now - t0 > TRUNCATE_SECONDS:
                truncated = True
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except OSError:
                    proc.kill()
                break
        time.sleep(0.5)
    downloaded = dir_bytes(CACHE)   # before communicate(), the child may still be writing
    stdout, stderr = proc.communicate()
    wall = time.time() - t0
    downloaded = max(downloaded, dir_bytes(CACHE))

    result = {"ok": False, "error": "no result line"}
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if truncated:
        result = {"ok": True, "error": None}

    try:
        from unsloth_zoo.hf_xet_tuning import scan_xet_log

        xet_errors = scan_xet_log(log_dir, max_messages = 3)
    except Exception:
        xet_errors = []

    row = {
        "repo": repo, "filename": filename, "transport": transport, "profile": profile,
        "auth": auth, "truncated": truncated,
        "ok": bool(result.get("ok")), "error": result.get("error"),
        "wall_seconds": round(wall, 2), "bytes": downloaded,
        "peak_rss_gb": round(peak / 1e9, 3),
        "throughput_mbps": round(downloaded * 8 / 1e6 / wall, 1) if wall > 0 else 0,
        "xet_log_errors": xet_errors,
        "stderr_tail": (stderr or "")[-800:],
    }
    print(f"    ok={row['ok']} truncated={truncated} peak_rss={row['peak_rss_gb']}GB "
          f"wall={row['wall_seconds']}s tput={row['throughput_mbps']}Mbps "
          f"xet_errors={len(xet_errors)}", flush = True)
    shutil.rmtree(CACHE, ignore_errors = True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help = "run only this repo's cells")
    args = ap.parse_args()

    cells = [c for c in CELLS if not args.repo or c[0] == args.repo]
    if not cells:
        print(f"no cell for {args.repo}", flush = True)
        return 1

    # Only present if the repo owner deliberately configured the secret. Absent by default, which is
    # the whole point of this job -- but when it IS present we get the token-vs-anonymous answer on
    # the same runner, same link, same minute, which no local comparison can claim.
    token = os.environ.get("BENCH_HF_TOKEN") or None

    RESULTS.unlink(missing_ok = True)
    failures = 0
    for repo, filename, approx_gb, truncate_gb in cells:
        need = (truncate_gb or approx_gb) * 1.5
        free_gb = shutil.disk_usage(".").free / 1e9
        if free_gb < need:
            print(f"skipping {repo}: {free_gb:.1f}GB free, need ~{need:.1f}GB", flush = True)
            continue
        combos = [("xet", "stock", None), ("xet", "tuned", None), ("http", "tuned", None)]
        if token:
            combos += [("xet", "tuned", token), ("http", "tuned", token)]
        for transport, profile, tok in combos:
            row = run_cell(repo, filename, transport, profile, truncate_gb, tok)
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
