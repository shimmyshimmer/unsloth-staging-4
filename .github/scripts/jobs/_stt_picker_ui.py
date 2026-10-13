"""Launch one tree's Studio from source and photograph the Transcribe model picker.

Used by mlx_stt_ui.py on Apple Silicon (base and head trees, one shared Hugging Face cache).
`python _stt_picker_ui.py --tree DIR --hub HUB --out PNG --port N --expect REPO` prints one
`UI_FACTS {...}` line: whether REPO is in the open picker's text, plus the picker text itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def _post(
    base,
    path,
    body,
    token = None,
):
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return json.load(urllib.request.urlopen(req, timeout = 60))


def _get(base, path, token):
    req = urllib.request.Request(base + path, headers = {"Authorization": "Bearer " + token})
    return json.load(urllib.request.urlopen(req, timeout = 300))


def _launch(tree, hub, port, log, python):
    home = Path(tempfile.mkdtemp(prefix = "stt_ui_home_"))
    env = dict(
        os.environ,
        UNSLOTH_STUDIO_HOME = str(home),
        HF_HUB_CACHE = str(hub),
        HF_HOME = str(Path(hub).parent),
        UNSLOTH_DISABLE_AUTO_UPDATES = "1",
        PYTHONUNBUFFERED = "1",
    )
    cmd = [
        python,
        str(Path(tree) / "studio" / "backend" / "run.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--frontend",
        str(Path(tree) / "studio" / "frontend" / "dist"),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd = str(Path(tree) / "studio" / "backend"),
        env = env,
        stdout = open(log, "w"),
        stderr = subprocess.STDOUT,
        start_new_session = True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Studio exited {proc.returncode}: {Path(log).read_text()[-2000:]}")
        try:
            urllib.request.urlopen(base + "/api/health", timeout = 5)
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("Studio never answered /api/health")
    boot = home / "auth" / ".bootstrap_password"
    for _ in range(60):
        if boot.is_file():
            break
        time.sleep(1)
    pw = boot.read_text().strip()
    tok = _post(base, "/api/auth/login", {"username": "unsloth", "password": pw})
    tok = _post(
        base,
        "/api/auth/change-password",
        {"current_password": pw, "new_password": "StagingUi-1!pw"},
        tok["access_token"],
    )
    return proc, base, tok


async def _shoot(base, tok, out_png, expect):
    from playwright.async_api import async_playwright

    seed = json.dumps(
        {
            "unsloth_auth_token": tok["access_token"],
            "unsloth_refresh_token": tok.get("refresh_token", ""),
        }
    )
    init = f"(() => {{ const s = {seed}; for (const k in s) localStorage.setItem(k, s[k]); }})();"
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport = {"width": 1400, "height": 900})
        await page.add_init_script(init)
        await page.goto(base + "/audio", wait_until = "domcontentloaded")
        await page.wait_for_timeout(6000)
        # The page opens on Speak; Transcribe is the pill beside it.
        tab = page.get_by_role("tab", name = "Transcribe")
        if await tab.count() == 0:
            tab = page.get_by_text("Transcribe", exact = True)
        await tab.first.click(timeout = 30_000)
        await page.wait_for_timeout(2000)
        await page.locator("[data-tour='audio-model']").first.click(timeout = 30_000)
        await page.wait_for_timeout(3000)
        await page.get_by_text("On Device", exact = True).first.click(timeout = 30_000)
        await page.wait_for_timeout(4000)
        dialog = page.get_by_role("dialog")
        scope = dialog.first if await dialog.count() else page
        text = await scope.inner_text()
        await page.screenshot(path = str(out_png))
        await browser.close()
    leaf = expect.split("/")[-1]
    return {"listed": leaf in text, "picker_text": text[:4000]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required = True)
    ap.add_argument("--hub", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--port", type = int, required = True)
    ap.add_argument("--expect", required = True)
    ap.add_argument("--studio-python", default = sys.executable)
    a = ap.parse_args()
    log = Path(a.out).with_suffix(".studio.log")
    proc, base, tok = _launch(a.tree, a.hub, a.port, log, a.studio_python)
    facts = {"tree": a.tree}
    try:
        try:
            rows = _get(base, "/api/hub/cached-models", tok["access_token"])
        except Exception as exc:
            rows = {"error": str(exc)}
        facts["inventory_row"] = _row_for(rows, a.expect)
        facts.update(asyncio.run(_shoot(base, tok, a.out, a.expect)))
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    print("UI_FACTS " + json.dumps(facts), flush = True)


def _row_for(data, repo):
    found = []

    def walk(x):
        if isinstance(x, dict):
            if str(x.get("repo_id", "")).lower() == repo.lower():
                found.append(
                    {k: x.get(k) for k in ("repo_id", "task", "library_name", "pipeline_tag")}
                    | {"can_chat": (x.get("capabilities") or {}).get("can_chat")}
                )
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return found[:1] or data if isinstance(data, dict) and "error" in data else found[:1]


if __name__ == "__main__":
    main()
