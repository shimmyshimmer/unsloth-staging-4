"""PR #7207 S3 driver -- Training (QLoRA 1-step smoke) + Projects.

Gates:
  G_train_page (HARD): /studio loads, Qwen3-0.6B selectable, GET
     /api/models/config/* returns 200 (backend defaults applied), Start appears.
  G_train_start (best-effort, important): upload a tiny dataset, set Max Steps=1,
     click Start Training -> POST /api/train/start 200 and the Loss stat / loss
     chart becomes populated.
  G_project (HARD): create a chat project on /projects.

Run: python -m studio_test_kit.examples.pr7207_s3_training \
        --base http://127.0.0.1:8933 --password PW --dataset PATH --out out/s3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from ..auth import login, seed_init_script
from ..ui import open_chat

TRAIN_MODEL = "Qwen3-0.6B"


async def run(base, password, dataset, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    res = {"base": base, "gates": {}}

    def log(m):
        print(f"[s3] {m}", flush=True)

    auth = await login(base, "unsloth", password)
    init = seed_init_script(auth, [])
    async with open_chat(
        base, init_scripts=[init], video_dir=out_dir / "video",
        video_name="s3", transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page
        cfg_ok = {"seen": False}
        train_start_ok = {"seen": False}

        def on_resp(r):
            if "/api/models/config/" in r.url and r.ok:
                cfg_ok["seen"] = True
            if r.url.rstrip("/").endswith("/api/train/start") and r.ok:
                train_start_ok["seen"] = True

        page.on("response", on_resp)

        # ---- G_train_page ----
        try:
            log("open /studio")
            await page.goto(base + "/studio", wait_until="domcontentloaded", timeout=25000)
            await page.wait_for_timeout(1500)
            try:
                await page.get_by_role("tab", name=re.compile(r"Configure", re.I)).first.click(timeout=6000)
            except Exception:
                pass
            model_box = page.get_by_placeholder(re.compile(r"Search models", re.I)).first
            await model_box.wait_for(state="visible", timeout=12000)
            await model_box.click()
            await model_box.fill(TRAIN_MODEL)
            await page.wait_for_timeout(1500)
            opt = page.get_by_text(re.compile(rf"unsloth/{re.escape(TRAIN_MODEL)}$", re.I)).first
            try:
                await opt.click(timeout=6000)
            except Exception:
                await model_box.press("Enter")
            for _ in range(20):
                if cfg_ok["seen"]:
                    break
                await asyncio.sleep(0.5)
            await sp.screenshot(out_dir / "shots" / "10_model_selected.png")
            start_btn = page.locator('[data-tour="studio-start"]')
            start_present = await start_btn.count() > 0
            res["gates"]["G_train_page"] = {
                "pass": bool(cfg_ok["seen"] and start_present),
                "config_200": cfg_ok["seen"],
                "start_present": start_present,
            }
            log(f"  G_train_page pass={res['gates']['G_train_page']['pass']} (cfg200={cfg_ok['seen']})")
        except Exception as e:
            res["gates"]["G_train_page"] = {"pass": False, "error": repr(e)}
            log(f"  G_train_page ERROR {e!r}")

        # ---- G_train_start (best-effort): upload dataset, steps=1, Start ----
        try:
            log("upload tiny dataset")
            # Local source, then the hidden file input.
            try:
                await page.locator('[data-tour="studio-dataset"]').get_by_role(
                    "radio", name=re.compile(r"Local", re.I)
                ).first.click(timeout=5000)
            except Exception:
                pass
            finp = page.locator('input[type="file"]').first
            await finp.set_input_files(dataset, timeout=8000)
            await page.wait_for_timeout(2500)
            await sp.screenshot(out_dir / "shots" / "11_dataset.png")

            log("set Max Steps = 1")
            params = page.locator('[data-tour="studio-params"]')
            try:
                await params.get_by_role("spinbutton").first.fill("1", timeout=5000)
            except Exception:
                pass

            log("click Start Training")
            await page.locator('[data-tour="studio-start"]').first.click(timeout=8000)
            # Wait for train/start 200 + a numeric Loss / loss chart.
            loss_chart = False
            for _ in range(120):
                if train_start_ok["seen"]:
                    if await page.locator('[data-tour="studio-training-loss"]').count() > 0:
                        loss_chart = True
                        break
                await asyncio.sleep(1)
            await sp.screenshot(out_dir / "shots" / "12_training.png")
            res["gates"]["G_train_start"] = {
                "pass": bool(train_start_ok["seen"]),
                "train_start_200": train_start_ok["seen"],
                "loss_chart": loss_chart,
            }
            log(f"  G_train_start pass={train_start_ok['seen']} loss_chart={loss_chart}")
            # Stop the run so the GPU frees for other checks.
            try:
                await page.locator('[data-tour="studio-training-stop"]').first.click(timeout=4000)
                await page.get_by_role("button", name=re.compile(r"Stop", re.I)).last.click(timeout=4000)
            except Exception:
                pass
        except Exception as e:
            res["gates"]["G_train_start"] = {"pass": False, "error": repr(e)}
            log(f"  G_train_start ERROR {e!r}")

        # ---- G_project: create a chat project ----
        try:
            log("create a project on /projects")
            await page.goto(base + "/projects", wait_until="domcontentloaded", timeout=20000)
            await page.get_by_role("button", name=re.compile(r"New project", re.I)).first.click(timeout=8000)
            await page.get_by_label(re.compile(r"Project name", re.I)).first.fill("s3-smoke", timeout=6000)
            await page.get_by_role("button", name=re.compile(r"^Create$", re.I)).first.click(timeout=6000)
            await page.wait_for_timeout(1200)
            made = await page.get_by_text("s3-smoke", exact=False).count() > 0
            res["gates"]["G_project"] = {"pass": bool(made)}
            log(f"  G_project pass={bool(made)}")
            await sp.screenshot(out_dir / "shots" / "13_project.png")
        except Exception as e:
            res["gates"]["G_project"] = {"pass": False, "error": repr(e)}
            log(f"  G_project ERROR {e!r}")

    (out_dir / "captures.json").write_text(json.dumps(res, indent=2))
    hard = res["gates"].get("G_train_page", {}).get("pass") and res["gates"].get("G_project", {}).get("pass")
    log(f"VERDICT hard(page+project)={bool(hard)} start={res['gates'].get('G_train_start', {}).get('pass')}")
    return 0 if hard else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="out/s3")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.base, a.password, a.dataset, a.out)))


if __name__ == "__main__":
    main()
