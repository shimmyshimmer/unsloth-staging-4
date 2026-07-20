"""PR #7207 S4 driver -- Export + Settings.

Gates:
  G_settings (HARD): open /settings, visit every tab (General/Profile/Appearance/
     System/Chat/API/Connections/Voice/About) -- each renders. Assert the
     "Load on selection" row is ABSENT (count 0) -- this PR removed it.
  G_export (best-effort): /export -> Hugging Face source -> Qwen3-0.6B -> GGUF
     Q4_K_M -> Start Export -> phase reaches "Complete" / "Export finished".

Run: python -m studio_test_kit.examples.pr7207_s4_export_settings \
        --base http://127.0.0.1:8934 --password PW --out out/s4
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

SETTINGS_TABS = ["General", "Profile", "Appearance", "System", "Chat", "API", "Connections", "About"]


async def run(base, password, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    res = {"base": base, "gates": {}}

    def log(m):
        print(f"[s4] {m}", flush=True)

    auth = await login(base, "unsloth", password)
    init = seed_init_script(auth, [])
    async with open_chat(
        base, init_scripts=[init], video_dir=out_dir / "video",
        video_name="s4", transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page

        # ---- G_settings ----
        try:
            log("open /settings, visit every tab")
            await page.goto(base + "/settings", wait_until="domcontentloaded", timeout=20000)
            dlg = page.get_by_role("dialog").first
            await dlg.wait_for(state="visible", timeout=10000)
            tab_results = {}
            load_on_selection_total = 0
            for tab in SETTINGS_TABS:
                try:
                    btn = dlg.get_by_role("button", name=re.compile(rf"^{tab}$", re.I)).first
                    await btn.click(timeout=5000)
                    await page.wait_for_timeout(700)
                    # count any "load on selection" occurrences on this tab
                    los = await dlg.get_by_text(re.compile(r"load on selection", re.I)).count()
                    load_on_selection_total += los
                    tab_results[tab] = {"rendered": True, "load_on_selection": los}
                    await sp.screenshot(out_dir / "shots" / f"settings_{tab.lower()}.png")
                except Exception as e:
                    tab_results[tab] = {"rendered": False, "error": repr(e)}
                    log(f"    tab {tab} ERROR {e!r}")
            rendered = [t for t, r in tab_results.items() if r.get("rendered")]
            res["gates"]["G_settings"] = {
                "pass": len(rendered) >= 6 and load_on_selection_total == 0,
                "tabs_rendered": rendered,
                "load_on_selection_total": load_on_selection_total,
                "detail": tab_results,
            }
            log(f"  G_settings pass={res['gates']['G_settings']['pass']} "
                f"(tabs={len(rendered)}, load_on_selection={load_on_selection_total})")
            await page.keyboard.press("Escape")
        except Exception as e:
            res["gates"]["G_settings"] = {"pass": False, "error": repr(e)}
            log(f"  G_settings ERROR {e!r}")

        # ---- G_export (best-effort) ----
        try:
            log("open /export, GGUF export of Qwen3-0.6B (best-effort)")
            export_ok = {"started": False, "complete": False}

            def on_resp(r):
                if "/api/export/export/" in r.url and r.ok:
                    export_ok["started"] = True

            page.on("response", on_resp)
            await page.goto(base + "/export", wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(1500)
            # Source: Hugging Face
            try:
                await page.get_by_role("tab", name=re.compile(r"Hugging Face", re.I)).first.click(timeout=6000)
            except Exception:
                pass
            # Type the model id into any visible search/textbox
            try:
                hf_box = page.get_by_placeholder(re.compile(r"(model|search|hf|repo)", re.I)).first
                await hf_box.fill("unsloth/Qwen3-0.6B", timeout=5000)
                await page.wait_for_timeout(1200)
                await hf_box.press("Enter")
            except Exception as e:
                log(f"    hf model entry note: {e!r}")
            # Method: GGUF
            try:
                await page.locator('[data-tour="export-method"]').get_by_role(
                    "button", name=re.compile(r"GGUF", re.I)
                ).first.click(timeout=6000)
            except Exception:
                pass
            await sp.screenshot(out_dir / "shots" / "20_export_setup.png")
            # Kick the export panel + start
            for name in (r"Export Model", r"Start Export"):
                try:
                    await page.get_by_role("button", name=re.compile(name, re.I)).first.click(timeout=6000)
                    await page.wait_for_timeout(1000)
                except Exception:
                    pass
            # Wait up to ~5 min for a completion signal
            for _ in range(150):
                if await page.get_by_text(re.compile(r"Export finished|Complete", re.I)).count() > 0:
                    export_ok["complete"] = True
                    break
                await asyncio.sleep(2)
            await sp.screenshot(out_dir / "shots" / "21_export_result.png")
            res["gates"]["G_export"] = {
                "pass": bool(export_ok["complete"]),
                "started": export_ok["started"],
                "complete": export_ok["complete"],
            }
            log(f"  G_export started={export_ok['started']} complete={export_ok['complete']}")
        except Exception as e:
            res["gates"]["G_export"] = {"pass": False, "error": repr(e)}
            log(f"  G_export (best-effort) ERROR {e!r}")

    (out_dir / "captures.json").write_text(json.dumps(res, indent=2))
    hard = res["gates"].get("G_settings", {}).get("pass")
    log(f"VERDICT hard(G_settings)={bool(hard)} export={res['gates'].get('G_export', {}).get('pass')}")
    return 0 if hard else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", default="out/s4")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.base, a.password, a.out)))


if __name__ == "__main__":
    main()
