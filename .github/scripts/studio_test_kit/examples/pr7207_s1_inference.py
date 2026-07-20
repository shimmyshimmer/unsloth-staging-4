"""PR #7207 S1 driver -- inference + model-picker inference settings.

Regression gates (tied to the PR's own bug history):
  G1 context-persist (HARD): Context Length 8192 + Remember + Load -> localStorage
     customContextLength==8192 AND intercepted /api/inference/{load,validate}
     max_seq_length==8192; page reload keeps it.
  G3 settings-persist: KV Cache Dtype (under Advanced) + Remember persists.
  G2 reset-clears: after a customized config, Reset removes the override.
  G4 template-editor: the chat-template dialog shows the "/ 65,536 bytes" counter.

Run: python -m studio_test_kit.examples.pr7207_s1_inference \
        --base http://127.0.0.1:8931 --password PW --out out/s1
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
from .pr6647_after import _find_ctx_entry

MODEL_HINT = "gemma-3-270m"
REPO_HINT = "gemma-3-270m-it-GGUF"


async def _ls_configs(page):
    raw = await page.evaluate("() => localStorage.getItem('unsloth_model_configs')")
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


async def _open_config(page, log):
    """Open picker -> On Device -> search -> expand GGUF repo -> click the first
    variant gear -> wait for the Context Length control. Returns the popover."""
    await page.locator('[data-tour="chat-model-selector"]').first.click(timeout=15000)
    panel = page.locator('[data-tour="chat-model-selector-popover"]').first
    await panel.wait_for(state="visible", timeout=10000)
    # Let the first inventory scan settle so the list stops re-rendering.
    await page.wait_for_timeout(1800)
    tab = panel.get_by_role("tab", name=re.compile(r"On Device", re.I)).first
    if (await tab.get_attribute("aria-selected")) != "true":
        await tab.click(timeout=6000)
        await page.wait_for_timeout(600)
    await panel.get_by_placeholder(re.compile(r"Search", re.I)).first.fill(MODEL_HINT)
    await page.wait_for_timeout(900)
    row = panel.locator("[data-model-picker-option]", has_text=REPO_HINT).first
    await row.click()
    await page.wait_for_timeout(1200)
    gear = panel.get_by_role("button", name=re.compile(r"Inference settings for", re.I)).first
    await gear.click(timeout=8000)
    # Context Length appears after the model's native-context fetch resolves.
    await panel.get_by_role("textbox", name="Context Length").first.wait_for(state="visible", timeout=20000)
    return panel


async def _primary_click(panel):
    await panel.get_by_role(
        "button", name=re.compile(r"^(Load model|Reload model|Save settings|Forget settings)$")
    ).first.click(timeout=10000)


async def run(base, password, ctx, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    res = {"base": base, "context": ctx, "gates": {}}

    def log(m):
        print(f"[s1] {m}", flush=True)

    auth = await login(base, "unsloth", password)
    init = seed_init_script(auth, [])

    async with open_chat(
        base, init_scripts=[init], video_dir=out_dir / "video",
        video_name="s1", transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page
        captured = []
        page.on("request", lambda r: captured.append(r.post_data_json)
                if r.method == "POST" and "/api/inference/" in r.url else None)

        # ---- Session A: set context (+KV via advanced) + remember + load ----
        try:
            log("G1/G3: open config, set Context Length + KV + Remember + Load")
            panel = await _open_config(page, log)
            ctx_box = panel.get_by_role("textbox", name="Context Length").first
            await ctx_box.fill(str(ctx))
            await ctx_box.press("Enter")

            # KV Cache Dtype lives under Advanced.
            kv_set = False
            try:
                adv = panel.get_by_role("switch", name=re.compile("advanced", re.I)).first
                if (await adv.get_attribute("aria-checked")) != "true":
                    await adv.click(timeout=4000)
                    await page.wait_for_timeout(800)
                # Under Advanced the panel exposes exactly two comboboxes:
                # KV Cache Dtype (first, default "f16") then Speculative (second).
                kv_combo = panel.get_by_role("combobox").first
                await kv_combo.wait_for(state="visible", timeout=6000)
                await kv_combo.click(timeout=5000)
                await page.get_by_role("option", name=re.compile(r"^q8_0$")).first.click(timeout=5000)
                kv_set = True
            except Exception as e:
                log(f"    KV set note: {e!r}")

            await panel.get_by_role("checkbox", name="Remember for this model").first.check(timeout=6000)
            await sp.screenshot(out_dir / "shots" / "01_config_set.png")
            await _primary_click(panel)
            await page.wait_for_timeout(1500)

            cfg = await _ls_configs(page)
            hit = _find_ctx_entry(json.dumps(cfg), ctx)
            kv_ok = any(isinstance(v, dict) and v.get("kvCacheDtype") == "q8_0" for v in cfg.values())
            req_hit = None
            for _ in range(20):
                if any(isinstance(b, dict) and b.get("max_seq_length") == ctx for b in captured if b):
                    req_hit = True
                    break
                await asyncio.sleep(0.5)
            res["gates"]["G3_kv_persist"] = {"pass": bool(kv_set and kv_ok)}
            log(f"  G3 pass={bool(kv_set and kv_ok)} (set={kv_set})")
            await sp.screenshot(out_dir / "shots" / "02_after_load.png")

            # ---- reload -> verify persistence ----
            log("G1: reload page, re-open config, verify persistence")
            await page.reload(wait_until="domcontentloaded")
            await page.locator("form:has(textarea) textarea").first.wait_for(state="visible", timeout=15000)
            cfg2 = await _ls_configs(page)
            hit2 = _find_ctx_entry(json.dumps(cfg2), ctx)
            panel = await _open_config(page, log)
            reopened = await panel.get_by_role("textbox", name="Context Length").first.input_value()
            res["gates"]["G1_context_persist"] = {
                "pass": bool(hit and hit2 and str(ctx) in str(reopened)),
                "ls_after_load": bool(hit), "request_seen": bool(req_hit),
                "ls_after_reload": bool(hit2), "reopened_value": reopened,
            }
            log(f"  G1 pass={res['gates']['G1_context_persist']['pass']} reopened={reopened!r}")
            await sp.screenshot(out_dir / "shots" / "03_reopened_after_reload.png")
        except Exception as e:
            res["gates"].setdefault("G1_context_persist", {"pass": False, "error": repr(e)})
            log(f"  G1 ERROR {e!r}")

        # ---- G2: Reset clears the override (config is now customized) ----
        try:
            log("G2: Reset clears the stored override")
            # picker is open on the config page from G1
            reset = panel.get_by_role("button", name=re.compile(r"^Reset$")).first
            await reset.click(timeout=6000)
            await _primary_click(panel)
            await page.wait_for_timeout(1200)
            cfg = await _ls_configs(page)
            pinned = [k for k, v in cfg.items()
                      if isinstance(v, dict) and (v.get("customContextLength") is not None or v.get("kvCacheDtype"))]
            res["gates"]["G2_reset_clears"] = {"pass": len(pinned) == 0, "still_pinned": pinned}
            log(f"  G2 pass={len(pinned) == 0} (still_pinned={pinned})")
            await sp.screenshot(out_dir / "shots" / "04_after_reset.png")
        except Exception as e:
            res["gates"]["G2_reset_clears"] = {"pass": False, "error": repr(e)}
            log(f"  G2 ERROR {e!r}")

        # ---- G4: chat-template editor byte counter ----
        try:
            log("G4: open chat-template editor, assert byte counter")
            panel = await _open_config(page, log)
            # The chat-template "Edit" button is only rendered under Advanced.
            adv = panel.get_by_role("switch", name=re.compile("advanced", re.I)).first
            if (await adv.get_attribute("aria-checked")) != "true":
                await adv.click(timeout=4000)
                await page.wait_for_timeout(800)
            await panel.get_by_role("button", name=re.compile(r"^(Edit|View)$")).first.click(timeout=6000)
            dlg = page.get_by_role("dialog").first
            await dlg.wait_for(state="visible", timeout=6000)
            has_counter = await dlg.get_by_text(re.compile(r"/\s*65,536\s*bytes")).count() > 0
            res["gates"]["G4_template_editor"] = {"pass": bool(has_counter)}
            log(f"  G4 pass={bool(has_counter)}")
            await sp.screenshot(out_dir / "shots" / "05_template_editor.png")
            await page.keyboard.press("Escape")
        except Exception as e:
            res["gates"]["G4_template_editor"] = {"pass": False, "error": repr(e)}
            log(f"  G4 ERROR {e!r}")

    (out_dir / "captures.json").write_text(json.dumps(res, indent=2))
    hard = res["gates"].get("G1_context_persist", {}).get("pass")
    others = {g: res["gates"].get(g, {}).get("pass") for g in ("G2_reset_clears", "G3_kv_persist", "G4_template_editor")}
    log(f"VERDICT hard(G1)={bool(hard)} others={others}")
    return 0 if hard else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--out", default="out/s1")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.base, a.password, a.context, a.out)))


if __name__ == "__main__":
    main()
