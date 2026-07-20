"""PR #7207 S2 driver -- Hub + hidden infra models + GGUF load.

Regression gates:
  G_hidden (HARD): the RAG embedder ("bge-small-en-v1.5") and the llama.cpp
     validation probe ("stories260k") must NOT appear -- with the search box
     empty -- on /hub?tab=discover, /hub?tab=downloaded, nor in the chat picker
     (Recommended + On Device). This is the resurfacing regression the PR fixes.
  G_tabs: the hub Discover/On Device radios switch views.
  G_load (best-effort): the pre-cached gemma GGUF loads from the picker and the
     model streams a reply.

Run: python -m studio_test_kit.examples.pr7207_s2_hub \
        --base http://127.0.0.1:8932 --password PW --out out/s2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from ..auth import login, seed_init_script
from ..ui import open_chat, send_prompt, wait_for_stream
from .pr7207_s1_inference import _open_config

HIDDEN = ["bge-small-en-v1.5", "stories260k"]
MODEL_HINT = "gemma-3-270m"


async def _count_hidden(scope):
    out = {}
    for needle in HIDDEN:
        try:
            out[needle] = await scope.get_by_text(needle, exact=False).count()
        except Exception:
            out[needle] = -1
    return out


async def run(base, password, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    res = {"base": base, "gates": {}}

    def log(m):
        print(f"[s2] {m}", flush=True)

    auth = await login(base, "unsloth", password)
    init = seed_init_script(auth, [])
    async with open_chat(
        base, init_scripts=[init], video_dir=out_dir / "video",
        video_name="s2", transcode_mp4=True, viewport=(1440, 900), headless=True,
    ) as sp:
        page = sp.page
        hidden_report = {}

        # ---- G_hidden on hub discover + downloaded (empty query) ----
        try:
            for tab in ("discover", "downloaded"):
                await page.goto(base + f"/hub?tab={tab}", wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(2500)
                hidden_report[f"hub_{tab}"] = await _count_hidden(page)
                await sp.screenshot(out_dir / "shots" / f"hub_{tab}.png")
                log(f"  hub {tab}: {hidden_report[f'hub_{tab}']}")
        except Exception as e:
            hidden_report["hub_error"] = repr(e)
            log(f"  hub hidden-check ERROR {e!r}")

        # ---- G_tabs: Discover / On Device radios ----
        try:
            await page.goto(base + "/hub?tab=discover", wait_until="domcontentloaded", timeout=20000)
            await page.get_by_role("radio", name=re.compile(r"On Device", re.I)).first.click(timeout=8000)
            await page.wait_for_timeout(800)
            await page.get_by_role("radio", name=re.compile(r"Discover", re.I)).first.click(timeout=8000)
            res["gates"]["G_tabs"] = {"pass": True}
        except Exception as e:
            res["gates"]["G_tabs"] = {"pass": False, "error": repr(e)}
            log(f"  G_tabs ERROR {e!r}")

        # ---- G_hidden in the chat picker (Recommended + On Device) ----
        try:
            await page.goto(base + "/chat", wait_until="domcontentloaded", timeout=20000)
            await page.locator("form:has(textarea) textarea").first.wait_for(state="visible", timeout=15000)
            await page.locator('[data-tour="chat-model-selector"]').first.click(timeout=15000)
            panel = page.locator('[data-tour="chat-model-selector-popover"]').first
            await panel.wait_for(state="visible", timeout=10000)
            for tabname in ("Recommended", "On Device"):
                try:
                    await panel.get_by_role("tab", name=re.compile(tabname, re.I)).first.click(timeout=6000)
                    await page.wait_for_timeout(1200)
                    hidden_report[f"picker_{tabname}"] = await _count_hidden(panel)
                    log(f"  picker {tabname}: {hidden_report[f'picker_{tabname}']}")
                except Exception as e:
                    hidden_report[f"picker_{tabname}"] = f"err {e!r}"
            await sp.screenshot(out_dir / "shots" / "picker.png")
            await page.keyboard.press("Escape")
        except Exception as e:
            hidden_report["picker_error"] = repr(e)
            log(f"  picker hidden-check ERROR {e!r}")

        # Hard gate: every observed count must be 0 (no -1 errors, no >0 hits).
        all_counts = []
        for scope, rep in hidden_report.items():
            if isinstance(rep, dict):
                all_counts += [c for c in rep.values()]
        g_hidden_pass = len(all_counts) > 0 and all(c == 0 for c in all_counts)
        res["gates"]["G_hidden"] = {"pass": bool(g_hidden_pass), "report": hidden_report}
        log(f"  G_hidden pass={bool(g_hidden_pass)}")

        # ---- G_load (best-effort): load gemma GGUF + stream a reply ----
        try:
            log("G_load: open config for gemma GGUF -> Load model -> prompt")
            await page.goto(base + "/chat", wait_until="domcontentloaded", timeout=20000)
            await page.locator("form:has(textarea) textarea").first.wait_for(state="visible", timeout=15000)
            panel = await _open_config(page, log)
            await panel.get_by_role(
                "button", name=re.compile(r"^(Load model|Reload model)$")
            ).first.click(timeout=10000)
            # Wait for the load to settle (popover closes, composer usable again).
            for _ in range(60):
                if await panel.get_by_role("textbox", name="Context Length").count() == 0:
                    break
                await asyncio.sleep(1)
            await page.wait_for_timeout(2000)
            await sp.screenshot(out_dir / "shots" / "loaded_model.png")
            await send_prompt(sp, "Say hello in exactly one word.")
            await wait_for_stream(sp, timeout_ms=180000)
            # Grab the last assistant bubble text via a broad, frontend-agnostic scrape.
            body = await page.evaluate(
                """() => {
                    const nodes = Array.from(document.querySelectorAll(
                      '[data-message-role="assistant"],[data-role="assistant"],'
                      + '.prose, [class*="assistant"], [class*="message"]'));
                    const txt = nodes.map(n => (n.innerText||'').trim()).filter(Boolean);
                    return txt.length ? txt[txt.length-1] : '';
                }"""
            )
            res["gates"]["G_load"] = {"pass": bool(body and body.strip()), "reply_len": len(body or ""), "reply": (body or "")[:120]}
            log(f"  G_load pass={bool(body and body.strip())} reply_len={len(body or '')}")
            await sp.screenshot(out_dir / "shots" / "loaded_reply.png")
        except Exception as e:
            res["gates"]["G_load"] = {"pass": False, "error": repr(e)}
            log(f"  G_load (best-effort) ERROR {e!r}")

    (out_dir / "captures.json").write_text(json.dumps(res, indent=2))
    hard = res["gates"].get("G_hidden", {}).get("pass")
    log(f"VERDICT hard(G_hidden)={bool(hard)}")
    return 0 if hard else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", default="out/s2")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.base, a.password, a.out)))


if __name__ == "__main__":
    main()
