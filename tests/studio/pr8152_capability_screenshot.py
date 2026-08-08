# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Screenshot the Train/Video capability gate on a real Apple Silicon runner.

Evidence collector, not a gate: it asserts nothing and always exits 0, so the artifacts
upload whatever the run turns out to show.

Nothing here is simulated. The runner is Apple Silicon, `install.sh --local --no-torch`
leaves no MLX stack behind, so `import mlx.core` genuinely fails, detection genuinely
settles on chat_only / "mlx_unavailable", and utils.mlx_repair genuinely shells out to
`uv pip install mlx mlx-lm mlx-vlm`. What the sidebar does over that window is the whole
question, so this samples the rows and /api/health once a second from first paint and
shoots the sidebar on every state change.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
import playwright_mac_tab_capabilities as mac  # noqa: E402
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_view_transition_killer,
)

ART = mac.ART
BASE = mac.BASE
VARIANT = os.environ.get("PR8152_VARIANT", "unknown")
SAMPLE_S = float(os.environ.get("PR8152_SAMPLE_S", "300"))
ROWS = ("train", "projects", "hub", "images")

MORE_ITEMS_JS = """() => {
    const items = Array.from(document.querySelectorAll('[role="menuitem"], [role="option"]'));
    return items.map(e => ({
        text: (e.innerText || '').trim(),
        disabled: e.hasAttribute('disabled') || e.getAttribute('aria-disabled') === 'true',
        spinner: e.getAttribute('data-spinner') === 'true'
            || !!e.querySelector('.animate-spin, [data-spinner="true"]'),
    })).filter(i => i.text);
}"""

NAV_BOX_JS = """() => {
    const rows = Array.from(document.querySelectorAll('[data-testid^="nav-row-"]'));
    const more = Array.from(document.querySelectorAll('button'))
        .find(b => (b.innerText || '').trim() === 'More');
    const els = more ? rows.concat([more]) : rows;
    if (!els.length) return null;
    const rs = els.map(e => e.getBoundingClientRect());
    return {
        x: Math.min(...rs.map(r => r.left)),
        y: Math.min(...rs.map(r => r.top)),
        width: Math.max(...rs.map(r => r.right)) - Math.min(...rs.map(r => r.left)),
        height: Math.max(...rs.map(r => r.bottom)) - Math.min(...rs.map(r => r.top)),
    };
}"""


def info(msg):
    print(f"[pr8152/{VARIANT}] {msg}", flush = True)


def health():
    try:
        with urllib.request.urlopen(f"{BASE}/api/health", timeout = 10) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return {}


def nav_clip(page, extra_width = 0.0):
    box = page.evaluate(NAV_BOX_JS)
    if not box:
        return None
    return {
        "x": max(0.0, box["x"] - 14),
        "y": max(0.0, box["y"] - 14),
        "width": min(1440.0 - max(0.0, box["x"] - 14), box["width"] + 28 + extra_width),
        "height": box["height"] + 28,
    }


def shoot(page, name):
    clip = nav_clip(page)
    try:
        if clip:
            page.screenshot(path = str(ART / f"{VARIANT}_{name}_nav.png"), clip = clip)
        page.screenshot(path = str(ART / f"{VARIANT}_{name}.png"), full_page = False)
    except Exception as exc:
        info(f"screenshot {name} failed: {exc!r}")


def main() -> int:
    ART.mkdir(parents = True, exist_ok = True)
    timeline = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args = chromium_launch_args(sys.platform))
        ctx = browser.new_context(viewport = {"width": 1440, "height": 900})
        install_view_transition_killer(ctx)
        page = ctx.new_page()
        if not mac.log_in(page):
            info("could not sign in; nothing to shoot")
            (ART / f"{VARIANT}_timeline.json").write_text(json.dumps({"error": "login"}))
            return 0

        page.goto(f"{BASE}/chat", wait_until = "domcontentloaded", timeout = 60000)
        try:
            page.wait_for_selector('[data-testid="nav-row-train"]', timeout = 60000)
        except Exception as exc:
            info(f"sidebar never rendered: {exc!r}")
            return 0

        began = time.monotonic()
        last = None
        marks = {5.0, 20.0, 60.0, 120.0, 240.0}
        shot_marks = set()
        while True:
            elapsed = time.monotonic() - began
            try:
                rows = mac.row_states(page, ROWS)
            except Exception as exc:
                info(f"row read failed: {exc!r}")
                break
            h = health()
            sample = {
                "t": round(elapsed, 1),
                "rows": rows,
                "chat_only": h.get("chat_only"),
                "hardware_detecting": h.get("hardware_detecting"),
                "torch_warm_in_progress": h.get("torch_warm_in_progress"),
            }
            timeline.append(sample)
            key = json.dumps([rows.get("train"), h.get("hardware_detecting")])
            if key != last:
                info(f"t={elapsed:6.1f}s train={rows.get('train')} "
                     f"detecting={h.get('hardware_detecting')} chat_only={h.get('chat_only')}")
                shoot(page, f"t{int(elapsed):04d}_change")
                last = key
            for m in sorted(marks):
                if m not in shot_marks and elapsed >= m:
                    shot_marks.add(m)
                    shoot(page, f"t{int(m):04d}")
            if elapsed >= SAMPLE_S:
                break
            time.sleep(1.0)

        # The hint the user reads on a greyed row, and the Video row in the More flyout.
        try:
            page.locator('[data-testid="nav-row-train"]').hover(timeout = 10000, force = True)
            page.wait_for_timeout(2500)
            clip = nav_clip(page, extra_width = 380.0)
            if clip:
                page.screenshot(path = str(ART / f"{VARIANT}_tooltip.png"), clip = clip)
            tips = page.evaluate(
                """() => Array.from(document.querySelectorAll('[role="tooltip"]'))
                            .map(e => (e.innerText || '').trim()).filter(Boolean)"""
            )
            info(f"tooltips: {tips}")
        except Exception as exc:
            info(f"tooltip capture failed: {exc!r}")
            tips = []

        more_items = []
        try:
            # Hover, not click: the flyout opens on pointer enter and a click toggles it shut.
            page.get_by_role("button", name = "More").first.hover(timeout = 10000)
            page.wait_for_timeout(2500)
            more_items = page.evaluate(MORE_ITEMS_JS)
            info(f"more items: {json.dumps(more_items)}")
            clip = nav_clip(page, extra_width = 380.0)
            if clip:
                page.screenshot(path = str(ART / f"{VARIANT}_more.png"), clip = clip)
        except Exception as exc:
            info(f"more-flyout capture failed: {exc!r}")

        final_rows = mac.row_states(page, ROWS)
        ctx.close()
        browser.close()

    (ART / f"{VARIANT}_timeline.json").write_text(
        json.dumps(
            {
                "variant": VARIANT,
                "final_rows": final_rows,
                "tooltips": tips,
                "more_items": more_items,
                "samples": timeline,
            },
            indent = 1,
        ),
        encoding = "utf-8",
    )
    info(f"final train row: {json.dumps(final_rows.get('train'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
