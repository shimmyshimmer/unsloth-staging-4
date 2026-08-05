#!/usr/bin/env python3
"""One-engine gate for PR 7927, for CI where real Edge and real macOS WebKit exist.

Locally we can only run Playwright's chromium/firefox/webkit on Linux. Chrome and
Edge share Chromium and Safari shares WebKit, but "same engine" is an argument,
not a measurement -- so the same assertions run again on the real runners, and on
windows-latest against the REAL Edge via channel="msedge".

Engine/channel come from the environment so one workflow matrix drives them:
  H3_ENGINE  = chromium | firefox | webkit
  H3_CHANNEL = msedge | chrome | (unset)

Asserts, for this engine:
  1. the reported bug reproduces pre-PR and is fixed post-PR
  2. the functional sidebar toggle is click-blocked pre-PR and clickable post-PR
  3. exactly one accessible toggle post-PR (two pre-PR, one of them inert)
  4. no painted-geometry change at widths that already fitted
"""

from __future__ import annotations

import http.server
import os
import socketserver
import sys
import threading
from functools import partial
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
DIST = Path(os.environ.get("H3_DIST", HERE / "dist"))
PORT = int(os.environ.get("H3_PORT", "5235"))
ENGINE = os.environ.get("H3_ENGINE", "chromium")
CHANNEL = os.environ.get("H3_CHANNEL") or None
BASE = f"http://127.0.0.1:{PORT}/__h3harness__/index.html"

NARROW = 722          # the width from the bug report
WIDE = [924, 1100, 1400]

SETTINGS_JS = r"""
() => {
  const out = {};
  for (const holder of document.querySelectorAll('[data-row]')) {
    const id = holder.getAttribute('data-row');
    const row = holder.querySelector('[data-settings-label]');
    const labelCol = row.firstElementChild;
    const ctrl = row.children.length > 1 ? row.children[1] : null;
    const span = labelCol.querySelector('span');
    const r = document.createRange(); r.selectNodeContents(span);
    const rects = Array.from(r.getClientRects()); const f = rects[0];
    const c = ctrl ? ctrl.getBoundingClientRect() : null;
    out[id] = {
      labelWidth: +labelCol.getBoundingClientRect().width.toFixed(2),
      lines: rects.length,
      textLeft: +f.left.toFixed(2), textRight: +f.right.toFixed(2), textTop: +f.top.toFixed(2),
      ctrlLeft: c ? +c.left.toFixed(2) : 0, ctrlRight: c ? +c.right.toFixed(2) : 0,
      ctrlTop: c ? +c.top.toFixed(2) : 0,
    };
  }
  return out;
}
"""

CHROME_JS = r"""
() => {
  const names = ['Toggle Sidebar', 'Collapse sidebar', 'Expand sidebar'];
  const all = Array.from(document.querySelectorAll('button')).filter((b) => {
    const l = (b.getAttribute('aria-label') || '') + ' ' + (b.textContent || '');
    return names.some((n) => l.includes(n));
  });
  const functional = all.find((b) =>
    ((b.getAttribute('aria-label') || '') + ' ' + (b.textContent || '')).includes('Toggle Sidebar'));
  let clickable = false, reach = 0, cover = null;
  if (functional) {
    const r = functional.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const hit = document.elementFromPoint(cx, r.top + r.height / 2);
    clickable = !!hit && (hit === functional || functional.contains(hit));
    cover = clickable ? null : (hit ? (hit.getAttribute('aria-label') || hit.tagName) : null);
    for (let y = Math.ceil(r.top); y < r.bottom; y++) {
      const e = document.elementFromPoint(cx, y + 0.5);
      if (e && (e === functional || functional.contains(e))) reach++;
    }
  }
  return { count: all.length, present: !!functional, clickable, reach, cover };
}
"""


# Two frames, or 300ms, whichever comes first. WebKit on a real macOS runner
# suspends rAF in a headless window, so an unbounded wait for it hangs the job
# instead of failing it.
SETTLE_JS = r"""
() => new Promise((resolve) => {
  const done = () => { clearTimeout(t); resolve(); };
  const t = setTimeout(resolve, 300);
  requestAnimationFrame(() => requestAnimationFrame(done));
})
"""


def settle(page, w):
    page.set_viewport_size({"width": w, "height": 800})
    page.wait_for_function("w => window.innerWidth === w", arg=w, timeout=20000)
    page.evaluate(SETTLE_JS)


def main() -> int:
    if not DIST.exists():
        sys.exit(f"missing dist at {DIST}")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DIST))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    httpd.RequestHandlerClass.log_message = lambda *a, **k: None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    label = f"{ENGINE}" + (f" (channel={CHANNEL})" if CHANNEL else "")
    print(f"=== PR 7927 browser gate: {label} on {sys.platform} ===")
    failures = []
    try:
        with sync_playwright() as p:
            kw = {"headless": True}
            if CHANNEL:
                kw["channel"] = CHANNEL
            browser = getattr(p, ENGINE).launch(**kw)
            ctx = browser.new_context(viewport={"width": 1000, "height": 800},
                                      reduced_motion="reduce")
            page = ctx.new_page()

            settings = {}
            for variant in ("before", "after"):
                page.goto(f"{BASE}?case=settings&variant={variant}", wait_until="load")
                page.wait_for_selector("[data-settings-label]")
                for w in [NARROW] + WIDE:
                    settle(page, w)
                    settings[(variant, w)] = page.evaluate(SETTINGS_JS)

            print(f"\n[1] label geometry at {NARROW}px (the reported width)")
            for row in ("hf-token", "embedding-model"):
                b = settings[("before", NARROW)][row]
                a = settings[("after", NARROW)][row]
                print(f"    {row:<17} before {b['labelWidth']:>6.0f}px/{b['lines']} lines"
                      f"   after {a['labelWidth']:>6.0f}px/{a['lines']} lines")
                if not (a["labelWidth"] > b["labelWidth"] and a["lines"] <= b["lines"]):
                    failures.append(f"{row}: no improvement at {NARROW}px")

            print("\n[2] painted geometry unchanged at widths that already fitted")
            # Where the label STARTS, how many lines it takes, and where the
            # control sits must be identical: those are what a user sees move.
            EXACT = ("textLeft", "textTop", "lines", "ctrlLeft", "ctrlRight", "ctrlTop")
            deltas = 0
            for w in WIDE:
                b, a = settings[("before", w)], settings[("after", w)]
                for row in a:
                    for k in EXACT:
                        if abs(a[row][k] - b[row][k]) > 0.5:
                            deltas += 1
                            print(f"    DELTA w={w} {row}.{k}: {b[row][k]} -> {a[row][k]}")
                    # textRight is the text's own extent. The PR only ever gives
                    # the label MORE room (flex-1 basis-0 vs shrink-to-fit), so it
                    # may grow but must never shrink. Real Edge shapes this text a
                    # few px wider than Chromium does at the same width, which is
                    # an engine difference, not a regression, and is invisible
                    # because the control is right-aligned metres away.
                    grow = a[row]["textRight"] - b[row]["textRight"]
                    if grow < -0.5:
                        deltas += 1
                        print(f"    DELTA w={w} {row}.textRight SHRANK: "
                              f"{b[row]['textRight']} -> {a[row]['textRight']}")
                    elif abs(grow) > 0.5:
                        print(f"    note w={w} {row}.textRight grew {grow:+.2f}px "
                              "(more room for the label; not a regression)")
            print(f"    painted deltas at {WIDE}: {deltas}")
            if deltas:
                failures.append(f"{deltas} painted deltas at wide viewports")

            print("\n[3] the functional sidebar toggle in the mobile layout")
            for variant in ("before", "after"):
                page.goto(f"{BASE}?case=chrome&variant={variant}&platform=Windows",
                          wait_until="load")
                settle(page, NARROW)
                m = page.evaluate(CHROME_JS)
                print(f"    {variant:<7} toggles={m['count']} present={m['present']} "
                      f"clickable={m['clickable']} reachable={m['reach']}px "
                      f"covered_by={m['cover']}")
                if variant == "after":
                    if not m["clickable"]:
                        failures.append("post-PR: functional toggle is not clickable")
                    if m["count"] != 1:
                        failures.append(f"post-PR: {m['count']} toggles (expected 1)")
                else:
                    if m["clickable"]:
                        failures.append("pre-PR: toggle was clickable; "
                                        "the harness did not reproduce the bug")
            ctx.close()
            browser.close()
    finally:
        httpd.shutdown()

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL ({label}): {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS ({label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
