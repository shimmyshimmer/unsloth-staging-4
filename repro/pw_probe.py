"""Cross-check driver: load repro/probe.html in Playwright engines.

Used alongside the native drivers (WebKitGTK / WKWebView / WebView2) so every
OS reports at least two engines.

  python repro/pw_probe.py chromium webkit
"""

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
URL = (HERE / "probe.html").as_uri()


def run(spec):
    engine, _, channel = spec.partition(":")
    out = {"driver": "playwright", "engine": engine, "channel": channel or None}
    try:
        with sync_playwright() as p:
            kwargs = {"channel": channel} if channel else {}
            browser = getattr(p, engine).launch(**kwargs)
            page = browser.new_context(viewport={"width": 900, "height": 700}).new_page()
            page.goto(URL)
            page.wait_for_function("window.__RESULT !== undefined", timeout=30000)
            out.update(page.evaluate("window.__RESULT"))
            browser.close()
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    return out


if __name__ == "__main__":
    specs = sys.argv[1:] or ["chromium"]
    results = [run(s) for s in specs]
    print(json.dumps(results, indent=2))
    dest = os.environ.get("REPRO_OUT")
    if dest:
        Path(dest).write_text(json.dumps(results, indent=2))
