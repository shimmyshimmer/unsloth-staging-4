"""Staging-only: prove Unsloth Studio is reachable AND usable in a real browser.

Boots against an already-running Studio (BASE_URL). Steps:
  1. Chromium navigates to the Studio URL -> HTTP 200, page renders.
  2. Complete first-run account setup (/change-password, current pw pre-seeded)
     and wait for the app shell (the chat composer) to render -- i.e. you can
     actually get INTO Studio through the browser.
  3. Hit /api/health from inside the browser context (page fetch) -> healthy.
Screenshots are written to PW_ART_DIR for visual evidence.

Self-contained (no repo imports) so it survives on a throwaway staging branch.
"""

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = os.environ["BASE_URL"].rstrip("/")
NEWPW = os.environ.get("STUDIO_NEW_PW", "BrowserCheck-Aa1!7Xy")
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_browser"))
ART.mkdir(parents=True, exist_ok=True)

COMPOSER = 'textarea[aria-label="Message input"]'


def log(m):
    print(f"[browser] {m}", flush=True)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page_errors = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        # 1. Navigate to Studio -- proves the server is reachable and the SPA renders.
        resp = page.goto(BASE, wait_until="domcontentloaded", timeout=60_000)
        status = resp.status if resp else None
        log(f"GET {BASE} -> HTTP {status}")
        if not resp or not resp.ok:
            raise SystemExit(f"FAIL: navigating to Studio returned HTTP {status}")
        log(f"page title: {page.title()!r}")
        page.screenshot(path=str(ART / "10-landing.png"), full_page=True)

        # 2. First-run account setup, then wait for the app shell.
        # The form is React-rendered after hydration, so WAIT for #new-password
        # rather than checking count() immediately (which races the mount and
        # returns 0 -- the bug in the first attempt).
        try:
            page.goto(f"{BASE}/change-password", wait_until="domcontentloaded", timeout=60_000)
            has_form = False
            try:
                page.wait_for_selector("#new-password", state="visible", timeout=25_000)
                has_form = True
            except Exception:  # noqa: BLE001
                log("no #new-password after waiting -- account may already be set up")
            if has_form:
                page.fill("#new-password", NEWPW, timeout=30_000)
                page.fill("#confirm-password", NEWPW, timeout=30_000)
                page.screenshot(path=str(ART / "11-setup-filled.png"), full_page=True)
                with page.expect_response(
                    lambda r: "/api/auth/change-password" in r.url, timeout=60_000
                ):
                    page.locator('button[type="submit"]').click()
                log("submitted first-run account setup")
        except Exception as e:  # noqa: BLE001
            log(f"setup step note: {e}")

        logged_in = False
        for attempt in range(3):
            try:
                page.wait_for_selector(COMPOSER, timeout=45_000)
                logged_in = True
                break
            except Exception:  # noqa: BLE001
                log(f"app shell not visible yet (attempt {attempt + 1}); re-navigating")
                try:
                    page.goto(BASE, wait_until="domcontentloaded", timeout=60_000)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(2)

        if logged_in:
            page.screenshot(path=str(ART / "12-app-shell.png"), full_page=True)
            log("app shell rendered (chat composer present) -- Studio is usable in a browser")
        else:
            page.screenshot(path=str(ART / "12-login-or-setup.png"), full_page=True)
            log("WARN: reached Studio in the browser but did not get to the app shell")

        # 3. /api/health from inside the browser context.
        health = page.evaluate(
            "async () => { const r = await fetch('/api/health'); "
            "return { http: r.status, body: await r.json() }; }"
        )
        log(f"/api/health via browser fetch: {json.dumps(health)}")
        healthy = isinstance(health, dict) and health.get("body", {}).get("status") == "healthy"

        ctx.close()
        browser.close()

    if page_errors:
        log(f"page errors observed: {page_errors}")
    if not healthy:
        raise SystemExit("FAIL: /api/health was not healthy from the browser context")
    if not logged_in:
        raise SystemExit("FAIL: could not reach the Studio app shell in the browser")
    log("PASS: Studio runs and is accessible + usable via a real browser")


if __name__ == "__main__":
    sys.exit(main())
