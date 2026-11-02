from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import monotonic
from urllib.parse import urlparse

from playwright.async_api import async_playwright

from studio_test_kit.auth import StudioAuth, seed_init_script


async def drive(session, out_dir: Path, label: str, **kwargs):
    out_dir.mkdir(parents = True, exist_ok = True)
    records = []
    health_done = asyncio.Event()
    credential_paths = {"/api/settings/hugging-face-token", "/api/providers/"}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        context = await browser.new_context(viewport = {"width": 1440, "height": 900})
        await context.add_init_script(
            seed_init_script(
                StudioAuth(session.access_token, session.refresh_token, session.base_url), []
            )
        )
        page = await context.new_page()
        started = monotonic()

        async def delayed(route):
            path = urlparse(route.request.url).path
            records.append(
                {"path": path, "event": "start", "ms": round((monotonic() - started) * 1000)}
            )
            response = await route.fetch()
            await asyncio.sleep(2 if path == "/api/health" else 1.5)
            await route.fulfill(response = response)
            records.append(
                {"path": path, "event": "end", "ms": round((monotonic() - started) * 1000)}
            )
            if path == "/api/health":
                health_done.set()

        await page.route("**/api/health", delayed)
        await page.route("**/api/settings/hugging-face-token", delayed)
        await page.route("**/api/providers/", delayed)
        await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
        await asyncio.wait_for(health_done.wait(), 60)
        await page.wait_for_timeout(500)
        early = out_dir / f"{label.lower()}_startup.png"
        await page.screenshot(path = str(early))
        early_ready = await page.locator("textarea").count() > 0
        await page.locator("textarea").first.wait_for(state = "visible", timeout = 60000)
        await page.wait_for_timeout(2000)
        final = out_dir / f"{label.lower()}_settled.png"
        await page.screenshot(path = str(final))
        health_end = next(
            r["ms"] for r in records if r["path"] == "/api/health" and r["event"] == "end"
        )
        starts = [r for r in records if r["path"] in credential_paths and r["event"] == "start"]
        assert starts, records
        facts = {
            "credential_requests": len(starts),
            "credentials_before_health": sum(r["ms"] < health_end for r in starts),
            "composer_ready_after_health_500ms": early_ready,
            "settled_composer": True,
        }
        (out_dir / "request_trace.json").write_text(json.dumps(records, indent = 2))
        await context.close()
        await browser.close()
    return [early, final], facts
