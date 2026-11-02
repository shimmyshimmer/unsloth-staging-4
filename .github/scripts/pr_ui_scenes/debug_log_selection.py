from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.async_api import async_playwright

from studio_test_kit.auth import seed_init_script


SELECT = """() => {
  const n = document.querySelector('[data-testid="debug-log-pane"]').firstChild;
  const r = document.createRange(); r.setStart(n, 0); r.setEnd(n, 24);
  const s = getSelection(); s.removeAllRanges(); s.addRange(r);
  return s.toString();
}"""


async def drive(
    session,
    out_dir: Path,
    label: str,
    engines = None,
    **kwargs,
):
    engines = engines or ["chromium", "firefox", "webkit"]
    auth = type(
        "Auth", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
    )()
    init = seed_init_script(
        auth,
        [],
        extra_local_storage = {
            "unsloth_debug_log_refresh_mode": "live",
        },
    )
    facts, shots = {}, []
    async with async_playwright() as pw:
        for engine in engines:
            browser = await getattr(pw, engine).launch(headless = True)
            context = await browser.new_context(viewport = {"width": 1440, "height": 1000})
            try:
                await context.add_init_script(init)
                await context.add_init_script("""window.__copied = null;
                  Object.defineProperty(navigator, 'clipboard', {value: {
                    writeText: async text => {window.__copied = text;}
                  }});""")
                page = await context.new_page()
                state = {"polls": 0, "advance": False}

                async def logs(route):
                    path = urlparse(route.request.url)
                    if path.path.endswith("/sources"):
                        await route.fulfill(
                            json = {
                                "sources": [
                                    {
                                        "id": source,
                                        "family": "server",
                                        "label": source + ".log",
                                        "realpath": "/logs/" + source + ".log",
                                        "is_current": source == "server",
                                    }
                                    for source in ("server", "previous")
                                ],
                                "default_source_id": "server",
                                "log_root": "/logs",
                            }
                        )
                        return
                    state["polls"] += 1
                    query = parse_qs(path.query)
                    source = query.get("source", ["server"])[0]
                    first = not query.get("cursor")
                    lines = (
                        [f"{source}: diagnostic line {i:02d}" for i in range(1, 9)] if first else []
                    )
                    if state["advance"]:
                        lines += [f"{source}: appended poll {state['polls']}"]
                    await route.fulfill(
                        json = {
                            "status": "ok",
                            "source_id": source,
                            "realpath": "/logs/" + source + ".log",
                            "lines": lines,
                            "cursor": str(state["polls"]),
                            "reset": first,
                        }
                    )

                await page.route("**/api/settings/debug/logs**", logs)
                await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
                await page.get_by_role("button", name = "Settings", exact = True).click(timeout = 60000)
                dialog = page.get_by_role("dialog")
                await dialog.get_by_role("button", name = re.compile(r"^Logs\b")).click()
                pane = page.get_by_test_id("debug-log-pane")
                await page.wait_for_function(
                    "() => document.querySelector('[data-testid=debug-log-pane]')?.textContent.includes('diagnostic line 01')"
                )
                initial = await pane.inner_text()
                selected = await page.evaluate(SELECT)
                await page.wait_for_function("() => getSelection().toString().length > 0")
                poll_start = state["polls"]
                state["advance"] = True
                await page.wait_for_timeout(2300)
                after = await page.evaluate("() => getSelection().toString()")
                held = await pane.inner_text()
                f = {
                    "selection_before": selected,
                    "selection_after_poll": after,
                    "poll_exercised": state["polls"] > poll_start,
                    "held_text_unchanged": initial == held,
                    "selection_retained": selected == after,
                }
                if not f["poll_exercised"]:
                    raise AssertionError("No poll reached the mounted component")
                if engine == "chromium":
                    shot = out_dir / f"{label.lower()}_selection.png"
                    await dialog.screenshot(path = str(shot))
                    shots.append(shot)
                await dialog.get_by_role("button", name = "Copy visible log", exact = True).click()
                f["copy_matches_display"] = await page.evaluate("() => window.__copied") == held
                await page.evaluate("() => getSelection().removeAllRanges()")
                await page.wait_for_function(
                    "() => document.querySelector('[data-testid=debug-log-pane]').textContent.includes('appended poll')"
                )
                f["clear_resumes"] = True
                state["advance"] = False
                await page.get_by_test_id("debug-log-mode-manual").click()
                await page.evaluate(SELECT)
                await dialog.get_by_role("button", name = "Refresh now", exact = True).click()
                await page.wait_for_timeout(200)
                f["refresh_clears_selection"] = await page.evaluate(
                    "() => getSelection().isCollapsed"
                )
                await page.evaluate(SELECT)
                await page.get_by_test_id("debug-log-source").click()
                await page.get_by_role("option", name = re.compile("previous.log")).click()
                await page.wait_for_function(
                    "() => document.querySelector('[data-testid=debug-log-pane]').textContent.includes('previous: diagnostic')"
                )
                f["source_switches"] = True
                f["source_clears_selection"] = await page.evaluate(
                    "() => getSelection().isCollapsed"
                )
                await page.get_by_test_id("debug-log-mode-live").click()
                state["advance"] = True
                await page.wait_for_timeout(1200)
                box = await pane.bounding_box()
                await page.mouse.click(box["x"] + 8, box["y"] + 8)
                click_text = await pane.inner_text()
                await page.wait_for_timeout(1200)
                f["plain_click_resumes"] = click_text != await pane.inner_text()
                state["advance"] = False
                await page.wait_for_timeout(200)
                await page.evaluate(
                    "() => {const p = document.querySelector('[data-testid=debug-log-pane]'); p.scrollTop = 0;}"
                )
                box = await pane.bounding_box()
                await page.mouse.move(box["x"] + 8, box["y"] + 12)
                await page.mouse.down()
                await page.mouse.move(box["x"] + 100, box["y"] + 12, steps = 8)
                drag_before = await page.evaluate("() => getSelection().toString()")
                if not drag_before:
                    raise AssertionError("Mouse drag did not select log text")
                state["advance"] = True
                await page.wait_for_timeout(1200)
                f["drag_retained"] = drag_before == await page.evaluate(
                    "() => getSelection().toString()"
                )
                await page.mouse.up()
                await page.evaluate("() => getSelection().removeAllRanges()")
                drag_text = await pane.inner_text()
                await page.wait_for_timeout(1200)
                f["drag_release_resumes"] = drag_text != await pane.inner_text()
                facts[engine] = f
            except Exception:
                if "page" in locals():
                    await page.screenshot(path = str(out_dir / f"{label.lower()}_{engine}_error.png"))
                raise
            finally:
                await context.close()
                await browser.close()
    (out_dir / f"{label.lower()}_selection_facts.json").write_text(
        json.dumps(facts, indent = 2), encoding = "utf-8"
    )
    return shots, facts
