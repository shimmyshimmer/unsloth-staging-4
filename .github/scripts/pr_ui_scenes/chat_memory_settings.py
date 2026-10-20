"""Scene: Settings > Chat memory editor (PR 14357).

Both sides get the same global AGENTS.md written into their Studio home before the
browser opens, so the only variable is the build. The Chat tab is photographed from the
Conversations heading to the Display heading: BEFORE has nothing between them (the notes
reach every chat but are invisible in the app), AFTER has the Memory section showing them.
Facts carry the API reading (404 before, the saved text after) and whether the section
rendered, so the pair is not judged from pixels alone.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

NOTES = "- I write Python and use uv.\n- Keep answers short and skip the preamble.\n- I live in Sydney (AEST).\n"


def _get(session: Session, path: str):
    req = urllib.request.Request(
        session.base_url + path, headers = {"Authorization": f"Bearer {session.access_token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout = 60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, None


async def _shoot(page, session, out_dir, label, facts, shots):
    await page.goto(f"{session.base_url}/chat", wait_until = "domcontentloaded")
    tab = page.locator('[data-testid="settings-tab-chat"]').first
    try:
        # /settings opens the dialog before the app shell mounts and loses it on this build;
        # the Mod+Comma shortcut opens it from the mounted page.
        await page.get_by_text("New chat", exact = True).first.wait_for(
            state = "visible", timeout = 60_000
        )
        for attempt in range(4):
            if attempt < 2:
                await page.keyboard.press("Control+Comma")
            else:
                # WebKit on Linux does not deliver the shortcut: use the sidebar footer's gear instead.
                await page.get_by_role("button", name = "Settings", exact = True).last.click(
                    timeout = 10_000
                )
            try:
                await tab.wait_for(state = "visible", timeout = 5_000)
                break
            except Exception:
                continue
        await tab.click(timeout = 30_000)
    except Exception:
        await page.screenshot(path = str(out_dir / f"{label.lower()}_debug_no_settings.png"))
        facts["url_at_failure"] = page.url
        (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 1))
        raise
    conversations = page.locator('section[data-settings-label="Conversations"]').first
    display = page.locator('section[data-settings-label="Display"]').first
    await display.wait_for(state = "visible", timeout = 30_000)
    memory = page.locator('section[data-settings-label="Memory"]')
    facts["memory_section_rendered"] = await memory.count() > 0
    if facts["memory_section_rendered"]:
        # The editor fills after its own fetch; wait for the seeded text, not a fixed sleep.
        await page.wait_for_function(
            "el => (el.querySelector('textarea')?.value || '').includes('uv')",
            arg = await memory.first.element_handle(),
            timeout = 30_000,
        )
        facts["memory_section_text"] = " | ".join(
            t.strip() for t in (await memory.first.inner_text()).splitlines() if t.strip()
        )[:600]
        facts["memory_editor_value"] = await memory.first.locator("textarea").input_value()
        facts["memory_switch_checked"] = await memory.first.get_by_role("switch").get_attribute(
            "aria-checked"
        )
    # Same fixed-height crop on both sides, from the last Conversations row down, so neither
    # half is rescaled: BEFORE shows Display there, AFTER shows the Memory section.
    anchor = page.locator('[data-settings-label="Auto-title new chats"]').first
    await anchor.evaluate("el => el.scrollIntoView({block: 'start'})")
    await page.wait_for_timeout(300)
    top = await anchor.bounding_box()
    box = await conversations.bounding_box()
    clip = {"x": box["x"] - 16, "y": top["y"] - 12, "width": box["width"] + 32, "height": 600}
    shot = out_dir / f"{label.lower()}_chat_memory_{facts['browser']}.png"
    await page.screenshot(path = str(shot), clip = clip)
    shots.append(shot)
    if facts["memory_section_rendered"]:
        # Drive it, not just render it: edit + Save, then the switch, each read back from the API.
        mem = page.locator('section[data-settings-label="Memory"]').first
        await mem.locator("textarea").fill(NOTES + "- Added in " + facts["browser"] + ".\n")
        await mem.get_by_role("button", name = "Save").click()
        # The saved file is the fact: poll the API for it rather than a button state.
        for _ in range(30):
            saved = _get(session, "/api/settings/chat-memory")[1]["text"]
            if "Added in" in saved:
                break
            await page.wait_for_timeout(500)
        facts["after_save_api_text"] = saved
        await mem.get_by_role("switch").click()
        await page.wait_for_timeout(1500)
        facts["after_switch_off_api_enabled"] = _get(session, "/api/settings/chat-memory")[1][
            "enabled"
        ]
        facts["after_switch_off_agents_md_sent"] = bool(
            (_get(session, "/api/chat/agents-md")[1] or {}).get("text", "").strip()
        )
        await mem.get_by_role("switch").click()
        await page.wait_for_timeout(1500)
        facts["after_switch_on_api_enabled"] = _get(session, "/api/settings/chat-memory")[1][
            "enabled"
        ]
        (Path(session.home) / "AGENTS.md").write_text(NOTES, encoding = "utf-8")


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    browser: str = "chromium",
    **_kwargs,
):
    (Path(session.home) / "AGENTS.md").write_text(NOTES, encoding = "utf-8")
    facts: dict = {}
    code, body = _get(session, "/api/settings/chat-memory")
    facts["chat_memory_api_status"] = code
    facts["chat_memory_api_text"] = (body or {}).get("text")
    facts["chat_memory_api_enabled"] = (body or {}).get("enabled")
    _, agents = _get(session, "/api/chat/agents-md")
    facts["agents_md_sent_to_chats"] = bool((agents or {}).get("text", "").strip())

    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    shots: list[Path] = []
    facts["browser"] = browser
    if browser == "chromium":
        async with open_chat(
            session.base_url, init_scripts = [init], viewport = (1400, 1600), headless = True
        ) as sp:
            await _shoot(sp.page, session, out_dir, label, facts, shots)
    else:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            engine = await getattr(pw, browser).launch(headless = True)
            try:
                ctx = await engine.new_context(viewport = {"width": 1400, "height": 1600})
                await ctx.add_init_script(init)
                await _shoot(await ctx.new_page(), session, out_dir, label, facts, shots)
            finally:
                await engine.close()
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 1))
    return shots, facts
