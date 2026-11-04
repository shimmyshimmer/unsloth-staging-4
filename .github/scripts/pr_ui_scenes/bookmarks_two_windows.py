"""Scene: turn bookmarks made in two windows of one Studio (PR 16578).

One seeded three-turn chat is open in two pages of the same browser context (they
share localStorage, so `storage` events cross between them like two real windows).
Window B bookmarks turn 1, then window A bookmarks turn 3. Facts: which turns each
window shows as bookmarked, the persisted list, and what A shows after a reload.
The shot is window A after the reload, with a harness-drawn caption of the stored
list (Studio never paints it).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

THREAD_ID = "uidiff-bookmarks-" + uuid.uuid4().hex[:8]
CREATED_AT_MS = 1_767_225_600_000
VIEWPORT = (1440, 900)
STORE_KEY = "unsloth_bookmarked_turns"
TURNS = [
    ("What is a stag beetle?", "A beetle with large mandibles."),
    ("Where do they live?", "Mostly in old deciduous woodland."),
    ("What do larvae eat?", "Decaying wood, for several years."),
]


def _put(session: Session, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        session.base_url + path,
        data = json.dumps(payload).encode(),
        method = "PUT",
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session.access_token}",
        },
    )
    with urllib.request.urlopen(req, timeout = 60) as resp:
        return json.loads(resp.read() or b"{}")


def _seed(session: Session) -> list[str]:
    api_post(
        session,
        "/api/chat/threads",
        {
            "id": THREAD_ID,
            "title": "Stag beetles",
            "modelType": "base",
            "modelId": "unsloth/Qwen3-0.6B",
            "createdAt": CREATED_AT_MS,
            "updatedAt": CREATED_AT_MS,
        },
    )
    parent, n, user_ids = None, 0, []
    for i, (q, a) in enumerate(TURNS):
        for role, text in (("user", q), ("assistant", a)):
            n += 1
            mid = f"{THREAD_ID}-{role[0]}{i + 1}"
            _put(
                session,
                f"/api/chat/threads/{THREAD_ID}/messages/{mid}",
                {
                    "id": mid,
                    "threadId": THREAD_ID,
                    "parentId": parent,
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                    "createdAt": CREATED_AT_MS + n,
                },
            )
            parent = mid
            if role == "user":
                user_ids.append(mid)
    api_get(session, f"/api/chat/threads/{THREAD_ID}")
    return user_ids


async def _open(page, base_url: str) -> None:
    await page.goto(f"{base_url}/chat?thread={THREAD_ID}", wait_until = "domcontentloaded")
    await page.get_by_text(TURNS[-1][1]).first.wait_for(state = "visible", timeout = 60_000)


async def _pressed(page) -> list[bool]:
    """aria-pressed of each user turn's bookmark button, in turn order."""
    out = []
    for q, _ in TURNS:
        msg = page.get_by_text(q, exact = True).first.locator(
            "xpath=ancestor::*[.//button[@aria-pressed]][1]"
        )
        button = msg.locator("button[aria-pressed]").first
        out.append((await button.get_attribute("aria-pressed")) == "true")
    return out


async def _toggle(page, turn: int) -> None:
    q = TURNS[turn - 1][0]
    text = page.get_by_text(q, exact = True).first
    await text.hover()
    msg = text.locator("xpath=ancestor::*[.//button[@aria-pressed]][1]")
    await msg.locator("button[aria-pressed]").first.click(force = True)
    await page.wait_for_timeout(800)


async def _stored(page) -> dict:
    raw = await page.evaluate(f"() => localStorage.getItem('{STORE_KEY}')")
    return (json.loads(raw) if raw else {}).get("state", {}).get("bookmarkedByThread", {})


async def drive(session: Session, out_dir: Path, label: str, **_: object):
    user_ids = _seed(session)
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    facts: dict = {"_thread_id": THREAD_ID}
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = VIEWPORT, headless = True
    ) as sp:
        a = sp.page
        await _open(a, session.base_url)
        b = await a.context.new_page()
        await b.set_viewport_size({"width": VIEWPORT[0], "height": VIEWPORT[1]})
        await _open(b, session.base_url)

        await _toggle(b, 1)
        facts["b_after_b_bookmarks_turn1"] = await _pressed(b)
        facts["a_after_b_bookmarks_turn1"] = await _pressed(a)
        await _toggle(a, 3)
        facts["a_after_a_bookmarks_turn3"] = await _pressed(a)
        stored = (await _stored(a)).get(THREAD_ID, [])
        facts["stored_turns"] = [user_ids.index(m) + 1 for m in stored if m in user_ids]

        await b.close()
        await _open(a, session.base_url)
        facts["a_after_reload"] = await _pressed(a)
        await a.evaluate(
            """(text) => {
              const el = document.createElement('div');
              el.textContent = text;
              Object.assign(el.style, {position: 'fixed', left: '50%', top: '16px',
                transform: 'translateX(-50%)', zIndex: 2147483647, padding: '10px 16px',
                background: '#111', color: '#fff', font: '600 16px system-ui',
                borderRadius: '8px', boxShadow: '0 4px 16px rgba(0,0,0,.4)'});
              document.body.appendChild(el);
            }""",
            "Window B bookmarked turn 1, then window A bookmarked turn 3. "
            f"Saved bookmarks: turns {facts['stored_turns']}",
        )
        await a.wait_for_timeout(400)
        shot = out_dir / f"{label.lower()}_bookmarks_two_windows.png"
        await a.screenshot(path = str(shot), full_page = False)
    if facts["b_after_b_bookmarks_turn1"][0] is not True:
        raise RuntimeError(f"{label}: window B's click never bookmarked turn 1: {facts}")
    return [shot], facts
