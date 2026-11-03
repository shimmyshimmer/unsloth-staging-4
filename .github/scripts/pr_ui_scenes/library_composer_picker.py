from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))

import httpx

from pr_ui_scenes._common import Session, api_get, api_post, studio_session
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat

EXPECT = (
    "BEFORE: the chat attachment menu has no Library picker and the saved note cannot be "
    "attached from it. AFTER: the menu opens a searchable Library picker and selecting "
    "quarterly-report-16485.txt attaches its original bytes to the existing conversation "
    "while preserving the draft and thread."
)
NOTE_NAME = "quarterly-report-16485.txt"
NOTE_TEXT = "Quarterly report: revenue grew by 12 percent. Saved Library fixture 16485.\n"
THREAD_ID = "library-composer-evidence-16485"
PROJECT_ID = "library-composer-project-16485"
PROJECT_NAME = "Library attachment project"
DRAFT = "Please summarize this saved report without leaving this conversation."
PROJECT_EXPECT = (
    "The project landing composer attaches quarterly-report-16485.txt from Library "
    "without creating a conversation, leaving the project, or changing its draft."
)


def seed(session: Session, project: bool = False) -> dict:
    headers = {"Authorization": f"Bearer {session.access_token}"}
    snapshot = api_get(session, "/api/library")
    items = [item for item in snapshot["items"] if item["name"] == NOTE_NAME]
    if not items:
        with httpx.Client(timeout = 30) as client:
            response = client.post(
                f"{session.base_url}/api/library/uploads",
                headers = headers,
                files = {"files": (NOTE_NAME, NOTE_TEXT.encode(), "text/plain")},
            )
            response.raise_for_status()
        snapshot = api_get(session, "/api/library")
        items = [item for item in snapshot["items"] if item["name"] == NOTE_NAME]
    assert len(items) == 1, items
    now = int(time.time() * 1000)
    if project:
        api_post(
            session,
            "/api/chat/projects",
            {
                "id": PROJECT_ID,
                "name": PROJECT_NAME,
                "instructions": "",
                "createdAt": now,
                "updatedAt": now,
            },
        )
        return items[0]
    api_post(
        session,
        "/api/chat/threads",
        {
            "id": THREAD_ID,
            "title": "Quarterly report discussion",
            "modelType": "base",
            "modelId": "",
            "createdAt": now,
            "updatedAt": now,
        },
    )
    with httpx.Client(timeout = 30) as client:
        for ident, role, text, parent in [
            ("library-question-16485", "user", "Let's discuss the quarterly report.", None),
            (
                "library-answer-16485",
                "assistant",
                "Please attach the saved report.",
                "library-question-16485",
            ),
        ]:
            response = client.put(
                f"{session.base_url}/api/chat/threads/{THREAD_ID}/messages/{ident}",
                headers = headers,
                json = {
                    "id": ident,
                    "threadId": THREAD_ID,
                    "parentId": parent,
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                    "createdAt": now,
                },
            )
            response.raise_for_status()
    return items[0]


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    menu_label: str = "Attach from Library",
    dialog_title: str = "Attach from Library",
    search_placeholder: str = "Search Library files...",
    project: bool = False,
    remove_reattach: bool = True,
    **_: object,
) -> tuple[list[Path], dict]:
    out_dir.mkdir(parents = True, exist_ok = True)
    item = await asyncio.to_thread(seed, session, project)
    facts = {
        "expect": PROJECT_EXPECT if project else EXPECT,
        "seeded_name": item["name"],
        "seeded_id": item["id"],
        "project_landing": project,
    }
    shots = []
    init = seed_init_script(session, [], connections_enabled = False)
    async with open_chat(
        session.base_url,
        init_scripts = [init],
        video_dir = out_dir / "video",
        video_name = f"{label.lower()}_library_picker",
        viewport = (1280, 900),
    ) as sp:
        page = sp.page
        route = f"project={PROJECT_ID}" if project else f"thread={THREAD_ID}"
        await page.goto(f"{session.base_url}/chat?{route}", wait_until = "domcontentloaded")
        composer = page.locator("form:has(textarea)").filter(visible = True).first
        text = composer.locator("textarea").first
        await text.wait_for(state = "visible", timeout = 60_000)
        if project:
            await page.get_by_role("heading", name = PROJECT_NAME, exact = True).wait_for(
                timeout = 60_000
            )
        else:
            await page.get_by_text("Please attach the saved report.", exact = True).wait_for(
                timeout = 60_000
            )
        await text.fill(DRAFT)
        original_url = page.url
        plus = composer.get_by_role("button", name = "Tools and attachments", exact = True)
        await plus.focus()
        await plus.press("Space")
        upload_entry = page.get_by_role("menuitem", name = "Add photos & files", exact = True)
        await upload_entry.wait_for()
        plus_menu = upload_entry.locator("xpath=ancestor::*[@role='menu'][1]")
        composer_clip = {
            "x": 250,
            "y": 180 if project else 360,
            "width": 1000,
            "height": 650 if project else 540,
        }
        picker_entry = page.get_by_role("menuitem", name = menu_label, exact = True)
        facts["picker_available"] = await picker_entry.count() == 1
        menu_shot = out_dir / f"{label.lower()}_library_menu.png"
        await page.screenshot(path = str(menu_shot), animations = "disabled", clip = composer_clip)
        shots.append(menu_shot)
        if not facts["picker_available"]:
            if label.upper() != "BEFORE":
                raise AssertionError(f"Missing Library picker menu entry {menu_label!r}")
            await page.keyboard.press("Escape")
            await plus_menu.wait_for(state = "hidden")
            facts.update(
                attached = False,
                draft_preserved = await text.input_value() == DRAFT,
                thread_preserved = page.url == original_url,
            )
        else:
            await picker_entry.click()
            dialog = page.get_by_role("dialog", name = dialog_title, exact = True)
            await dialog.wait_for()
            search = dialog.get_by_placeholder(search_placeholder, exact = True)
            await search.fill("quarterly-report-16485")
            await dialog.get_by_text(NOTE_NAME, exact = True).wait_for()
            picker_shot = out_dir / f"{label.lower()}_library_picker.png"
            await page.screenshot(path = str(picker_shot), animations = "disabled", full_page = False)
            facts["picker_screenshot"] = picker_shot.name
            await page.keyboard.press("Escape")
            await dialog.wait_for(state = "hidden")
            facts["escape_preserves_draft"] = await text.input_value() == DRAFT
            assert facts["escape_preserves_draft"]
            await plus.click()
            await page.get_by_role("menuitem", name = menu_label, exact = True).click()
            await dialog.wait_for()
            search = dialog.get_by_placeholder(search_placeholder, exact = True)
            await search.fill("quarterly-report-16485")
            entry = dialog.get_by_text(NOTE_NAME, exact = True)
            await entry.wait_for()
            read_path = urlsplit(item["fileUrl"]).path
            async with page.expect_response(
                lambda response: urlsplit(response.url).path == read_path
                and response.request.method == "GET"
            ) as read_response:
                await search.press("ArrowDown")
                await search.press("Enter")
            response = await read_response.value
            data = await response.body()
            facts["read_path"] = read_path
            facts["read_status"] = response.status
            facts["read_sha256"] = hashlib.sha256(data).hexdigest()
            facts["read_text"] = data.decode()
            assert response.ok and data == NOTE_TEXT.encode(), facts
            await dialog.wait_for(state = "hidden")
            await composer.get_by_role(
                "button", name = re.compile(re.escape(NOTE_NAME))
            ).first.wait_for(timeout = 30_000)
            facts["attachment_count"] = await composer.get_by_role(
                "button", name = "Remove file", exact = True
            ).count()
            facts["attached"] = facts["attachment_count"] == 1
            facts["draft_preserved"] = await text.input_value() == DRAFT
            facts["thread_preserved"] = page.url == original_url
            assert (
                facts["attached"] and facts["draft_preserved"] and facts["thread_preserved"]
            ), facts
            if remove_reattach:
                await composer.get_by_role("button", name = "Remove file", exact = True).click()
                await composer.get_by_role("button", name = "Remove file", exact = True).wait_for(
                    state = "hidden"
                )
                await plus.click()
                await page.get_by_role("menuitem", name = menu_label, exact = True).click()
                await dialog.wait_for()
                await dialog.get_by_placeholder(search_placeholder, exact = True).fill(
                    "quarterly-report-16485"
                )
                await dialog.get_by_text(NOTE_NAME, exact = True).click()
                await dialog.wait_for(state = "hidden")
                await composer.get_by_role("button", name = "Remove file", exact = True).wait_for()
                facts["remove_reattach_count"] = await composer.get_by_role(
                    "button", name = "Remove file", exact = True
                ).count()
                facts["remove_reattach_ok"] = (
                    facts["remove_reattach_count"] == 1
                    and await text.input_value() == DRAFT
                    and page.url == original_url
                )
                assert facts["remove_reattach_ok"], facts
        final_shot = out_dir / f"{label.lower()}_library_attachment.png"
        await plus_menu.wait_for(state = "hidden")
        await page.screenshot(path = str(final_shot), animations = "disabled", clip = composer_clip)
        shots.append(final_shot)
        facts["thread_id"] = None if project else THREAD_ID
        if project:
            facts["project_id"] = PROJECT_ID
            facts["project_preserved"] = page.url == original_url
            project_threads = api_get(session, f"/api/chat/threads?project_id={PROJECT_ID}")
            facts["project_threads_created"] = sum(
                row.get("projectId") == PROJECT_ID for row in project_threads.get("threads", [])
            )
            assert facts["project_preserved"] and facts["project_threads_created"] == 0, facts
        facts["draft"] = await text.input_value()
    facts["video"] = str(sp.video_webm) if sp.video_webm else None
    (out_dir / f"{label.lower()}_library_facts.json").write_text(
        json.dumps(facts, indent = 2), encoding = "utf-8"
    )
    return shots, facts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required = True)
    parser.add_argument("--home", type = Path, required = True)
    parser.add_argument("--password", required = True)
    parser.add_argument("--out", type = Path, required = True)
    parser.add_argument("--label", choices = ["BEFORE", "AFTER"], required = True)
    parser.add_argument("--menu-label", default = "Attach from Library")
    parser.add_argument("--dialog-title", default = "Attach from Library")
    parser.add_argument("--search-placeholder", default = "Search Library files...")
    parser.add_argument("--project", action = "store_true")
    args = parser.parse_args()
    session = studio_session(args.url, args.home, args.password)
    print(
        asyncio.run(
            drive(
                session,
                args.out,
                args.label,
                menu_label = args.menu_label,
                dialog_title = args.dialog_title,
                search_placeholder = args.search_placeholder,
                project = args.project,
            )
        )
    )
