from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
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
GENERATED_EXPECT = (
    "A file in the current conversation's real sandbox appears as Generated in Library "
    "and attaches its original bytes without leaving the conversation or changing the draft."
)


async def check_route_lifecycle(
    page,
    composer,
    out_dir: Path,
    label: str,
    target_id: str,
    menu_label: str,
    dialog_title: str,
    expect_closed: bool,
) -> dict:
    origin = urlsplit(page.url)
    await page.goto(f"{origin.scheme}://{origin.netloc}/library", wait_until = "domcontentloaded")
    library_heading = page.locator("h1").filter(has_text = re.compile(r"^Library$"))
    await library_heading.wait_for(state = "visible")
    await page.locator(f'[data-testid="recent-thread"][data-thread-id="{target_id}"]').click()
    await page.wait_for_url(
        lambda url: urlsplit(url).path == "/chat" and target_id in urlsplit(url).query
    )
    await composer.locator("textarea").first.wait_for(state = "visible")
    document_token = uuid.uuid4().hex
    await page.evaluate("token => { window.__librarySceneDocumentToken = token; }", document_token)
    retained_composer = await composer.element_handle()
    await composer.get_by_role("button", name = "Tools and attachments", exact = True).click()
    await page.get_by_role("menuitem", name = "More", exact = True).click()
    await page.get_by_role("menuitem", name = menu_label, exact = True).click()
    dialog = page.get_by_role("dialog", name = dialog_title, exact = True)
    await dialog.wait_for(state = "visible")
    await page.go_back(wait_until = "domcontentloaded")
    await page.wait_for_url(lambda url: urlsplit(url).path == "/library")
    await library_heading.wait_for(state = "visible")
    await page.evaluate(
        "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
    )
    if expect_closed:
        try:
            await dialog.wait_for(state = "hidden", timeout = 5_000)
        except Exception:
            pass
    facts = {
        "route_after_back": urlsplit(page.url).path,
        "library_heading_visible": await library_heading.is_visible(),
        "dialog_hidden_after_back": not await dialog.is_visible(),
        "same_document": await page.evaluate(
            "token => window.__librarySceneDocumentToken === token", document_token
        ),
        "chat_composer_retained": await page.evaluate(
            "node => node.isConnected", retained_composer
        ),
        "expected_dialog_hidden": expect_closed,
    }
    screenshot = out_dir / f"{label.lower()}_library_route_back.png"
    await page.screenshot(path = str(screenshot), animations = "disabled", full_page = False)
    facts["screenshot"] = screenshot.name
    (out_dir / f"{label.lower()}_route_lifecycle.json").write_text(
        json.dumps(facts, indent = 2), encoding = "utf-8"
    )
    assert (
        facts["same_document"]
        and facts["chat_composer_retained"]
        and facts["library_heading_visible"]
    ), facts
    assert facts["dialog_hidden_after_back"] == expect_closed, facts
    return facts


def seed(
    session: Session,
    project: bool = False,
    generated: bool = False,
) -> tuple[dict, str]:
    headers = {"Authorization": f"Bearer {session.access_token}"}
    if project and generated:
        raise ValueError("Run project landing and generated-file scenes separately")
    items = []
    if not generated:
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
        projects = api_get(session, "/api/chat/projects").get("projects", [])
        existing = next((entry for entry in projects if entry["name"] == PROJECT_NAME), None)
        if existing is not None:
            return items[0], existing["id"]
        project_id = f"{PROJECT_ID}-{uuid.uuid4().hex}"
        api_post(
            session,
            "/api/chat/projects",
            {
                "id": project_id,
                "name": PROJECT_NAME,
                "instructions": "",
                "createdAt": now,
                "updatedAt": now,
            },
        )
        return items[0], project_id
    nonce = uuid.uuid4().hex
    thread_id = f"{THREAD_ID}-{nonce}"
    question_id = f"library-question-{nonce}"
    answer_id = f"library-answer-{nonce}"
    api_post(
        session,
        "/api/chat/threads",
        {
            "id": thread_id,
            "title": "Quarterly report discussion",
            "modelType": "base",
            "modelId": "",
            "createdAt": now,
            "updatedAt": now,
        },
    )
    with httpx.Client(timeout = 30) as client:
        for ident, role, text, parent in [
            (question_id, "user", "Let's discuss the quarterly report.", None),
            (answer_id, "assistant", "Please attach the saved report.", question_id),
        ]:
            response = client.put(
                f"{session.base_url}/api/chat/threads/{thread_id}/messages/{ident}",
                headers = headers,
                json = {
                    "id": ident,
                    "threadId": thread_id,
                    "parentId": parent,
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                    "createdAt": now,
                },
            )
            response.raise_for_status()
        if generated:
            name = f"generated-quarterly-report-16485-{nonce[:8]}.txt"
            response = client.post(
                f"{session.base_url}/api/inference/sandbox/{thread_id}/upload",
                headers = headers,
                files = {"file": (name, NOTE_TEXT.encode(), "text/plain")},
            )
            response.raise_for_status()
            expected_url = f"/api/inference/sandbox/{thread_id}/{name}"
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                snapshot = api_get(session, "/api/library", timeout = 10)
                items = [
                    item
                    for item in snapshot["items"]
                    if item["fileUrl"] == expected_url and item["source"] == "generated"
                ]
                if items:
                    break
                time.sleep(0.5)
            assert len(items) == 1, f"Sandbox file absent from Library: {expected_url}"
    return items[0], thread_id


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    menu_label: str = "Attach from Library",
    dialog_title: str = "Attach from Library",
    search_placeholder: str = "Search Library files...",
    project: bool = False,
    generated: bool = False,
    remove_reattach: bool = True,
    route_lifecycle: bool = False,
    expect_route_closed: bool = True,
    expect_pointer_unlocked: bool = True,
    **_: object,
) -> tuple[list[Path], dict]:
    out_dir.mkdir(parents = True, exist_ok = True)
    item, target_id = await asyncio.to_thread(seed, session, project, generated)
    note_name = item["name"]
    facts = {
        "expect": PROJECT_EXPECT if project else GENERATED_EXPECT if generated else EXPECT,
        "seeded_name": item["name"],
        "seeded_id": item["id"],
        "project_landing": project,
        "generated": generated,
        "seeded_source": item["source"],
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
        route = f"project={target_id}" if project else f"thread={target_id}"
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
        await page.get_by_role("menuitem", name = "More", exact = True).click()
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
            await search.fill(note_name)
            await dialog.get_by_text(note_name, exact = True).wait_for()
            if generated:
                await (
                    dialog.get_by_role("option")
                    .filter(has_text = note_name)
                    .get_by_text("Generated", exact = True)
                    .wait_for()
                )
                facts["generated_label_visible"] = True
            picker_shot = out_dir / f"{label.lower()}_library_picker.png"
            await page.screenshot(path = str(picker_shot), animations = "disabled", full_page = False)
            facts["picker_screenshot"] = picker_shot.name
            await page.keyboard.press("Escape")
            await dialog.wait_for(state = "hidden")
            facts["escape_preserves_draft"] = await text.input_value() == DRAFT
            assert facts["escape_preserves_draft"]
            await plus.click()
            await page.get_by_role("menuitem", name = "More", exact = True).click()
            await page.get_by_role("menuitem", name = menu_label, exact = True).click()
            await dialog.wait_for()
            search = dialog.get_by_placeholder(search_placeholder, exact = True)
            await search.fill(note_name)
            entry = dialog.get_by_text(note_name, exact = True)
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
                "button", name = re.compile(re.escape(note_name))
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
                await page.get_by_role("menuitem", name = "More", exact = True).click()
                await page.get_by_role("menuitem", name = menu_label, exact = True).click()
                await dialog.wait_for()
                await dialog.get_by_placeholder(search_placeholder, exact = True).fill(note_name)
                await dialog.get_by_text(note_name, exact = True).click()
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
        facts["thread_id"] = None if project else target_id
        if project:
            facts["project_id"] = target_id
            facts["project_preserved"] = page.url == original_url
            project_threads = api_get(session, f"/api/chat/threads?project_id={target_id}")
            facts["project_threads_created"] = sum(
                row.get("projectId") == target_id for row in project_threads.get("threads", [])
            )
            assert facts["project_preserved"] and facts["project_threads_created"] == 0, facts
        facts["draft"] = await text.input_value()
        if facts["picker_available"]:
            pointer = await page.evaluate("""() => ({
                body: getComputedStyle(document.body).pointerEvents,
                html: getComputedStyle(document.documentElement).pointerEvents,
                inlineBody: document.body.style.pointerEvents,
                inlineHtml: document.documentElement.style.pointerEvents,
                dialogs: [...document.querySelectorAll('[role="dialog"]')].filter(
                    element => element.getBoundingClientRect().width > 0
                        && getComputedStyle(element).visibility !== 'hidden'
                ).length,
            })""")
            pointer["unlocked"] = pointer["body"] != "none"
            pointer["expected_unlocked"] = expect_pointer_unlocked
            facts["post_attachment_pointer"] = pointer
            (out_dir / f"{label.lower()}_attachment_pointer.json").write_text(
                json.dumps(pointer, indent = 2),
                encoding = "utf-8",
            )
            assert pointer["unlocked"] == expect_pointer_unlocked, pointer
            if expect_pointer_unlocked:
                await page.get_by_test_id("nav-row-library").click(timeout = 5_000)
                await (
                    page.locator("h1")
                    .filter(has_text = re.compile(r"^Library$"))
                    .wait_for(state = "visible")
                )
                pointer["real_sidebar_click_succeeded"] = True
                (out_dir / f"{label.lower()}_attachment_pointer.json").write_text(
                    json.dumps(pointer, indent = 2),
                    encoding = "utf-8",
                )
        if route_lifecycle and facts["picker_available"]:
            if project:
                raise ValueError("Run the route-lifecycle scene with an ordinary chat")
            facts["route_lifecycle"] = await check_route_lifecycle(
                page,
                composer,
                out_dir,
                label,
                target_id,
                menu_label,
                dialog_title,
                expect_route_closed,
            )
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
    parser.add_argument("--generated", action = "store_true")
    parser.add_argument("--route-lifecycle", action = "store_true")
    parser.add_argument("--expect-route-open", action = "store_true")
    parser.add_argument("--expect-pointer-locked", action = "store_true")
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
                generated = args.generated,
                route_lifecycle = args.route_lifecycle,
                expect_route_closed = not args.expect_route_open,
                expect_pointer_unlocked = not args.expect_pointer_locked,
            )
        )
    )
