from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageDraw

from pr_ui_scenes._common import Session, api_get
from pr_ui_scenes.library_composer_picker import seed
from studio_test_kit.auth import seed_init_script
from studio_test_kit.compose import _font
from studio_test_kit.ui import open_chat

EXPECT = (
    "BEFORE: More cannot select saved Library files. AFTER: More opens the real picker, which "
    "lists sample uploads and a generated file, filters by filename, and attaches a sample "
    "image and document to the same draft without changing the conversation."
)
DRAFT = "Summarize the campaign brief and describe the attached artwork."
IMAGE_NAME = "spring-campaign.png"
BRIEF_NAME = "campaign-brief.md"
GENERATED_NAME = "generated-campaign-plan.txt"
BRIEF = b"# Spring campaign\n\nA fictional product launch for Aurora Outdoor.\nAudience: weekend hikers.\nMessage: Find your next trail.\nPalette: mint, evergreen and warm peach.\n"


def sample_image() -> bytes:
    image = Image.new("RGB", (960, 600))
    draw = ImageDraw.Draw(image)
    for y in range(600):
        ratio = y / 599
        draw.line(
            (0, y, 959, y),
            fill = (int(221 - 46 * ratio), int(243 - 24 * ratio), int(238 - 21 * ratio)),
        )
    draw.ellipse((690, 84, 820, 214), fill = "#f2bf9c")
    draw.polygon(
        [(0, 405), (245, 170), (540, 455), (780, 205), (960, 390), (960, 600), (0, 600)],
        fill = "#77b6aa",
    )
    draw.polygon(
        [(0, 490), (215, 365), (475, 535), (705, 345), (960, 475), (960, 600), (0, 600)],
        fill = "#285d54",
    )
    draw.rounded_rectangle((40, 38, 566, 150), radius = 20, fill = "#ffffff")
    draw.text((66, 58), "AURORA OUTDOOR", font = _font(30), fill = "#20483f")
    draw.text((66, 105), "Spring campaign | Sample artwork", font = _font(19), fill = "#50766e")
    draw.text((48, 534), "Find your next trail.", font = _font(30), fill = "white")
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG")
    return buffer.getvalue()


def seed_samples(session: Session, out_dir: Path) -> tuple[str, dict, dict[str, bytes]]:
    _, thread_id = seed(session)
    image = sample_image()
    (out_dir / IMAGE_NAME).write_bytes(image)
    fixtures = {
        IMAGE_NAME: (image, "image/png"),
        BRIEF_NAME: (BRIEF, "text/markdown"),
        "sales-summary.csv": (
            b"month,revenue\nJanuary,42000\nFebruary,47000\nMarch,53000\n",
            "text/csv",
        ),
    }
    headers = {"Authorization": f"Bearer {session.access_token}"}
    snapshot = api_get(session, "/api/library")["items"]
    with httpx.Client(timeout = 30) as client:
        for name, (content, mime) in fixtures.items():
            if any(item["name"] == name for item in snapshot):
                continue
            response = client.post(
                f"{session.base_url}/api/library/uploads",
                headers = headers,
                files = {"files": (name, content, mime)},
            )
            response.raise_for_status()
        if not any(item["name"] == GENERATED_NAME for item in snapshot):
            response = client.post(
                f"{session.base_url}/api/inference/sandbox/{thread_id}/upload",
                headers = headers,
                files = {
                    "file": (
                        GENERATED_NAME,
                        b"Fictional campaign plan: launch teaser, trail artwork and weekly updates.\n",
                        "text/plain",
                    )
                },
            )
            response.raise_for_status()
    expected = {*fixtures, GENERATED_NAME}
    deadline = time.monotonic() + 30
    while True:
        items = api_get(session, "/api/library")["items"]
        by_name = {item["name"]: item for item in items if item["name"] in expected}
        if expected <= by_name.keys():
            break
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"Missing sample files: {sorted(expected - by_name.keys())}; "
                f"Library returned {[item['name'] for item in items]}"
            )
        time.sleep(0.5)
    assert by_name[GENERATED_NAME]["source"] == "generated"
    return thread_id, by_name, {name: data for name, (data, _) in fixtures.items()}


async def drive(session: Session, out_dir: Path, label: str, **_: object):
    out_dir.mkdir(parents = True, exist_ok = True)
    thread_id, items, contents = await asyncio.to_thread(seed_samples, session, out_dir)
    facts = {
        "expect": EXPECT,
        "samples": sorted(items),
        "source": {name: item["source"] for name, item in items.items()},
        "read_sha256": {},
        "captures": {},
    }
    async with open_chat(
        session.base_url,
        init_scripts = [seed_init_script(session, [], connections_enabled = False)],
        viewport = (1280, 900),
        video_dir = out_dir / "video",
        video_name = f"{label.lower()}_picker_walkthrough",
    ) as sp:
        page = sp.page
        await page.goto(
            f"{session.base_url}/chat?thread={thread_id}", wait_until = "domcontentloaded"
        )
        await page.get_by_text("Please attach the saved report.", exact = True).wait_for(
            timeout = 60_000
        )
        composer = page.locator("form:has(textarea)").filter(visible = True).first
        text = composer.locator("textarea").first
        await text.fill(DRAFT)
        original_url = page.url
        plus = composer.get_by_role("button", name = "Tools and attachments", exact = True)
        await plus.click()
        upload = page.get_by_role("menuitem", name = "Add photos & files", exact = True)
        await upload.wait_for()
        option = page.get_by_role("menuitem", name = "Attach from Library", exact = True)
        facts["library_in_main_menu"] = await option.count() == 1
        assert not facts["library_in_main_menu"]
        await page.get_by_role("menuitem", name = "More", exact = True).click()
        available = await option.count() == 1
        facts["picker_available"] = available
        facts["menu_path"] = "+ > More > Attach from Library"
        surface = out_dir / f"{label.lower()}_picker_surface.png"
        attached = out_dir / f"{label.lower()}_composer_attachments.png"
        clip = {"x": 250, "y": 160, "width": 1000, "height": 720}
        if not available:
            assert label.upper() == "BEFORE"
            await page.screenshot(path = str(surface), animations = "disabled", clip = clip)
            await page.keyboard.press("Escape")
            await page.keyboard.press("Escape")
        else:
            await option.wait_for()
            await option.hover()
            await page.screenshot(path = str(surface), animations = "disabled", clip = clip)
            menu = out_dir / "after_more_menu.png"
            boxes = [
                await page.get_by_role("menu").nth(i).bounding_box()
                for i in range(await page.get_by_role("menu").count())
            ]
            boxes = [box for box in boxes if box]
            left = max(0, min(box["x"] for box in boxes) - 12)
            top = max(0, min(box["y"] for box in boxes) - 12)
            right = min(1280, max(box["x"] + box["width"] for box in boxes) + 12)
            bottom = min(900, max(box["y"] + box["height"] for box in boxes) + 12)
            await page.screenshot(
                path = str(menu),
                animations = "disabled",
                clip = {"x": left, "y": top, "width": right - left, "height": bottom - top},
            )
            facts["captures"]["more_menu"] = menu.name
            await option.click()
            dialog = page.get_by_role("dialog", name = "Attach from Library", exact = True)
            await dialog.wait_for()
            await dialog.get_by_text(IMAGE_NAME, exact = True).wait_for()
            await dialog.get_by_text(GENERATED_NAME, exact = True).wait_for()
            assert await dialog.get_by_role("option").count() >= 5
            await dialog.get_by_text("Generated", exact = True).wait_for()
            all_files = out_dir / "after_picker_all_files.png"
            await dialog.screenshot(path = str(all_files), animations = "disabled")
            facts["captures"]["all_files"] = all_files.name
            search = dialog.get_by_placeholder("Search Library files...", exact = True)
            for name, query in [(BRIEF_NAME, "campaign-brief"), (IMAGE_NAME, "spring")]:
                if name == IMAGE_NAME:
                    await plus.click()
                    await page.get_by_role("menuitem", name = "More", exact = True).click()
                    await page.get_by_role(
                        "menuitem", name = "Attach from Library", exact = True
                    ).click()
                    await dialog.wait_for()
                await search.fill(query)
                await dialog.get_by_text(name, exact = True).wait_for()
                assert await dialog.get_by_role("option").count() == 1
                filtered = out_dir / (
                    "after_picker_search_image.png"
                    if name == IMAGE_NAME
                    else "after_picker_search_document.png"
                )
                await dialog.screenshot(path = str(filtered), animations = "disabled")
                facts["captures"][name] = filtered.name
                path = urlsplit(items[name]["fileUrl"]).path
                async with page.expect_response(
                    lambda response: urlsplit(response.url).path == path
                    and response.request.method == "GET"
                ) as pending:
                    await dialog.get_by_text(name, exact = True).click()
                response = await pending.value
                data = await response.body()
                assert response.ok and data == contents[name]
                facts["read_sha256"][name] = hashlib.sha256(data).hexdigest()
                await dialog.wait_for(state = "hidden")
                await (
                    composer.get_by_role("button", name = "Remove file", exact = True)
                    .nth(1 if name == IMAGE_NAME else 0)
                    .wait_for(timeout = 30_000)
                )
                assert await text.input_value() == DRAFT and page.url == original_url
            preview = composer.get_by_role("img", name = IMAGE_NAME, exact = True)
            await preview.wait_for()
            assert await preview.evaluate("img => img.complete && img.naturalWidth > 0")
            assert await composer.get_by_role("button", name = "Remove file", exact = True).count() == 2
            facts["image_preview_loaded"] = True
        await page.get_by_role("menuitem", name = "Add photos & files", exact = True).wait_for(
            state = "hidden"
        )
        facts["attachment_count"] = await composer.get_by_role(
            "button", name = "Remove file", exact = True
        ).count()
        facts["draft_preserved"] = await text.input_value() == DRAFT
        facts["thread_preserved"] = page.url == original_url
        assert facts["attachment_count"] == (2 if available else 0)
        assert facts["draft_preserved"] and facts["thread_preserved"]
        await page.mouse.move(1050, 120)
        await page.get_by_role("tooltip").wait_for(state = "hidden")
        detail = out_dir / f"{label.lower()}_composer_detail.png"
        await composer.screenshot(path = str(detail), animations = "disabled")
        capture = Image.open(detail).convert("RGB")
        canvas = Image.new(
            "RGB", (max(800, capture.width + 32), max(340, capture.height + 32)), "white"
        )
        canvas.paste(
            capture, ((canvas.width - capture.width) // 2, canvas.height - capture.height - 16)
        )
        canvas.save(attached)
        facts["captures"]["composer_detail"] = detail.name
        facts["captures"]["composer"] = attached.name
    facts["video"] = str(sp.video_webm) if sp.video_webm else None
    (out_dir / "walkthrough_facts.json").write_text(json.dumps(facts, indent = 2), encoding = "utf-8")
    return [surface, attached], facts
