"""Photograph cached-vendor training access facts without loading a model."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pr_ui_scenes import image_train_base_picker as picker
from pr_ui_scenes._common import Session, api_get, open_menu
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    family_label: str = "FLUX.2 Klein",
    family_key: str = "flux.2-klein",
    target_base: str = "FLUX.2-klein-base-9B",
    vendor: str = "black-forest-labs/FLUX.2-klein-base-9B",
    mirror: str = "unsloth/FLUX.2-klein-base-9B",
    **kwargs: object,
) -> tuple[list[Path], dict]:
    facts = {"family_key": family_key, "target_base": target_base}
    info = api_get(session, "/api/train/diffusion/info")
    family = next(f for f in info["families"] if f["name"] == family_key)
    access = next(
        value for repo, value in family["base_specs"].items() if repo.lower() == vendor.lower()
    )
    facts.update(
        api_gated = access["gated"],
        api_note = access["note"],
        api_precision_modes = family.get("precision_modes"),
        capability_fixture = kwargs.get("capability_fixture", "none"),
    )
    auth = type(
        "Auth",
        (),
        {"access_token": session.access_token, "refresh_token": session.refresh_token},
    )()
    async with open_chat(
        session.base_url,
        init_scripts = [seed_init_script(auth, [])],
        viewport = (1500, 1000),
        headless = True,
    ) as sp:
        page = sp.page
        async with page.expect_response(
            lambda response: "/api/train/diffusion/info" in response.url and response.status == 200,
            timeout = 60_000,
        ):
            await page.goto(f"{session.base_url}/images", wait_until = "domcontentloaded")
            await page.get_by_role("tab", name = "Train", exact = True).click()
        family_trigger = page.get_by_role("combobox", name = "Model family").first
        await open_menu(
            page,
            family_trigger,
            page.get_by_role("option", name = picker.FAMILY_ANCHOR).first,
        )
        await page.get_by_role("option", name = re.compile(re.escape(family_label))).first.click()
        assert family_label in await family_trigger.inner_text()
        base_trigger = page.get_by_role("combobox", name = "Base model").first
        await open_menu(
            page,
            base_trigger,
            page.get_by_role("option", name = picker.BASE_ANCHOR).first,
        )
        await page.get_by_role("option", name = re.compile(re.escape(target_base))).first.click()
        assert target_base in await base_trigger.inner_text()
        facts["base_after_pick"] = (await base_trigger.inner_text()).strip()

        warning = page.locator("p").filter(has_text = "Needs its license and your HF token.")
        note = page.locator("p").filter(has_text = f"Downloads from {mirror}.")
        badges = page.get_by_text("Gated", exact = True)
        facts["ui_gated_badge"] = any([await badge.is_visible() for badge in await badges.all()])
        facts["ui_token_warning"] = any([await item.is_visible() for item in await warning.all()])
        facts["ui_mirror_note"] = any([await item.is_visible() for item in await note.all()])
        assert facts["ui_gated_badge"] == facts["api_gated"]
        assert facts["ui_token_warning"] == facts["api_gated"]
        assert facts["ui_mirror_note"] == (f"Downloads from {mirror}." in access["note"])
        out_dir.mkdir(parents = True, exist_ok = True)
        panel = out_dir / f"{label.lower()}_panel.png"
        await page.screenshot(
            path = str(panel),
            clip = kwargs.get("clip") or {"x": 300, "y": 140, "width": 700, "height": 300},
        )

    print(f"[{label}] {json.dumps(facts, sort_keys=True)}", flush = True)
    return [panel], facts
