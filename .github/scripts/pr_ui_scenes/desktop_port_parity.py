from pathlib import Path

from playwright.async_api import async_playwright

from pr_ui_scenes._common import Session
from studio_test_kit.auth import seed_init_script


async def drive(session: Session, out_dir: Path, label: str, **kwargs):
    auth = type(
        "Auth", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
    )()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless = True, args = ["--disable-gpu"])
        try:
            context = await browser.new_context(viewport = {"width": 1440, "height": 900})
            await context.add_init_script(seed_init_script(auth, []))
            page = await context.new_page()
            await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
            await page.get_by_role("button", name = "Settings", exact = True).click(timeout = 60_000)
            dialog = page.get_by_role("dialog")
            await dialog.wait_for(state = "visible")
            await dialog.get_by_role("button", name = "Connections", exact = True).click()
            await page.get_by_text("Enable connections", exact = True).wait_for(state = "visible")
            facts = {"connections_tab": await dialog.inner_text()}
            shot = out_dir / f"{label.lower()}_connections.png"
            await page.screenshot(path = str(shot), full_page = False)
            return [shot], facts
        finally:
            await browser.close()
