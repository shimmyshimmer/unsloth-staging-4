from pathlib import Path

from playwright.async_api import async_playwright

from pr_ui_scenes._common import api_get


async def drive(session, out_dir: Path, label: str, **kwargs):
    schema = api_get(session, "/openapi.json")
    description = schema["paths"]["/api/health"]["get"]["description"]
    out_dir.mkdir(parents = True, exist_ok = True)
    shot = out_dir / "health_docs.png"
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        context = await browser.new_context(
            viewport = {"width": 1200, "height": 900},
            extra_http_headers = {"Authorization": f"Bearer {session.access_token}"},
        )
        page = await context.new_page()
        await page.goto(f"{session.base_url}/docs")
        operation = page.locator("#operations-default-health_check_api_health_get")
        await operation.wait_for(state = "visible", timeout = 60000)
        await operation.locator(".opblock-summary").click()
        rendered = operation.locator(".opblock-description-wrapper").first
        await rendered.wait_for(state = "visible")
        assert " ".join(description.split()) in " ".join((await rendered.inner_text()).split())
        await operation.screenshot(path = str(shot))
        await context.close()
        await browser.close()
    return [shot], {"health_description": description}
