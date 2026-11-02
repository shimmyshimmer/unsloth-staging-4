"""Real Desktop account onboarding with a browser adapter for native IPC."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pr_ui_scenes._common import Session, _post
from pr_ui_scenes.desktop_seeded_owner import adapter_script


def prepare_session(base_url: str, home: Path, _password: str) -> Session:
    password = (home / "auth" / ".bootstrap_password").read_text(encoding = "utf-8").strip()
    python = home / "unsloth_studio" / "bin" / "python"
    script = """
import json
from unsloth_cli._inference import ensure_studio_backend_path
ensure_studio_backend_path(seed_cache_env=False)
from auth import storage
assert storage.requires_password_change('unsloth')
assert storage.account_counts() == (1, 0)
print('SCENE_SEED=' + json.dumps({'secret': storage.create_desktop_secret()}))
"""
    result = subprocess.run(
        [str(python), "-c", script],
        env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home)},
        capture_output = True,
        text = True,
        check = True,
        timeout = 120,
    )
    seed = json.loads(
        next(
            line.removeprefix("SCENE_SEED=")
            for line in result.stdout.splitlines()
            if line.startswith("SCENE_SEED=")
        )
    )
    desktop = _post(base_url + "/api/auth/desktop-login", {"secret": seed["secret"]})
    session = Session(base_url, home, desktop["access_token"], desktop["refresh_token"], password)
    session.desktop_scene_secret = seed["secret"]
    return session


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    browser_name = "chromium",
    exercise_cancel = False,
    **_: object,
):
    from playwright.async_api import async_playwright
    async with async_playwright() as playwright:
        browser_type = getattr(playwright, "chromium" if browser_name == "msedge" else browser_name)
        options = {"channel": "msedge"} if browser_name == "msedge" else {}
        browser = await browser_type.launch(headless = True, **options)
        try:
            context = await browser.new_context(viewport = {"width": 1440, "height": 900})
            await context.add_init_script(
                adapter_script(session.base_url, session.desktop_scene_secret)
            )
            page = await context.new_page()
            await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
            await page.locator('button[aria-label="Settings"]:visible').click(timeout = 120_000)
            await page.get_by_role("button", name = "Accounts", exact = True).click()
            await page.get_by_role("heading", name = "Accounts", exact = True).wait_for()
            await page.get_by_role("button", name = "Create account", exact = True).click()
            await page.wait_for_function("""() =>
                document.querySelector('#cp-new') || document.querySelector('#new-account-username')
            """)
            password_first = await page.locator("#cp-new").is_visible()
            if password_first and exercise_cancel:
                await (
                    page.locator('[role="dialog"]:has(#cp-new)')
                    .get_by_role("button", name = "Close", exact = True)
                    .click()
                )
                await page.locator("#cp-new").wait_for(state = "hidden")
                count = await page.evaluate("""async () => (await (await fetch('/api/accounts', {
                    headers: {Authorization: 'Bearer ' + localStorage.getItem('unsloth_auth_token')}
                })).json()).accounts.length""")
                if count != 1:
                    raise RuntimeError("Cancelling password setup created an account")
                await page.get_by_role("button", name = "Create account", exact = True).click()
                await page.locator("#cp-new").wait_for()
            await page.wait_for_function("""() => Array.from(document.querySelectorAll('[role=dialog]'))
                .filter(el => el.getClientRects().length)
                .every(el => getComputedStyle(el).opacity === '1' &&
                    !el.getAnimations().some(animation => animation.playState === 'running'))
            """)
            shot = out_dir / "desktop_owner_onboarding.png"
            await page.screenshot(path = str(shot))
            if password_first:
                await page.locator("#cp-new").fill("Owner-onboarding-demo42")
                await page.locator("#cp-confirm").fill("Owner-onboarding-demo42")
                await page.locator('[role="dialog"]:has(#cp-new) button[type="submit"]').click()
            await page.locator("#new-account-username").wait_for()
            await page.locator("#new-account-username").fill("ui-managed")
            await page.locator(
                '[role="dialog"]:has(#new-account-username) button[type="submit"]'
            ).click()
            await page.get_by_test_id("account-setup-code").wait_for()
            facts = await page.evaluate("""async () => {
                const token = localStorage.getItem('unsloth_auth_token');
                const headers = {Authorization: 'Bearer ' + token};
                const response = await fetch('/api/accounts', {headers});
                if (!response.ok) throw new Error('Account list failed: ' + response.status);
                const accounts = (await response.json()).accounts;
                const status = await (await fetch('/api/auth/status')).json();
                return {account_count: accounts.length, login_mode: status.login_mode,
                        requires_password_change: status.requires_password_change};
            }""")
            facts["password_setup_before_account"] = password_first
            if exercise_cancel:
                facts["cancel_and_retry_preserved_single_owner"] = True
            if facts["account_count"] != 2 or facts["login_mode"] != "multi":
                raise RuntimeError("Account creation did not finish on the real backend")
            refused = _post(
                session.base_url + "/api/auth/desktop-login",
                {
                    "secret": session.desktop_scene_secret,
                    "seeded_owner_login": True,
                },
            )
            facts["desktop_secret_requires_login"] = refused.get("login_required") is True
            if not facts["desktop_secret_requires_login"]:
                raise RuntimeError("Desktop secret unexpectedly bypassed shared-install login")
            if password_first:
                login = _post(
                    session.base_url + "/api/auth/login",
                    {
                        "username": "unsloth",
                        "password": "Owner-onboarding-demo42",
                    },
                )
                if not login.get("access_token"):
                    raise RuntimeError("Chosen owner password cannot recover the owner session")
                facts["chosen_owner_password_login"] = True
            facts["adapter_limit"] = "Real frontend/backend; browser adapter replaces Rust IPC."
            return [shot], facts
        finally:
            await browser.close()
