"""Drive a LIVE Unsloth Studio with Playwright, in one named browser, and record it.

Why not `studio_test_kit._smoke_ui`: that spins up a FAKE HTTP server that mimics Studio's
/chat surface. It proves the kit works; it says nothing about Studio. And `open_chat` launches
chromium only, so it cannot answer "Firefox, Safari, Chrome and Edge all work".

Browser targets and what each one stands for:
  chromium  the bundled Chromium engine (the baseline every runner has)
  firefox   Firefox
  webkit    Safari -- WebKit is Safari's engine; Playwright cannot drive Safari.app itself,
            and on Linux/Windows this is the only way to exercise the Safari code path at all
  chrome    real Google Chrome, via Playwright's `channel`
  msedge    real Microsoft Edge, via `channel`

Exit code is the gate: 0 only if the browser launched, Studio answered, the SPA mounted a
composer, and a non-trivial screenshot came out. Facts land in <out>/<label>.json so a green
job can be checked rather than believed.

usage: studio_browser_drive.py --base-url http://127.0.0.1:8888 --browser firefox \
           --out logs/pw --password-file <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

# Studio is an SSE/WebSocket SPA, so `networkidle` never fires: wait for the DOM, then for a
# real element (studio_test_kit README pitfall 1).
READY_SELECTORS = (
    "form:has(textarea) textarea",
    "textarea",
    "[contenteditable='true']",
    "main",
)


def _post(base_url: str, path: str, body: dict, token=None, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(base_url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def login(base_url: str, username: str, password: str, timeout: float = 30.0) -> dict:
    return _post(base_url, "/api/auth/login", {"username": username, "password": password},
                 timeout=timeout)


def authenticate(base_url: str, username: str, bootstrap: str, new_pw: str) -> tuple:
    """Return (token, how). Studio forces a password change on first login, so a token minted
    from the BOOTSTRAP password lands every route on `Change Password - Unsloth` and the chat
    composer never renders -- a run that looks green and photographs a login form. Trade the
    bootstrap password for a real one first, and tolerate a home where that already happened.
    """
    try:
        return login(base_url, username, new_pw)["access_token"], "already-rotated"
    except Exception:  # noqa: BLE001 - first run, the new password does not exist yet
        pass
    if not bootstrap:
        raise RuntimeError("no bootstrap password file and the rotated password did not work")
    tok = login(base_url, username, bootstrap)["access_token"]
    _post(base_url, "/api/auth/change-password",
          {"current_password": bootstrap, "new_password": new_pw}, tok)
    return login(base_url, username, new_pw)["access_token"], "rotated"


def wait_healthz(base_url: str, timeout: float = 300.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(base_url + "/healthz", timeout=10) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888")
    ap.add_argument("--browser", required=True,
                    choices=["chromium", "firefox", "webkit", "chrome", "msedge"])
    ap.add_argument("--out", default="logs/pw")
    ap.add_argument("--password-file", default=None)
    ap.add_argument("--password", default=os.environ.get("STUDIO_PW"))
    ap.add_argument("--username", default="unsloth")
    ap.add_argument("--new-password", default=os.environ.get("STUDIO_NEW_PW", "XplatCI-Studio-1!"))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "video").mkdir(parents=True, exist_ok=True)
    label = args.browser
    facts: dict = {"browser": label, "base_url": args.base_url}

    if not wait_healthz(args.base_url):
        facts["error"] = "studio /healthz never answered"
        (out / f"{label}.json").write_text(json.dumps(facts, indent=2))
        print(json.dumps(facts, indent=2))
        return 1

    password = args.password
    if not password and args.password_file:
        # Trap 2 in the UI-evidence workflow: the bootstrap password is written to a FILE under
        # <home>/auth/, not to the log, so a log regex returns empty and every route 403s.
        p = Path(args.password_file)
        if p.is_file():
            password = p.read_text().strip()
    token = None
    # Always attempt auth, even with no bootstrap file. Studio DELETES
    # <home>/auth/.bootstrap_password once the password has been changed, so from the second
    # browser onward the file is gone -- which silently made every browser after the first one
    # drive Studio logged out and photograph the login page (and the job stayed green).
    if True:
        try:
            token, how = authenticate(args.base_url, args.username, password, args.new_password)
            facts["auth_path"] = how
        except Exception as e:  # noqa: BLE001 - any auth failure is just "drive it logged out"
            facts["login_error"] = f"{type(e).__name__}: {e}"
    facts["authenticated"] = bool(token)

    engine = "chromium" if args.browser in ("chromium", "chrome", "msedge") else args.browser
    channel = args.browser if args.browser in ("chrome", "msedge") else None

    with sync_playwright() as p:
        launcher = getattr(p, engine)
        kwargs = {"headless": True}
        if channel:
            kwargs["channel"] = channel
        browser = launcher.launch(**kwargs)
        facts["engine"] = engine
        facts["channel"] = channel
        facts["browser_version"] = browser.version
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(out / "video"),
            record_video_size={"width": 1440, "height": 900},
        )
        if token:
            # JWT lives in localStorage, not cookies, and must be seeded BEFORE the first goto
            # (kit pitfall 5).
            ctx.add_init_script(
                "window.localStorage.setItem('unsloth_auth_token', %s);" % json.dumps(token))
        page = ctx.new_page()
        try:
            page.goto(args.base_url, wait_until="domcontentloaded", timeout=90_000)
            facts["title"] = page.title()
            found = None
            for sel in READY_SELECTORS:
                try:
                    page.wait_for_selector(sel, timeout=20_000, state="attached")
                    found = sel
                    break
                except Exception:  # noqa: BLE001 - try the next selector
                    continue
            facts["ready_selector"] = found
            page.wait_for_timeout(2500)
            shot = out / f"{label}_home.png"
            page.screenshot(path=str(shot), full_page=False)
            facts["home_png_bytes"] = shot.stat().st_size

            page.goto(args.base_url.rstrip("/") + "/chat",
                      wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_timeout(3000)
            shot2 = out / f"{label}_chat.png"
            page.screenshot(path=str(shot2), full_page=False)
            facts["chat_png_bytes"] = shot2.stat().st_size
            facts["chat_title"] = page.title()
            # Type into the composer if there is one: a keystroke reaching the SPA is the
            # difference between "the page rendered" and "the browser can drive it".
            typed = False
            try:
                box = page.locator("form:has(textarea) textarea").first
                if box.count() == 0:
                    box = page.locator("textarea").first
                box.click(timeout=10_000)
                box.type(f"cross-browser check from {label}", delay=25)
                typed = box.input_value().endswith(label)
            except Exception as e:  # noqa: BLE001
                facts["type_error"] = f"{type(e).__name__}: {str(e)[:160]}"
            facts["typed_into_composer"] = typed
            shot3 = out / f"{label}_composer.png"
            page.screenshot(path=str(shot3), full_page=False)
            facts["composer_png_bytes"] = shot3.stat().st_size
            facts["console_errors"] = []
        finally:
            video = page.video
            ctx.close()  # video only flushes here
            if video:
                try:
                    dest = out / "video" / f"{label}.webm"
                    video.save_as(str(dest))
                    facts["video_webm"] = str(dest)
                    facts["video_bytes"] = dest.stat().st_size
                except Exception as e:  # noqa: BLE001
                    facts["video_error"] = f"{type(e).__name__}: {e}"
            browser.close()

    # `chat_title` still reading "Change Password" means the SPA never got past the forced
    # rotation, so the screenshots are of a login form and prove nothing about Studio.
    ok = (facts.get("ready_selector") is not None
          and facts.get("home_png_bytes", 0) > 5000
          and facts.get("chat_png_bytes", 0) > 5000
          and "Change Password" not in (facts.get("chat_title") or "")
          # Logged out, every route renders the login page. A screenshot of that says nothing
          # about Studio in this browser.
          and "Login" not in (facts.get("chat_title") or "")
          and facts.get("authenticated") is True)
    facts["ok"] = ok
    (out / f"{label}.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
