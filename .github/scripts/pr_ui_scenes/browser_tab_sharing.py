from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.async_api import async_playwright

from pr_ui_scenes._common import _post, api_delete, api_get, pick_free_ports
from studio_test_kit.auth import ProviderSeed, StudioAuth, seed_init_script


HTML = """<!doctype html><html><head><title>Browser handoff fixture</title></head>
<body style="font:18px system-ui;padding:40px;background:#f3f7fb;color:#153348">
<h1>Browser handoff fixture</h1><p>This form has unsaved state in the current page.</p>
<label>Draft note <input id="draft" style="display:block;width:90%;padding:12px;margin:12px 0" /></label>
<button style="padding:12px 18px" type="button" onclick="document.querySelector('#applied').textContent=document.querySelector('#draft').value">Apply note</button>
<p>Applied: <output id="applied"></output></p></body></html>"""
NOTE = "UNSAVED-16274: keep this live form"


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int):
        super().__init__(("127.0.0.1", port), FixtureHandler)
        self.requests = []
        self.turns = set()


class FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "object": "list",
                    "data": [{"id": "gpt-4o-mini", "object": "model", "owned_by": "fixture"}],
                }
            ).encode()
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        messages = body.get("messages", [])
        self.server.requests.append(body)
        users = [m.get("content", "") for m in messages if m.get("role") == "user"]
        key = json.dumps(users[-1] if users else "")
        joined = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "tool")
        tools = {t.get("function", {}).get("name") for t in body.get("tools", [])}
        first = key not in self.server.turns
        self.server.turns.add(key)
        clicking = "Click Apply note" in key
        name = "browser_click" if clicking else "browser_snapshot"
        if first and name in tools:
            refs = re.findall(r'\[(r\d+)\] button "Apply note', joined)
            arguments = {"ref": refs[-1] if refs else "r1"} if clicking else {}
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"fixture-{len(self.server.requests)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            marker = "Browser click complete." if clicking else "Browser read complete."
            delta = {"role": "assistant", "content": marker}
            finish = "stop"
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/event-stream" if body.get("stream") else "application/json"
        )
        self.end_headers()
        chunk = {
            "id": "fixture-response",
            "object": "chat.completion.chunk",
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        if body.get("stream"):
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish}]
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
        else:
            self.wfile.write(
                json.dumps(
                    {
                        "id": "fixture-response",
                        "object": "chat.completion",
                        "model": "gpt-4o-mini",
                        "choices": [{"index": 0, "message": delta, "finish_reason": finish}],
                    }
                ).encode()
            )
        self.wfile.flush()


async def form_frame(page):
    for _ in range(150):
        for frame in page.frames[1:]:
            if await frame.locator("#draft").count():
                return frame
        await asyncio.sleep(0.1)
    raise AssertionError("The real browser frame did not show the fixture form")


async def send(page, text: str, marker: str):
    previous = await page.get_by_text(marker, exact = True).count()
    box = page.locator("form:has(textarea) textarea").first
    await box.fill(text)
    await box.press("Enter")
    for _ in range(600):
        allow = page.get_by_role("button", name = "Allow", exact = True)
        if await allow.count() and await allow.first.is_visible():
            await allow.first.click()
        if await page.get_by_text(marker, exact = True).count() > previous:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f'Chat did not finish: {await page.locator("body").inner_text()}')


async def drive(session, out_dir: Path, label: str, **kwargs):
    out_dir.mkdir(parents = True, exist_ok = True)
    server = FixtureServer(pick_free_ports(1)[0])
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    provider_id = None
    facts = {}
    try:
        public_key = out_dir / "fixture_public.pem"
        public_key.write_text(
            api_get(session, "/api/providers/public-key")["public_key"], encoding = "utf-8"
        )
        encrypted = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-encrypt",
                "-pubin",
                "-inkey",
                str(public_key),
                "-pkeyopt",
                "rsa_padding_mode:oaep",
                "-pkeyopt",
                "rsa_oaep_md:sha256",
                "-pkeyopt",
                "rsa_mgf1_md:sha256",
            ],
            input = b"fixture-key",
            capture_output = True,
            check = True,
        ).stdout
        provider = _post(
            session.base_url + "/api/providers/",
            {
                "provider_type": "custom",
                "display_name": "Tab sharing fixture",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "models": ["gpt-4o-mini"],
                "available_models": ["gpt-4o-mini"],
                "encrypted_api_key": base64.b64encode(encrypted).decode(),
            },
            session.access_token,
        )
        provider_id = provider["id"]
        seed = ProviderSeed(
            "custom",
            "Tab sharing fixture",
            f"http://127.0.0.1:{server.server_port}/v1",
            ["gpt-4o-mini"],
            "fixture-key",
            id = provider_id,
        )
        auth = StudioAuth(session.access_token, session.refresh_token, session.base_url)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless = True, args = ["--disable-gpu"])
            context = await browser.new_context(viewport = {"width": 1500, "height": 1000})
            await context.add_init_script(seed_init_script(auth, [seed]))
            page = await context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))

            async def fixture(route):
                payload = route.request.post_data_json
                if "handoff.example" in payload.get("url", ""):
                    await route.fulfill(
                        status = 200,
                        headers = {
                            "Content-Type": "application/json",
                            "X-Unsloth-Browser-Kind": "html",
                        },
                        body = json.dumps(
                            {
                                "url": payload["url"],
                                "base": payload["url"],
                                "ownBase": False,
                                "anyOrigin": False,
                                "refresh": None,
                                "html": HTML,
                            }
                        ),
                    )
                else:
                    await route.continue_()

            await page.route("**/api/browser/fetch", fixture)
            await page.goto(session.base_url)
            await page.get_by_text("Select model", exact = True).first.click()
            await page.get_by_text("Connected", exact = True).first.click()
            await page.get_by_text("gpt-4o-mini", exact = True).last.click()
            await page.get_by_role("button", name = "New tab", exact = True).first.click()
            address = page.get_by_role("textbox", name = "Search or enter a URL", exact = True)
            await address.fill("https://handoff.example/form")
            await address.press("Enter")
            frame = await form_frame(page)
            await frame.locator("#draft").fill(NOTE)
            identity = await frame.evaluate("window.fixtureIdentity = Math.random().toString(36)")
            fixture_tab = page.get_by_role("tab", name = re.compile("Browser handoff fixture")).first
            tab_bounds = await fixture_tab.bounding_box()
            await fixture_tab.click(button = "right")
            share = page.get_by_role("menuitem", name = "Share tab with this chat", exact = True)
            available = await share.count() > 0
            facts["share_action_available"] = available
            menu_shot = out_dir / "tab_menu.png"
            await page.get_by_role("menu").evaluate(
                "(el) => Promise.all(el.getAnimations({subtree: true}).map(a => a.finished.catch(() => {})))"
            )
            left = max(0, tab_bounds["x"] - 12)
            await page.screenshot(
                path = str(menu_shot), clip = {"x": left, "y": 0, "width": 1500 - left, "height": 520}
            )
            if label == "AFTER":
                assert available, "The new share action is absent"
            else:
                assert not available, "Base unexpectedly already supports sharing"
            if available:
                await share.click()
                await page.get_by_text(
                    "Shared with Unsloth: Browser handoff fixture", exact = True
                ).wait_for()
            else:
                await page.keyboard.press("Escape")
                await page.get_by_role("button", name = "Tools and attachments", exact = True).click()
                await page.get_by_role("menuitem", name = "Browser use", exact = True).click()
            facts["identity_after_share"] = (
                await frame.evaluate("window.fixtureIdentity") == identity
            )
            facts["draft_after_share"] = await frame.locator("#draft").input_value()
            await send(
                page,
                "Read the shared page at https://handoff.example/form without navigating.",
                "Browser read complete.",
            )
            results = [
                str(m.get("content", ""))
                for request in server.requests
                for m in request.get("messages", [])
                if m.get("role") == "tool"
            ]
            facts["model_tool_results"] = results
            if available:
                assert any(NOTE in result for result in results), results
                facts["draft_after_first_message"] = await frame.locator("#draft").input_value()
                facts["identity_after_first_message"] = (
                    await frame.evaluate("window.fixtureIdentity") == identity
                )
                assert facts["identity_after_first_message"]
                assert facts["draft_after_first_message"] == NOTE
            else:
                assert any("No page is open" in result for result in results), results
            card = page.get_by_text("Used tool: browser_snapshot", exact = False).last
            if await card.count():
                await card.click()
            result = page.locator('[data-slot="tool-fallback-result"]').last
            await result.wait_for(state = "visible")
            await result.evaluate(
                '(el) => Promise.all(el.closest("[data-slot=tool-fallback-content]").getAnimations({subtree: true}).map(a => a.finished.catch(() => {})))'
            )
            read_shot = out_dir / "shared_read.png"
            await page.screenshot(
                path = str(read_shot), clip = {"x": 280, "y": 0, "width": 1220, "height": 640}
            )
            if available:
                await send(
                    page,
                    "Click Apply note in the shared page at https://handoff.example/form.",
                    "Browser click complete.",
                )
                await frame.locator("#applied").filter(has_text = NOTE).wait_for()
                facts["applied_note"] = await frame.locator("#applied").inner_text()
                await page.get_by_role("button", name = "Stop sharing tab", exact = True).click()
                await send(
                    page,
                    "Read after revoking the shared page at https://handoff.example/form.",
                    "Browser read complete.",
                )
                final = [
                    str(m.get("content", ""))
                    for m in server.requests[-1].get("messages", [])
                    if m.get("role") == "tool"
                ]
                facts["revoked_result"] = final[-1] if final else ""
                assert "No page is open" in facts["revoked_result"], final
            facts["page_errors"] = errors
            (out_dir / "body.txt").write_text(
                await page.locator("body").inner_text(), encoding = "utf-8"
            )
            await context.close()
            await browser.close()
            return [menu_shot, read_shot], facts
    finally:
        (out_dir / "provider_requests.json").write_text(
            json.dumps(server.requests, indent = 2), encoding = "utf-8"
        )
        (out_dir / "facts.json").write_text(json.dumps(facts, indent = 2), encoding = "utf-8")
        server.shutdown()
        server.server_close()
        if provider_id:
            api_delete(session, f"/api/providers/{provider_id}")
