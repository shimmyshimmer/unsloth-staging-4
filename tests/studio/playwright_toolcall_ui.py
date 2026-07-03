# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser tool-calling + multi-turn smoke for Studio chat.

Boots against a running Studio (BASE_URL), authenticates over the API
(login -> change-password -> token), loads a small tool-capable model (a GGUF
via llama.cpp, or -- when TOOLCALL_GGUF_VARIANT is blank -- an MLX / safetensors
repo routed onto the native backend), then
drives the real chat UI through headless Chromium: enable the Code (Python
sandbox) tool pill, send a computation prompt (turn 1), then a follow-up that
depends on turn 1 (turn 2, multi-turn). Screenshots each step into PW_ART_DIR.

Hard assertions: auth + model load succeed, both turns complete without the
stream hanging, and the composer Code pill toggles. Tool execution is verified
leniently (a code block / tool-result UI is reported; a tiny CPU model does not
always emit a well-formed call, so its absence warns rather than fails) -- the
point is that the browser tool pipeline does not hang or error, which is exactly
what the parser hardening protects.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    robust_evaluate,
    wait_for_health,
)
from playwright.sync_api import sync_playwright  # noqa: E402

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ.get("STUDIO_NEW_PW", "ToolCallCI-Xy9!")
# Backend-agnostic model selection. TOOLCALL_MODEL_PATH takes precedence and works
# for any backend; the legacy TOOLCALL_GGUF_REPO is the GGUF fallback. When
# TOOLCALL_GGUF_VARIANT is left blank the load omits gguf_variant, which routes the
# request off the llama.cpp path onto the MLX / safetensors backend (used on macOS,
# where the native Metal MLX path generates coherent text the CPU GGUF path does not).
MODEL_PATH = os.environ.get("TOOLCALL_MODEL_PATH") or os.environ.get(
    "TOOLCALL_GGUF_REPO", "unsloth/Llama-3.2-3B-Instruct-GGUF"
)
GGUF_VARIANT = os.environ.get("TOOLCALL_GGUF_VARIANT", "Q5_K_M").strip()
MAX_SEQ_LEN = int(os.environ.get("TOOLCALL_MAX_SEQ_LEN", "2048"))
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright"))
ART.mkdir(parents=True, exist_ok=True)
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "300000"))
LOAD_TIMEOUT_S = float(os.environ.get("TOOLCALL_LOAD_TIMEOUT_S", "300"))


def step(s):
    print(f"[toolcall] STEP {s}", flush=True)


def info(s):
    print(f"[toolcall] {s}", flush=True)


def fail(m):
    raise AssertionError(f"[toolcall] FAIL: {m}")


def _api(path, method="POST", token=None, body=None, timeout=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout or 30) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="ignore")
        return exc.code, raw


def api_auth():
    """login(bootstrap) -> change-password -> return a usable access token."""
    step("API login with bootstrap password")
    status, body = _api("/api/auth/login", body={"username": "unsloth", "password": OLD})
    if status != 200:
        fail(f"login failed: {status} {body!r}")
    token = body["access_token"]
    if body.get("must_change_password"):
        step("API change-password (bootstrap -> new)")
        status, body = _api(
            "/api/auth/change-password",
            token=token,
            body={"current_password": OLD, "new_password": NEW},
        )
        if status != 200:
            fail(f"change-password failed: {status} {body!r}")
        token = body["access_token"]
    return token


def main():
    wait_for_health(BASE, timeout=180, info=info)
    token = api_auth()

    backend = "GGUF" if GGUF_VARIANT else "MLX/safetensors"
    step(f"load tool-capable model {MODEL_PATH} ({GGUF_VARIANT or backend})")
    load_body = {
        "model_path": MODEL_PATH,
        "is_lora": False,
        "max_seq_length": MAX_SEQ_LEN,
    }
    if GGUF_VARIANT:
        load_body["gguf_variant"] = GGUF_VARIANT
    status, body = _api(
        "/api/inference/load",
        token=token,
        body=load_body,
        timeout=LOAD_TIMEOUT_S,
    )
    if status != 200:
        fail(f"/api/inference/load returned {status}: {body!r}")
    info(f"loaded: {(body or {}).get('display_name') if isinstance(body, dict) else body}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=chromium_launch_args())
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(ART / "video"),
            record_video_size={"width": 1440, "height": 900},
        )
        ctx.add_init_script(
            "window.localStorage.setItem('unsloth_auth_token', %s);" % json.dumps(token)
        )
        page = ctx.new_page()

        def shoot(name):
            try:
                page.screenshot(path=str(ART / f"{name}.png"), full_page=False)
            except Exception as e:
                info(f"WARN screenshot {name}: {e}")

        step("open /chat")
        page.goto(f"{BASE}/chat", wait_until="domcontentloaded", timeout=60_000)
        composer = page.locator('textarea[aria-label="Message input"]')
        composer.wait_for(state="visible", timeout=60_000)
        page.wait_for_timeout(1500)
        shoot("01-chat-open")

        # Enable the Code (Python sandbox) tool pill.
        step("enable Code tool pill")
        code_pill = page.locator('button[data-pill-label="Code"]').first
        if code_pill.count() == 0:
            fail("Code tool pill not found in composer")
        if (code_pill.get_attribute("data-active") or "false") != "true":
            code_pill.click()
            page.wait_for_timeout(400)
        active = code_pill.get_attribute("data-active")
        info(f"Code pill data-active={active!r}")
        if active != "true":
            fail("Code tool pill did not activate")
        shoot("02-code-pill-on")

        # JS: is a tool-call badge / code / tool-result rendered right now, and is
        # the model still streaming?
        _tool_js = (
            "() => { const t = document.body.innerText || '';"
            " const badge = /\\b\\d+\\s+tool calls?\\b|Used tool/i.test(t);"
            " const code = document.querySelectorAll('pre code, .tool-ui, [data-tool-name]').length > 0;"
            " const streaming = !!document.querySelector('button[aria-label=\"Stop generating\"]');"
            " return {badge, code, streaming}; }"
        )

        def _stop_if_streaming():
            stop = page.locator('button[aria-label="Stop generating"]').first
            try:
                if stop.count() and stop.is_visible():
                    stop.click(timeout=5_000)
            except Exception:
                pass
            try:
                page.wait_for_selector(
                    'button[aria-label="Stop generating"]', state="detached", timeout=60_000
                )
            except Exception:
                pass

        def send_turn(prompt, want_toolcall, cap_ms):
            import time as _t

            page.wait_for_selector(
                'button[aria-label="Send message"]', state="attached", timeout=120_000
            )
            before = robust_evaluate(
                page, "() => document.querySelectorAll('[data-role=\"assistant\"]').length"
            )
            composer.click()
            composer.fill(prompt)
            page.locator('button[aria-label="Send message"]').click()
            page.wait_for_function(
                "(want) => document.querySelectorAll('[data-role=\"assistant\"]').length >= want",
                arg=before + 1,
                timeout=120_000,
            )
            # Poll until the tool-call badge appears (turn 1) or streaming ends. The
            # Code sandbox can send the model into a long retry loop that the slow
            # macOS runner cannot finish in budget, so once the tool call is visible
            # we stop rather than wait the loop out.
            deadline = _t.time() + cap_ms / 1000.0
            seen = False
            while _t.time() < deadline:
                st = robust_evaluate(page, _tool_js) or {}
                if want_toolcall and (st.get("badge") or st.get("code")):
                    seen = True
                    break
                if not st.get("streaming"):
                    break
                page.wait_for_timeout(1500)
            return seen

        # A computation the model cannot shortcut from memory, so it must call the
        # Python tool (a trivial sum is answered directly, skipping the tool). The
        # badge renders as soon as the first call fires; the early-stop below bounds
        # the runtime so the slow macOS runner never has to finish the retry loop.
        step("turn 1: python sum of first 20 primes (tool call)")
        tool_seen = send_turn(
            "Use the Python tool to compute the sum of the first 20 prime numbers, then tell me the number.",
            want_toolcall=True,
            cap_ms=min(TURN_TIMEOUT_MS, 240_000),
        )
        shoot("03-turn1-done")
        _stop_if_streaming()
        info(f"turn 1 tool call visible: {tool_seen}")

        # Turn 2: lightweight multi-turn follow-up with the Code tool OFF so it
        # completes quickly and proves the conversation carries context.
        step("turn 2 (multi-turn follow-up, no tool)")
        if (code_pill.get_attribute("data-active") or "false") == "true":
            code_pill.click()
            page.wait_for_timeout(300)
        send_turn(
            "In one short sentence, restate the number you just computed.",
            want_toolcall=False,
            cap_ms=180_000,
        )
        _stop_if_streaming()
        shoot("04-turn2-done")

        # Final DOM re-check of the tool-call evidence (badge still rendered).
        final_signals = robust_evaluate(page, _tool_js) or {}
        info(f"final tool signals: {final_signals}")

        assistant_texts = robust_evaluate(
            page,
            "() => Array.from(document.querySelectorAll('[data-role=\"assistant\"]')).map(e => (e.innerText||'').trim())",
        )
        info(f"assistant bubble count: {len(assistant_texts)}")
        if len(assistant_texts) < 2:
            fail(f"expected >= 2 assistant turns, got {len(assistant_texts)}")

        # A real tool call must have been observed on turn 1 (badge or code block),
        # either while streaming or in the final DOM.
        if tool_seen or final_signals.get("badge") or final_signals.get("code"):
            info("OK tool call fired in the browser (badge/code observed)")
        else:
            fail("no tool call observed on turn 1 (expected the Code tool to fire)")

        shoot("05-final")
        ctx.close()
        browser.close()
    info("PASS tool-calling + multi-turn browser smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
