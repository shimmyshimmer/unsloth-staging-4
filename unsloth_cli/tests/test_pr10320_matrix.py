"""Cross-platform simulation matrix for the openclaw memory.search config (PR #10320).

Repo-native copy of the workspace matrix, so the Windows and macOS runners execute the
branches a Linux-only lane never reaches: _print_env's PowerShell form, the recipe
quoting, and the locale-encoded JSON writer.
"""

import io
import json
import os
import urllib.error

import pytest

import typer
import unsloth_cli.commands.start as start
from typer.testing import CliRunner

from test_start import (
    BASE,
    MODEL,
    _simulate_windows,
    fake_studio,  # noqa: F401  (pytest fixture, used by name)
)

SEARCH = ("memory", "search")


def _config(tmp_path):
    return json.loads((tmp_path / "agents" / "openclaw" / "openclaw.json").read_text(encoding = "utf-8"))


def _run_openclaw(tmp_path, monkeypatch, *extra):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch", *extra])
    assert result.exit_code == 0, result.output
    return result


def _override_embedding_route(monkeypatch, responder):
    """Re-patch _http_json after fake_studio, per the PR's own test idiom."""
    real = start._http_json

    def routed(method, url, *args, **kwargs):
        if url.endswith("/api/settings/embedding-model"):
            return responder()
        return real(method, url, *args, **kwargs)

    monkeypatch.setattr(start, "_http_json", routed)


def _raiser(exc):
    def _raise():
        raise exc

    return _raise


def _value(payload):
    return lambda: payload


# --------------------------------------------------------------------------------------
# H1. The probe's failure matrix. `_studio_embedding_model` catches bare Exception; every
# row here is a transport or payload shape that a real Studio (or a proxy in front of one)
# can produce. The invariant: the CLI never crashes and always writes a usable config.
# --------------------------------------------------------------------------------------

def _http_error(code):
    return urllib.error.HTTPError(
        f"{BASE}/api/settings/embedding-model", code, "boom", {}, io.BytesIO(b"{}")
    )


PROBE_FAILURES = {
    "http_401": _http_error(401),
    "http_403": _http_error(403),
    "http_404": _http_error(404),
    "http_500": _http_error(500),
    "http_502": _http_error(502),
    # urlopen_no_redirect turns a 3xx into an HTTPError carrying the 3xx code.
    "refused_redirect_302": _http_error(302),
    "conn_refused": urllib.error.URLError(ConnectionRefusedError(111, "refused")),
    "dns": urllib.error.URLError("nodename nor servname provided"),
    "timeout": TimeoutError("timed out"),
    "non_json_body": json.JSONDecodeError("Expecting value", "<html>", 0),
    "non_utf8_body": UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    "os_error": OSError("socket closed"),
}


@pytest.mark.parametrize("label", sorted(PROBE_FAILURES))
def test_probe_failure_still_writes_a_usable_config(label, fake_studio, tmp_path, monkeypatch):
    _override_embedding_route(monkeypatch, _raiser(PROBE_FAILURES[label]))
    _run_openclaw(tmp_path, monkeypatch)
    search = _config(tmp_path)["memory"]["search"]
    # Whatever the fallback shape is, it must never name a model the server cannot serve
    # while also pointing at that server with no fallback of its own.
    assert search["fallback"] == "none"
    if search.get("provider") == "openai-compatible":
        assert search["model"] not in ("", None)


PROBE_ODD_BODIES = {
    "empty_object": ({}, "default"),
    "null_body": (None, "default"),
    "list_body": ([1, 2, 3], "default"),
    "string_body": ("unsloth/bge-small-en-v1.5", "default"),
    "int_body": (42, "default"),
    "missing_key": ({"model": "x"}, "default"),
    "empty_name": ({"embedding_model": ""}, "default"),
    "blank_name": ({"embedding_model": "   "}, "default"),
    "int_name": ({"embedding_model": 42}, "default"),
    "null_name": ({"embedding_model": None}, "default"),
    "list_name": ({"embedding_model": ["a"]}, "default"),
}


@pytest.mark.parametrize("label", sorted(PROBE_ODD_BODIES))
def test_probe_odd_body_falls_back(label, fake_studio, tmp_path, monkeypatch):
    body, expected = PROBE_ODD_BODIES[label]
    _override_embedding_route(monkeypatch, _value(body))
    _run_openclaw(tmp_path, monkeypatch)
    search = _config(tmp_path)["memory"]["search"]
    assert search.get("model", expected) == expected or search.get("provider") == "none"


def test_probe_value_is_stripped(fake_studio, tmp_path, monkeypatch):
    """H2. A padded Settings value must not reach OpenClaw with its padding."""
    _override_embedding_route(monkeypatch, _value({"embedding_model": "  unsloth/bge-small-en-v1.5  "}))
    _run_openclaw(tmp_path, monkeypatch)
    assert _config(tmp_path)["memory"]["search"]["model"] == "unsloth/bge-small-en-v1.5"


def test_probe_huge_body_is_handled(fake_studio, tmp_path, monkeypatch):
    _override_embedding_route(monkeypatch, _value({"embedding_model": "m", "junk": "x" * 5_000_000}))
    _run_openclaw(tmp_path, monkeypatch)
    assert _config(tmp_path)["memory"]["search"]["model"] == "m"


def test_probe_does_not_swallow_typer_exit(fake_studio, tmp_path, monkeypatch):
    """H3. typer.Exit is a RuntimeError subclass, so a bare `except Exception` converts a
    deliberate CLI abort into a silent fallback. The probe must let it through."""
    _override_embedding_route(monkeypatch, _raiser(typer.Exit(1)))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code != 0


def test_probe_does_not_swallow_keyboard_interrupt(fake_studio, tmp_path, monkeypatch):
    _override_embedding_route(monkeypatch, _raiser(KeyboardInterrupt()))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch"])
    assert result.exit_code != 0


def test_probe_uses_the_session_key_and_the_right_route(fake_studio, tmp_path, monkeypatch):
    _run_openclaw(tmp_path, monkeypatch)
    probes = [c for c in fake_studio if c[1].endswith("/api/settings/embedding-model")]
    assert len(probes) == 1, probes
    assert probes[0][0] == "GET"


# --------------------------------------------------------------------------------------
# H4. Config merge semantics. The block is written with .update() on a _subdict, so it
# overwrites its own four keys and must preserve everything else.
# --------------------------------------------------------------------------------------

def test_stale_external_fallback_is_replaced(fake_studio, tmp_path, monkeypatch):
    path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    path.parent.mkdir(parents = True)
    path.write_text(json.dumps({"memory": {"search": {"fallback": "openai", "provider": "openai"}}}))
    _run_openclaw(tmp_path, monkeypatch)
    search = _config(tmp_path)["memory"]["search"]
    assert search["fallback"] == "none"
    assert search["provider"] == "openai-compatible"


def test_sibling_keys_under_memory_search_survive(fake_studio, tmp_path, monkeypatch):
    path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    path.parent.mkdir(parents = True)
    path.write_text(json.dumps({
        "memory": {"search": {"sources": ["memory", "sessions"], "cache": {"enabled": True}},
                   "somethingElse": 1},
        "unrelatedTopLevel": {"keep": "me"},
    }))
    _run_openclaw(tmp_path, monkeypatch)
    config = _config(tmp_path)
    assert config["memory"]["search"]["sources"] == ["memory", "sessions"]
    assert config["memory"]["search"]["cache"] == {"enabled": True}
    assert config["memory"]["somethingElse"] == 1
    assert config["unrelatedTopLevel"] == {"keep": "me"}


@pytest.mark.parametrize("bad", ["not-a-dict", 5, [1, 2], None])
def test_non_dict_memory_is_replaced_not_crashed(bad, fake_studio, tmp_path, monkeypatch):
    path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    path.parent.mkdir(parents = True)
    path.write_text(json.dumps({"memory": bad}))
    _run_openclaw(tmp_path, monkeypatch)
    assert isinstance(_config(tmp_path)["memory"]["search"], dict)


@pytest.mark.parametrize("bad", ["not-a-dict", 5, [1, 2], None])
def test_non_dict_memory_search_is_replaced(bad, fake_studio, tmp_path, monkeypatch):
    path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    path.parent.mkdir(parents = True)
    path.write_text(json.dumps({"memory": {"search": bad}}))
    _run_openclaw(tmp_path, monkeypatch)
    assert isinstance(_config(tmp_path)["memory"]["search"], dict)


def test_second_run_is_byte_identical(fake_studio, tmp_path, monkeypatch):
    _run_openclaw(tmp_path, monkeypatch)
    path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    before = path.read_bytes()
    _run_openclaw(tmp_path, monkeypatch)
    assert path.read_bytes() == before


def test_config_mode_still_0600_with_two_key_copies(fake_studio, tmp_path, monkeypatch):
    _run_openclaw(tmp_path, monkeypatch)
    path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    config = _config(tmp_path)
    key = config["models"]["providers"]["unsloth"]["apiKey"]
    assert config["memory"]["search"]["remote"]["apiKey"] == key
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_corrupt_config_is_left_untouched(fake_studio, tmp_path, monkeypatch):
    path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    path.parent.mkdir(parents = True)
    path.write_bytes(b"{ not json")
    _run_openclaw(tmp_path, monkeypatch)
    assert path.read_bytes() == b"{ not json"


def test_corrupt_config_still_probes_the_server(fake_studio, tmp_path, monkeypatch):
    """H5. The probe is evaluated as an argument to write_openclaw_config, so it runs even
    when the config turns out to be unparseable and nothing is written. One wasted GET on
    a path that already ends in a warning: documented, not worth restructuring the call
    site for. Pinned so the cost is a known one."""
    path = tmp_path / "agents" / "openclaw" / "openclaw.json"
    path.parent.mkdir(parents = True)
    path.write_bytes(b"{ not json")
    _run_openclaw(tmp_path, monkeypatch)
    probes = [c for c in fake_studio if c[1].endswith("/api/settings/embedding-model")]
    assert len(probes) == 1


def test_baseurl_matches_the_chat_provider(fake_studio, tmp_path, monkeypatch):
    _run_openclaw(tmp_path, monkeypatch)
    config = _config(tmp_path)
    assert config["memory"]["search"]["remote"]["baseUrl"] == config["models"]["providers"]["unsloth"]["baseUrl"]
    assert config["memory"]["search"]["remote"]["baseUrl"] == f"{BASE}/v1"


# --------------------------------------------------------------------------------------
# H6. Odd model names. The Settings field is free text, so the value is server-controlled
# and reaches a JSON file that the writer opens in locale encoding.
# --------------------------------------------------------------------------------------

ODD_NAMES = {
    "cjk": "组织/嵌入模型-v1",
    "emoji": "org/model-\U0001f600",
    "rtl": "org/עברית",
    "accents": "org/modèle-français",
    "quote": 'org/model"quoted',
    "dollar": "org/model$HOME",
    "backtick": "org/model`x`",
    "space": "org/model with space",
    "backslash": "org\\model",
    "very_long": "org/" + "m" * 512,
    "leading_dash": "-org/model",
    "local_path_style": "model-3f2a1b9c",
    "colon_quant": "org/model:Q8_0",
    "percent": "org/model%3Aweird",
    "newline": "org/model\nsecond",
}


@pytest.mark.parametrize("label", sorted(ODD_NAMES))
def test_odd_embedding_model_name_round_trips(label, fake_studio, tmp_path, monkeypatch):
    name = ODD_NAMES[label]
    _override_embedding_route(monkeypatch, _value({"embedding_model": name}))
    _run_openclaw(tmp_path, monkeypatch)
    # The file must be readable exactly the way the CLI's own reader reads it: utf-8.
    assert _config(tmp_path)["memory"]["search"]["model"] == name.strip()


@pytest.mark.parametrize("label", ["cjk", "emoji", "accents"])
def test_odd_name_survives_a_windows_locale_writer(label, fake_studio, tmp_path, monkeypatch):
    """H7. _write_private_json opens the file with no encoding=, so it writes in the locale
    encoding (cp1252 on a typical Windows), while every reader hardcodes utf-8. This looked
    like a live Windows bug for a server-controlled model name, and it is NOT one: the
    payload goes through json.dumps, whose default ensure_ascii=True escapes every
    non-ASCII character to \\uXXXX, so the bytes on disk are pure ASCII whatever the
    locale. Pinning that here so the reasoning is not re-derived, and so a future switch
    to ensure_ascii=False fails loudly instead of breaking Windows users silently."""
    name = ODD_NAMES[label]
    _override_embedding_route(monkeypatch, _value({"embedding_model": name}))

    real_fdopen = os.fdopen

    def cp1252_fdopen(fd, mode = "r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "cp1252"
        return real_fdopen(fd, mode, *args, **kwargs)

    monkeypatch.setattr(start.os, "fdopen", cp1252_fdopen)
    _run_openclaw(tmp_path, monkeypatch)
    assert _config(tmp_path)["memory"]["search"]["model"] == name


def test_written_file_parses_on_every_platform(fake_studio, tmp_path, monkeypatch):
    """H7b. No newline= either, so on Windows the trailing \\n becomes \\r\\n. That changes
    the bytes but not the JSON, and both OpenClaw and the CLI's own reader parse it. What
    matters is that the file round-trips, not that the bytes match across platforms."""
    _run_openclaw(tmp_path, monkeypatch)
    raw = (tmp_path / "agents" / "openclaw" / "openclaw.json").read_bytes()
    assert json.loads(raw.decode("utf-8"))["memory"]["search"]["provider"] == "openai-compatible"
    assert raw.decode("utf-8").isascii()


# --------------------------------------------------------------------------------------
# H8. Cross-platform recipe. The recipe helpers branch on the real os.name, so the
# PowerShell branch of _print_env is never executed on Linux CI.
# --------------------------------------------------------------------------------------

def _run_openclaw_as_windows(tmp_path, monkeypatch, *extra):
    """Drive the whole command under a simulated `os.name == "nt"`, then restore it before
    asserting. pytest itself builds Paths while reporting, and a live WindowsPath flavour
    on a POSIX host crashes the reporter rather than the test."""
    _simulate_windows(monkeypatch)
    monkeypatch.chdir(tmp_path)
    try:
        result = CliRunner().invoke(start.start_app, ["openclaw", "--no-launch", *extra])
    finally:
        monkeypatch.undo()
    assert result.exit_code == 0, result.output
    return result


def test_windows_recipe_contains_the_workspace_and_config(fake_studio, tmp_path, monkeypatch):
    result = _run_openclaw_as_windows(tmp_path, monkeypatch)
    assert "$env:OPENCLAW_WORKSPACE_DIR = (Get-Location).Path" in result.output
    assert "$env:OPENCLAW_CONFIG_PATH" in result.output
    assert "export OPENCLAW_CONFIG_PATH" not in result.output


def test_windows_recipe_still_writes_memory_search(fake_studio, tmp_path, monkeypatch):
    _run_openclaw_as_windows(tmp_path, monkeypatch)
    assert _config(tmp_path)["memory"]["search"]["provider"] == "openai-compatible"


def test_posix_recipe_is_a_single_runnable_line(fake_studio, tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX recipe")
    result = _run_openclaw(tmp_path, monkeypatch)
    last = [line for line in result.output.splitlines() if line.strip()][-1]
    assert "openclaw" in last
    assert "OPENCLAW_CONFIG_PATH=" in last


def test_wsl_bridge_keeps_memory_search(fake_studio, tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("WSL scenario")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(
        start.shutil, "which", lambda _: "/mnt/c/Users/x/AppData/Roaming/npm/openclaw"
    )
    _run_openclaw(tmp_path, monkeypatch)
    assert _config(tmp_path)["memory"]["search"]["provider"] == "openai-compatible"


def test_config_path_is_absolute_on_every_platform(fake_studio, tmp_path, monkeypatch):
    result = _run_openclaw(tmp_path, monkeypatch)
    assert "openclaw.json" in result.output


# --------------------------------------------------------------------------------------
# H9. Persist reuse. The written model string is resolved once, at connect time.
# --------------------------------------------------------------------------------------

def test_settings_change_between_runs_rewrites_the_model(fake_studio, tmp_path, monkeypatch):
    _override_embedding_route(monkeypatch, _value({"embedding_model": "org/model-a"}))
    _run_openclaw(tmp_path, monkeypatch)
    assert _config(tmp_path)["memory"]["search"]["model"] == "org/model-a"
    _override_embedding_route(monkeypatch, _value({"embedding_model": "org/model-b"}))
    _run_openclaw(tmp_path, monkeypatch)
    assert _config(tmp_path)["memory"]["search"]["model"] == "org/model-b"


def test_yolo_and_plain_both_carry_memory_search(fake_studio, tmp_path, monkeypatch):
    _run_openclaw(tmp_path, monkeypatch, "--yolo")
    assert _config(tmp_path)["memory"]["search"]["provider"] == "openai-compatible"
    _run_openclaw(tmp_path, monkeypatch)
    assert _config(tmp_path)["memory"]["search"]["provider"] == "openai-compatible"


def test_write_openclaw_config_default_kwarg_is_still_accepted(tmp_path):
    """H10. Every existing unit caller omits the new kwarg; that must keep working."""
    path = tmp_path / "openclaw.json"
    start.write_openclaw_config(BASE, "sk-unsloth-abc", MODEL, path)
    config = json.loads(path.read_text(encoding = "utf-8"))
    assert "search" in config.get("memory", {})
