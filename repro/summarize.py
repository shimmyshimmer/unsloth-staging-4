"""Render the per-OS repro JSON as a markdown table for the CI step summary."""

import glob
import json
import sys
from pathlib import Path


def rows(label, payload):
    if isinstance(payload, list):
        for item in payload:
            yield from rows(label, item)
        return
    if not isinstance(payload, dict):
        return
    engine = payload.get("engine") or payload.get("driver") or "native"
    if payload.get("channel"):
        engine = f"{engine}/{payload['channel']}"
    if payload.get("webkit_version"):
        engine = f"WebKitGTK {payload['webkit_version']}"
    if payload.get("nsScrollerStyle"):
        engine = f"WKWebView ({payload['nsScrollerStyle']})"
    if payload.get("error"):
        yield [label, engine, "ERROR", payload["error"][:80], "", "", "", ""]
        return
    before = payload.get("gearBefore") or {}
    after = payload.get("gearAfter") or {}
    stable = payload.get("gearStable") or {}
    yield [
        label,
        engine,
        str(payload.get("layoutScrollbarPx")),
        f"{before.get('hit')}/{before.get('total')}",
        str(payload.get("measuredGutter")),
        f"{after.get('hit')}/{after.get('total')}",
        f"{stable.get('hit')}/{stable.get('total')}",
        "YES" if payload.get("reproduced") else "no",
    ]


def main():
    label = sys.argv[1]
    patterns = sys.argv[2:] or ["repro-out-*.json"]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))

    header = [
        "run", "engine", "layout px", "gear clickable (before)",
        "measured gutter", "gear clickable (after)", "gear in scrollbar-gutter:stable", "reproduced",
    ]
    print(f"## PR 8031 overlay-scrollbar repro - {label}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    any_row = False
    for f in files:
        try:
            payload = json.loads(Path(f).read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"| {f} | - | parse error | {exc!r} | | | | |")
            continue
        stem = Path(f).stem.replace("repro-out-", "")
        for row in rows(stem, payload):
            any_row = True
            print("| " + " | ".join(str(c) for c in row) + " |")
    if not any_row:
        print("| (no results) | | | | | | | |")

    print(
        "\n`reproduced` = the scrollbar takes 0 layout width **and** the gear's "
        "right-hand pixel columns hit-test to the scroller instead of the button."
    )


if __name__ == "__main__":
    main()
