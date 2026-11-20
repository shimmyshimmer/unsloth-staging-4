"""Which studio_regress code a run is: a content digest, and the files that differ from scripts main.

A workspace keeps its own copy of studio_regress. One that was not refreshed kept exiting 3 on UI
noise after scripts #880 fixed that on main, and nothing in its output said the code was old.
`run` now logs one line, and records it in the report as `source`:

    source: studio_regress 3f2a9c1d07be (/ws): same as scripts main
    source: studio_regress 81c0d2e4a5f6 (/ws): STALE or modified, 3 files differ from scripts main
            (run.py, diff.py, journeys/crawl.py): a stale copy gives verdicts main may already have fixed; ...

Comparison is by git blob SHA (computed locally, no git needed) against scripts main's
`scripts/studio_regress` tree and `scripts/studio_regress.py`. Main's listing is one `gh api` call,
cached for CACHE_S (6 h) under $WORKSPACE/temp/studio_regress/, so a run costs no network most of
the time; when gh is missing or fails, only the digest is logged. Off with
STUDIO_REGRESS_SOURCE_CHECK=0, and never on a remote VM (STUDIO_REGRESS_PIN_SHAS: no gh there).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = "unslothai/scripts"
REF = "main"
CACHE_S = 6 * 3600
FETCH_TIMEOUT_S = 20


def blob_sha(data: bytes) -> str:
    """git's blob SHA-1 of `data` (what `git hash-object` prints)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _tracked(rel: str) -> bool:
    """Files that decide what a run does: code and config, not tests or caches."""
    parts = rel.split("/")
    return (
        "__pycache__" not in parts
        and not parts[-1].startswith("test_")
        and parts[-1] != "conftest.py"
        and Path(rel).suffix in (".py", ".toml", ".json")
    )


def local_blobs(scripts = SCRIPTS) -> dict:
    """{path relative to scripts/: blob sha} for studio_regress.py and the package."""
    scripts = Path(scripts)
    out = {}
    top = scripts / "studio_regress.py"
    if top.is_file():
        out["studio_regress.py"] = blob_sha(top.read_bytes())
    pkg = scripts / "studio_regress"
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(scripts).as_posix()
        if p.is_file() and _tracked(rel):
            out[rel] = blob_sha(p.read_bytes())
    return out


def digest(blobs: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(blobs):
        h.update(f"{k}\0{blobs[k]}\n".encode())
    return h.hexdigest()[:12]


def _gh_json(path):
    r = subprocess.run(["gh", "api", path], capture_output = True, text = True, timeout = FETCH_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:200] or f"gh api exit {r.returncode}")
    return json.loads(r.stdout)


def fetch_main_blobs(gh_json = _gh_json) -> dict:
    """{path relative to scripts/: blob sha} on scripts main (two gh api calls)."""
    tree = gh_json(f"repos/{REPO}/git/trees/{REF}:scripts/studio_regress?recursive=1")
    out = {
        f"studio_regress/{e['path']}": e["sha"]
        for e in tree.get("tree") or []
        if e.get("type") == "blob" and _tracked(f"studio_regress/{e['path']}")
    }
    top = gh_json(f"repos/{REPO}/contents/scripts/studio_regress.py?ref={REF}")
    if top.get("sha"):
        out["studio_regress.py"] = top["sha"]
    if not out:
        raise RuntimeError("scripts main listed no studio_regress files")
    return out


def cache_file() -> Path:
    base = os.environ.get("STUDIO_REGRESS_SOURCE_CACHE")
    if base:
        return Path(base)
    return (
        Path(os.environ.get("WORKSPACE") or SCRIPTS)
        / "temp"
        / "studio_regress"
        / "scripts_main_blobs.json"
    )


def main_blobs(
    fetch = fetch_main_blobs,
    cache = None,
    now = None,
    max_age_s = CACHE_S,
):
    """Main's blobs from the cache when it is younger than max_age_s, else fetched (and cached).
    -> (blobs, age_s) or (None, why)."""
    cache = Path(cache) if cache else cache_file()
    now = time.time() if now is None else now
    try:
        got = json.loads(cache.read_text())
        if now - float(got["t"]) < max_age_s and got.get("blobs"):
            return got["blobs"], now - float(got["t"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        blobs = fetch()
    except Exception as e:  # noqa: BLE001 - a version check never fails a run
        return None, f"{type(e).__name__}: {e}"[:200]
    try:
        cache.parent.mkdir(parents = True, exist_ok = True)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"t": now, "blobs": blobs}))
        os.replace(tmp, cache)
    except OSError:
        pass
    return blobs, 0.0


def enabled() -> bool:
    if os.environ.get("STUDIO_REGRESS_SOURCE_CHECK", "1") == "0" or os.environ.get(
        "STUDIO_REGRESS_PIN_SHAS"
    ):
        return False
    return shutil.which("gh") is not None


def check(
    scripts = SCRIPTS,
    fetch = fetch_main_blobs,
    cache = None,
    now = None,
    compare = None,
    log = print,
) -> dict:
    """Log and return {"digest", "path", "differs": [...] | None, "stale": bool | None, "note"}.
    `compare` None = enabled()."""
    mine = local_blobs(scripts)
    rec = {
        "digest": digest(mine),
        "path": str(Path(scripts).resolve()),
        "differs": None,
        "stale": None,
        "note": "",
    }
    if compare is None:
        compare = enabled()
    if compare:
        main, age = main_blobs(fetch = fetch, cache = cache, now = now)
        if main is None:
            rec["note"] = f"not compared with scripts main ({age})"
        else:
            differs = sorted(k for k in set(main) | set(mine) if main.get(k) != mine.get(k))
            rec.update(differs = differs, stale = bool(differs), main_age_s = round(age))
    head = f"source: studio_regress {rec['digest']} ({rec['path']})"
    if rec["stale"]:
        names = ", ".join(k.removeprefix("studio_regress/") for k in rec["differs"][:8])
        more = f", +{len(rec['differs']) - 8} more" if len(rec["differs"]) > 8 else ""
        log(
            f"{head}: STALE or modified, {len(rec['differs'])} files differ from scripts main ({names}{more}): "
            "a stale copy gives verdicts main may already have fixed; refresh it"
        )
    elif rec["stale"] is False:
        log(f"{head}: same as scripts main")
    else:
        log(f"{head}" + (f": {rec['note']}" if rec["note"] else ""))
    return rec
