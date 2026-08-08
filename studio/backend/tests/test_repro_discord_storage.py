# Repro tests for the Discord report about Unsloth Studio file locations.
#
# These are DIAGNOSTIC tests: they assert the CURRENT (buggy) behaviour so the
# same file can be run on Windows / macOS / Linux runners and show that the
# problem is platform independent (paths) or platform specific (cleanup).
#
# Run: PYTHONPATH=studio/backend python -m pytest studio/backend/tests/test_repro_discord_storage.py -q -s

import os
import sys
import platform
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. The tool sandbox ignores UNSLOTH_STUDIO_HOME and always lands in ~
# ---------------------------------------------------------------------------
def test_sandbox_root_ignores_studio_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "userprofile"
    studio_home = tmp_path / "custom_studio_home"
    fake_home.mkdir()
    studio_home.mkdir()

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))

    from core.inference import tools

    tools._workdirs.clear()
    wd = Path(tools.get_sandbox_workdir("__LOCALID_aB3xY7q"))
    print(f"\n[{platform.system()}] sandbox workdir = {wd}")

    assert wd.parent.name == "studio_sandbox"
    # The bug: it is under the user home, not under the configured studio home.
    assert str(wd).startswith(str(fake_home)), wd
    assert not str(wd).startswith(str(studio_home)), (
        "sandbox now honours UNSLOTH_STUDIO_HOME -- update this repro"
    )
    # The directory is named after an internal frontend thread id.
    assert wd.name.startswith("__LOCALID_")


# ---------------------------------------------------------------------------
# 2. Only images can be fetched back out of the sandbox
# ---------------------------------------------------------------------------
def test_only_images_are_servable():
    from routes.inference import _SANDBOX_MEDIA_TYPES

    print(f"\nservable extensions = {sorted(_SANDBOX_MEDIA_TYPES)}")
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        assert ext in _SANDBOX_MEDIA_TYPES
    # Everything a user would actually want to download is rejected with 403.
    for ext in (".csv", ".txt", ".py", ".json", ".pdf", ".zip", ".xlsx", ".md"):
        assert ext not in _SANDBOX_MEDIA_TYPES, ext


def test_no_sandbox_listing_route():
    """Nothing can enumerate a session sandbox, so the UI cannot show its files."""
    from routes.inference import router

    sandbox_routes = [r.path for r in router.routes if "sandbox" in r.path]
    print(f"\nsandbox routes = {sandbox_routes}")
    assert sandbox_routes == ["/sandbox/{session_id}/{filename}"]


# ---------------------------------------------------------------------------
# 3. Only _python_exec reports created files, and only images
# ---------------------------------------------------------------------------
def test_only_python_exec_reports_created_files():
    from core.inference import tools

    src_py = tools._python_exec.__doc__ or ""
    import inspect

    py_src = inspect.getsource(tools._python_exec)
    bash_src = inspect.getsource(tools._bash_exec)

    assert "__IMAGES__" in py_src, "python exec no longer emits the image sentinel"
    assert "__IMAGES__" not in bash_src, "bash exec now reports files -- update this repro"
    assert "__FILES__" not in py_src and "__FILES__" not in bash_src, (
        "a non-image file sentinel now exists -- update this repro"
    )
    del src_py


def test_tool_description_never_names_the_working_directory():
    from core.inference import tools

    note = tools._SANDBOX_PATHS_NOTE
    print(f"\nsandbox note = {note!r}")
    assert "studio_sandbox" not in note
    assert "current working directory" in note.lower()


# ---------------------------------------------------------------------------
# 4. The compiled cache is CWD-relative and the cleaner cannot see it
# ---------------------------------------------------------------------------
def test_compiled_cache_default_is_cwd_relative():
    default = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
    assert not os.path.isabs(default)
    resolved = Path(os.path.abspath(default))
    print(f"\n[{platform.system()}] compiled cache would land in {resolved}")
    # On Windows the launcher starts Studio with -WorkingDirectory $env:USERPROFILE
    # (install.ps1), so this resolves inside the user profile.
    assert resolved.parent == Path.cwd()


def test_cache_cleanup_only_knows_repo_relative_dirs(tmp_path, monkeypatch):
    from utils import cache_cleanup

    # Studio is launched with CWD = user profile on Windows (install.ps1), so
    # stand somewhere that is not the source tree.
    monkeypatch.chdir(tmp_path)

    dirs = [str(d) for d in cache_cleanup._CACHE_DIRS]
    print("\ncache_cleanup dirs:")
    for d in dirs:
        print("   ", d)
    # Every candidate is derived from the source tree location, so a cache
    # created in the launcher's CWD (the user profile on Windows) is invisible.
    repo_root = Path(cache_cleanup.__file__).resolve().parents[3]
    assert all(str(repo_root) in d for d in dirs), dirs
    assert all(Path(os.path.abspath("unsloth_compiled_cache")) != Path(d) for d in dirs)
    src = Path(cache_cleanup.__file__).read_text(encoding="utf-8")
    assert "UNSLOTH_COMPILE_LOCATION" not in src, (
        "cleanup now honours the env var -- update this repro"
    )
    assert "getcwd" not in src


# ---------------------------------------------------------------------------
# 5. Deleting a chat leaves its sandbox behind
# ---------------------------------------------------------------------------
def test_thread_delete_does_not_touch_the_filesystem():
    import inspect
    from storage import studio_db

    src = inspect.getsource(studio_db.delete_chat_threads)
    print("\ndelete_chat_threads source:\n" + src)
    for token in ("rmtree", "shutil", "os.remove", "unlink", "sandbox"):
        assert token not in src, f"{token} now referenced -- update this repro"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
