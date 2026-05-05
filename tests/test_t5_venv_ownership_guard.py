"""_ensure_venv_dir must refuse to wipe a non-empty directory under a
custom UNSLOTH_STUDIO_HOME unless the dir or root is sentinel-marked.
Mirrors the install.sh / setup.sh ownership semantics on the runtime
T5-sidecar repair path (studio/backend/utils/transformers_version.py)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TV_PATH = REPO_ROOT / "studio" / "backend" / "utils" / "transformers_version.py"


def _load_transformers_version(custom_root: Path):
    """Load transformers_version.py with stubs so module-level imports
    (loggers, structlog, utils.* helpers) do not require the full Studio
    backend; studio_root() is monkeypatched to ``custom_root``."""
    fake_loggers = types.ModuleType("loggers")
    fake_loggers.get_logger = lambda name: types.SimpleNamespace(
        warning = lambda *a, **k: None,
        info = lambda *a, **k: None,
        error = lambda *a, **k: None,
    )
    fake_structlog = types.ModuleType("structlog")
    fake_native = types.ModuleType("utils.native_path_leases")
    fake_native.child_env_without_native_path_secret = lambda: {}
    fake_subprocess_compat = types.ModuleType("utils.subprocess_compat")
    fake_subprocess_compat.windows_hidden_subprocess_kwargs = lambda: {}
    fake_paths = types.ModuleType("utils.paths")
    fake_storage_roots = types.ModuleType("utils.paths.storage_roots")
    fake_storage_roots.studio_root = lambda: custom_root
    fake_utils = types.ModuleType("utils")

    overrides = {
        "loggers": fake_loggers,
        "structlog": fake_structlog,
        "utils": fake_utils,
        "utils.paths": fake_paths,
        "utils.paths.storage_roots": fake_storage_roots,
        "utils.native_path_leases": fake_native,
        "utils.subprocess_compat": fake_subprocess_compat,
    }
    saved = {k: sys.modules.get(k) for k in overrides}
    for k, v in overrides.items():
        sys.modules[k] = v
    try:
        sys.modules.pop("tv_under_test", None)
        spec = importlib.util.spec_from_file_location("tv_under_test", TV_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tv_under_test"] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def test_ensure_venv_dir_refuses_to_wipe_unowned_custom_root_dir(tmp_path):
    custom_root = tmp_path / "custom_studio"
    custom_root.mkdir()
    venv_dir = custom_root / ".venv_t5_530"
    venv_dir.mkdir()
    (venv_dir / "important.txt").write_text("user data")
    mod = _load_transformers_version(custom_root)
    with mock.patch.object(mod, "_venv_dir_is_valid", return_value = False):
        result = mod._ensure_venv_dir(str(venv_dir), ("transformers==5.3.0",), "t5_530")
    assert result is False
    assert (venv_dir / "important.txt").exists()


def test_ensure_venv_dir_proceeds_when_marker_is_present(tmp_path):
    custom_root = tmp_path / "custom_studio"
    custom_root.mkdir()
    venv_dir = custom_root / ".venv_t5_530"
    venv_dir.mkdir()
    (venv_dir / ".unsloth-studio-owned").write_text("")
    (venv_dir / "stale.txt").write_text("stale")
    mod = _load_transformers_version(custom_root)
    with mock.patch.object(mod, "_venv_dir_is_valid", return_value = False), \
            mock.patch.object(mod, "_install_to_dir", return_value = True):
        result = mod._ensure_venv_dir(str(venv_dir), ("transformers==5.3.0",), "t5_530")
    assert result is True
    assert not (venv_dir / "stale.txt").exists()
    assert (venv_dir / ".unsloth-studio-owned").is_file()


def test_ensure_venv_dir_proceeds_when_share_studio_conf_present(tmp_path):
    custom_root = tmp_path / "custom_studio"
    custom_root.mkdir()
    (custom_root / "share").mkdir()
    (custom_root / "share" / "studio.conf").write_text("")
    venv_dir = custom_root / ".venv_t5_530"
    venv_dir.mkdir()
    (venv_dir / "stale.txt").write_text("stale")
    mod = _load_transformers_version(custom_root)
    with mock.patch.object(mod, "_venv_dir_is_valid", return_value = False), \
            mock.patch.object(mod, "_install_to_dir", return_value = True):
        result = mod._ensure_venv_dir(str(venv_dir), ("transformers==5.3.0",), "t5_530")
    assert result is True
    assert not (venv_dir / "stale.txt").exists()


def test_ensure_venv_dir_proceeds_under_legacy_default_root(tmp_path):
    legacy = Path.home() / ".unsloth" / "studio"
    venv_dir = tmp_path / ".venv_t5_530"
    venv_dir.mkdir()
    (venv_dir / "stale.txt").write_text("stale")
    mod = _load_transformers_version(legacy)
    with mock.patch.object(mod, "_venv_dir_is_valid", return_value = False), \
            mock.patch.object(mod, "_install_to_dir", return_value = True):
        result = mod._ensure_venv_dir(str(venv_dir), ("transformers==5.3.0",), "t5_530")
    assert result is True
    assert not (venv_dir / "stale.txt").exists()


def test_ensure_venv_dir_proceeds_when_target_does_not_exist(tmp_path):
    custom_root = tmp_path / "custom_studio"
    custom_root.mkdir()
    venv_dir = custom_root / ".venv_t5_530"
    mod = _load_transformers_version(custom_root)
    with mock.patch.object(mod, "_venv_dir_is_valid", return_value = False), \
            mock.patch.object(mod, "_install_to_dir", return_value = True):
        result = mod._ensure_venv_dir(str(venv_dir), ("transformers==5.3.0",), "t5_530")
    assert result is True
    assert venv_dir.is_dir()
    assert (venv_dir / ".unsloth-studio-owned").is_file()
