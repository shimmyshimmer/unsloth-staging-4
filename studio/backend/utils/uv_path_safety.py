# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hand uv a space-free constraints / requirements / override file path.

uv's `-c`/`--constraint` and `--override`/`UV_OVERRIDE` arguments split their
value on whitespace, so a path containing a space truncates at the first space
(https://github.com/unslothai/unsloth/issues/6503,
https://github.com/astral-sh/uv/issues/12639). `-r`/`--requirement` does not
split, but routing it through here too is harmless and keeps every uv file-path
call site uniform.

On Windows resolve the 8.3 short form. macOS/Linux have no 8.3 equivalent, so
copy the small flat file into a space-free temp dir and point uv at the copy;
the temp dirs are removed at process exit. Any error falls back to the original
path, so this is never worse than passing the path through unchanged.

This is the single chokepoint shared by install_python_stack.py (the `-c`/`-r`
call sites and the macOS-arm64 `UV_OVERRIDE`) and utils.mlx_repair (the MLX
self-heal `UV_OVERRIDE`).
"""

from __future__ import annotations

import atexit
import os
import platform
import shutil
import tempfile

IS_WINDOWS = platform.system() == "Windows"

# Space-free temp copies handed to uv on POSIX; removed at process exit.
_UV_SAFE_PATH_TMPDIRS: list[str] = []


@atexit.register
def _cleanup_uv_safe_path_tmpdirs() -> None:
    while _UV_SAFE_PATH_TMPDIRS:
        shutil.rmtree(_UV_SAFE_PATH_TMPDIRS.pop(), ignore_errors = True)


def uv_safe_path(path: object) -> str:
    # uv 0.11.x truncates a `-c`/`--override` path at the first space, so hand it
    # a space-free path instead (https://github.com/unslothai/unsloth/issues/6503).
    s = str(path)
    if " " not in s:
        return s
    if IS_WINDOWS:
        # Windows: resolve to the 8.3 short form, no temp copy needed.
        try:
            import ctypes
            from ctypes import wintypes

            get_short = ctypes.windll.kernel32.GetShortPathNameW
            get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            get_short.restype = wintypes.DWORD
            buf = ctypes.create_unicode_buffer(32768)
            rc = get_short(s, buf, 32768)
            if 0 < rc < 32768 and " " not in buf.value:
                return buf.value
        except Exception:
            pass
        return s
    # macOS/Linux have no 8.3 equivalent: copy the constraints/requirements file
    # into a space-free temp dir and point uv there. On any error, return s.
    tmp_dir = None
    try:
        if not os.path.isfile(s):
            return s
        tmp_dir = tempfile.mkdtemp(prefix = "unsloth_uv_")
        if " " in tmp_dir:  # extremely unusual (e.g. TMPDIR has a space)
            shutil.rmtree(tmp_dir, ignore_errors = True)
            return s
        dst = os.path.join(tmp_dir, (os.path.basename(s) or "uv_args.txt").replace(" ", "_"))
        shutil.copyfile(s, dst)
        _UV_SAFE_PATH_TMPDIRS.append(tmp_dir)
        tmp_dir = None
        return dst
    except Exception:
        # Don't leak the just-created temp dir if the copy (or anything after
        # mkdtemp) failed before we registered it for cleanup.
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors = True)
        return s
