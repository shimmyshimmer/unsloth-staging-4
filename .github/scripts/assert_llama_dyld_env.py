# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Assert the real llama-server child environment on macOS (unslothai/unsloth#8566).

Runs against an actually-installed llama.cpp runtime, not a monkeypatched
sys.platform, so it can tell whether _llama_server_env_for_binary hands dyld a
usable search path. Exits non-zero with a diagnosis when it does not, which is
what the pre-fix revision of llama_cpp.py is expected to do.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

sys.path.insert(0, "studio/backend")

from core.inference.llama_cpp import LlamaCppBackend, _llama_lib_dir  # noqa: E402

failures: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


binary = LlamaCppBackend._find_llama_server_binary()
if not binary:
    sys.exit("no llama-server found; the install step did not produce a runtime")

lib_dir = str(_llama_lib_dir(binary))
env = LlamaCppBackend._llama_server_env_for_binary(binary)
dyld = env.get("DYLD_LIBRARY_PATH", "")

print(f"binary            : {binary}")
print(f"resolved lib dir  : {lib_dir}")
print(f"DYLD_LIBRARY_PATH : {dyld or '<unset>'}")
print(f"LD_LIBRARY_PATH   : {env.get('LD_LIBRARY_PATH', '<unset>')}")
print()

check(bool(dyld), "the child env sets DYLD_LIBRARY_PATH at all")
check(
    bool(dyld) and dyld.split(os.pathsep)[0] == lib_dir,
    "the runtime's own lib dir comes first on DYLD_LIBRARY_PATH",
)
check(
    bool(list(pathlib.Path(lib_dir).glob("libllama*.dylib"))),
    "that dir really holds the llama.cpp dylibs",
)

# A child process must actually receive it. Use this interpreter (a runner tool
# install, so not SIP-protected -- macOS strips DYLD_* when exec'ing anything
# under /bin, /usr/bin or /System, which a shell wrapper would hit).
probe = subprocess.run(
    [sys.executable, "-c", "import os;print(os.environ.get('DYLD_LIBRARY_PATH',''))"],
    env = env,
    capture_output = True,
    text = True,
)
child_dyld = probe.stdout.strip()
print(f"\nchild DYLD_LIBRARY_PATH: {child_dyld or '<unset>'}")
check(
    bool(child_dyld) and child_dyld.split(os.pathsep)[0] == lib_dir,
    "a spawned child inherits it",
)

# The runtime must still start with this environment.
version = subprocess.run(
    [binary, "--version"], env = env, capture_output = True, text = True, timeout = 120
)
print(f"\nllama-server --version rc={version.returncode}")
print((version.stdout + version.stderr).strip()[:2000])
check(version.returncode == 0, "llama-server still starts with this environment")

print()
if failures:
    sys.exit(f"{len(failures)} check(s) failed: {failures}")
print("all checks passed")
