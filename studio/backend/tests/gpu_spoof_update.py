#!/usr/bin/env python3
"""Spoofed NVIDIA / AMD GPU validation of the in-app update path.

GitHub runners have no GPU, so we spoof the host profile and exercise the REAL
code two ways (no GPU binary is executed):

  A. Asset selection: run install_llama_prebuilt.py's real host -> asset choice
     against the real b9585 manifest, for each spoofed NVIDIA / AMD host. This is
     the bundle the update would download per GPU.
  B. Command construction: drive utils.llama_cpp_update._run_update (installer
     subprocess stubbed) for each platform's marker asset, asserting the update
     re-runs the installer with the right --published-repo and ROCm forwarding
     (--rocm-gfx gfxNNN for gfx bundles, --has-rocm for hip / version bundles),
     and never --simple-policy / --cpu-fallback.

Runs identically on ubuntu-latest and windows-latest (host is constructed, not
detected), proving the GPU update paths build correctly regardless of runner OS.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDIO = ROOT / "studio"
INSTALLER = STUDIO / "install_llama_prebuilt.py"
BACKEND = STUDIO / "backend"
TAG = "b9585"
U = "unslothai/llama.cpp"
G = "ggml-org/llama.cpp"

failures: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""), flush=True)
    if not cond:
        failures.append(label)
    return cond


# ---- load installer module from file ----
spec = importlib.util.spec_from_file_location("ilp", INSTALLER)
ilp = importlib.util.module_from_spec(spec)
sys.modules["ilp"] = ilp
spec.loader.exec_module(ilp)

# Neutralize the torch-cuda-runtime preference so this measures the host-driven
# (driver + compute cap) asset choice deterministically. That reorder matches
# the bundle's CUDA runtime to the local torch and is validated separately; it
# would otherwise make the choice depend on whatever torch the runner has. On
# the CI runner (structlog only, no torch) it is already None.
ilp.detect_torch_cuda_runtime_preference = lambda host: ilp.CudaRuntimePreference(
    runtime_line=None, selection_log=[]
)

# Linux CUDA selection (linux_cuda_choice_from_release) probes the CUDA runtime
# libraries physically present on the machine and only offers a cuda line the
# host can actually run. A GPU-less CI runner has none, so it returns None. We
# are spoofing an NVIDIA host, which by definition has the CUDA runtime, so spoof
# the probe to expose both lines; driver + SM bounds then decide cuda13 vs cuda12.
# (Windows selection keys off the driver version only, so it needs no spoof.)
ilp.detected_linux_runtime_lines = lambda: (
    ["cuda13", "cuda12"],
    {"cuda13": ["<spoofed-runtime>"], "cuda12": ["<spoofed-runtime>"]},
)


def host(**kw):
    base = dict(
        system="Linux", machine="x86_64",
        is_windows=False, is_linux=False, is_macos=False,
        is_x86_64=False, is_arm64=False,
        nvidia_smi=None, driver_cuda_version=None, compute_caps=[],
        visible_cuda_devices=None, has_physical_nvidia=False, has_usable_nvidia=False,
        has_rocm=False, rocm_gfx_target=None, macos_version=None,
    )
    base.update(kw)
    return ilp.HostInfo(**base)


def select(h, repo):
    res = next(iter(ilp.iter_resolved_published_releases(TAG, repo, TAG)))
    bundle, checksums = res.bundle, res.checksums
    if h.is_linux:
        attempts = ilp.apply_approved_hashes(
            ilp._linux_published_attempts(h, bundle, "latest"), checksums
        )
    else:
        attempts = ilp.resolve_release_asset_choice(
            h, bundle.upstream_tag, bundle, checksums, requested_tag="latest"
        )
    return attempts[0].name if attempts else None


# ============ A. spoofed-host asset selection (real manifest) ============
print(f"=== A. spoofed NVIDIA/AMD host -> real {TAG} asset selection ===", flush=True)
A_CASES = [
    ("Linux NVIDIA x64 B200 sm_100 drv13.1",
     host(system="Linux", machine="x86_64", is_linux=True, is_x86_64=True,
          nvidia_smi="x", driver_cuda_version=(13, 1), compute_caps=["10.0"],
          has_physical_nvidia=True, has_usable_nvidia=True), U, "linux-x64-cuda13"),
    ("Linux NVIDIA arm64 Spark sm_121 drv13.0",
     host(system="Linux", machine="aarch64", is_linux=True, is_arm64=True,
          nvidia_smi="x", driver_cuda_version=(13, 0), compute_caps=["12.1"],
          has_physical_nvidia=True, has_usable_nvidia=True), U, "linux-arm64-cuda13"),
    ("Linux AMD ROCm gfx110X (RDNA3)",
     host(system="Linux", machine="x86_64", is_linux=True, is_x86_64=True,
          has_rocm=True, rocm_gfx_target="gfx110x"), U, "rocm-gfx110X"),
    ("Linux AMD ROCm gfx120X (RDNA4)",
     host(system="Linux", machine="x86_64", is_linux=True, is_x86_64=True,
          has_rocm=True, rocm_gfx_target="gfx120x"), U, "rocm-gfx120X"),
    ("Linux AMD ROCm gfx1151 (Strix Halo)",
     host(system="Linux", machine="x86_64", is_linux=True, is_x86_64=True,
          has_rocm=True, rocm_gfx_target="gfx1151"), U, "rocm-gfx1151"),
    ("Windows NVIDIA x64 Ada sm_89 drv13.1",
     host(system="Windows", machine="AMD64", is_windows=True, is_x86_64=True,
          nvidia_smi="x", driver_cuda_version=(13, 1), compute_caps=["8.9"],
          has_physical_nvidia=True, has_usable_nvidia=True), U, "windows-x64-cuda13"),
    ("Windows NVIDIA x64 Blackwell sm_120 drv13.1",
     host(system="Windows", machine="AMD64", is_windows=True, is_x86_64=True,
          nvidia_smi="x", driver_cuda_version=(13, 1), compute_caps=["12.0"],
          has_physical_nvidia=True, has_usable_nvidia=True), U, "windows-x64-cuda13"),
    ("Windows AMD ROCm gfx110X",
     host(system="Windows", machine="AMD64", is_windows=True, is_x86_64=True,
          has_rocm=True, rocm_gfx_target="gfx110x"), U, "rocm-gfx110X"),
    ("Windows AMD ROCm gfx1151",
     host(system="Windows", machine="AMD64", is_windows=True, is_x86_64=True,
          has_rocm=True, rocm_gfx_target="gfx1151"), U, "rocm-gfx1151"),
]
for label, h, repo, expect in A_CASES:
    try:
        name = select(h, repo)
        ok = name is not None and expect in name
    except Exception as exc:  # noqa: BLE001
        name, ok = f"ERROR: {exc}", False
    check(ok, f"{label:42s} -> {expect}", f"({name})")


# ============ B. update command construction (installer stubbed) ============
print(f"\n=== B. update re-runs installer with correct args per GPU ===", flush=True)
# Fail-open backend stub so _run_update skips load coordination.
_rp = types.ModuleType("routes"); _rp.__path__ = []
_ri = types.ModuleType("routes.inference")
_ri.get_llama_cpp_backend = lambda: (_ for _ in ()).throw(RuntimeError("no backend"))
sys.modules["routes"] = _rp
sys.modules["routes.inference"] = _ri
sys.path.insert(0, str(BACKEND))
import utils.llama_cpp_update as upd  # noqa: E402

SCRIPT = STUDIO / "install_llama_prebuilt.py"  # value only; never executed


def capture_cmd(tmp: Path, repo: str, asset):
    install_dir = tmp / "llama.cpp"
    bindir = install_dir / "build" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    binary = bindir / "llama-server"
    binary.write_text("stub")
    marker = {"tag": "b9000", "release_tag": "b9000", "published_repo": repo,
              "installed_at_utc": "2020-01-01T00:00:00Z"}
    if asset is not None:
        marker["asset"] = asset
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(marker))
    upd._find_binary = lambda b=str(binary): b
    captured = {}

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        return _Proc()

    upd.subprocess.run = _fake_run
    upd._run_update(install_dir, repo, asset, SCRIPT)
    return captured["cmd"]


B_CASES = [
    ("Linux NVIDIA cuda13", U, "app-b9585-linux-x64-cuda13-newer.tar.gz", []),
    ("Linux AMD gfx110X (app bundle)", U, "app-b9585-linux-x64-rocm-gfx110X.tar.gz", ["--rocm-gfx", "gfx110x"]),
    ("Linux AMD gfx110X (lemonade)", U, "llama-b1292-ubuntu-rocm-gfx110X-x64.zip", ["--rocm-gfx", "gfx110x"]),
    ("Linux AMD gfx1151 (Strix Halo)", U, "app-b9585-linux-x64-rocm-gfx1151.tar.gz", ["--rocm-gfx", "gfx1151"]),
    ("Windows NVIDIA cuda13", U, "app-b9585-windows-x64-cuda13-newer.zip", []),
    ("Windows AMD gfx110X (lemonade)", U, "llama-b1292-windows-rocm-gfx110X-x64.zip", ["--rocm-gfx", "gfx110x"]),
    ("Windows AMD gfx1151 (app bundle)", U, "app-b9585-windows-x64-rocm-gfx1151.zip", ["--rocm-gfx", "gfx1151"]),
    ("Windows AMD HIP radeon (fallback)", U, "llama-b9585-bin-win-hip-radeon-x64.zip", ["--has-rocm"]),
    ("Linux AMD fork rocm-version (fallback)", U, "llama-b9585-bin-ubuntu-rocm-6.4-x64.tar.gz", ["--has-rocm"]),
    ("Windows CPU (ggml-org)", G, "llama-b9585-bin-win-cpu-x64.zip", []),
    ("Older marker, no asset field", U, None, []),
]
for label, repo, asset, expect_extra in B_CASES:
    with tempfile.TemporaryDirectory() as td:
        cmd = capture_cmd(Path(td), repo, asset)
    base_ok = ("--llama-tag" in cmd and "latest" in cmd
               and "--published-repo" in cmd and repo in cmd)
    after_repo = cmd[cmd.index("--published-repo") + 2:]
    extra_ok = after_repo == expect_extra
    clean = "--simple-policy" not in cmd and "--cpu-fallback" not in cmd
    ok = base_ok and extra_ok and clean
    extra_str = " ".join(expect_extra) if expect_extra else "(none)"
    check(ok, f"{label:42s} repo={repo.split('/')[0]:9s} extra={extra_str}",
          "" if ok else f"cmd={cmd}")


print("\n" + "=" * 60, flush=True)
if failures:
    print(f"GPU SPOOF FAIL ({len(failures)}): {failures}", flush=True)
    sys.exit(1)
print("GPU SPOOF PASS: NVIDIA + AMD update asset selection and command "
      "construction correct on Windows and Linux (no GPU executed).", flush=True)
sys.exit(0)
