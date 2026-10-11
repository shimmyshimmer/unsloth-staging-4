# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Selects the diffusion engine (diffusers vs native sd.cpp) for the live route.

One engine at a time. On a CUDA/ROCm/XPU GPU it is the diffusers ``DiffusionBackend`` (the default,
the only path with the torchao fast-quant / compile stack); with no usable GPU (CPU, or MPS when
enabled) the native ``SdCppDiffusionBackend`` (faster, lighter on RAM there). Chosen once at load
and remembered, so ``generate`` / ``unload`` / ``status`` / progress all act on the same engine.

Built on the pure ``select_diffusion_engine`` decision; this module adds the policy (env opt-out,
MPS gating, per-family native-asset support, lazy binary availability) and records fallbacks.

Env knobs:
  UNSLOTH_DIFFUSION_ENGINE=auto|diffusers|sd_cpp   force an engine (auto = decide)
  UNSLOTH_DIFFUSION_SD_CPP=auto|0|1                 enable/disable the native route
  UNSLOTH_DIFFUSION_SD_CPP_MPS=0|1                  allow native on Apple MPS (default off)
  UNSLOTH_DIFFUSION_SD_CPP_INSTALL=auto|0|1         allow lazy binary install (in sd_cpp_backend)
  UNSLOTH_DIFFUSION_SD_CPP_DEVICE=nvidia[:N]        run native on a card torch cannot see
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from core.inference.diffusion_device import resolve_diffusion_device_target
from core.inference.diffusion_families import (
    DiffusionFamily,
    family_pipeline_available,
    family_sd_cpp_supported,
    pipeline_available_family_names,
)
from core.inference.sd_cpp_backend import (
    _card_lookup_inventory,
    _install_allowed,
    _managed_tree_in_use,
    _server_binary_runnable,
    ensure_sd_cpp_binary,
    ensure_sd_server_binary,
    note_unlaunchable_accelerator_build,
    off_torch_build_mismatch,
    preferred_accelerator,
    sd_cpp_binary_runs_family,
    usable_or_recorded_failure,
)
from core.inference.sd_cpp_engine import (
    ENGINE_DIFFUSERS,
    ENGINE_SD_CPP,
    SdCppEngine,
    select_diffusion_engine,
)
from loggers import get_logger

logger = get_logger(__name__)

_DISABLE_TOKENS = frozenset({"0", "off", "false", "no"})
_ENABLE_TOKENS = frozenset({"1", "on", "true", "yes"})

# Force-native GPU loads only: without it the installer defaults to "cpu" and runs on CPU.
_INSTALL_ACCELERATOR = {"rocm": "rocm", "cuda": "cuda", "xpu": "vulkan"}


def _install_accelerator_for(backend: str) -> str:
    return _INSTALL_ACCELERATOR.get(backend, "auto")


_OFF_TORCH_VENDOR_ACCELERATOR = {"nvidia": "cuda"}
_off_torch_warned: set[str] = set()


@dataclass(frozen = True)
class OffTorchDevice:
    """A card torch cannot see. ``index`` is nvidia-smi's row, never a torch ordinal."""

    vendor: str
    index: int
    accelerator: str

    @property
    def label(self) -> str:
        return f"{self.vendor}:{self.index}"

    def child_env(self) -> dict[str, str]:
        # nvidia-smi numbers cards in PCI order; CUDA's default is fastest-first.
        return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(self.index)}


def _warn_off_torch_once(raw: str, message: str, *args: Any) -> None:
    if raw in _off_torch_warned:
        return
    _off_torch_warned.add(raw)
    logger.warning(message, *args)


def _physical_inventory() -> dict:
    return _card_lookup_inventory() or {}


def off_torch_sd_cpp_device(backend: Optional[str] = None) -> Optional[OffTorchDevice]:
    """The card ``UNSLOTH_DIFFUSION_SD_CPP_DEVICE`` asks native image generation to run on, when
    torch cannot drive it (an NVIDIA card beside a ROCm torch). None keeps today's routing: unset,
    unparseable, a vendor torch already serves, or an inventory that answered without that card."""
    raw = os.environ.get("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "").strip().lower()
    if not raw:
        return None
    vendor, _, index_text = raw.partition(":")
    vendor = vendor.strip()
    accelerator = _OFF_TORCH_VENDOR_ACCELERATOR.get(vendor)
    try:
        index = int(index_text) if index_text.strip() else 0
    except ValueError:
        index = -1
    if accelerator is None or index < 0:
        _warn_off_torch_once(
            raw,
            "UNSLOTH_DIFFUSION_SD_CPP_DEVICE=%r is not understood (expected nvidia or "
            "nvidia:<index>); ignoring it",
            raw,
        )
        return None
    if backend is None:
        backend = resolve_diffusion_device_target().backend
    if _INSTALL_ACCELERATOR.get(backend) == accelerator:
        return None
    inventory = _physical_inventory()
    unanswered = set(inventory.get("unanswered") or ())
    if vendor not in unanswered and not (inventory.get("unknown") and not unanswered):
        present = any(
            isinstance(device, dict)
            and device.get("vendor") == vendor
            and device.get("index") == index
            for device in inventory.get("devices") or ()
        )
        if not present:
            _warn_off_torch_once(
                raw,
                "UNSLOTH_DIFFUSION_SD_CPP_DEVICE=%s: this host lists no such card; ignoring it",
                raw,
            )
            return None
    return OffTorchDevice(vendor = vendor, index = index, accelerator = accelerator)


def image_install_accelerator(backend: str) -> str:
    """The sd.cpp build the IMAGE engine wants: the off-torch card's when one is set, else torch's.
    Video keeps ``_install_accelerator_for``, since torch places its loads."""
    off_torch = off_torch_sd_cpp_device(backend)
    return off_torch.accelerator if off_torch is not None else _install_accelerator_for(backend)


_lock = threading.Lock()
# Serializes a whole engine switch; _lock is released during the slow unload().
_transition_lock = threading.Lock()
_active_engine_name: str = ENGINE_DIFFUSERS
_fallback_reason: Optional[str] = None


def _engine_config() -> tuple[str, str, bool]:
    forced = os.environ.get("UNSLOTH_DIFFUSION_ENGINE", "auto").strip().lower()
    sd_cpp = os.environ.get("UNSLOTH_DIFFUSION_SD_CPP", "auto").strip().lower()
    mps = os.environ.get("UNSLOTH_DIFFUSION_SD_CPP_MPS", "0").strip().lower() in _ENABLE_TOKENS
    return forced, sd_cpp, mps


def engine_for(name: str) -> Any:
    """The engine object a name refers to, WITHOUT activating it: activating unloads the resident
    model, so /images/load's gated-repo preflight needs the pending engine before the switch."""
    if name == ENGINE_SD_CPP:
        from core.inference.sd_cpp_backend import get_sd_cpp_backend
        return get_sd_cpp_backend()
    from core.inference.diffusion import get_diffusion_backend

    return get_diffusion_backend()


def get_active_diffusion_engine() -> Any:
    """The engine object the active selection points at (defaults to diffusers)."""
    return engine_for(_active_engine_name)


def cancel_generation_for_account(account_id: str) -> bool:
    """Checks both engines via sys.modules, not imports, since a deselected engine can still be draining."""
    cancelled = False
    for module_name, attribute in (
        ("core.inference.diffusion", "_diffusion_backend"),
        ("core.inference.sd_cpp_backend", "_sd_cpp_backend"),
    ):
        module = sys.modules.get(module_name)
        engine = getattr(module, attribute, None) if module is not None else None
        if engine is None or engine._active_generate_account != account_id:
            continue
        if engine.cancel_generate(expected_account = account_id):
            cancelled = True
    return cancelled


def retire_load_for_account(account_id: str) -> bool:
    """Tear down an in-flight image load ``account_id`` started; True when one was found."""
    from hub.services.models.account_access import retire_media_load

    retired = False
    for module_name, attribute in (
        ("core.inference.diffusion", "_diffusion_backend"),
        ("core.inference.sd_cpp_backend", "_sd_cpp_backend"),
    ):
        module = sys.modules.get(module_name)
        engine = getattr(module, attribute, None) if module is not None else None
        if retire_media_load("diffusion", account_id, engine):
            retired = True
    return retired


def active_engine_name() -> str:
    return _active_engine_name


def _activate(name: str, reason: Optional[str]) -> Any:
    global _active_engine_name, _fallback_reason
    with _transition_lock:
        # Unload the old engine first (else it leaks VRAM), outside _lock since unload is slow.
        engine_to_unload = None
        old_name = None
        with _lock:
            if name != _active_engine_name:
                engine_to_unload = get_active_diffusion_engine()
                old_name = _active_engine_name
            else:
                _fallback_reason = reason if name == ENGINE_DIFFUSERS else None
        if engine_to_unload is not None:
            # Publish only after the old one unloads, or the evictor may evict the new empty engine.
            try:
                engine_to_unload.unload()
            except Exception as exc:
                # Do not publish after a failed teardown: the old model would leak, hidden from the evictor.
                logger.error("failed to unload previous engine %s: %s", old_name, exc)
                raise RuntimeError(
                    f"Could not switch the diffusion engine to {name}: unloading the current "
                    f"{old_name} model failed ({exc}). The current model is still loaded; "
                    "unload it and try again."
                ) from exc
            with _lock:
                _active_engine_name = name
                _fallback_reason = reason if name == ENGINE_DIFFUSERS else None
        if name == ENGINE_SD_CPP:
            logger.info("diffusion engine: sd_cpp")
        else:
            logger.info("diffusion engine: diffusers (%s)", reason or "selected")
        return get_active_diffusion_engine()


def begin_load_on(expected_engine: Any, start: Callable[[], Any]) -> Any:
    """Re-checks the engine under the transition lock; a second load can switch engines in the start gap."""
    with _transition_lock:
        if expected_engine is not get_active_diffusion_engine():
            raise RuntimeError(
                "The diffusion engine changed while this load was starting. Retry the load."
            )
        return start()


def _selected_card(gpu_ordinal) -> Optional[str]:
    """The card at an already RESOLVED ordinal, or ``None``, meaning every record applies. Never
    re-derived from the id list: free-VRAM ranking can name a different card the second time."""
    if gpu_ordinal is None:
        return None
    try:
        from core.inference.sd_cpp_backend import selected_card_identity
        return selected_card_identity(gpu_ordinal)
    except Exception:  # noqa: BLE001
        return None


def select_and_activate_engine(
    fam: DiffusionFamily,
    *,
    hf_token: Optional[str] = None,
    model_kind: Optional[str] = None,
    gpu_ordinal: Optional[int] = None,
    before_fallback: Optional[Callable[[], None]] = None,
) -> Any:
    """Falls back to diffusers before any slow load, so a fallback never strands a half-native load."""
    if model_kind and model_kind != "gguf":
        return _activate(ENGINE_DIFFUSERS, f"non-GGUF load ({model_kind}) requires diffusers")

    forced, sd_cpp_pref, mps_enabled = _engine_config()

    if forced == ENGINE_DIFFUSERS:
        return _activate(ENGINE_DIFFUSERS, "forced (UNSLOTH_DIFFUSION_ENGINE=diffusers)")

    prefer_native = forced == ENGINE_SD_CPP
    if sd_cpp_pref in _DISABLE_TOKENS and not prefer_native:
        return _activate(ENGINE_DIFFUSERS, "native engine disabled (UNSLOTH_DIFFUSION_SD_CPP=0)")

    target = resolve_diffusion_device_target()
    backend = target.backend
    off_torch = off_torch_sd_cpp_device(backend)
    if off_torch is not None:
        prefer_native = True
        gpu_ordinal = None
    policy_eligible = backend == "cpu" or (backend == "mps" and mps_enabled) or prefer_native
    fam_ok = family_sd_cpp_supported(fam)

    binary = None
    server_binary = None
    incapable_build = False
    if policy_eligible and fam_ok:
        selected_card = _selected_card(gpu_ordinal)
        install_accelerator = preferred_accelerator(
            _install_accelerator_for(backend), selected_card
        )
        if off_torch is not None:
            install_accelerator = preferred_accelerator(off_torch.accelerator, selected_card)
        # Offline an ensure returns the condemned ROCm build; a deferred upgrade keeps native.
        upgrade_is_deferred = _managed_tree_in_use() and _install_allowed()

        def _accept(candidate):
            if candidate and upgrade_is_deferred:
                return candidate
            wrong_build = off_torch_build_mismatch(off_torch, candidate)
            if wrong_build:
                logger.warning(
                    "%s is the %s sd.cpp build; %s needs %s",
                    candidate,
                    wrong_build,
                    off_torch.label,
                    off_torch.accelerator,
                )
                return None
            return usable_or_recorded_failure(candidate, install_accelerator, selected_card)

        server_binary = _accept(
            ensure_sd_server_binary(
                allow_install = _install_allowed(),
                accelerator = install_accelerator,
            )
        )
        unlaunchable_server: Optional[str] = None
        if server_binary and not _server_binary_runnable(server_binary):
            logger.warning(
                "sd-server at %s is present but not runnable; not using it", server_binary
            )
            unlaunchable_server = server_binary
            server_binary = None
        # Probe runnability first, else a non-runnable binary fails inside the background load.
        binary = _accept(
            ensure_sd_cpp_binary(
                allow_install = _install_allowed() and server_binary is None,
                accelerator = install_accelerator,
            )
        )
        unlaunchable_cli: Optional[str] = None
        if binary and SdCppEngine(binary = binary).version() is None:
            logger.warning("sd-cli at %s is present but not runnable; not using it", binary)
            unlaunchable_cli = binary
            binary = None
        if binary is None and server_binary is None and (unlaunchable_cli or unlaunchable_server):
            # One strike per bundle, only when neither executable runs.
            note_unlaunchable_accelerator_build(
                unlaunchable_cli or unlaunchable_server, card = selected_card
            )
        # Runnable is not capable: an old build is never upgraded and fails late on new families.
        for name, candidate in (("sd-server", server_binary), ("sd-cli", binary)):
            if candidate and not sd_cpp_binary_runs_family(candidate, fam):
                incapable_build = True
                logger.warning(
                    "%s at %s predates %s support; using diffusers. Reinstall the native engine "
                    "(delete the managed stable-diffusion.cpp install, or point SD_CLI_PATH at a "
                    "build from after upstream added it) to use it here",
                    name,
                    candidate,
                    fam.name,
                )
                if name == "sd-server":
                    server_binary = None
                else:
                    binary = None

    native_available = bool(binary or server_binary) and policy_eligible and fam_ok
    choice = select_diffusion_engine(
        backend, native_available = native_available, prefer_native = prefer_native
    )
    if choice == ENGINE_SD_CPP:
        return _activate(ENGINE_SD_CPP, None)

    if not policy_eligible:
        reason = f"GPU backend '{backend}' uses diffusers"
    elif not fam_ok:
        reason = f"family '{fam.name}' has no native sd.cpp asset mapping"
    elif incapable_build:
        reason = f"the installed sd.cpp build predates '{fam.name}' support"
    elif not (binary or server_binary):
        reason = "native sd.cpp binary unavailable"
    else:
        reason = "diffusers selected"
    if off_torch is not None:
        reason = f"{reason}; UNSLOTH_DIFFUSION_SD_CPP_DEVICE={off_torch.label} not honoured"
        logger.warning("image load stays on torch's device: %s", reason)
    if before_fallback is not None:
        before_fallback()
    return _activate(ENGINE_DIFFUSERS, reason)


def native_binary_installed(
    *, gpu_ordinal: Optional[int] = None, fam: Optional[DiffusionFamily] = None
) -> bool:
    """Answers whether a runnable sd.cpp binary exists on disk, never installing one, unlike the
    prediction."""
    backend = resolve_diffusion_device_target().backend
    off_torch = off_torch_sd_cpp_device(backend)
    if off_torch is not None:
        gpu_ordinal = None
    selected_card = _selected_card(gpu_ordinal)
    install_accelerator = preferred_accelerator(image_install_accelerator(backend), selected_card)

    def _usable(candidate):
        if off_torch_build_mismatch(off_torch, candidate):
            return None
        return usable_or_recorded_failure(candidate, install_accelerator, selected_card)

    server_binary = _usable(
        ensure_sd_server_binary(allow_install = False, accelerator = install_accelerator)
    )
    if (
        server_binary
        and _server_binary_runnable(server_binary)
        and (fam is None or sd_cpp_binary_runs_family(server_binary, fam))
    ):
        return True
    binary = _usable(ensure_sd_cpp_binary(allow_install = False, accelerator = install_accelerator))
    if fam is not None and binary and not sd_cpp_binary_runs_family(binary, fam):
        return False
    return bool(binary and SdCppEngine(binary = binary).version() is not None)


def predict_engine(
    fam: DiffusionFamily,
    *,
    model_kind: Optional[str] = None,
    gpu_ordinal: Optional[int] = None,
) -> str:
    """Same policy as selection with no side effects, so the download plan stages the matching files."""
    if model_kind and model_kind != "gguf":
        return ENGINE_DIFFUSERS

    forced, sd_cpp_pref, mps_enabled = _engine_config()
    if forced == ENGINE_DIFFUSERS:
        return ENGINE_DIFFUSERS
    prefer_native = forced == ENGINE_SD_CPP
    if sd_cpp_pref in _DISABLE_TOKENS and not prefer_native:
        return ENGINE_DIFFUSERS

    backend = resolve_diffusion_device_target().backend
    if off_torch_sd_cpp_device(backend) is not None:
        prefer_native = True
        gpu_ordinal = None
    policy_eligible = backend == "cpu" or (backend == "mps" and mps_enabled) or prefer_native
    if not (policy_eligible and family_sd_cpp_supported(fam)):
        return ENGINE_DIFFUSERS

    # A resident build that cannot run this family is never upgraded, so an install counts only with none.
    native_available = native_binary_installed(gpu_ordinal = gpu_ordinal, fam = fam) or (
        _install_allowed() and not native_binary_installed(gpu_ordinal = gpu_ordinal)
    )
    return select_diffusion_engine(
        backend, native_available = native_available, prefer_native = prefer_native
    )


def family_buildable_here(fam: Optional[DiffusionFamily], *, model_kind: Optional[str]) -> bool:
    """Unbuildable only when neither engine can build it; the diffusers class alone hides native GGUFs."""
    if fam is None:
        return False
    if family_pipeline_available(fam):
        return True
    if model_kind != "gguf" or not family_sd_cpp_supported(fam):
        return False
    try:
        return predict_engine(fam, model_kind = "gguf") == ENGINE_SD_CPP
    except Exception:  # noqa: BLE001 -- a probe failure must not hide/refuse a usable model
        return False


@functools.cache
def _supported_family_capabilities() -> tuple[str, ...]:
    return tuple(pipeline_available_family_names())


def annotate_status(status: dict[str, Any]) -> dict[str, Any]:
    """Tag a backend status dict with the active engine + any fallback reason."""
    out = dict(status)
    out["engine"] = _active_engine_name
    out["fallback_reason"] = _fallback_reason
    out["supported_families"] = list(_supported_family_capabilities())
    return out


def active_status() -> dict[str, Any]:
    """The active engine's status, annotated with which engine + any fallback reason."""
    return annotate_status(get_active_diffusion_engine().status())
