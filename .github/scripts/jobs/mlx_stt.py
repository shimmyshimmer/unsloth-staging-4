"""Studio dictation of a downloaded mlx-audio checkpoint on Apple Silicon (unsloth#10272).

Runs the checked-out tree's own Studio backend: the HF cache inventory row, the Transformers
dictation sidecar and its spawned worker, against a real mlx-community Qwen3-ASR quant and speech
rendered by macOS `say`. Nothing is patched.

    python jobs/mlx_stt.py                 # mlx-community/Qwen3-ASR-0.6B-8bit
    python jobs/mlx_stt.py --model mlx-community/Qwen3-ASR-1.7B-8bit
    python jobs/mlx_stt.py --base-sha <merge base>   # also run the base tree: it must NOT transcribe

Installs the Studio backend requirements and the Mac Studio pins (transformers 5.5.0 and the
tokenizers it accepts, mlx-vlm <= 0.7.4, which brings mlx-audio) on first run.

Checks: inventory lists the repo as ASR under library mlx-audio and not chattable; the sidecar sees
it as downloaded; a short and a > 30 s clip transcribe with the spoken words; unload frees it.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"
SHORT = "In the beginning God created the heaven and the earth."
LONG = (
    "And the earth was without form, and void; and darkness was upon the face of the deep. "
    "And the Spirit of God moved upon the face of the waters. And God said, Let there be light: "
    "and there was light. And God saw the light, that it was good: and God divided the light from "
    "the darkness. And God called the light Day, and the darkness he called Night. And the evening "
    "and the morning were the first day. And God said, Let there be a firmament in the midst of the "
    "waters, and let it divide the waters from the waters. And God made the firmament, and divided the "
    "waters which were under the firmament from the waters which were above the firmament: and it was so. "
    "And God called the firmament Heaven. And the evening and the morning were the second day."
)  # read aloud this runs past one 30 s dictation window, so the sidecar splits it
STUDIO_PINS = [
    "mlx==0.32.3",
    "mlx-metal==0.32.3",
    "mlx-lm==0.31.3",
    "mlx-vlm<=0.7.4",
    "av",
    "soundfile",
]
STUDIO_OVERRIDES = ["transformers==5.5.0", "tokenizers==0.22.2"]


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def _recall(expected, got):
    want, have = _words(expected), set(_words(got))
    return sum(w in have for w in want) / max(1, len(want))


def _say(
    text,
    out_wav,
    rate = 16000,
):
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-o", str(aiff), text], check = True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1", str(aiff), str(out_wav)],
        check = True,
    )
    with wave.open(str(out_wav), "rb") as w:
        return w.getnframes() / w.getframerate()


def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check = True)


def _arm_checks(rec, a, backend, hub, audio_dir, clips):
    """Every check against one tree's backend, in this (fresh) process."""
    sys.path.insert(0, str(backend))
    from core.inference import stt_sidecar
    from hub.services.models import cache_inventory

    try:
        from core.inference import stt_mlx
        rec.check(
            "mlx_stt_available", stt_mlx.mlx_stt_available(), "Apple Silicon + mlx_audio importable"
        )
    except ImportError:
        rec.check("mlx_stt_available", False, "tree has no core.inference.stt_mlx")
    rows = [
        r
        for r in cache_inventory._scan_cached_models(active_hub_cache = hub)
        if r["repo_id"].lower() == a.model.lower()
    ]
    row = rows[0] if rows else {}
    rec.check(
        "inventory_row_is_asr",
        row.get("task") == "automatic-speech-recognition"
        and row.get("library_name") == "mlx-audio",
        {k: row.get(k) for k in ("task", "library_name", "pipeline_tag")},
    )
    rec.check(
        "inventory_row_cannot_chat",
        (row.get("capabilities") or {}).get("can_chat") is False,
        row.get("capabilities"),
    )
    rec.check("sidecar_sees_download", stt_sidecar.is_model_downloaded(a.model), a.model)
    if "long" in clips:
        rec.check("long_clip_spans_two_windows", clips["long"][1] > 30, f"{clips['long'][1]:.1f}s")
    sidecar = stt_sidecar.WhisperSttSidecar(keep_alive_seconds = 0)
    try:
        for name, (text, seconds) in clips.items():
            t0 = time.perf_counter()
            try:
                result = sidecar.transcribe((audio_dir / f"{name}.wav").read_bytes(), model = a.model)
            except Exception as exc:
                rec.check(
                    f"transcribes_{name}_clip", False, f"{type(exc).__name__}: {str(exc)[:300]}"
                )
                continue
            took = time.perf_counter() - t0
            recall = _recall(text, result.get("text") or "")
            rec.step(
                step = len(rec.data["steps"]) + 1,
                clip = name,
                audio_s = round(seconds, 1),
                wall_s = round(took, 2),
                text = result.get("text"),
                word_recall = round(recall, 3),
                device = sidecar.device,
            )
            rec.check(
                f"transcribes_{name}_clip",
                recall >= 0.8,
                f"recall {recall:.2f} over {seconds:.1f}s: {result.get('text')!r}",
            )
        rec.check(
            "model_resident_on_engine",
            sidecar.loaded_model == a.model,
            f"loaded={sidecar.loaded_model} device={sidecar.device}",
        )
    finally:
        sidecar.unload()
    rec.check("unload_releases_model", sidecar.loaded_model is None, sidecar.loaded_model)


def _base_tree(repo_root, sha):
    """The merge base checked out beside the PR tree (staging clones are shallow)."""
    dest = Path(tempfile.mkdtemp(prefix = "mlx_stt_base_")) / "tree"
    git = ["git", "-C", str(repo_root)]
    if subprocess.run(
        [*git, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output = True
    ).returncode:
        subprocess.run(
            [*git, "fetch", "-q", "--depth", "1", "https://github.com/unslothai/unsloth", sha],
            check = True,
        )
    subprocess.run([*git, "worktree", "add", "-q", "--detach", str(dest), sha], check = True)
    return dest


def main(argv = None):
    p = C.base_parser("mlx_stt", DEFAULT_MODEL, DEFAULT_MODEL, default_steps = 0)
    p.add_argument("--repo-root", default = ".", help = "checkout whose studio/backend runs")
    p.add_argument(
        "--base-sha", default = None, help = "also run this tree first; it must not transcribe"
    )
    p.add_argument("--no-install", action = "store_true")
    p.add_argument("--arm", action = "store_true", help = argparse_hidden())
    p.add_argument("--hub", default = None, help = argparse_hidden())
    p.add_argument("--audio-dir", default = None, help = argparse_hidden())
    a = C.resolve_args(p, argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint = "studio-stt-mlx") as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        backend = (Path(a.repo_root) / "studio" / "backend").resolve()
        if not (backend / "core" / "inference" / "stt_sidecar.py").is_file():
            raise SystemExit(f"no Studio backend at {backend}")
        if a.arm:
            os.environ["HF_HUB_CACHE"] = a.hub
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["UNSLOTH_STUDIO_HOME"] = tempfile.mkdtemp(prefix = "mlx_stt_home_")
            audio_dir = Path(a.audio_dir)
            clips = {
                "short": (SHORT, _wav_seconds(audio_dir / "short.wav")),
                "long": (LONG, _wav_seconds(audio_dir / "long.wav")),
            }
            _arm_checks(rec, a, backend, Path(a.hub), audio_dir, clips)
            return
        if not a.no_install:
            _pip("-r", str(backend / "requirements" / "studio.txt"))
            _pip(*STUDIO_PINS)
            _pip("--no-deps", *STUDIO_OVERRIDES)
        rec.data["summary"]["stack"] = {
            name: (md_version(name))
            for name in (
                "mlx",
                "mlx-lm",
                "mlx-vlm",
                "mlx-audio",
                "transformers",
                "tokenizers",
                "torch",
            )
        }

        hub = Path(tempfile.mkdtemp(prefix = "mlx_stt_hub_"))
        from huggingface_hub import snapshot_download

        t0 = time.perf_counter()
        snapshot_download(a.model, cache_dir = str(hub))
        rec.data["summary"]["download_s"] = round(time.perf_counter() - t0, 1)
        audio_dir = Path(tempfile.mkdtemp(prefix = "mlx_stt_audio_"))
        rec.data["summary"]["clip_seconds"] = {
            "short": round(_say(SHORT, audio_dir / "short.wav"), 1),
            "long": round(_say(LONG, audio_dir / "long.wav"), 1),
        }

        def arm(root, label):
            out = Path(a.out).with_name(Path(a.out).stem + f"_{label}.json")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--arm",
                "--repo-root",
                str(root),
                "--model",
                a.model,
                "--hub",
                str(hub),
                "--audio-dir",
                str(audio_dir),
                "--out",
                str(out),
                "--compile-cache",
                "keep",
            ]
            proc = subprocess.run(cmd)
            import json

            data = json.loads(out.read_text()) if out.is_file() else {}
            rec.data.setdefault("arms", {})[label] = {
                "exit": proc.returncode,
                "passed": data.get("passed"),
                "checks": data.get("checks"),
                "steps": data.get("steps"),
                "error": (data.get("error") or "")[-1500:] or None,
            }
            return data

        if a.base_sha:
            base = arm(_base_tree(Path(a.repo_root), a.base_sha), "base")
            checks = base.get("checks") or {}
            lists = (checks.get("inventory_row_is_asr") or {}).get("ok")
            heard = any(
                (c or {}).get("ok") for k, c in checks.items() if k.startswith("transcribes_")
            )
            rec.check(
                "base_reproduces_issue",
                lists is False and not heard,
                f"base lists as ASR={lists}, base transcribed={heard}: "
                f"{ {k: c.get('detail') for k, c in checks.items() if not c.get('ok')} }",
            )
        head = arm(Path(a.repo_root), "head")
        for name, c in (head.get("checks") or {}).items():
            rec.check(f"head_{name}", c.get("ok"), c.get("detail"))
        for step in head.get("steps") or []:
            rec.step(**step)
        if not head.get("checks"):
            rec.check("head_ran", False, head.get("error"))


def argparse_hidden():
    import argparse
    return argparse.SUPPRESS


def _wav_seconds(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def md_version(name):
    import importlib.metadata as md
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
