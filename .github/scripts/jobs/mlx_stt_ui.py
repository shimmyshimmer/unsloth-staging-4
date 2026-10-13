"""Before/after of the Transcribe picker on Apple Silicon with an MLX speech model on disk (unsloth#10272).

Builds the frontend of the merge base and of this checkout, runs each tree's Studio from source on
one shared Hugging Face cache holding mlx-community/Qwen3-ASR-0.6B-8bit, opens Audio > Transcribe >
model picker > On Device, and photographs it. The composite and both shots are embedded (base64)
in the job JSON, since staging uploads only job_*.json.

    python jobs/mlx_stt_ui.py --base-sha <merge base>

Checks: base does not list the model, head does; the cache row is ASR / mlx-audio only on head.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import mlx_stt as S  # noqa: E402

DRIVER = Path(__file__).resolve().parent / "_stt_picker_ui.py"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_frontend(tree):
    fe = Path(tree) / "studio" / "frontend"
    subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd = fe, check = True)
    subprocess.run(["npx", "vite", "build"], cwd = fe, check = True)


def _arm(tree, hub, out_png, model):
    proc = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--tree",
            str(tree),
            "--hub",
            str(hub),
            "--out",
            str(out_png),
            "--port",
            str(_free_port()),
            "--expect",
            model,
        ],
        capture_output = True,
        text = True,
        timeout = 1800,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("UI_FACTS ")), None)
    if line is None:
        log = Path(out_png).with_suffix(".studio.log")
        raise RuntimeError(
            f"driver rc={proc.returncode}: {proc.stderr[-2500:]}\n"
            f"studio log: {log.read_text()[-2500:] if log.is_file() else '-'}"
        )
    return json.loads(line[len("UI_FACTS ") :])


def _composite(before, after, out):
    from PIL import Image, ImageDraw, ImageFont

    a, b = Image.open(before).convert("RGB"), Image.open(after).convert("RGB")
    crop = (280, 0, 900, 520)  # the picker, not the whole window
    a, b = a.crop(crop), b.crop(crop)
    pad, head = 16, 44
    canvas = Image.new(
        "RGB", (a.width + b.width + 3 * pad, max(a.height, b.height) + head + pad), "white"
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except OSError:
        font = ImageFont.load_default()
    for i, (img, label) in enumerate(((a, "BEFORE (main)"), (b, "AFTER (this PR)"))):
        x = pad + i * (a.width + pad)
        draw.text((x, 10), label, fill = "black", font = font)
        canvas.paste(img, (x, head))
    canvas.save(out)


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def main(argv = None):
    p = C.base_parser("mlx_stt_ui", S.DEFAULT_MODEL, S.DEFAULT_MODEL, default_steps = 0)
    p.add_argument("--repo-root", default = ".")
    p.add_argument("--base-sha", required = True)
    p.add_argument("--no-install", action = "store_true")
    a = C.resolve_args(p, argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint = "studio-ui") as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        head_tree = Path(a.repo_root).resolve()
        backend = head_tree / "studio" / "backend"
        if not a.no_install:
            S._pip("-r", str(backend / "requirements" / "studio.txt"))
            S._pip(*S.STUDIO_PINS, "playwright<2", "pillow")
            S._pip("--no-deps", *S.STUDIO_OVERRIDES)
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check = True)
        hub = Path(tempfile.mkdtemp(prefix = "stt_ui_hf_")) / "hub"
        from huggingface_hub import snapshot_download

        snapshot_download(a.model, cache_dir = str(hub))
        base_tree = S._base_tree(head_tree, a.base_sha)
        out = Path(a.out).resolve().parent
        shots = {}
        for label, tree in (("base", base_tree), ("head", head_tree)):
            _build_frontend(tree)
            png = out / f"stt_picker_{label}.png"
            facts = _arm(tree, hub, png, a.model)
            shots[label] = png
            rec.data.setdefault("arms", {})[label] = {k: v for k, v in facts.items() if k != "tree"}
        arms = rec.data["arms"]
        rec.check(
            "base_does_not_list_model",
            arms["base"]["listed"] is False,
            re.sub(r"\s+", " ", arms["base"]["picker_text"])[:400],
        )
        rec.check(
            "head_lists_model_on_device",
            arms["head"]["listed"] is True,
            re.sub(r"\s+", " ", arms["head"]["picker_text"])[:400],
        )
        composite = out / "stt_picker_before_after.png"
        _composite(shots["base"], shots["head"], composite)
        rec.data["images_b64"] = {
            "composite": _b64(composite),
            "before": _b64(shots["base"]),
            "after": _b64(shots["head"]),
        }
        rec.step(step = 1, composite = str(composite))


if __name__ == "__main__":
    main()
