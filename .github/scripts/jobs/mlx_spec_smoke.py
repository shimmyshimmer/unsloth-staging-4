"""Studio's MLX speculative decoding end to end on Apple Silicon (unsloth#13014 + unsloth-zoo#1621).

Each mode loads the model in a FRESH interpreter through Studio's own ``MLXInferenceBackend``
(``load_model(speculative_type=...)``), then decodes greedy single replies and a two-row batch.
Checks per speculative mode: the drafter it reports attached, drafted tokens were accepted, and every
reply is token-identical to the Off arm (greedy speculation must not change the target's pick).

    python jobs/mlx_spec_smoke.py                 # Qwen3.5-0.8B (built-in MTP head): off, ngram, mtp, auto
    python jobs/mlx_spec_smoke.py --tiny          # same model, fewer tokens

Exit 3 off Apple Silicon. Every child gets a scratch UNSLOTH_STUDIO_HOME under the --out directory.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"  # ships its MTP head (zoo#1621 metal tests use it)
MODES = ("off", "ngram", "mtp", "auto")
CODE = (
    "def merge_sort(values):\n    if len(values) <= 1:\n        return values\n"
    "    middle = len(values) // 2\n    left = merge_sort(values[:middle])\n"
    "    right = merge_sort(values[middle:])\n    return merge(left, right)\n"
)
PROMPTS = [
    f"Copy this code exactly, then add a docstring to it:\n{CODE}",
    "Explain what a hash map is in two sentences.",
]
EXPECT_KIND = {"off": None, "ngram": "ngram", "mtp": "mtp", "auto": "mtp"}


def _backend_dir():
    root = Path.cwd()
    for cand in (root, *root.parents):
        if (cand / "studio" / "backend" / "core" / "inference" / "mlx_inference.py").is_file():
            return cand / "studio" / "backend"
    raise SystemExit("run from an unsloth checkout (studio/backend not found)")


def child(model, mode, tokens, out_path):
    os.chdir(_backend_dir())
    sys.path.insert(0, os.getcwd())
    from core.inference.mlx_inference import MLXInferenceBackend
    from utils.models import ModelConfig

    res = {"mode": mode, "single": [], "batch": None}
    backend = MLXInferenceBackend()
    t = time.perf_counter()
    config = ModelConfig.from_identifier(model_id = model)
    ok = backend.load_model(config, max_seq_length = 4096, load_in_4bit = False, speculative_type = mode)
    res["load_s"] = round(time.perf_counter() - t, 2)
    info = dict(backend.models.get(config.identifier) or {})
    res["loaded"] = bool(ok)
    res["is_vlm"] = bool(getattr(backend, "_is_vlm", False))
    for key in ("speculative_type", "spec_drafter_kind", "spec_fallback_reason"):
        res[key] = info.get(key)
    res["draft_attached"] = backend._speculative_draft is not None
    greedy = dict(temperature = 0.0, top_p = 1.0, top_k = 0, min_p = 0.0, seed = 0)
    for prompt in PROMPTS:
        t = time.perf_counter()
        text = ""
        for text in backend.generate_chat_response(
            [{"role": "user", "content": prompt}],
            max_new_tokens = tokens,
            enable_thinking = False,
            **greedy,
        ):
            pass
        wall = time.perf_counter() - t
        timings = (backend.last_generation_stats or {}).get("timings") or {}
        res["single"].append(
            {
                "text": text,
                "wall_s": round(wall, 3),
                "predicted_n": timings.get("predicted_n"),
                "tok_s": timings.get("predicted_per_second"),
                "draft_n": timings.get("draft_n"),
                "draft_n_accepted": timings.get("draft_n_accepted"),
            }
        )
    requests = [
        dict(
            messages = [{"role": "user", "content": p}],
            max_new_tokens = tokens,
            enable_thinking = False,
            **greedy,
        )
        for p in PROMPTS
    ]
    reason = backend.batch_unavailable_reason(requests)
    if reason is None:
        replies = {i: "" for i in range(len(requests))}
        for row, snap in backend.generate_chat_batch(requests):
            if snap is not None:
                replies[row] = snap
        stats = [
            ((s or {}).get("timings") or {}) for s in (backend.last_batch_generation_stats or [])
        ]
        res["batch"] = {
            "texts": [replies[i] for i in range(len(requests))],
            "draft_n": [s.get("draft_n") for s in stats],
            "draft_n_accepted": [s.get("draft_n_accepted") for s in stats],
        }
    else:
        res["batch"] = {"unavailable": reason}
    Path(out_path).write_text(json.dumps(res, indent = 1))


def main():
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        child(sys.argv[i + 1], sys.argv[i + 2], int(sys.argv[i + 3]), sys.argv[i + 4])
        return
    p = C.base_parser("mlx_spec_smoke", DEFAULT_MODEL, DEFAULT_MODEL, default_steps = 0)
    p.add_argument("--modes", default = ",".join(MODES))
    p.add_argument("--tokens", type = int, default = None)
    a = C.resolve_args(p)
    C.reject_cfg(a)
    tokens = a.tokens or (64 if a.tiny else 160)
    modes = a.modes.split(",")
    with C.JobRecorder(a, backend_hint = "mlx") as rec:
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            rec.skip("needs Apple Silicon")
        from huggingface_hub import snapshot_download

        snapshot_download(a.model)
        scratch = Path(a.out).resolve().parent / "mlx_spec_smoke_scratch"
        shutil.rmtree(scratch, ignore_errors = True)
        scratch.mkdir(parents = True)
        env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(scratch / "studio_home"))
        results = {}
        for mode in modes:
            out = scratch / f"{mode}.json"
            try:
                rc = subprocess.run(
                    [
                        sys.executable,
                        os.path.abspath(__file__),
                        "--child",
                        a.model,
                        mode,
                        str(tokens),
                        str(out),
                    ],
                    env = env,
                    timeout = 1800,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            if rc != 0 or not out.is_file():
                rec.check(f"child_ok[{mode}]", False, f"exit {rc}")
                continue
            res = results[mode] = json.loads(out.read_text())
            for i, row in enumerate(res["single"]):
                rec.step(mode = mode, prompt = i, **{k: v for k, v in row.items() if k != "text"})
            rec.check(f"loaded[{mode}]", res["loaded"], res.get("spec_fallback_reason"))
            rec.check(
                f"drafter_kind[{mode}]",
                res["spec_drafter_kind"] == EXPECT_KIND.get(mode, res["spec_drafter_kind"]),
                {k: res[k] for k in ("spec_drafter_kind", "spec_fallback_reason", "is_vlm")},
            )
            if mode != "off":
                drafted = sum(r["draft_n"] or 0 for r in res["single"])
                accepted = sum(r["draft_n_accepted"] or 0 for r in res["single"])
                rec.check(f"drafted[{mode}]", drafted > 0 and accepted > 0, [drafted, accepted])
            rec.summary(
                **{
                    f"{mode}.{k}": res[k]
                    for k in ("load_s", "spec_drafter_kind", "spec_fallback_reason", "is_vlm")
                }
            )
            rec.summary(
                **{
                    f"{mode}.tok_s": [r["tok_s"] for r in res["single"]],
                    f"{mode}.batch": res["batch"],
                }
            )
        base = results.get("off")
        for mode, res in results.items():
            if mode == "off" or base is None:
                continue
            same = [r["text"] == b["text"] for r, b in zip(res["single"], base["single"])]
            rec.check(
                f"greedy_identical_to_off[{mode}]",
                all(same),
                [
                    (r["text"][:200], b["text"][:200])
                    for r, b, s in zip(res["single"], base["single"], same)
                    if not s
                ],
            )
            bt, ot = (res.get("batch") or {}).get("texts"), (base.get("batch") or {}).get("texts")
            if bt is not None and ot is not None:
                rec.check(
                    f"batch_greedy_identical_to_off[{mode}]",
                    bt == ot,
                    [(x[:120], y[:120]) for x, y in zip(bt, ot) if x != y],
                )


if __name__ == "__main__":
    main()
