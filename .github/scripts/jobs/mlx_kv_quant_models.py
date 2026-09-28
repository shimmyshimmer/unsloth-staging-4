"""Studio MLX KV cache quantization across small mlx-community models (Apple Silicon only).

    python jobs/mlx_kv_quant_models.py [--models A,B] [--quants auto,8,4,tq-4,tq-3.5,tq-3]
                                       [--max-tokens 64] [--out mlx_kv_quant_models.json]

Each (model, setting) runs in its own process through Studio's real MLXInferenceBackend
(load_model + generate_chat_response, greedy) with the real Zoo loader. Records the eligibility
verdict, tok/s, peak Metal memory and text, plus plain mlx-vlm TurboQuant as the reference for
whether a degenerate reply is the scheme or Studio. Exit 1 if any Studio cell crashed, 3 off Apple
Silicon, else 0.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

MODELS = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/gemma-3-1b-it-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/SmolLM2-1.7B-Instruct-4bit",
    "mlx-community/Qwen3.5-2B-4bit",
]
QUANTS = ["auto", "8", "4", "tq-4", "tq-3.5", "tq-3"]
PROMPT = "List the first ten prime numbers, then explain in two sentences why 1 is not prime."
BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "studio", "backend")


def _degenerate(text):
    words = text.split()
    grams = [" ".join(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    return bool(grams) and max(grams.count(g) for g in set(grams)) >= 5


def _cell(model, quant, max_tokens, raw):
    """One cell, in this process: JSON on the last stdout line."""
    import mlx.core as mx

    res = {"model": model, "quant": quant, "raw_mlx_vlm": raw}
    msgs = [{"role": "user", "content": PROMPT}]
    try:
        mx.reset_peak_memory()
        if raw:
            from mlx_vlm import load, stream_generate

            m, proc = load(model)
            tok = getattr(proc, "tokenizer", proc)
            try:
                prompt = tok.apply_chat_template(msgs, tokenize = False, add_generation_prompt = True, enable_thinking = False)
            except TypeError:
                prompt = tok.apply_chat_template(msgs, tokenize = False, add_generation_prompt = True)
            ids = mx.array([tok.encode(prompt, add_special_tokens = False)])
            bits = float(quant.split("-")[1])
            t = time.time(); text = ""; n = 0
            for r in stream_generate(m, proc, prompt, input_ids = ids, max_tokens = max_tokens, temperature = 0.0,
                                     kv_bits = bits, kv_quant_scheme = "turboquant", quantized_kv_start = 0):
                text += r.text; n += 1
            res.update(text = text, gen_s = round(time.time() - t, 3), tokens = n)
        else:
            sys.path.insert(0, os.path.abspath(BACKEND))
            from types import SimpleNamespace
            from core.inference.mlx_inference import MLXInferenceBackend

            b = MLXInferenceBackend()
            cfg = SimpleNamespace(identifier = model, is_vision = False, is_lora = False)
            t = time.time()
            b.load_model(cfg, max_seq_length = 0, kv_quant = None if quant == "auto" else quant)
            res["load_s"] = round(time.time() - t, 2)
            kv = getattr(b, "_kv_quant", None) or {}
            res["kv"] = {k: kv.get(k) for k in ("kv_bits", "eligibility", "reason", "note")}
            res["turboquant_active"] = bool(getattr(b, "_turboquant", False)) and kv.get("kv_bits") is not None
            res["served_by_mlx_vlm"] = bool(getattr(b, "_is_vlm", False))
            t = time.time(); text = ""
            for text in b.generate_chat_response(msgs, temperature = 0.0, top_p = 1.0, top_k = 0,
                                                 max_new_tokens = max_tokens, seed = 0, enable_thinking = False):
                pass
            res.update(text = text, gen_s = round(time.time() - t, 3))
            res["tokens"] = len(b._tokenizer.encode(text)) if text else 0
        res["tok_s"] = round(res["tokens"] / res["gen_s"], 1) if res.get("gen_s") else None
        res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
        res["degenerate"] = _degenerate(res.get("text", ""))
    except Exception as e:  # noqa: BLE001 - a crashed cell is the result, recorded and scored
        import traceback
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["tb"] = traceback.format_exc()[-2000:]
    print("CELL " + json.dumps(res), flush = True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default = ",".join(MODELS))
    ap.add_argument("--quants", default = ",".join(QUANTS))
    ap.add_argument("--max-tokens", type = int, default = 64)
    ap.add_argument("--out", default = "mlx_kv_quant_models.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging_ci)")
    ap.add_argument("--cell", nargs = 3, metavar = ("MODEL", "QUANT", "RAW"), help = argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cell:
        return _cell(a.cell[0], a.cell[1], a.max_tokens, a.cell[2] == "1")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("JOB_RESULT " + json.dumps({"skipped": "Apple Silicon only"}))
        return 3
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "structlog", "pydantic", "fastapi",
                    "httpx", "pyjwt", "python-multipart"], check = False)
    cells = []
    for model in a.models.split(","):
        plan = [(q, "0") for q in a.quants.split(",")] + [("tq-4", "1")]
        for quant, raw in plan:
            t = time.time()
            p = subprocess.run([sys.executable, __file__, "--cell", model, quant, raw,
                                "--max-tokens", str(a.max_tokens)], capture_output = True, text = True, timeout = 1200)
            line = [x for x in p.stdout.splitlines() if x.startswith("CELL ")]
            cell = json.loads(line[-1][5:]) if line else {
                "model": model, "quant": quant, "raw_mlx_vlm": raw == "1",
                "error": f"no result (rc={p.returncode}): {(p.stderr or '')[-600:]}"}
            cell["wall_s"] = round(time.time() - t, 1)
            cells.append(cell)
            kv = cell.get("kv") or {}
            print(f"{model:45s} {quant:7s} raw={raw} elig={kv.get('eligibility')} bits={kv.get('kv_bits')} "
                  f"tok/s={cell.get('tok_s')} peak={cell.get('peak_gb')}GB degen={cell.get('degenerate')} "
                  f"{cell.get('error') or repr((cell.get('text') or '')[:90])}", flush = True)
            with open(a.out, "w") as f:
                json.dump({"job": "mlx_kv_quant_models", "cells": cells, "finished": False}, f, indent = 1)
    crashed = [c for c in cells if c.get("error") and not c.get("raw_mlx_vlm")]
    summary = {"cells": len(cells), "studio_crashed": len(crashed),
               "crashed": [f"{c['model']} {c['quant']}: {c['error'][:160]}" for c in crashed]}
    with open(a.out, "w") as f:
        json.dump({"job": "mlx_kv_quant_models", "cells": cells, "summary": summary, "finished": True}, f, indent = 1)
    print("JOB_RESULT " + json.dumps(summary), flush = True)
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
