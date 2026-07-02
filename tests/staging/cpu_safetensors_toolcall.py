# SPDX-License-Identifier: AGPL-3.0-only
"""Small safetensors model on CPU: generate a tool-calling turn and run the
output through the Studio parser, guarding against the DoS/loop the parser
fixes address.

Loads a tiny unsloth/* instruct model (no GPU), prompts it with a single tool
schema, generates a bounded number of tokens, then parses the completion with
``parse_tool_calls_from_text`` inside a wall-clock guard so a parser hang fails
loudly instead of blocking CI. Coherence (no token loop) is asserted on the raw
generation; a parsed tool call is reported but not required (a 0.5B model does
not always emit a well-formed call).
"""
import sys
import threading
from pathlib import Path

MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
_BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"
sys.path.insert(0, str(_BACKEND))

from core.inference.tool_call_parser import parse_tool_calls_from_text  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
PROMPT = "What is the weather in Paris right now? Use the tool."


def _coherent(text: str) -> bool:
    words = text.split()
    if len(words) < 4:
        return True  # a short/empty completion is not a loop
    return len(set(w.lower() for w in words)) >= max(4, len(words) // 4)


def _parse_with_timeout(text: str, seconds: float = 10.0):
    box: dict = {}
    t = threading.Thread(
        target=lambda: box.setdefault("calls", parse_tool_calls_from_text(text)), daemon=True
    )
    t.start()
    t.join(timeout=seconds)
    if t.is_alive():
        raise AssertionError("parse_tool_calls_from_text hung on model output")
    return box.get("calls", [])


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {MODEL} on CPU ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    completion = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    print("=== COMPLETION ===", flush=True)
    print(completion, flush=True)

    assert _coherent(completion), "generation looped / degenerate (not coherent)"
    calls = _parse_with_timeout(completion)
    print(f"=== PARSED {len(calls)} tool call(s): {[c['function']['name'] for c in calls]}", flush=True)
    print("CPU-SAFETENSORS-TOOLCALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
