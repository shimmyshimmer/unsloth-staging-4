"""Unit test for the offload_embedding device-safe forward hooks in
unsloth/models/vision.py (_install_offload_embedding_hooks).

Regression guard for the two bugs it fixes:
  1. offloaded (CPU) embedding + CUDA input_ids -> device mismatch (the original bug).
  2. a bf16 embedding pulled BACK onto the GPU by a later model.to(...) while the hook
     hard-codes moving the input to CPU -> the reverse device mismatch.

AST-extracts the real helper (no unsloth/CUDA import needed for the CPU-only checks),
matching tests/saving/test_is_gpt_oss_detection.py. Cross-device asserts run only when
CUDA is available (skipped on CPU-only CI runners).
"""
import ast, os
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")


def _load_installer():
    src = open(VISION).read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_install_offload_embedding_hooks":
            ns = {"torch": torch}
            exec(ast.get_source_segment(src, node), ns)
            return ns["_install_offload_embedding_hooks"]
    raise AssertionError("_install_offload_embedding_hooks not found in vision.py")


install = _load_installer()


def _fresh_emb():
    return nn.Embedding(32, 8)


def test_install_and_idempotent():
    emb = _fresh_emb()
    assert install(emb) is True
    assert emb._unsloth_offload_hooks_installed is True
    n_pre = len(emb._forward_pre_hooks)
    n_post = len(emb._forward_hooks)
    assert install(emb) is True  # idempotent
    assert len(emb._forward_pre_hooks) == n_pre and len(emb._forward_hooks) == n_post
    assert install(None) is False


def test_cpu_noop_forward():
    # weight on cpu, input on cpu -> pre-hook is a no-op, forward works.
    emb = _fresh_emb()
    install(emb)
    x = torch.randint(0, 32, (2, 5))
    out = emb(x)
    assert out.shape == (2, 5, 8)
    assert out.device.type == "cpu"


def test_cuda_offloaded_weight_roundtrip():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available"); return
    # BUG 1: weight offloaded to CPU, input on CUDA -> must not raise; output back on CUDA.
    emb = _fresh_emb().to("cpu")
    install(emb)
    x = torch.randint(0, 32, (2, 5), device="cuda")
    out = emb(x)
    assert out.device.type == "cuda", out.device
    assert emb._unsloth_saved_device.type == "cuda"


def test_cuda_weight_pulled_back_to_gpu():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available"); return
    # BUG 2: weight on CUDA (bf16 pulled back), input on CUDA -> must be a no-op, not send
    # the index to CPU. Hard-coded ".to('cpu')" would crash here.
    emb = _fresh_emb().to("cuda")
    install(emb)
    x = torch.randint(0, 32, (2, 5), device="cuda")
    out = emb(x)
    assert out.device.type == "cuda", out.device


if __name__ == "__main__":
    test_install_and_idempotent(); print("[PASS] install + idempotent")
    test_cpu_noop_forward(); print("[PASS] cpu no-op forward")
    test_cuda_offloaded_weight_roundtrip(); print("[PASS] cuda offloaded-weight roundtrip (bug 1)")
    test_cuda_weight_pulled_back_to_gpu(); print("[PASS] cuda weight-on-gpu no-op (bug 2)")
    print("OK: offload embedding hooks are device-safe both directions")
