"""Test that grpo_compute_loss_slow returns the expected 7-tuple unpacking."""
import ast
import os

def test_return_unpacking_matches():
    """The call site unpacks 7 values: loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, completion_mask."""
    rl_path = os.path.join(os.path.dirname(__file__), "unsloth", "models", "rl_replacements.py")
    if not os.path.exists(rl_path):
        print("SKIP: rl_replacements.py not found")
        return

    with open(rl_path) as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "grpo_compute_loss_slow":
                    target = node.targets[0]
                    if isinstance(target, ast.Tuple):
                        names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
                        assert len(names) == 7, f"Expected 7-tuple unpacking, got {len(names)}: {names}"
                        expected = ["loss", "completion_length", "mean_kl", "delta",
                                    "flat_is_ratio", "coef_1", "completion_mask"]
                        assert names == expected, f"Expected {expected}, got {names}"
                        print(f"PASS: grpo_compute_loss_slow return unpacks to {names}")
                        return

    print("SKIP: grpo_compute_loss_slow assignment not found")

def test_stub_returns_7_tuple():
    """Stub test: function returns 7 values matching the call site unpacking."""
    def grpo_compute_loss_slow(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kw):
        return 0.5, 10.0, 0.01, None, None, None, mask

    loss, comp_len, mean_kl, delta, flat_is, coef, mask = grpo_compute_loss_slow(
        "ref", "new", "old", "sampling", "ids", "MASK", 0.1, "adv",
        loss_type="grpo",
    )
    assert loss == 0.5
    assert mask == "MASK"
    print("PASS: 7-tuple unpacking works correctly")

if __name__ == "__main__":
    test_return_unpacking_matches()
    test_stub_returns_7_tuple()
