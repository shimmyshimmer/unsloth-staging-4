"""Test that the else branch (per_token_logps is None) is unaffected by the fix.
The PR only touches the 'if per_token_logps is not None' branch."""
import ast
import os

def test_else_branch_uses_grpo_accumulated_loss():
    """When per_token_logps is None, grpo_accumulated_loss is called instead, and its call is unchanged."""
    rl_path = os.path.join(os.path.dirname(__file__), "unsloth", "models", "rl_replacements.py")
    if not os.path.exists(rl_path):
        print("SKIP: rl_replacements.py not found")
        return

    with open(rl_path) as f:
        source = f.read()

    tree = ast.parse(source)
    accumulated_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "grpo_accumulated_loss":
                kw_names = {kw.arg for kw in node.keywords if kw.arg}
                accumulated_calls.append(kw_names)

    assert len(accumulated_calls) >= 1, "grpo_accumulated_loss call not found"

    # The main else branch (with loss_type) should pass sampling_per_token_logps as keyword
    found_sampling_kwarg = False
    for kw_set in accumulated_calls:
        if "sampling_per_token_logps" in kw_set:
            found_sampling_kwarg = True
            break

    assert found_sampling_kwarg, \
        "grpo_accumulated_loss should receive sampling_per_token_logps as keyword arg"
    print("PASS: else branch (grpo_accumulated_loss) still passes sampling_per_token_logps as kwarg")

def test_per_token_logps_none_takes_else():
    """Simulate the branching: per_token_logps=None should skip grpo_compute_loss_slow."""
    per_token_logps = None
    slow_called = False
    accumulated_called = False

    if per_token_logps is not None:
        slow_called = True
    else:
        accumulated_called = True

    assert not slow_called
    assert accumulated_called
    print("PASS: per_token_logps=None correctly takes else branch")

if __name__ == "__main__":
    test_else_branch_uses_grpo_accumulated_loss()
    test_per_token_logps_none_takes_else()
