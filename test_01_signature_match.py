"""Test that grpo_compute_loss_slow call site positional args match the function signature."""
import ast
import sys
import os

def get_function_signature_params(source_path, func_name):
    """Extract positional parameter names from a function definition."""
    with open(source_path) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            params = []
            for arg in node.args.args:
                params.append(arg.arg)
            return params
    return None

def get_call_site_args(source_path, func_name):
    """Extract positional and keyword arg info from a call to func_name."""
    with open(source_path) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.name == func_name:
                positional_count = len(node.args)
                keyword_names = [kw.arg for kw in node.keywords if kw.arg is not None]
                return positional_count, keyword_names
    return None, None

def test_signature_positional_count():
    # The function signature has 8 positional params before **kwargs:
    # ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages
    zoo_path = None
    for root, dirs, files in os.walk(os.path.join(sys.prefix, "lib")):
        for f in files:
            if f == "rl_replacements.py" and "unsloth_zoo" in root:
                zoo_path = os.path.join(root, f)
                break
    if zoo_path is None:
        # Fallback: check site-packages directly
        import glob
        candidates = glob.glob(os.path.join(sys.prefix, "lib/*/site-packages/unsloth_zoo/rl_replacements.py"))
        if candidates:
            zoo_path = candidates[0]

    if zoo_path is None:
        print("SKIP: unsloth_zoo not found in environment")
        return

    params = get_function_signature_params(zoo_path, "grpo_compute_loss")
    assert params is not None, "grpo_compute_loss function not found"
    # Should be: ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages
    assert len(params) >= 8, f"Expected at least 8 positional params, got {len(params)}: {params}"
    assert params[3] == "sampling_per_token_logps", \
        f"4th param should be 'sampling_per_token_logps', got '{params[3]}'"
    assert params[4] == "input_ids", f"5th param should be 'input_ids', got '{params[4]}'"
    assert params[5] == "mask", f"6th param should be 'mask', got '{params[5]}'"
    assert params[6] == "beta", f"7th param should be 'beta', got '{params[6]}'"
    assert params[7] == "advantages", f"8th param should be 'advantages', got '{params[7]}'"
    print("PASS: grpo_compute_loss signature has correct positional param order")

def test_call_site_no_duplicate_keyword():
    """The call site must NOT pass sampling_per_token_logps as keyword (it's positional now)."""
    call_site_path = os.path.join(os.path.dirname(__file__), "unsloth", "models", "rl_replacements.py")
    if not os.path.exists(call_site_path):
        print("SKIP: call site file not found")
        return

    with open(call_site_path) as f:
        source = f.read()

    # Parse the function that contains the call
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "grpo_compute_loss_slow":
                kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]
                assert "sampling_per_token_logps" not in kw_names, \
                    "sampling_per_token_logps should be positional, not keyword"
                assert len(node.args) == 8, \
                    f"Expected 8 positional args in call, got {len(node.args)}"
                print("PASS: grpo_compute_loss_slow call has 8 positional args, no duplicate keyword")
                return
    print("SKIP: grpo_compute_loss_slow call not found in AST")

if __name__ == "__main__":
    test_signature_positional_count()
    test_call_site_no_duplicate_keyword()
