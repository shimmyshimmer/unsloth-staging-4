"""Verify the slow path call site matches the fast path (UnslothEfficientGRPO.forward) arg order."""
import ast
import os

def extract_call_positional_args(source, func_name):
    """Find a call to func_name and return AST nodes for positional args."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == func_name:
                return [ast.dump(a) for a in node.args]
    return None

def test_fast_slow_path_arg_order():
    rl_path = os.path.join(os.path.dirname(__file__), "unsloth", "models", "rl_replacements.py")
    if not os.path.exists(rl_path):
        print("SKIP: rl_replacements.py not found")
        return

    with open(rl_path) as f:
        source = f.read()

    tree = ast.parse(source)
    slow_call_args = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "grpo_compute_loss_slow":
                slow_call_args = node.args
                slow_call_kwargs = {kw.arg for kw in node.keywords if kw.arg}
                break

    assert slow_call_args is not None, "grpo_compute_loss_slow call not found"
    assert len(slow_call_args) == 8, f"Expected 8 positional args, got {len(slow_call_args)}"

    # Verify sampling_per_token_logps is NOT in kwargs (it should be positional)
    assert "sampling_per_token_logps" not in slow_call_kwargs, \
        "sampling_per_token_logps should not be in kwargs"

    # Check the 4th positional arg is sampling_per_token_logps (Name node)
    fourth_arg = slow_call_args[3]
    assert isinstance(fourth_arg, ast.Name), f"4th arg should be a Name, got {type(fourth_arg)}"
    assert fourth_arg.id == "sampling_per_token_logps", \
        f"4th positional arg should be 'sampling_per_token_logps', got '{fourth_arg.id}'"

    print("PASS: slow path has sampling_per_token_logps as 4th positional arg, matching fast path")

if __name__ == "__main__":
    test_fast_slow_path_arg_order()
