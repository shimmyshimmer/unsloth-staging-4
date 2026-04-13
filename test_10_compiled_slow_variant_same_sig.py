"""Test that grpo_compute_loss_slow is torch.compiled grpo_compute_loss with same signature."""
import os
import re

def test_slow_is_compiled_variant():
    """grpo_compute_loss_slow should be a @torch.compile'd copy of grpo_compute_loss with same params."""
    # Check unsloth_zoo source
    zoo_path = None
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), "..")):
        for f in files:
            if f == "rl_replacements.py" and "unsloth_zoo" in root:
                zoo_path = os.path.join(root, f)
                break
        if zoo_path:
            break

    # Fallback to site-packages
    if not zoo_path:
        import sys, glob
        candidates = glob.glob(os.path.join(sys.prefix, "lib/*/site-packages/unsloth_zoo/rl_replacements.py"))
        if candidates:
            zoo_path = candidates[0]

    if not zoo_path:
        print("SKIP: unsloth_zoo rl_replacements.py not found")
        return

    with open(zoo_path) as f:
        content = f.read()

    # Verify grpo_compute_loss_slow is built from grpo_compute_loss source
    assert 'RL_REPLACEMENTS["grpo_compute_loss_slow"]' in content, \
        "grpo_compute_loss_slow not defined in RL_REPLACEMENTS"

    # Check it references torch.compile
    slow_section = content[content.index('"grpo_compute_loss_slow"'):]
    assert "torch.compile" in slow_section[:500], \
        "grpo_compute_loss_slow should use torch.compile"

    # Check it's built from grpo_compute_loss source
    assert "inspect.getsource(grpo_compute_loss)" in slow_section[:500], \
        "grpo_compute_loss_slow should be built from grpo_compute_loss source"

    print("PASS: grpo_compute_loss_slow is a torch.compiled variant of grpo_compute_loss")

def test_rename_in_compiled_source():
    """The compiled slow variant renames the function from grpo_compute_loss to grpo_compute_loss_slow."""
    zoo_path = None
    import sys, glob
    candidates = glob.glob(os.path.join(sys.prefix, "lib/*/site-packages/unsloth_zoo/rl_replacements.py"))
    if candidates:
        zoo_path = candidates[0]

    if not zoo_path:
        print("SKIP: unsloth_zoo not found")
        return

    with open(zoo_path) as f:
        content = f.read()

    # It should replace "def grpo_compute_loss" with "def grpo_compute_loss_slow"
    assert '.replace(\n        "def grpo_compute_loss"' in content or \
           ".replace(\"def grpo_compute_loss\"" in content or \
           '.replace(\n        "def grpo_compute_loss",' in content, \
        "Should rename grpo_compute_loss to grpo_compute_loss_slow in compiled variant"
    print("PASS: compiled variant correctly renames function")

if __name__ == "__main__":
    test_slow_is_compiled_variant()
    test_rename_in_compiled_source()
