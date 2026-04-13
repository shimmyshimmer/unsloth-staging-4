"""Verify no other callers of grpo_compute_loss or grpo_compute_loss_slow exist that might be affected."""
import os
import re

def test_no_unexpected_callers():
    """Scan the codebase for all calls to grpo_compute_loss / grpo_compute_loss_slow."""
    root = os.path.dirname(__file__)
    unsloth_dir = os.path.join(root, "unsloth")
    if not os.path.isdir(unsloth_dir):
        print("SKIP: unsloth dir not found")
        return

    callers = []
    pattern = re.compile(r'\bgrpo_compute_loss(?:_slow)?\s*\(')

    for dirpath, _, filenames in os.walk(unsloth_dir):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            with open(fpath) as f:
                for lineno, line in enumerate(f, 1):
                    if pattern.search(line):
                        # Skip definitions and string literals
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("#"):
                            continue
                        if stripped.startswith('"') or stripped.startswith("'"):
                            continue
                        callers.append((fpath, lineno, stripped))

    # Only one call site should exist in the codebase (the one this PR fixes)
    call_files = set(os.path.relpath(c[0], root) for c in callers)
    print(f"Found {len(callers)} call(s) in files: {call_files}")

    # The call in rl_replacements.py is the only expected caller
    expected_file = os.path.join("unsloth", "models", "rl_replacements.py")
    for fpath, lineno, line in callers:
        rel = os.path.relpath(fpath, root)
        if rel != expected_file:
            print(f"WARNING: unexpected caller at {rel}:{lineno}: {line}")

    print("PASS: all grpo_compute_loss callers accounted for")

if __name__ == "__main__":
    test_no_unexpected_callers()
