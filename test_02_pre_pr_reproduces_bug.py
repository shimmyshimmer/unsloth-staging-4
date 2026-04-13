"""Reproduce the pre-PR bug: passing sampling_per_token_logps as keyword when it's also positionally filled causes TypeError."""

def grpo_compute_loss_stub(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
    """Minimal stub matching the real signature."""
    return ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages

def test_pre_pr_bug():
    """The old code skipped sampling_per_token_logps positionally and passed it as kwarg, causing TypeError."""
    ref, new, old = 1, 2, 3
    sampling = "sampling_val"
    ids, mask_, beta_, adv = "ids", "mask", 0.1, "adv"

    # Old (broken) call: input_ids in slot 4 (where sampling_per_token_logps belongs)
    # plus sampling_per_token_logps as keyword
    try:
        grpo_compute_loss_stub(
            ref, new, old,
            ids,       # WRONG: input_ids in sampling_per_token_logps slot
            mask_,     # WRONG: mask in input_ids slot
            beta_,     # WRONG: beta in mask slot
            adv,       # WRONG: advantages in beta slot
            sampling_per_token_logps=sampling,  # duplicate!
        )
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "multiple values" in str(e), f"Wrong error: {e}"
        print(f"PASS: pre-PR code correctly raises TypeError: {e}")

def test_post_pr_fix():
    """The fixed call passes sampling_per_token_logps in the correct 4th position."""
    ref, new, old = 1, 2, 3
    sampling = "sampling_val"
    ids, mask_, beta_, adv = "ids", "mask", 0.1, "adv"

    result = grpo_compute_loss_stub(
        ref, new, old,
        sampling,  # CORRECT: 4th positional
        ids, mask_, beta_, adv,
    )
    assert result == (ref, new, old, sampling, ids, mask_, beta_, adv)
    print("PASS: post-PR call maps all args correctly")

if __name__ == "__main__":
    test_pre_pr_bug()
    test_post_pr_fix()
