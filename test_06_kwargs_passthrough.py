"""Test that keyword arguments after positional args pass through correctly."""

def grpo_compute_loss(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
    return kwargs

def test_all_kwargs_pass_through():
    expected_kwargs = {
        "pixel_values": "pv",
        "image_grid_thw": "igt",
        "loss_type": "grpo",
        "importance_sampling_level": "token",
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "max_completion_length": 512,
        "delta": None,
        "temperature": 1.0,
        "max_left_pad": 0,
        "logit_softcapping": 0,
        "logit_scale_multiply": 0,
        "logit_scale_divide": 0,
        "num_items_in_batch": 4,
        "current_gradient_accumulation_steps": 1,
        "num_processes": 1,
    }
    result = grpo_compute_loss(
        "ref", "new", "old", "sampling", "ids", "mask", 0.04, "adv",
        **expected_kwargs,
    )
    for k, v in expected_kwargs.items():
        assert result[k] == v, f"kwarg {k}: expected {v}, got {result.get(k)}"
    # sampling_per_token_logps must NOT appear in kwargs
    assert "sampling_per_token_logps" not in result, \
        "sampling_per_token_logps leaked into kwargs"
    print("PASS: all kwargs pass through correctly, sampling_per_token_logps not in kwargs")

if __name__ == "__main__":
    test_all_kwargs_pass_through()
