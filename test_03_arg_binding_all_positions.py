"""Verify all 8 positional arguments bind to the correct parameters."""
import inspect

def grpo_compute_loss(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
    return {
        "ref": ref, "new": new, "old": old,
        "sampling_per_token_logps": sampling_per_token_logps,
        "input_ids": input_ids, "mask": mask,
        "beta": beta, "advantages": advantages,
        "kwargs": kwargs,
    }

def test_all_positions():
    vals = {
        "ref": "REF", "new": "NEW", "old": "OLD",
        "sampling_per_token_logps": "SAMPLING",
        "input_ids": "IDS", "mask": "MASK",
        "beta": 0.04, "advantages": "ADV",
    }
    result = grpo_compute_loss(
        vals["ref"], vals["new"], vals["old"],
        vals["sampling_per_token_logps"],
        vals["input_ids"], vals["mask"],
        vals["beta"], vals["advantages"],
        loss_type="grpo", epsilon_low=0.2,
    )
    for key in ["ref", "new", "old", "sampling_per_token_logps", "input_ids", "mask", "beta", "advantages"]:
        assert result[key] == vals[key], f"{key}: expected {vals[key]}, got {result[key]}"
    assert result["kwargs"]["loss_type"] == "grpo"
    assert result["kwargs"]["epsilon_low"] == 0.2
    print("PASS: all 8 positional args + kwargs bind correctly")

def test_shifted_args_wrong_binding():
    """If sampling_per_token_logps is skipped, input_ids ends up in its slot."""
    vals = {"ref": "R", "new": "N", "old": "O", "input_ids": "IDS",
            "mask": "M", "beta": 0.1, "advantages": "A"}
    result = grpo_compute_loss(
        vals["ref"], vals["new"], vals["old"],
        vals["input_ids"],  # WRONG slot
        vals["mask"], vals["beta"], vals["advantages"],
        "extra_positional",
    )
    # input_ids lands in sampling_per_token_logps slot
    assert result["sampling_per_token_logps"] == "IDS", "Shifted binding confirmed"
    assert result["input_ids"] == "M", "mask lands in input_ids slot"
    print("PASS: shifted args produce wrong bindings as expected")

if __name__ == "__main__":
    test_all_positions()
    test_shifted_args_wrong_binding()
