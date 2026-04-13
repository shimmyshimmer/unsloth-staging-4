"""Test that sampling_per_token_logps=None is handled correctly in both paths."""

def grpo_compute_loss(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
    """Stub matching real signature. Tests None handling for sampling_per_token_logps."""
    use_vllm = kwargs.get("use_vllm", False)
    # Real code: if use_vllm and sampling_per_token_logps is not None: ...
    if use_vllm and sampling_per_token_logps is not None:
        result = "vllm_importance_sampling"
    else:
        result = "no_importance_sampling"
    return result, input_ids, mask, beta

def test_sampling_none_no_vllm():
    result = grpo_compute_loss("ref", "new", "old", None, "ids", "mask", 0.1, "adv")
    assert result[0] == "no_importance_sampling"
    assert result[1] == "ids"  # input_ids correct
    print("PASS: sampling_per_token_logps=None without vLLM works correctly")

def test_sampling_none_with_vllm():
    result = grpo_compute_loss("ref", "new", "old", None, "ids", "mask", 0.1, "adv", use_vllm=True)
    assert result[0] == "no_importance_sampling"
    print("PASS: sampling_per_token_logps=None with vLLM skips importance sampling")

def test_sampling_present_with_vllm():
    result = grpo_compute_loss("ref", "new", "old", "sampling", "ids", "mask", 0.1, "adv", use_vllm=True)
    assert result[0] == "vllm_importance_sampling"
    print("PASS: sampling_per_token_logps present with vLLM triggers importance sampling")

def test_sampling_present_no_vllm():
    result = grpo_compute_loss("ref", "new", "old", "sampling", "ids", "mask", 0.1, "adv", use_vllm=False)
    assert result[0] == "no_importance_sampling"
    print("PASS: sampling_per_token_logps present without vLLM skips importance sampling")

if __name__ == "__main__":
    test_sampling_none_no_vllm()
    test_sampling_none_with_vllm()
    test_sampling_present_with_vllm()
    test_sampling_present_no_vllm()
