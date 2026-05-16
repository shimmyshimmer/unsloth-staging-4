"""Extended guardrail coverage: int8 8-bit payload accounting, mlp.gate_proj
partial-bypass detection, user-supplied quantization_config resolution, and
the correct bnb class name in the total-bypass warning."""

import warnings

import torch
import torch.nn as nn

from unsloth.models.vision import (
    _GUARDRAIL_BULK_WEIGHT_NUMEL,
    _warn_if_quantization_silently_dropped,
)


class _PretendLinear4bit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(_GUARDRAIL_BULK_WEIGHT_NUMEL + 1, dtype = torch.uint8),
            requires_grad = False,
        )


_PretendLinear4bit.__name__ = "Linear4bit"


class _PretendLinear8bitLt(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(_GUARDRAIL_BULK_WEIGHT_NUMEL + 1, dtype = torch.int8),
            requires_grad = False,
        )


_PretendLinear8bitLt.__name__ = "Linear8bitLt"


class _LargeBF16Param(nn.Module):
    def __init__(self, name = "extra_param"):
        super().__init__()
        setattr(
            self,
            name,
            nn.Parameter(
                torch.zeros(_GUARDRAIL_BULK_WEIGHT_NUMEL + 1, dtype = torch.bfloat16),
                requires_grad = False,
            ),
        )


def test_silent_when_8bit_int8_payload_succeeded():
    model = nn.Sequential(_PretendLinear8bitLt())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = True,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs
    assert not any("0.00 GB" in m for m in msgs), msgs


def test_fires_on_partial_8bit_with_fused_expert_in_bf16():
    model = nn.Sequential(_PretendLinear8bitLt(), _LargeBF16Param("gate_up_proj"))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = True,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs
    assert any("gate_up_proj" in m for m in msgs), msgs


def test_fires_on_partial_quant_with_mlp_gate_proj():
    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Parameter(
                torch.zeros(_GUARDRAIL_BULK_WEIGHT_NUMEL + 1, dtype = torch.bfloat16),
                requires_grad = False,
            )

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = _PretendLinear4bit()
            self.mlp = _MLP()

    model = _Block()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs
    assert any("mlp.gate_proj" in m for m in msgs), msgs


def test_fires_on_partial_quant_with_mlp_gate_up_proj():
    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(
                torch.zeros(_GUARDRAIL_BULK_WEIGHT_NUMEL + 1, dtype = torch.bfloat16),
                requires_grad = False,
            )

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = _PretendLinear4bit()
            self.mlp = _MLP()

    model = _Block()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs


def test_router_and_block_sparse_moe_gate_still_skipped():
    class _Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Parameter(
                torch.zeros(_GUARDRAIL_BULK_WEIGHT_NUMEL + 1, dtype = torch.bfloat16),
                requires_grad = False,
            )

    class _MoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = _PretendLinear4bit()
            self.router = nn.Parameter(
                torch.zeros(_GUARDRAIL_BULK_WEIGHT_NUMEL + 1, dtype = torch.bfloat16),
                requires_grad = False,
            )
            self.block_sparse_moe = _Router()

    model = _MoEBlock()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs


def test_quantization_config_object_revives_total_bypass_warning():
    class _BNB:
        load_in_4bit = True
        load_in_8bit = False

    model = nn.Sequential(nn.Linear(4, 4))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = False,
            full_finetuning = False,
            quantization_config = _BNB(),
        )
    msgs = [str(w.message) for w in caught]
    assert any("load_in_4bit=True was requested" in m for m in msgs), msgs


def test_quantization_config_dict_revives_total_bypass_warning():
    model = nn.Sequential(nn.Linear(4, 4))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = False,
            full_finetuning = False,
            quantization_config = {"load_in_8bit": True},
        )
    msgs = [str(w.message) for w in caught]
    assert any("load_in_8bit=True was requested" in m for m in msgs), msgs


def test_total_bypass_message_uses_linear8bitlt_name():
    model = nn.Sequential(nn.Linear(4, 4))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = True,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("Linear8bitLt" in m for m in msgs), msgs
    assert not any("Linear8bit modules" in m for m in msgs), msgs


def test_total_bypass_message_uses_linear4bit_name():
    model = nn.Sequential(nn.Linear(4, 4))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("Linear4bit" in m for m in msgs), msgs


def test_partial_bypass_silent_when_quantized_bytes_zero():
    class _ExoticQuant(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.zeros(1, dtype = torch.float16),
                requires_grad = False,
            )

    _ExoticQuant.__name__ = "Linear4bit"
    model = nn.Sequential(_ExoticQuant(), _LargeBF16Param("extra_param"))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs
