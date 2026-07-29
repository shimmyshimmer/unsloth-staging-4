# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One shared answer to "is bitsandbytes usable on this host?".

`import bitsandbytes` succeeding is not that answer. A wheel whose native side
failed to load still imports, and what it leaves behind depends on the version:

  * no `functional` at all, so reading it raises at import - how `main` went red
    with "module 'bitsandbytes' has no attribute 'functional'"
  * `functional.lib is None` (0.45.5, the floor in pyproject.toml), so the ctypes
    binds raise "'NoneType' object has no attribute cdequantize_blockwise_fp32"
  * `functional.lib` is an ErrorHandlerMockBNBNativeLibrary, or a real wrapper
    over a .so missing that symbol (0.46 onwards). These do NOT raise:
    BNBNativeLibrary.__getattr__ returns a plain `throw_on_call` closure, so the
    binds succeed and 4bit dies later inside a kernel with a misleading
    "not available in CPU-only version of bitsandbytes".

The last shape is why attribute access alone is not a probe: a real handle is a
ctypes function pointer, a deferred failure is a Python function.

device_type.py, _gpu_init.py and kernels/utils.py all have to agree on the
answer. If they drift, ALLOW_BITSANDBYTES stays true while the kernels fall back
to the stub, and loader.py forwards a 4bit request that fails downstream instead
of taking the advertised 16bit fallback.

Deliberately a leaf module: it imports nothing from unsloth - not device_type,
which is imported very early and would be a cycle - and takes the device type as
an argument instead. bitsandbytes itself is imported inside a function, so
`import unsloth` never hard-requires it.
"""

__all__ = [
    "bitsandbytes_symbols",
    "check_bitsandbytes",
    "probe_bitsandbytes",
]

# The ctypes handles kernels/utils.py binds at module scope. Keep in step with
# the `bnb.functional.lib.*` reads there - a test asserts the two match.
_C_SYMBOLS = (
    "cdequantize_blockwise_fp32",
    "cdequantize_blockwise_fp16_nf4",
    "cdequantize_blockwise_bf16_nf4",
)
# 4bit inference is a gemv on xpu and a naive gemm everywhere else, so the symbol
# set is device dependent - probing the xpu names on cuda would write off a
# perfectly good wheel.
_C_SYMBOLS_XPU = (
    "cgemv_4bit_inference_fp16",
    "cgemv_4bit_inference_bf16",
)
_C_SYMBOLS_GEMM = (
    "cgemm_4bit_inference_naive_fp16",
    "cgemm_4bit_inference_naive_bf16",
)


def bitsandbytes_symbols(device_type):
    """Names kernels/utils.py reads off `bitsandbytes.functional.lib`."""
    tail = _C_SYMBOLS_XPU if device_type == "xpu" else _C_SYMBOLS_GEMM
    return _C_SYMBOLS + tail


def check_bitsandbytes(bnb, device_type):
    """Raise unless `bnb` can serve every module-scope read kernels/utils.py makes.

    Resolving each symbol is safe to repeat: ctypes caches a function object on
    the first attribute lookup and bitsandbytes memoizes its wrapper, so the real
    handles bound later are the very objects this resolved.
    """
    if bnb is None:
        raise ImportError("Unsloth: `bitsandbytes` is not installed.")
    _version = bnb.__version__  # kernels/utils.py gates HAS_CUDA_STREAM on it
    functional = bnb.functional
    _get_ptr = functional.get_ptr
    lib = functional.lib  # None on a 0.45.5 native-load failure
    for symbol in bitsandbytes_symbols(device_type):
        # `restype` is a ctypes foreign function; bitsandbytes hands back a plain
        # Python closure that defers the failure to call time instead of raising.
        if not hasattr(getattr(lib, symbol), "restype"):
            raise AttributeError(
                f"Unsloth: `bitsandbytes.functional.lib.{symbol}` is not a native "
                "handle - the bitsandbytes native library did not load."
            )


def probe_bitsandbytes(device_type):
    """The bitsandbytes module when it is usable here, else None."""
    try:
        import bitsandbytes
        check_bitsandbytes(bitsandbytes, device_type)
    except Exception:
        return None
    return bitsandbytes
