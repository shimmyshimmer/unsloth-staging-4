"""`import unsloth` must survive a missing bitsandbytes.

device_type.py already tells the user "bitsandbytes is not installed - 4bit QLoRA
unallowed, but 16bit and full finetuning works", and the gfx906 install path
(#7354) deliberately removes the generic wheel because it carries no gfx906
kernels. Any module-level `import bitsandbytes` on the import chain turns that
into an unimportable package instead.

peft's 4bit LoRA layer is exported only when bnb is importable, so
`from peft.tuners.lora import Linear4bit` fails on the same hosts and is checked
here too.
"""

# Path | None below is a PEP 604 union; the project still supports Python 3.9.
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_MODULE = "unsloth"


def _module_path(name: str) -> Path | None:
    base = REPO_ROOT / Path(*name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _bnb_dependent(node: ast.stmt) -> bool:
    """True for an import that raises when bitsandbytes is absent."""
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == "bitsandbytes" for a in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        module = node.module or ""
        if module.split(".")[0] == "bitsandbytes":
            return True
        # peft re-exports Linear4bit only when bnb imported cleanly.
        if module.startswith("peft.tuners.lora"):
            return any(a.name == "Linear4bit" for a in node.names)
    return False


def _allow_bitsandbytes_gated(test: ast.expr) -> bool:
    """device_type.py sets ALLOW_BITSANDBYTES=False exactly when the import failed,
    so a branch keyed on it cannot run without bnb."""
    return any(isinstance(n, ast.Name) and n.id == "ALLOW_BITSANDBYTES" for n in ast.walk(test))


def _scan(path: Path, module: str):
    """Yield (lineno, source) for unguarded top-level imports.

    Imports inside a `try`, or under an ALLOW_BITSANDBYTES branch, are guarded.
    Other `if` bodies are not: the condition may well be true on a host without bnb.
    """
    is_package = path.name == "__init__.py"
    package = module if is_package else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    risky, edges = [], []

    def walk(body, guarded):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not guarded and _bnb_dependent(node):
                    risky.append((node.lineno, ast.unparse(node)))
                if isinstance(node, ast.Import):
                    edges.extend(a.name for a in node.names)
                elif node.level:
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - (node.level - 1)])
                    edges.append(f"{base}.{node.module}" if node.module else base)
                else:
                    edges.append(node.module or "")
            elif isinstance(node, ast.Try):
                walk(node.body, True)
                for handler in node.handlers:
                    walk(handler.body, True)
                walk(node.orelse, True)
                walk(node.finalbody, guarded)
            elif isinstance(node, ast.If):
                walk(node.body, guarded or _allow_bitsandbytes_gated(node.test))
                walk(node.orelse, guarded)

    walk(tree.body, False)
    return risky, edges


def test_no_unguarded_bitsandbytes_import_on_the_unsloth_import_chain():
    seen, pending, offenders = set(), [(ROOT_MODULE, [])], []
    while pending:
        module, chain = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        risky, edges = _scan(path, module)
        for lineno, source in risky:
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel}:{lineno}  {source}\n    via {' -> '.join(chain + [module])}")
        pending.extend(
            (edge, chain + [module]) for edge in edges if edge.split(".")[0] == ROOT_MODULE
        )

    assert len(seen) > 20, f"import chain walk collapsed, only reached {seen}"
    assert not offenders, (
        "`import unsloth` must not hard-require bitsandbytes. Wrap these in "
        "try/except and fall back to a placeholder:\n  " + "\n  ".join(offenders)
    )


def test_missing_bnb_leaves_a_callable_that_reports_the_real_cause():
    """The 4bit ctypes handles degrade to a stub, not a NameError later on."""
    src = (REPO_ROOT / "unsloth" / "kernels" / "utils.py").read_text(encoding = "utf-8")
    assert "def _bnb_required(" in src
    assert "get_ptr = _bnb_required" in src
    for name in (
        "cdequantize_blockwise_fp32",
        "cdequantize_blockwise_fp16_nf4",
        "cdequantize_blockwise_bf16_nf4",
        "cgemm_4bit_inference_naive_fp16",
        "cgemm_4bit_inference_naive_bf16",
    ):
        assert f"{name} = _bnb_required" in src, f"{name} has no bnb-less fallback"


def test_capability_flags_come_from_a_guarded_import_not_find_spec():
    """device_type.py, _gpu_init.py and kernels/utils.py must share one probe.

    If they drift, an installed-but-unusable wheel leaves ALLOW_BITSANDBYTES true
    while the kernels fall back to the stub, and loader.py forwards the default
    load_in_4bit=True instead of taking the advertised 16bit fallback.
    """
    head = (REPO_ROOT / "unsloth" / "device_type.py").read_text(encoding = "utf-8")
    head = head.split('if DEVICE_TYPE == "hip":')[0]
    assert "probe_bitsandbytes(DEVICE_TYPE)" in head
    assert 'find_spec("bitsandbytes")' not in head, "find_spec cannot see a broken wheel"
    assert head.count("ALLOW_BITSANDBYTES = False") >= 1

    kernels = (REPO_ROOT / "unsloth" / "kernels" / "utils.py").read_text(encoding = "utf-8")
    assert "bnb = BITSANDBYTES" in kernels, "the kernels must reuse the shared result"
    reimports = [node.lineno for node in ast.walk(ast.parse(kernels)) if _bnb_dependent(node)]
    assert not reimports, f"a second bnb import can disagree with the flags: {reimports}"

    gpu_init = (REPO_ROOT / "unsloth" / "_gpu_init.py").read_text(encoding = "utf-8")
    assert "_check_bitsandbytes(bnb, DEVICE_TYPE)" in gpu_init


def _bnb_guards():
    src = (REPO_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    return src, [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Name) and n.id == "ALLOW_BITSANDBYTES" for n in ast.walk(node.test)
        )
    ]


def test_bitsandbytes_guard_is_not_gated_on_use_exact_model_name():
    """use_exact_model_name suppresses repo-name remapping; it cannot make bnb
    available. Gating on it left the default load_in_4bit=True set on a host
    without bitsandbytes."""
    _, guards = _bnb_guards()
    assert len(guards) == 2, f"expected both loader guards, found {len(guards)}"
    for guard in guards:
        names = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
        assert (
            "use_exact_model_name" not in names
        ), f"guard at line {guard.lineno} still gates the capability check on naming"


def test_bitsandbytes_guard_drops_a_bnb_quantization_config():
    """A BitsAndBytesConfig in kwargs re-sets the flags downstream, so clearing
    load_in_4bit/8bit alone still builds the bnb quantizer in Transformers. A
    non-bnb config (GPTQ/AWQ/fp8) must not be touched."""
    _, guards = _bnb_guards()
    for guard in guards:
        # ast.unparse normalises quotes, so match on the call shape instead.
        def _is_pop(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "kwargs"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "quantization_config"
            )

        assert any(
            _is_pop(n) for n in ast.walk(guard)
        ), f"guard at line {guard.lineno} leaves the bnb config in kwargs"
        # the pop must be conditional on the config actually asking for bnb
        pops = [
            node
            for node in ast.walk(guard)
            if isinstance(node, ast.If) and any(_is_pop(n) for n in ast.walk(node))
        ]
        assert pops, f"guard at line {guard.lineno} pops unconditionally"
        assert any(
            isinstance(n, ast.Name) and n.id == "_wants_bnb"
            for node in pops
            for n in ast.walk(node.test)
        ), f"guard at line {guard.lineno} does not gate the pop on a bnb request"


def test_bitsandbytes_guard_clears_8bit_as_well_as_4bit():
    """8bit is bitsandbytes too: leaving load_in_8bit set sends the request to
    Transformers, which builds the bnb quantizer and fails there instead."""
    src = (REPO_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Name) and n.id == "ALLOW_BITSANDBYTES" for n in ast.walk(node.test)
        )
    ]
    assert len(guards) == 2, f"expected both loader guards, found {len(guards)}"
    for guard in guards:
        cleared = {
            target.id
            for stmt in guard.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is False
        }
        assert {
            "load_in_4bit",
            "load_in_8bit",
        } <= cleared, f"guard at line {guard.lineno} clears only {sorted(cleared)}"


def test_capability_fallback_precedes_the_mutually_exclusive_mode_check():
    """load_in_4bit defaults to True, so load_in_16bit=True trips the
    "can only load in 4bit or 8bit or 16bit" RuntimeError unless the unavailable
    4bit request is cleared first. That check must come after the fallback."""
    src, _ = _bnb_guards()
    tree = ast.parse(src)
    checked = 0
    # Scope to the enclosing function: the other loader's guard sits earlier in the
    # file and would otherwise satisfy a plain line-number comparison.
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        raises = [
            node.lineno
            for node in ast.walk(func)
            if isinstance(node, ast.Raise)
            and "Can only load in 4bit or 8bit or 16bit" in ast.unparse(node)
        ]
        if not raises:
            continue
        guards = [
            node.lineno
            for node in ast.walk(func)
            if isinstance(node, ast.If)
            and any(
                isinstance(n, ast.Name) and n.id == "ALLOW_BITSANDBYTES"
                for n in ast.walk(node.test)
            )
        ]
        for lineno in raises:
            checked += 1
            assert any(g < lineno for g in guards), (
                f"{func.name}: the mode check at line {lineno} runs before this "
                "function's ALLOW_BITSANDBYTES fallback, so load_in_16bit=True on a "
                "bnb-less host raises instead of taking the 16bit path"
            )
    assert checked, "mode-exclusivity check not found"


def test_bitsandbytes_compile_patch_is_never_called_unguarded():
    """unsloth_zoo's patch_compiling_bitsandbytes imports bitsandbytes
    unconditionally, so an unwrapped call raises on a bnb-less host before any
    fallback can run."""
    src = (REPO_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "patch_compiling_bitsandbytes"
    ]
    assert calls, "call sites not found"
    guarded = {
        call.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "patch_compiling_bitsandbytes"
    }
    unguarded = sorted({c.lineno for c in calls} - guarded)
    assert not unguarded, f"patch_compiling_bitsandbytes called unguarded at {unguarded}"


def _load_shared_probe():
    """Import unsloth/bnb_availability.py straight off disk.

    By file path, not as ``unsloth.bnb_availability``: that would run the package
    __init__ and pull in torch. It only works because the module is a leaf, which
    is the property that lets device_type.py (imported very early) and
    _gpu_init.py both use it without a cycle.
    """
    path = REPO_ROOT / "unsloth" / "bnb_availability.py"
    spec = importlib.util.spec_from_file_location("unsloth_bnb_availability_undertest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shared_probe_covers_every_module_scope_bitsandbytes_symbol():
    """kernels/utils.py binds these ctypes handles off ``bnb.functional.lib`` at
    import time, so a wheel missing any one of them raised there. Probing only
    ``get_ptr`` covers one broken shape out of several."""
    tree = ast.parse((REPO_ROOT / "unsloth" / "kernels" / "utils.py").read_text(encoding = "utf-8"))
    bound = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "lib"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "functional"
    }
    probe = _load_shared_probe()
    xpu, cuda = set(probe.bitsandbytes_symbols("xpu")), set(probe.bitsandbytes_symbols("cuda"))
    # xpu binds the gemv pair and every other device the naive gemm pair, so the
    # union is what kernels/utils.py can ask for across devices.
    assert bound == xpu | cuda, f"probe and module-scope binds differ: {bound ^ (xpu | cuda)}"
    # and neither device probes the other's symbols, which its wheel will not have
    assert xpu - cuda and cuda - xpu, "the device split collapsed"


def test_the_probe_rejects_a_handle_that_only_fails_when_called():
    """bitsandbytes 0.46 onwards does not raise for a symbol its .so lacks.

    ``BNBNativeLibrary.__getattr__`` returns a plain ``throw_on_call`` closure,
    and a wheel whose native library failed to load is replaced wholesale by
    ``ErrorHandlerMockBNBNativeLibrary``, which does that for every name. So the
    ctypes binds succeed, ALLOW_BITSANDBYTES stays true, and 4bit dies inside a
    kernel instead of falling back to 16bit. Run against the real classes.
    """
    if importlib.util.find_spec("bitsandbytes") is None:
        return
    try:
        from bitsandbytes.cextension import ErrorHandlerMockBNBNativeLibrary
    except ImportError:
        # 0.45.5, the floor in pyproject.toml, predates the mock class. That
        # release fails differently (functional.lib is None), which the
        # lib_is_none shape already covers.
        return

    probe = _load_shared_probe()
    functional = types.ModuleType("bitsandbytes.functional")
    functional.get_ptr = lambda tensor: None
    functional.lib = ErrorHandlerMockBNBNativeLibrary(
        "libbitsandbytes_cuda128.so: cannot open shared object file"
    )
    bnb = types.ModuleType("bitsandbytes")
    bnb.__version__ = "0.50.0"
    bnb.functional = functional

    assert not hasattr(functional.lib.cdequantize_blockwise_fp32, "restype")
    for device in ("cuda", "xpu"):
        try:
            probe.check_bitsandbytes(bnb, device)
        except Exception:
            continue
        raise AssertionError(f"the probe accepted a dead native library on {device}")


# A wheel whose native side failed to load. It imports fine and only raises when
# the kernels are read, which is how `Core` went red on all three legs with
# "module 'bitsandbytes' has no attribute 'functional'".
#
#   no_functional       `functional` is never bound
#   lib_is_none         `functional.lib is None`, the bitsandbytes 0.45.5 fallback
#                       when the native library fails to load
#   lib_missing_kernel  `functional.get_ptr` resolves but `functional.lib` has no
#                       cgemm_4bit_inference_naive_bf16
#   lib_defers_failure  every lookup resolves to a closure that raises only when
#                       called - what bitsandbytes 0.46 onwards leaves behind, via
#                       BNBNativeLibrary.__getattr__ / ErrorHandlerMockBNBNativeLibrary
_BROKEN_WHEEL_PROBE = """
import importlib
import importlib.machinery
import os
import sys
import types

SHAPE = os.environ["BROKEN_BNB_SHAPE"]


def _apply_conftest_cpu_harness():
    \"\"\"Run tests/conftest.py here too.

    Its accelerator spoof is in-process, so a subprocess inherits none of it and
    `import unsloth` raises "cannot find any torch accelerator" on the CPU-only
    repo runners, before any assertion below is reached. Executing the very same
    file, rather than restating it, keeps the two from drifting; it no-ops on a
    host with a real accelerator, so DEVICE_TYPE and the device-dependent symbol
    sets stay honest. The stand-in is installed first, so the device_type it
    pre-loads probes the broken wheel.
    \"\"\"
    import runpy

    root = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    runpy.run_path(os.path.join(root, "tests", "conftest.py"))


# ctypes raises AttributeError for a symbol the .so does not export, and caches
# the function object on the first attribute lookup. Its handles carry `restype`,
# which is how the probe tells them from a deferred failure.
class _FakeCDLL:
    def __init__(self, symbols):
        self.__dict__["_symbols"] = symbols

    def __getattr__(self, name):
        if name.startswith("_") or name not in self._symbols:
            raise AttributeError(f"{name} not found")

        def handle(*args, **kwargs):
            return None

        handle.__name__ = name
        handle.restype = None
        setattr(self, name, handle)
        return handle


# bitsandbytes >= 0.46: a missing symbol, or a native library that never loaded,
# resolves to a plain Python closure that raises only when it is called.
class _DeferredFailureLib:
    def __getattr__(self, name):
        def throw_on_call(*args, **kwargs):
            raise RuntimeError(f"Method '{name}' not available in CPU-only version")

        return throw_on_call


# A real loader, so the stand-in survives a reload. importlib.reload re-runs
# sys.meta_path instead of reusing the module's __spec__, so a stand-in without
# one is silently repopulated from the real installed wheel -- and _gpu_init.py
# reloads bnb whenever os.geteuid() == 0.
class _BrokenLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__version__ = "0.50.0"
        module.__dict__.pop("functional", None)
        sys.modules.pop("bitsandbytes.functional", None)
        if SHAPE == "no_functional":
            return
        functional = types.ModuleType("bitsandbytes.functional")
        functional.get_ptr = lambda tensor: None
        if SHAPE == "lib_is_none":
            functional.lib = None
        elif SHAPE == "lib_defers_failure":
            functional.lib = _DeferredFailureLib()
        else:
            functional.lib = _FakeCDLL({
                "cdequantize_blockwise_fp32",
                "cdequantize_blockwise_fp16_nf4",
                "cdequantize_blockwise_bf16_nf4",
                "cgemm_4bit_inference_naive_fp16",
            })
        module.functional = functional
        sys.modules["bitsandbytes.functional"] = functional


class _BrokenFinder:
    def find_spec(self, fullname, path = None, target = None):
        if fullname != "bitsandbytes":
            return None
        return importlib.machinery.ModuleSpec(
            fullname, _BrokenLoader(), origin = "<broken bitsandbytes wheel>",
        )


sys.meta_path.insert(0, _BrokenFinder())

# _gpu_init.py's recovery path shells out to ldconfig. A root CI container would
# rebuild the host's linker cache on every one of these runs, so stub it always.
os.system = lambda command: 0
if os.environ.get("BROKEN_BNB_FAKE_ROOT") == "1":
    # That path is taken as root, which training containers usually are. Keep its
    # importlib.reload(bnb) live -- that is the point of the case.
    os.geteuid = lambda: 0

import bitsandbytes

reloaded = importlib.reload(bitsandbytes)
assert reloaded is sys.modules["bitsandbytes"] is bitsandbytes, reloaded
# The spec has no location, so import never sets __file__. The real wheel would.
assert not hasattr(bitsandbytes, "__file__"), bitsandbytes.__file__
assert hasattr(bitsandbytes, "functional") == (SHAPE != "no_functional"), SHAPE

_apply_conftest_cpu_harness()

import unsloth.kernels.utils as utils
from unsloth.device_type import ALLOW_BITSANDBYTES, ALLOW_PREQUANTIZED_MODELS

assert utils.bnb is None, utils.bnb
assert utils.get_ptr.__name__ == "_bnb_required", utils.get_ptr
assert utils.HAS_CUDA_STREAM is False, utils.HAS_CUDA_STREAM
for _name in (
    "cdequantize_blockwise_fp32",
    "cdequantize_blockwise_fp16_nf4",
    "cdequantize_blockwise_bf16_nf4",
    "cgemm_4bit_inference_naive_fp16",
    "cgemm_4bit_inference_naive_bf16",
):
    assert getattr(utils, _name).__name__ == "_bnb_required", _name
assert ALLOW_BITSANDBYTES is False, "the capability flag disagrees with the kernels"
assert ALLOW_PREQUANTIZED_MODELS is False
print("BROKEN_WHEEL_OK")
"""


def _run_broken_wheel_probe(shape, fake_root = False):
    env = dict(os.environ, BROKEN_BNB_SHAPE = shape)
    if fake_root:
        env["BROKEN_BNB_FAKE_ROOT"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _BROKEN_WHEEL_PROBE],
        cwd = REPO_ROOT,
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "BROKEN_WHEEL_OK" in proc.stdout


def test_a_wheel_that_imports_without_functional_degrades_to_the_stub():
    """Guarding only the import statement is not enough.

    ``import bitsandbytes`` can succeed while leaving ``bitsandbytes.functional``
    unbound, so reading it at module scope raised at import time, which is the
    failure the guard exists to prevent. Behavioural on purpose: the previous
    guard passed every source-level check in this file and still broke.
    """
    _run_broken_wheel_probe("no_functional")


def test_a_wheel_whose_lib_failed_to_load_degrades_to_the_stub():
    """bitsandbytes 0.45.x, the floor in pyproject.toml, sets
    ``functional.lib = None`` when the native library will not load. Probing
    ``functional.get_ptr`` alone still leaves the ctypes binds raising
    ``'NoneType' object has no attribute 'cdequantize_blockwise_fp32'``."""
    _run_broken_wheel_probe("lib_is_none")


def test_a_wheel_missing_one_kernel_degrades_to_the_stub():
    """A partially loaded or backend-mismatched wheel resolves ``functional.lib``
    and most of its symbols. ctypes raises AttributeError on the first one it does
    not export, so every symbol bound at module scope has to be probed."""
    _run_broken_wheel_probe("lib_missing_kernel")


def test_a_wheel_whose_lib_only_fails_on_call_degrades_to_the_stub():
    """The shape bitsandbytes 0.46 onwards actually produces. Nothing raises while
    the ctypes handles are bound, so a probe made only of attribute reads sees a
    healthy wheel and ALLOW_BITSANDBYTES stays true."""
    _run_broken_wheel_probe("lib_defers_failure")


def test_the_broken_wheel_stand_in_survives_the_root_recovery_reload():
    """_gpu_init.py reloads bitsandbytes on its ``os.geteuid() == 0`` path, and
    reload re-runs the finders rather than reusing the module's ``__spec__``. A
    stand-in without a real loader is therefore repopulated from the installed
    wheel there, making the test pass or fail on whether the container is root -
    and root GPU containers are the common case for training."""
    _run_broken_wheel_probe("no_functional", fake_root = True)
