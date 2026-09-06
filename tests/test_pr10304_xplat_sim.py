# SPDX-License-Identifier: AGPL-3.0-only
"""Torch-free variant of the PR #10304 simulation, for cross-Python-version sandboxes.

save.py cannot be imported without a GPU/torch stack, so both push_to_ollama AND the real
create_ollama_modelfile are ast-extracted and exec'd together against the real
ollama_template_mappers module (stdlib-only, loaded straight off disk). Only the two
subprocess boundaries are stubbed, so this exercises the same code the GPU run does.

  PR10304_TREE=<path to a checkout>  PR10304_STATE=base|head  pytest sim_nodeps.py
"""

import ast
import importlib.util
import inspect
import ntpath
import os
import posixpath
import sys

import pytest

TREE = os.environ["PR10304_TREE"]
STATE = os.environ.get("PR10304_STATE", "head")
SAVE = os.path.join(TREE, "unsloth", "save.py")
MAPPERS = os.path.join(TREE, "unsloth", "ollama_template_mappers.py")

_spec = importlib.util.spec_from_file_location("_ollama_mappers", MAPPERS)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
MAPPER = _m.MODEL_TO_OLLAMA_TEMPLATE_MAPPER
TEMPLATES = _m.OLLAMA_TEMPLATES

MAPPED = "unsloth/llama-3-8b-Instruct"
EOS_MAPPED = "unsloth/llama-2-7b"
UNMAPPED = "unsloth/definitely-not-a-real-model-xyz"


def _extract(*names):
    with open(SAVE, encoding = "utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    missing = set(names) - set(out)
    assert not missing, f"not found in save.py: {missing}"
    return out


def build(stub_modelfile = None):
    """Exec the real caller and the real callee side by side; stub only the subprocess edges."""
    calls = []
    ns = {
        "OLLAMA_TEMPLATES": TEMPLATES,
        "MODEL_TO_OLLAMA_TEMPLATE_MAPPER": MAPPER,
        "create_ollama_model": lambda **kw: calls.append(("create", kw)),
        "push_to_ollama_hub": lambda **kw: calls.append(("push", kw)),
    }
    srcs = _extract("push_to_ollama", "create_ollama_modelfile")
    exec(compile(srcs["create_ollama_modelfile"], "create_ollama_modelfile", "exec"), ns)
    if stub_modelfile is not ...:
        pass
    exec(compile(srcs["push_to_ollama"], "push_to_ollama", "exec"), ns)
    return ns, calls


class Tok:
    def __init__(self, eos = "<|eot_id|>"):
        self.eos_token = eos


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns, calls = build()
    return tmp_path, ns, calls


def call(ns, **kw):
    if STATE == "base":
        kw.pop("base_model_name", None)
    return ns["push_to_ollama"](**kw)


def test_state_matches_tree(sandbox):
    _, ns, _ = sandbox
    params = list(inspect.signature(ns["push_to_ollama"]).parameters)
    if STATE == "head":
        assert params == ["tokenizer", "base_model_name", "gguf_location",
                          "username", "model_name", "tag"]
    else:
        assert params == ["tokenizer", "gguf_location", "username", "model_name", "tag"]


def test_base_is_uncallable(sandbox):
    if STATE != "base":
        pytest.skip("base-only")
    _, ns, _ = sandbox
    with pytest.raises(TypeError, match = "gguf_location"):
        ns["push_to_ollama"](Tok(), "./m.gguf", "u", "n", "latest")


PATHS = [
    "./model.gguf", "model.gguf", "sub/dir/model.gguf", "../sibling/model.gguf",
    "/abs/path/model.gguf", "C:\\Users\\Daniel\\model.gguf", ".\\model.gguf",
    "//server/share/model.gguf", "my models/unsloth model.gguf",
    "\u6a21\u578b/\u6a21\u578b.gguf", "\u043c\u043e\u0434\u0435\u043b\u044c.gguf",
    "emoji-\U0001f9a5/model.gguf", "weird{braces}/model.gguf", "quo'te/mo\"del.gguf",
]


@pytest.mark.parametrize("gguf", PATHS)
def test_from_line_verbatim(sandbox, gguf):
    if STATE == "base":
        pytest.skip("unreachable on base")
    tmp, ns, _ = sandbox
    call(ns, tokenizer = Tok(), base_model_name = MAPPED, gguf_location = gguf,
         username = "u", model_name = "m", tag = "latest")
    text = (tmp / "Modelfile_m").read_text(encoding = "utf-8")
    assert f"FROM {gguf}" in text


def test_no_placeholder_survives(sandbox):
    if STATE == "base":
        pytest.skip("unreachable on base")
    tmp, ns, _ = sandbox
    call(ns, tokenizer = Tok(), base_model_name = MAPPED, gguf_location = "./m.gguf",
         username = "u", model_name = "m", tag = "latest")
    text = (tmp / "Modelfile_m").read_text(encoding = "utf-8")
    assert "__FILE_LOCATION__" not in text and "__EOS_TOKEN__" not in text


def test_eos_substituted(sandbox):
    if STATE == "base":
        pytest.skip("unreachable on base")
    tmp, ns, _ = sandbox
    call(ns, tokenizer = Tok("<|custom_eos|>"), base_model_name = EOS_MAPPED,
         gguf_location = "./m.gguf", username = "u", model_name = "m", tag = "latest")
    assert "<|custom_eos|>" in (tmp / "Modelfile_m").read_text(encoding = "utf-8")


def test_unmapped_raises_and_leaves_nothing(sandbox):
    tmp, ns, calls = sandbox
    exc = TypeError if STATE == "base" else RuntimeError
    with pytest.raises(exc):
        call(ns, tokenizer = Tok(), base_model_name = UNMAPPED, gguf_location = "./m.gguf",
             username = "u", model_name = "m", tag = "latest")
    assert not (tmp / "Modelfile_m").exists()
    assert calls == []


def test_call_order_and_args(sandbox):
    if STATE == "base":
        pytest.skip("unreachable on base")
    _, ns, calls = sandbox
    call(ns, tokenizer = Tok(), base_model_name = MAPPED, gguf_location = "./m.gguf",
         username = "danielhanchen", model_name = "my-model", tag = "v1")
    assert [c[0] for c in calls] == ["create", "push"]
    assert calls[0][1] == dict(username = "danielhanchen", model_name = "my-model",
                               tag = "v1", modelfile_path = "Modelfile_my-model")


def test_modelfile_name_portable(sandbox):
    if STATE == "base":
        pytest.skip("unreachable on base")
    tmp, ns, _ = sandbox
    call(ns, tokenizer = Tok(), base_model_name = MAPPED, gguf_location = "./m.gguf",
         username = "u", model_name = "my-model", tag = "latest")
    fname = "Modelfile_my-model"
    assert not set(fname) & set('<>:"/\\|?*')
    assert ntpath.basename(fname) == fname == posixpath.basename(fname)
    assert (tmp / fname).exists()


def test_every_mapped_model(sandbox):
    if STATE == "base":
        pytest.skip("unreachable on base")
    tmp, ns, calls = sandbox
    bad = []
    for i, name in enumerate(sorted(MAPPER)):
        calls.clear()
        try:
            call(ns, tokenizer = Tok(), base_model_name = name,
                 gguf_location = "./unsloth.gguf", username = "u",
                 model_name = f"m{i}", tag = "latest")
            text = (tmp / f"Modelfile_m{i}").read_text(encoding = "utf-8")
            if "FROM ./unsloth.gguf" not in text:
                bad.append((name, "no FROM"))
            elif "__FILE_LOCATION__" in text or "__EOS_TOKEN__" in text:
                bad.append((name, "placeholder"))
        except Exception as e:  # noqa: BLE001
            bad.append((name, f"{type(e).__name__}: {e}"))
    assert not bad, bad[:5]
    print(f"\nswept {len(MAPPER)} mapped ids on Python {sys.version.split()[0]}")
