"""PaliGemma loader behavior in FastModel.from_pretrained.

Covers the PEFT model_types enrichment, the PaliGemma v1/v2 version-guard
branch, env-var ordering, and the ImportError pass-through pattern that
matches the canonical AutoConfig.from_pretrained try blocks elsewhere in
unsloth/models/loader.py.
"""

import os
import unittest
from unittest.mock import MagicMock
from packaging.version import Version


def _run_peft_paligemma_enrichment(
    model_types,
    is_peft,
    peft_config,
    AutoConfig,
    get_transformers_model_type,
    logger,
    token = None,
    trust_remote_code = False,
    local_files_only = False,
):
    """Mirrors the PEFT PaliGemma enrichment block in loader.py."""
    if (
        is_peft
        and peft_config is not None
        and "paligemma" in model_types
        and not any(t in ("gemma", "gemma2") for t in model_types)
    ):
        try:
            _base_config = AutoConfig.from_pretrained(
                peft_config.base_model_name_or_path,
                token = token,
                trust_remote_code = trust_remote_code,
                local_files_only = local_files_only,
            )
            _base_model_types = get_transformers_model_type(
                _base_config,
                trust_remote_code = trust_remote_code,
            )
            model_types = sorted(set(model_types) | set(_base_model_types))
        except ImportError:
            raise
        except Exception as _e:
            logger.warning_once(
                "Unsloth: Could not inspect base model config for PaliGemma "
                f"adapter ({type(_e).__name__}: {_e}). "
                "Defaulting to PaliGemma 1 (Gemma backbone). If this is a "
                "PaliGemma 2 adapter, ensure the base model is accessible."
            )
    return model_types


def _run_paligemma_version_branch(
    model_types_all, transformers_version_str, SUPPORTS_GEMMA, env,
):
    """Mirrors the PaliGemma elif branch's version guards and env mutation."""
    if "paligemma" not in model_types_all:
        return
    if "gemma2" in model_types_all:
        if Version(transformers_version_str) < Version("4.42.3"):
            raise RuntimeError("PaliGemma 2 requires transformers >= 4.42.3.")
    else:
        if not SUPPORTS_GEMMA:
            raise RuntimeError("PaliGemma requires transformers >= 4.38.")
    env["UNSLOTH_HIGH_PRECISION_LAYERNORM"] = "1"


class TestPeftEnrichment(unittest.TestCase):
    def setUp(self):
        self.auto_config = MagicMock()
        self.get_types = MagicMock()
        self.logger = MagicMock()
        self.peft = MagicMock()
        self.peft.base_model_name_or_path = "google/paligemma2-3b-pt-224"

    def test_v2_peft_adapter_only_paligemma_gets_gemma2_merged(self):
        self.get_types.return_value = ["gemma2", "paligemma", "siglip"]
        result = _run_peft_paligemma_enrichment(
            model_types = ["paligemma"],
            is_peft = True,
            peft_config = self.peft,
            AutoConfig = self.auto_config,
            get_transformers_model_type = self.get_types,
            logger = self.logger,
        )
        self.assertIn("gemma2", result)
        self.auto_config.from_pretrained.assert_called_once()
        self.logger.warning_once.assert_not_called()

    def test_v1_peft_adapter_only_paligemma_gets_gemma_merged(self):
        self.peft.base_model_name_or_path = "google/paligemma-3b-pt-224"
        self.get_types.return_value = ["gemma", "paligemma", "siglip"]
        result = _run_peft_paligemma_enrichment(
            model_types = ["paligemma"],
            is_peft = True,
            peft_config = self.peft,
            AutoConfig = self.auto_config,
            get_transformers_model_type = self.get_types,
            logger = self.logger,
        )
        self.assertIn("gemma", result)

    def test_skip_lookup_when_backbone_already_present(self):
        result = _run_peft_paligemma_enrichment(
            model_types = ["gemma2", "paligemma", "siglip"],
            is_peft = True,
            peft_config = self.peft,
            AutoConfig = self.auto_config,
            get_transformers_model_type = self.get_types,
            logger = self.logger,
        )
        self.assertEqual(result, ["gemma2", "paligemma", "siglip"])
        self.auto_config.from_pretrained.assert_not_called()

    def test_skip_for_non_peft_load(self):
        result = _run_peft_paligemma_enrichment(
            model_types = ["paligemma"],
            is_peft = False,
            peft_config = None,
            AutoConfig = self.auto_config,
            get_transformers_model_type = self.get_types,
            logger = self.logger,
        )
        self.assertEqual(result, ["paligemma"])
        self.auto_config.from_pretrained.assert_not_called()

    def test_skip_for_non_paligemma_peft(self):
        result = _run_peft_paligemma_enrichment(
            model_types = ["llama"],
            is_peft = True,
            peft_config = self.peft,
            AutoConfig = self.auto_config,
            get_transformers_model_type = self.get_types,
            logger = self.logger,
        )
        self.assertEqual(result, ["llama"])
        self.auto_config.from_pretrained.assert_not_called()

    def test_revision_not_passed_to_base_lookup(self):
        self.get_types.return_value = ["gemma2", "paligemma", "siglip"]
        _run_peft_paligemma_enrichment(
            model_types = ["paligemma"],
            is_peft = True,
            peft_config = self.peft,
            AutoConfig = self.auto_config,
            get_transformers_model_type = self.get_types,
            logger = self.logger,
            token = "hf_xxx",
            trust_remote_code = True,
            local_files_only = True,
        )
        kwargs = self.auto_config.from_pretrained.call_args.kwargs
        self.assertNotIn("revision", kwargs)
        self.assertEqual(kwargs.get("token"), "hf_xxx")
        self.assertTrue(kwargs.get("trust_remote_code"))
        self.assertTrue(kwargs.get("local_files_only"))


class TestEnrichmentExceptions(unittest.TestCase):
    def setUp(self):
        self.auto_config = MagicMock()
        self.get_types = MagicMock()
        self.logger = MagicMock()
        self.peft = MagicMock()
        self.peft.base_model_name_or_path = "google/paligemma2-3b-pt-224"

    def test_import_error_propagates_from_autoconfig(self):
        self.auto_config.from_pretrained.side_effect = ImportError(
            "No module named 'some_required_extension'"
        )
        with self.assertRaises(ImportError):
            _run_peft_paligemma_enrichment(
                model_types = ["paligemma"],
                is_peft = True,
                peft_config = self.peft,
                AutoConfig = self.auto_config,
                get_transformers_model_type = self.get_types,
                logger = self.logger,
            )
        self.logger.warning_once.assert_not_called()

    def test_import_error_propagates_from_get_types(self):
        self.auto_config.from_pretrained.return_value = MagicMock()
        self.get_types.side_effect = ImportError("missing extension")
        with self.assertRaises(ImportError):
            _run_peft_paligemma_enrichment(
                model_types = ["paligemma"],
                is_peft = True,
                peft_config = self.peft,
                AutoConfig = self.auto_config,
                get_transformers_model_type = self.get_types,
                logger = self.logger,
            )

    def test_oserror_warns_and_falls_back(self):
        self.auto_config.from_pretrained.side_effect = OSError("offline")
        result = _run_peft_paligemma_enrichment(
            model_types = ["paligemma"],
            is_peft = True,
            peft_config = self.peft,
            AutoConfig = self.auto_config,
            get_transformers_model_type = self.get_types,
            logger = self.logger,
        )
        self.assertEqual(result, ["paligemma"])
        self.logger.warning_once.assert_called_once()


class TestPaligemmaVersionBranch(unittest.TestCase):
    def test_v2_supported_sets_env(self):
        env = {}
        _run_paligemma_version_branch(
            "gemma2,paligemma,siglip,", "4.57.6", SUPPORTS_GEMMA = True, env = env,
        )
        self.assertEqual(env["UNSLOTH_HIGH_PRECISION_LAYERNORM"], "1")

    def test_v1_supported_sets_env(self):
        env = {}
        _run_paligemma_version_branch(
            "gemma,paligemma,siglip,", "4.57.6", SUPPORTS_GEMMA = True, env = env,
        )
        self.assertEqual(env["UNSLOTH_HIGH_PRECISION_LAYERNORM"], "1")

    def test_v2_unsupported_raises_without_env_leak(self):
        env = {}
        with self.assertRaises(RuntimeError):
            _run_paligemma_version_branch(
                "gemma2,paligemma,siglip,", "4.42.2", SUPPORTS_GEMMA = True, env = env,
            )
        self.assertNotIn("UNSLOTH_HIGH_PRECISION_LAYERNORM", env)

    def test_v1_unsupported_raises_without_env_leak(self):
        env = {}
        with self.assertRaises(RuntimeError):
            _run_paligemma_version_branch(
                "gemma,paligemma,siglip,", "4.37.0", SUPPORTS_GEMMA = False, env = env,
            )
        self.assertNotIn("UNSLOTH_HIGH_PRECISION_LAYERNORM", env)


class TestLoaderSourceShape(unittest.TestCase):
    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "unsloth", "models", "loader.py")) as f:
            self.source = f.read()

    def _paligemma_block(self):
        marker = 'elif "paligemma" in model_types_all:'
        idx = self.source.find(marker)
        self.assertNotEqual(idx, -1)
        end = self.source.find("# Cohere", idx)
        self.assertNotEqual(end, -1)
        return self.source[idx:end]

    def _enrichment_block(self):
        marker = 'not any(t in ("gemma", "gemma2") for t in model_types)'
        idx = self.source.find(marker)
        self.assertNotEqual(idx, -1)
        end = self.source.find('model_types_all = ",".join', idx)
        self.assertNotEqual(end, -1)
        return self.source[idx:end]

    def test_branch_does_not_call_pre_patch(self):
        block = self._paligemma_block()
        self.assertNotIn("FastGemma2Model.pre_patch", block)
        self.assertNotIn("FastGemmaModel.pre_patch", block)

    def test_env_var_ordered_after_version_guards(self):
        block = self._paligemma_block()
        env_pos = block.find('os.environ["UNSLOTH_HIGH_PRECISION_LAYERNORM"] = "1"')
        v2_guard = block.find('Version("4.42.3")')
        v1_guard = block.find("not SUPPORTS_GEMMA")
        self.assertGreater(env_pos, v2_guard)
        self.assertGreater(env_pos, v1_guard)

    def test_enrichment_reraises_import_error_before_generic_catch(self):
        block = self._enrichment_block()
        import_pos = block.find("except ImportError:")
        generic_pos = block.find("except Exception as _e:")
        self.assertNotEqual(import_pos, -1)
        self.assertNotEqual(generic_pos, -1)
        self.assertLess(import_pos, generic_pos)
        self.assertIn("raise", block[import_pos:generic_pos])

    def test_enrichment_does_not_pass_revision(self):
        block = self._enrichment_block()
        self.assertIn("AutoConfig.from_pretrained", block)
        self.assertNotIn("revision", block)
