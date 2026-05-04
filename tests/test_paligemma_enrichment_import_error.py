"""Verify the PEFT PaliGemma enrichment block re-raises ImportError.

Other AutoConfig.from_pretrained try blocks in unsloth/models/loader.py
(lines 445-455 and 1060-1070) follow the canonical pattern of re-raising
ImportError before catching the generic Exception. The PEFT PaliGemma
enrichment block must follow the same pattern, otherwise a missing
dependency surfaced while loading a trust_remote_code base config gets
silently downgraded to a warning.
"""

import os
import unittest
from unittest.mock import MagicMock


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


class TestEnrichmentImportErrorPropagation(unittest.TestCase):
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
                trust_remote_code = True,
            )
        self.logger.warning_once.assert_not_called()

    def test_import_error_propagates_from_get_transformers_model_type(self):
        self.auto_config.from_pretrained.return_value = MagicMock()
        self.get_types.side_effect = ImportError("missing transformers extension")
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

    def test_oserror_still_warns_and_falls_back(self):
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

    def test_value_error_still_warns_and_falls_back(self):
        self.auto_config.from_pretrained.side_effect = ValueError("bad config")
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


class TestLoaderEnrichmentSourceShape(unittest.TestCase):
    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "unsloth", "models", "loader.py")) as f:
            self.source = f.read()

    def _enrichment_block(self):
        marker = 'not any(t in ("gemma", "gemma2") for t in model_types)'
        idx = self.source.find(marker)
        self.assertNotEqual(idx, -1, "PEFT PaliGemma enrichment block missing")
        end = self.source.find('model_types_all = ",".join', idx)
        self.assertNotEqual(end, -1)
        return self.source[idx:end]

    def test_enrichment_reraises_import_error_before_generic_catch(self):
        block = self._enrichment_block()
        import_pos = block.find("except ImportError:")
        generic_pos = block.find("except Exception as _e:")
        self.assertNotEqual(import_pos, -1, "Missing except ImportError: re-raise")
        self.assertNotEqual(generic_pos, -1)
        self.assertLess(import_pos, generic_pos)
        # 'raise' bare statement must follow the ImportError clause.
        between = block[import_pos:generic_pos]
        self.assertIn("raise", between)
