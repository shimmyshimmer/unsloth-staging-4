# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install-wide load policy is the owner's, so a managed account's load obeys it too.

The settings routes that write these values are owner-only, so an account-keyed read
finds nothing in the managed account's database and silently falls back to defaults.
"""

from __future__ import annotations

import pytest

from auth import policy
from utils import model_memory_settings, openai_auto_switch_settings, vram_budget_settings
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def studio_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    for module in (model_memory_settings, vram_budget_settings, openai_auto_switch_settings):
        monkeypatch.setattr(module, "_cache", {})
        if hasattr(module, "_generation"):
            monkeypatch.setattr(module, "_generation", {})
    yield


def test_a_managed_load_reads_the_owners_model_memory_policy():
    run_as(OWNER, model_memory_settings.set_model_memory_settings, True, False)
    assert run_as(ALICE, model_memory_settings.get_model_memory_settings) == (True, False)
    assert run_as(ALICE, model_memory_settings.should_mlock) is True


def test_a_managed_load_reads_the_owners_vram_budget():
    run_as(OWNER, vram_budget_settings.set_vram_budget_fraction, 0.85)
    assert run_as(ALICE, vram_budget_settings.get_vram_budget_fraction) == pytest.approx(0.85)


def test_a_managed_request_reads_the_owners_auto_switch_policy():
    run_as(OWNER, openai_auto_switch_settings.set_openai_auto_switch, True, None)
    assert run_as(ALICE, openai_auto_switch_settings.get_openai_auto_switch_enabled) is True
