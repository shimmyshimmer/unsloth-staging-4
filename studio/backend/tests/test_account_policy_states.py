# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The whole account-policy table, in one place.

Two predicates ride on the same counts and deliberately disagree. ``login_mode`` and
``installation_is_multi_user`` follow the ACTIVE count, so a deactivated account keeps the
single login form. ``installation_has_managed_accounts`` and the full-access gate follow
whether any managed account exists at all, because its files are still on disk. The
unreadable-database fallback splits them on purpose: one login form, closed host.

Collapsing these predicates breaks exactly this table, so the four states are asserted
together rather than one per test.
"""

import secrets

import pytest

from auth import policy, storage


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield storage
    policy.invalidate_account_cache()


def _state() -> tuple[str, bool, bool, bool]:
    policy.invalidate_account_cache()
    return (
        policy.login_mode(),
        policy.installation_is_multi_user(),
        policy.installation_has_managed_accounts(),
        policy.full_access_permitted(),
    )


def test_owner_only(auth_db):
    assert _state() == ("single", False, False, True)


def test_owner_plus_one_active_managed_account(auth_db):
    storage.issue_account_setup_code(username = "alice")
    assert _state() == ("multi", True, True, False)


def test_owner_plus_one_deactivated_managed_account(auth_db):
    account = storage.issue_account_setup_code(username = "alice")["account"]
    storage.set_account_active(account["account_id"], False)
    assert _state() == ("single", False, True, False)


def test_an_unreadable_auth_database(auth_db, monkeypatch):
    def boom():
        raise OSError("auth.db unreadable")

    monkeypatch.setattr(storage, "account_counts", boom)
    assert _state() == ("single", False, True, False)


def test_a_count_read_failure_keeps_a_bound_managed_account_isolated(auth_db, monkeypatch):
    """The fallback answers for the login form, which is unauthenticated; a request already
    bound to a managed account is proof of a multi-user install, so isolation stays on."""
    from utils.account_context import AccountContext, run_as

    def boom():
        raise OSError("auth.db unreadable")

    monkeypatch.setattr(storage, "account_counts", boom)
    policy.invalidate_account_cache()
    assert run_as(AccountContext("a" * 32, "alice"), policy.installation_is_multi_user) is True


def test_deactivating_the_last_managed_account_keeps_a_bound_request_isolated(
    auth_db, monkeypatch, tmp_path
):
    """A request that authenticated as alice can still be running when the owner
    deactivates her. The active count drops back to one, but the bound request is a
    managed one for its whole life and must not acquire the owner's model paths."""
    from fastapi import HTTPException
    from hub.services.models import account_access as access
    from utils.account_context import AccountContext, run_as

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "installation-token")
    account = storage.issue_account_setup_code(username = "alice")["account"]
    alice = AccountContext(account["account_id"], "alice")
    assert run_as(alice, access.managed_account) is True

    storage.set_account_active(account["account_id"], False)
    assert policy.installation_is_multi_user() is False
    assert run_as(alice, access.managed_account) is True
    assert run_as(alice, access.ambient_hf_token) is False
    with pytest.raises(HTTPException) as raised:
        run_as(alice, access.require_model_access, "/owner/private/checkpoint.gguf")
    assert raised.value.status_code == 404
