import types

import pytest

import ray_unsloth.config as config_module
import ray_unsloth.types.futures as futures_module
from ray_unsloth.config import RuntimeConfig
from ray_unsloth.types import FutureValueProxy


def test_legacy_modal_enabled_config_warns_once():
    config_module._WARNED_LEGACY_MODAL_SWITCH = False
    with pytest.warns(DeprecationWarning) as record:
        RuntimeConfig.from_dict({"modal": {"enabled": True}})
        RuntimeConfig.from_dict({"modal": {"enabled": True}})

    assert len(record) == 1
    assert "provider: modal" in str(record[0].message)


def test_future_value_proxy_getattr_warns_once():
    futures_module._WARNED_PROXY_GETATTR = False
    proxy = FutureValueProxy(types.SimpleNamespace(answer=42))

    with pytest.warns(DeprecationWarning) as record:
        assert proxy.answer == 42
        assert proxy.answer == 42

    assert len(record) == 1
    assert ".result()" in str(record[0].message)
