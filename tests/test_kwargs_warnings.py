import warnings

from ray_unsloth.clients.service import ServiceClient


def test_unknown_kwargs_warn_once_for_create_lora_training_client(tmp_path):
    service = ServiceClient(
        config={
            "provider": "fake",
            "checkpoint_root": str(tmp_path / "checkpoints"),
            "model": {"base_model": "Test/Test-1B"},
            "tracking": False,
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        service.create_lora_training_client(foo=1)
        service.create_lora_training_client(foo=1)

    assert len(caught) == 1
    assert "create_lora_training_client" in str(caught[0].message)
    assert "foo" in str(caught[0].message)


def test_known_create_lora_training_client_usage_does_not_warn(tmp_path):
    service = ServiceClient(
        config={
            "provider": "fake",
            "checkpoint_root": str(tmp_path / "checkpoints"),
            "model": {"base_model": "Test/Test-1B"},
            "tracking": False,
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        service.create_lora_training_client(seed=123)

    assert caught == []
