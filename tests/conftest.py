import pytest


@pytest.fixture(autouse=True)
def _default_store_root_in_tmp(tmp_path, monkeypatch):
    """Keep default-config run tracking out of the repository working tree.

    ServiceClient records runs to ``<store_root>/_store`` by default; tests that
    construct clients without an explicit checkpoint_root would otherwise write
    into ``./checkpoints`` in the repo. Tests that set their own checkpoint_root
    or tracking_root are unaffected.
    """
    monkeypatch.setenv("RAY_UNSLOTH_DEFAULT_STORE_ROOT", str(tmp_path / "default-store"))
