"""Unit tests for the sampler-download helpers."""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

import pytest

from ray_unsloth.download import (
    archive_relpath,
    load_or_create_secret,
    make_token,
    modal_volume_get_command,
    pack_lora_archive,
    resolve_archive,
    verify_token,
)


def test_load_or_create_secret_is_stable(tmp_path: Path) -> None:
    first = load_or_create_secret(tmp_path)
    second = load_or_create_secret(tmp_path)
    assert first == second
    assert len(first) == 32


def test_pack_lora_archive_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "adapter"
    src.mkdir()
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    (src / "adapter_config.json").write_text("{}")
    archive = pack_lora_archive(src)
    assert archive.suffix == ".gz" and archive.suffixes[-2:] == [".tar", ".gz"]
    with tarfile.open(archive, "r:gz") as tar:
        names = sorted(tar.getnames())
    assert "adapter/adapter_model.safetensors" in names
    assert "adapter/adapter_config.json" in names


def test_token_signs_and_verifies(tmp_path: Path) -> None:
    secret = load_or_create_secret(tmp_path)
    relpath = "sub/adapter.tar.gz"
    expires = int(time.time()) + 60
    token = make_token(relpath, expires, secret)
    assert verify_token(relpath, expires, token, secret)
    # tampered relpath fails
    assert not verify_token("other.tar.gz", expires, token, secret)
    # tampered token fails
    assert not verify_token(relpath, expires, token + "x", secret)
    # expired fails
    assert not verify_token(relpath, int(time.time()) - 10, token, secret)


def test_resolve_archive_blocks_traversal(tmp_path: Path) -> None:
    (tmp_path / "ok.tar.gz").write_bytes(b"x")
    resolved = resolve_archive("ok.tar.gz", tmp_path)
    assert resolved.name == "ok.tar.gz"
    with pytest.raises(PermissionError):
        resolve_archive("../escape.tar.gz", tmp_path)


def test_archive_relpath_under_root(tmp_path: Path) -> None:
    src = tmp_path / "adapter"
    src.mkdir()
    archive = pack_lora_archive(src)
    rel = archive_relpath(archive, tmp_path)
    assert rel == "adapter.tar.gz"


def test_modal_volume_get_command_uses_volume_name_and_relative_archive() -> None:
    command = modal_volume_get_command(
        "ray-unsloth-checkpoints",
        "train-b2c1e8b712df43a2a108330a14fa6e00/arc-math-tutor-qwen3.5-4b-sft.tar.gz",
    )

    assert command == (
        "modal volume get ray-unsloth-checkpoints \\\n"
        "  train-b2c1e8b712df43a2a108330a14fa6e00/arc-math-tutor-qwen3.5-4b-sft.tar.gz \\\n"
        "  ./arc-math-tutor-qwen3.5-4b-sft.tar.gz"
    )
