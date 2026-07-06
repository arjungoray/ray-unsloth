"""LoRA-archive download helpers: tarball packing + HMAC-signed URLs.

The signer (engine) and verifier (Modal web endpoint) both live behind the
same Modal Volume, so the shared secret is persisted as a single byte file
on that volume rather than baked into the image.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import os
import secrets
import shlex
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

from ray_unsloth.checkpoints import resolve_path

SECRET_FILENAME = ".ray_unsloth_download_secret"
ARCHIVE_SUFFIX = ".tar.gz"


@dataclass(frozen=True)
class SignedDownload:
    archive_path: Path
    token: str
    expires_at: int


def _secret_path(checkpoint_root: str | Path) -> Path:
    return resolve_path(checkpoint_root) / SECRET_FILENAME


def load_or_create_secret(checkpoint_root: str | Path) -> bytes:
    """Read the shared HMAC secret, creating one on first use."""
    path = _secret_path(checkpoint_root)
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(secret)
    os.replace(tmp, path)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return secret


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign(payload: str, secret: bytes) -> str:
    digest = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).digest()
    return _b64url(digest)


def make_token(archive_relpath: str, expires_at: int, secret: bytes) -> str:
    """Sign an archive-relative path and expiry into a token.

    Why: serialized inside the URL as a single opaque field so the endpoint
    only needs to receive (archive, expires, token) to verify.
    """
    payload = f"{archive_relpath}\n{expires_at}"
    return sign(payload, secret)


def verify_token(archive_relpath: str, expires_at: int, token: str, secret: bytes) -> bool:
    expected = make_token(archive_relpath, expires_at, secret)
    if not hmac.compare_digest(expected, token):
        return False
    return int(time.time()) <= int(expires_at)


def pack_lora_archive(source_dir: str | Path, archive_path: str | Path | None = None) -> Path:
    """Pack a LoRA checkpoint directory into a deterministic .tar.gz."""
    source = resolve_path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint directory not found: {source}")
    if archive_path is None:
        archive_path = source.with_name(source.name + ARCHIVE_SUFFIX)
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(str(source), arcname=source.name)
    os.replace(tmp, archive_path)
    return archive_path


def archive_relpath(archive_path: str | Path, checkpoint_root: str | Path) -> str:
    """Return the archive path relative to the volume mount root.

    Why: the Modal endpoint mounts the same volume, so it can resolve a
    volume-relative key without exposing absolute container paths.
    """
    archive = resolve_path(archive_path)
    root = resolve_path(checkpoint_root)
    try:
        return str(archive.relative_to(root))
    except ValueError:
        return str(archive)


def modal_volume_get_command(volume_name: str, archive_relpath: str, output_path: str | Path | None = None) -> str:
    output = str(output_path) if output_path is not None else f"./{Path(archive_relpath).name}"
    return (
        f"modal volume get {shlex.quote(volume_name)} \\\n  {shlex.quote(archive_relpath)} \\\n  {shlex.quote(output)}"
    )


def resolve_archive(archive_relpath: str, checkpoint_root: str | Path) -> Path:
    root = resolve_path(checkpoint_root)
    candidate = (root / archive_relpath).resolve()
    if not str(candidate).startswith(str(root)):
        raise PermissionError(f"Archive path escapes checkpoint root: {archive_relpath}")
    return candidate
