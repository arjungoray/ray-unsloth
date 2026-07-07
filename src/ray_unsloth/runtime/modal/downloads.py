from __future__ import annotations

import os
from pathlib import Path

DOWNLOAD_MOUNT_ENV_VAR = "RAY_UNSLOTH_DOWNLOAD_MOUNT_PATH"
DOWNLOAD_APP_SUFFIX = "-downloads"
DOWNLOAD_FUNCTION_NAME = "ray_unsloth_download_lora"


def ray_unsloth_download_lora(archive: str, expires: int, token: str):
    """Volume-streaming handler invoked by the Modal web endpoint."""

    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    from ray_unsloth.download import load_or_create_secret, resolve_archive, verify_token

    mount_path = os.environ.get(DOWNLOAD_MOUNT_ENV_VAR, "/checkpoints")
    secret = load_or_create_secret(mount_path)
    if not verify_token(archive, int(expires), token, secret):
        raise HTTPException(status_code=403, detail="Invalid or expired token")
    try:
        path = resolve_archive(archive, mount_path)
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Archive not found")
    return FileResponse(
        str(path),
        media_type="application/gzip",
        filename=path.name,
    )


def _build_download_app(*, modal, app_name: str, volume_name: str, mount_path: str, timeout: int, python_version: str):
    """Build (but don't deploy) the persistent download app."""

    app = modal.App(app_name)
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    source_root = Path(__file__).resolve().parents[3]
    package_dir = source_root / "ray_unsloth"
    image = (
        modal.Image.debian_slim(python_version=python_version)
        .pip_install("fastapi[standard]==0.115.0")
        .env({DOWNLOAD_MOUNT_ENV_VAR: mount_path, "PYTHONPATH": "/root/ray_unsloth_src"})
    )
    if hasattr(image, "add_local_dir"):
        image = image.add_local_dir(
            str(package_dir),
            remote_path="/root/ray_unsloth_src/ray_unsloth",
            copy=True,
        )

    endpoint_decorator = getattr(modal, "fastapi_endpoint", None) or getattr(modal, "web_endpoint", None)
    if endpoint_decorator is None:
        return app, None

    decorated = endpoint_decorator(method="GET", docs=False)(ray_unsloth_download_lora)
    fn = app.function(
        image=image,
        volumes={mount_path: volume},
        timeout=timeout,
        max_containers=2,
        name=DOWNLOAD_FUNCTION_NAME,
    )(decorated)
    return app, fn


def _ensure_download_endpoint(
    *, modal, app_name: str, volume_name: str, mount_path: str, timeout: int, python_version: str
):
    """Look up an existing deployed download function, or deploy it."""

    try:
        return modal.Function.from_name(app_name, DOWNLOAD_FUNCTION_NAME)
    except Exception:
        pass

    app, _fn = _build_download_app(
        modal=modal,
        app_name=app_name,
        volume_name=volume_name,
        mount_path=mount_path,
        timeout=timeout,
        python_version=python_version,
    )
    try:
        app.deploy(name=app_name)
    except TypeError:
        app.deploy()
    try:
        return modal.Function.from_name(app_name, DOWNLOAD_FUNCTION_NAME)
    except Exception:
        return None
