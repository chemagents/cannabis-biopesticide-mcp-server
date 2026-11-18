"""Strict PNG artifact persistence for local or S3-compatible storage.

The upload endpoint may use Docker DNS while the independent signing endpoint
is caller-reachable.  SigV4 signs the Host header, so a signed URL must never be
rewritten after generation.
"""
from __future__ import annotations

import hashlib
import io
import logging
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .config import get_settings

logger = logging.getLogger(__name__)
KEY_PREFIX = "cannabis_biopesticide"
CONTENT_TYPE = "image/png"


class ArtifactStorageError(RuntimeError):
    """Configured artifact storage could not persist or expose an artifact."""


def _client(endpoint_url: str):
    settings = get_settings()
    verify: bool | str = settings.s3_ca_bundle or True
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        verify=verify,
        config=Config(
            signature_version="s3v4",
            connect_timeout=3,
            read_timeout=10,
            retries={"max_attempts": 3, "mode": "standard"},
            s3={"addressing_style": settings.s3_addressing_style},
        ),
    )


@lru_cache(maxsize=1)
def _upload_client():
    return _client(get_settings().s3_endpoint_url)


@lru_cache(maxsize=1)
def _presign_client():
    return _client(get_settings().s3_presign_endpoint_url)


def clear_client_cache() -> None:
    _upload_client.cache_clear()
    _presign_client.cache_clear()


def ensure_storage_ready() -> None:
    """Fail startup when a configured bucket is unavailable or unauthorized."""
    settings = get_settings()
    if not settings.use_s3:
        logger.info("Artifact storage mode: local (%s)", settings.artifacts_dir)
        return
    try:
        _upload_client().head_bucket(Bucket=settings.s3_bucket_name)
    except (BotoCoreError, ClientError) as exc:
        message = f"S3 artifact bucket {settings.s3_bucket_name!r} is not ready"
        if settings.s3_allow_local_fallback:
            logger.error("%s; explicit local fallback is enabled (%s)", message, type(exc).__name__)
            return
        raise ArtifactStorageError(message) from exc


def _local_result(data: bytes, filename: str, warning: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    out_dir = Path(settings.artifacts_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        path.write_bytes(data)
    except OSError as exc:
        logger.warning("Artifacts dir %s not writable (%s); using temp dir", out_dir, exc)
        out_dir = Path(tempfile.gettempdir()) / KEY_PREFIX
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        path.write_bytes(data)
        warning = warning or "configured local artifact directory was unavailable; used temp storage"

    artifact = (
        f"{settings.artifact_url_base.rstrip('/')}/{filename}"
        if settings.artifact_url_base and path.parent == Path(settings.artifacts_dir)
        else str(path)
    )
    result: dict[str, Any] = {
        "artifact": artifact,
        "kind": "local",
        "sha256": hashlib.sha256(data).hexdigest(),
        "content_type": CONTENT_TYPE,
    }
    if warning:
        result["warning"] = warning
    return result


def store_png(data: bytes, name: str) -> dict[str, Any]:
    """Persist PNG bytes and return a verifiable artifact reference."""
    settings = get_settings()
    digest = hashlib.sha256(data).hexdigest()
    filename = f"{name}_{uuid.uuid4().hex[:8]}.png"
    if settings.use_s3:
        key = f"{KEY_PREFIX}/{filename}"
        try:
            _upload_client().put_object(
                Bucket=settings.s3_bucket_name,
                Key=key,
                Body=data,
                ContentType=CONTENT_TYPE,
                Metadata={"sha256": digest},
            )
            url = _presign_client().generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.s3_bucket_name, "Key": key},
                ExpiresIn=settings.s3_url_expiration,
            )
            return {
                "artifact": url,
                "kind": "s3",
                "bucket": settings.s3_bucket_name,
                "key": key,
                "sha256": digest,
                "content_type": CONTENT_TYPE,
                "expires_in": settings.s3_url_expiration,
            }
        except (BotoCoreError, ClientError) as exc:
            if not settings.s3_allow_local_fallback:
                raise ArtifactStorageError(
                    f"failed to store PNG in S3 bucket {settings.s3_bucket_name!r}"
                ) from exc
            warning = f"S3 upload failed ({type(exc).__name__}); explicit local fallback used"
            logger.error(warning)
            return _local_result(data, filename, warning)
    return _local_result(data, filename)


def save_figure(fig, name: str, *, dpi: int = 140) -> dict[str, Any]:
    """Render and close a matplotlib figure, then persist its PNG bytes."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return store_png(buf.getvalue(), name)
