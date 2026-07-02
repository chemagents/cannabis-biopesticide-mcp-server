"""Persist figures to a local artifacts dir or an S3-compatible bucket.

Mirrors the artifact handling of the other CoScientist MCP servers: if S3 is
configured a presigned URL is returned, otherwise a local file path (optionally
prefixed by ``HERACLEUM_ARTIFACT_URL_BASE`` to make it web-served).
"""
from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path

from .config import get_settings

logger = logging.getLogger(__name__)


def save_figure(fig, name: str) -> str:
    """Save a matplotlib figure; return an S3 presigned URL or a local path/URL."""
    settings = get_settings()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    data = buf.getvalue()
    import matplotlib.pyplot as plt

    plt.close(fig)

    filename = f"{name}_{uuid.uuid4().hex[:8]}.png"
    if settings.use_s3:
        try:
            import boto3

            s3 = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint_url,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
            )
            key = f"heracleum_tox/{filename}"
            s3.put_object(Bucket=settings.s3_bucket_name, Key=key, Body=data,
                          ContentType="image/png")
            return s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.s3_bucket_name, "Key": key},
                ExpiresIn=settings.s3_url_expiration,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("S3 upload failed (%s); falling back to local file.", exc)

    out_dir = Path(settings.artifacts_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        path.write_bytes(data)
    except OSError as exc:
        # Configured dir not writable (e.g. the container default /app/artifacts when
        # running locally): fall back to a writable temp dir so the tool still works.
        import tempfile

        logger.warning("Artifacts dir %s not writable (%s); using temp dir.", out_dir, exc)
        out_dir = Path(tempfile.gettempdir()) / "heracleum_tox"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        path.write_bytes(data)
        return str(path)
    if settings.artifact_url_base:
        return settings.artifact_url_base.rstrip("/") + "/" + filename
    return str(path)
