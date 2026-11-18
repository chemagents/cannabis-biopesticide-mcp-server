"""Strict S3 artifact behavior for the Cannabis biopesticide server."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from pydantic import ValidationError
from server import artifacts, canpest_server
from server.config import Settings


@pytest.fixture(autouse=True)
def _clear_clients():
    artifacts.clear_client_cache()
    yield
    artifacts.clear_client_cache()


def _settings(tmp_path: Path):
    return SimpleNamespace(
        use_s3=True,
        artifacts_dir=str(tmp_path),
        artifact_url_base="",
        s3_endpoint_url="http://minio:9000",
        s3_presign_endpoint_url="http://localhost:9000",
        s3_access_key="writer",
        s3_secret_key="secret",
        s3_bucket_name="coscientist-artifacts",
        s3_region="us-east-1",
        s3_addressing_style="path",
        s3_ca_bundle="",
        s3_allow_local_fallback=False,
        s3_url_expiration=3600,
    )


def test_partial_s3_configuration_is_rejected():
    with pytest.raises(ValidationError, match="partial S3 configuration"):
        Settings(
            _env_file=None,
            s3_endpoint_url="http://minio:9000",
            s3_access_key="writer",
            s3_secret_key="",
            s3_bucket_name="bucket",
        )


def test_presigned_reference_uses_public_endpoint_and_checksum(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(artifacts, "get_settings", lambda: settings)
    endpoints: list[str] = []
    uploaded: dict = {}

    class FakeClient:
        def __init__(self, endpoint: str):
            self.endpoint = endpoint

        def put_object(self, **kwargs):
            uploaded.update(kwargs)

        def generate_presigned_url(self, _method, *, Params, ExpiresIn):
            return f"{self.endpoint}/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"

    def fake_client(_service, **kwargs):
        endpoints.append(kwargs["endpoint_url"])
        return FakeClient(kwargs["endpoint_url"])

    monkeypatch.setattr(artifacts.boto3, "client", fake_client)
    data = b"\x89PNG\r\n\x1a\ncannabis"
    result = artifacts.store_png(data, "figure")
    assert endpoints == ["http://minio:9000", "http://localhost:9000"]
    assert result["artifact"].startswith("http://localhost:9000/")
    assert result["key"].startswith("cannabis_biopesticide/")
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert uploaded["Metadata"]["sha256"] == result["sha256"]


def test_upload_error_propagates_in_strict_mode(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(artifacts, "get_settings", lambda: settings)

    class FailingClient:
        def put_object(self, **_kwargs):
            raise EndpointConnectionError(endpoint_url=settings.s3_endpoint_url)

    monkeypatch.setattr(artifacts, "_upload_client", lambda: FailingClient())
    with pytest.raises(artifacts.ArtifactStorageError):
        artifacts.store_png(b"png", "strict")
    assert not list(tmp_path.iterdir())


def test_missing_bucket_fails_startup(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(artifacts, "get_settings", lambda: settings)

    class MissingBucketClient:
        def head_bucket(self, **_kwargs):
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    monkeypatch.setattr(artifacts, "_upload_client", lambda: MissingBucketClient())
    with pytest.raises(artifacts.ArtifactStorageError, match="not ready"):
        artifacts.ensure_storage_ready()


def test_tool_does_not_swallow_artifact_storage_error(monkeypatch):
    monkeypatch.setattr(canpest_server, "load_dataset", lambda: object())
    monkeypatch.setattr(
        canpest_server.docking,
        "active_vs_inactive",
        lambda _ds: {"per_protein": []},
    )

    def fail_storage(*_args, **_kwargs):
        raise artifacts.ArtifactStorageError("S3 unavailable")

    monkeypatch.setattr(canpest_server.plotting, "plot_docking", fail_storage)
    with pytest.raises(artifacts.ArtifactStorageError, match="S3 unavailable"):
        canpest_server.docking_analysis()


def test_chemical_space_does_not_swallow_artifact_storage_error(monkeypatch):
    monkeypatch.setattr(canpest_server, "load_dataset", lambda: object())

    def fail_storage(*_args, **_kwargs):
        raise artifacts.ArtifactStorageError("S3 unavailable")

    monkeypatch.setattr(canpest_server.plotting, "plot_chemical_space", fail_storage)
    with pytest.raises(artifacts.ArtifactStorageError, match="S3 unavailable"):
        canpest_server.chemical_space()


def test_artifact_fallback_warning_is_promoted_to_metadata(monkeypatch):
    monkeypatch.setattr(canpest_server, "load_dataset", lambda: object())
    monkeypatch.setattr(
        canpest_server.docking,
        "active_vs_inactive",
        lambda _ds: {"per_protein": []},
    )
    artifact = {
        "artifact": "/tmp/figure.png",
        "kind": "local",
        "sha256": "0" * 64,
        "warning": "S3 upload failed; explicit local fallback used",
    }
    monkeypatch.setattr(canpest_server.plotting, "plot_docking", lambda _rows: artifact)
    result = canpest_server.docking_analysis()
    assert result["metadata"]["figure"] == artifact
    assert result["metadata"]["warnings"] == [artifact["warning"]]


def test_generic_docking_plot_failure_is_disclosed_without_losing_answer(monkeypatch):
    monkeypatch.setattr(canpest_server, "load_dataset", lambda: object())
    monkeypatch.setattr(
        canpest_server.docking,
        "active_vs_inactive",
        lambda _ds: {"per_protein": [], "measured": 1},
    )

    def fail_plot(*_args, **_kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(canpest_server.plotting, "plot_docking", fail_plot)
    result = canpest_server.docking_analysis()
    assert result["answer"]["measured"] == 1
    assert result["answer"]["finding"]
    assert result["metadata"]["figure"] is None
    assert result["metadata"]["warnings"] == [
        "Docking figure could not be produced (RuntimeError): renderer unavailable"
    ]


def test_generic_chemical_space_plot_failure_is_disclosed(monkeypatch):
    monkeypatch.setattr(canpest_server, "load_dataset", lambda: object())
    monkeypatch.setattr(
        canpest_server.docking,
        "metabolite_pesticide_overlap",
        lambda _ds: {
            "frac_nn_active": 0.5,
            "n_metabolites": 2,
            "median_nn_similarity": 0.4,
            "frac_nn_similarity_above_0_4": 0.5,
        },
    )

    def fail_plot(*_args, **_kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(canpest_server.plotting, "plot_chemical_space", fail_plot)
    result = canpest_server.chemical_space()
    assert result["answer"]["finding"]
    assert result["metadata"]["figure"] is None
    assert result["metadata"]["warnings"] == [
        "Chemical-space figure could not be produced (RuntimeError): renderer unavailable"
    ]
