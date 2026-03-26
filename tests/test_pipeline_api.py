from __future__ import annotations

import json

from fastapi.testclient import TestClient

from mri_analysis.pipeline_api.main import create_app
from mri_analysis.pipeline_api.service import PipelineService
from mri_analysis.shared.schemas import ArtifactType, JobStatus


def test_pipeline_happy_path_for_h5_and_dicom(pipeline_service: PipelineService):
    app = create_app(pipeline_service=pipeline_service, start_worker=False)
    with TestClient(app) as client:
        for filename in ("brain_scan.h5", "brain_scan.dcm"):
            response = client.post("/v1/jobs", files={"file": (filename, b"raw-input", "application/octet-stream")})
            assert response.status_code == 202
            job_id = response.json()["job_id"]

            processed = pipeline_service.process_next_job()
            assert processed is True

            status_response = client.get(f"/v1/jobs/{job_id}")
            assert status_response.status_code == 200
            status_payload = status_response.json()
            assert status_payload["status"] == JobStatus.COMPLETED.value
            artifact_types = {artifact["artifact_type"] for artifact in status_payload["artifacts"]}
            assert artifact_types == {
                ArtifactType.INPUT.value,
                ArtifactType.RECONSTRUCTED_DICOM.value,
                ArtifactType.ANNOTATED_DICOM.value,
                ArtifactType.FINDINGS_JSON.value,
            }

            results_response = client.get(f"/v1/jobs/{job_id}/results")
            assert results_response.status_code == 200
            results_payload = results_response.json()
            assert len(results_payload["artifacts"]) == 3

            artifact_response = client.get(f"/v1/jobs/{job_id}/artifacts/{ArtifactType.FINDINGS_JSON.value}")
            assert artifact_response.status_code == 200
            artifact_payload = json.loads(artifact_response.content.decode("utf-8"))
            assert artifact_payload["job_id"] == job_id


def test_invalid_extension_is_rejected(pipeline_service: PipelineService):
    app = create_app(pipeline_service=pipeline_service, start_worker=False)
    with TestClient(app) as client:
        response = client.post("/v1/jobs", files={"file": ("brain_scan.zip", b"zip-bytes", "application/zip")})
        assert response.status_code == 400
        assert "Unsupported input format" in response.json()["detail"]


def test_results_before_completion_return_conflict(pipeline_service: PipelineService):
    app = create_app(pipeline_service=pipeline_service, start_worker=False)
    with TestClient(app) as client:
        response = client.post("/v1/jobs", files={"file": ("brain_scan.h5", b"raw-input", "application/octet-stream")})
        job_id = response.json()["job_id"]

        results_response = client.get(f"/v1/jobs/{job_id}/results")
        assert results_response.status_code == 409


def test_failed_detection_marks_job_failed(session_factory, storage, reconstruction_client, detection_client):
    from tests.conftest import AppInferenceClient

    class MixedInferenceClient(AppInferenceClient):
        def detect(self, request):
            del request
            raise TimeoutError("Detection timed out")

    pipeline_service = PipelineService(
        session_factory=session_factory,
        storage=storage,
        inference_client=MixedInferenceClient(reconstruction_client, detection_client),
    )
    app = create_app(pipeline_service=pipeline_service, start_worker=False)

    with TestClient(app) as client:
        response = client.post("/v1/jobs", files={"file": ("brain_scan.h5", b"raw-input", "application/octet-stream")})
        job_id = response.json()["job_id"]

        processed = pipeline_service.process_next_job()
        assert processed is True

        status_response = client.get(f"/v1/jobs/{job_id}")
        payload = status_response.json()
        assert payload["status"] == JobStatus.FAILED.value
        assert payload["error_code"] == "TimeoutError"
        assert "Detection timed out" in payload["error_message"]


def test_idempotent_worker_call_does_not_duplicate_artifacts(pipeline_service: PipelineService):
    app = create_app(pipeline_service=pipeline_service, start_worker=False)
    with TestClient(app) as client:
        response = client.post("/v1/jobs", files={"file": ("brain_scan.h5", b"raw-input", "application/octet-stream")})
        job_id = response.json()["job_id"]

        assert pipeline_service.process_next_job() is True
        assert pipeline_service.process_next_job() is False

        status_response = client.get(f"/v1/jobs/{job_id}")
        payload = status_response.json()
        assert len(payload["artifacts"]) == 4
