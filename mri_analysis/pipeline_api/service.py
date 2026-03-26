from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import sessionmaker

from mri_analysis.pipeline_api.clients import InferenceClient
from mri_analysis.pipeline_api.repository import JobRepository
from mri_analysis.shared.schemas import (
    ArtifactManifest,
    ArtifactType,
    DetectionRequest,
    JobDetail,
    JobResult,
    JobStatus,
    ReconstructionRequest,
    ResultArtifact,
)
from mri_analysis.shared.storage import StorageClient, guess_mime_type


class PipelineService:
    def __init__(self, session_factory: sessionmaker, storage: StorageClient, inference_client: InferenceClient) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.inference_client = inference_client

    def create_job(self, filename: str, content: bytes, correlation_id: Optional[str] = None) -> tuple[str, JobStatus]:
        input_format = self._normalize_input_format(filename)
        job_id = correlation_id or str(uuid.uuid4())
        sanitized_filename = self._sanitize_filename(filename)
        storage_key = f"jobs/{job_id}/input/{sanitized_filename}"
        input_uri = self.storage.upload_bytes(storage_key, content, guess_mime_type(sanitized_filename))

        with self.session_factory() as session:
            repo = JobRepository(session)
            job = repo.create_job(job_id, JobStatus.UPLOADED, input_format=input_format)
            repo.upsert_artifact(
                job,
                ArtifactManifest(
                    artifact_type=ArtifactType.INPUT,
                    uri=input_uri,
                    mime_type=guess_mime_type(sanitized_filename),
                    producer_stage="pipeline-api",
                ),
            )
            repo.set_job_status(job, JobStatus.QUEUED, input_uri=input_uri)
            session.commit()
        return job_id, JobStatus.QUEUED

    def get_job(self, job_id: str) -> Optional[JobDetail]:
        with self.session_factory() as session:
            repo = JobRepository(session)
            job = repo.get_job(job_id)
            if job is None:
                return None
            return repo.to_detail(job)

    def get_results(self, job_id: str, base_url: str) -> Optional[JobResult]:
        detail = self.get_job(job_id)
        if detail is None:
            return None
        if detail.status != JobStatus.COMPLETED:
            raise ValueError(f"Job {job_id} is not completed")
        artifacts = [
            ResultArtifact(
                artifact_type=artifact.artifact_type,
                uri=artifact.uri,
                mime_type=artifact.mime_type,
                download_url=f"{base_url.rstrip('/')}/v1/jobs/{job_id}/artifacts/{artifact.artifact_type.value}",
            )
            for artifact in detail.artifacts
            if artifact.artifact_type != ArtifactType.INPUT
        ]
        return JobResult(job_id=job_id, status=detail.status, artifacts=artifacts)

    def process_next_job(self) -> bool:
        with self.session_factory() as session:
            repo = JobRepository(session)
            job = repo.get_next_runnable_job()
            if job is None:
                return False
            correlation_id = job.job_id

            try:
                reconstructed_uri = self._run_reconstruction(repo, job, correlation_id)
                self._run_detection(repo, job, correlation_id, reconstructed_uri)
                repo.set_job_status(job, JobStatus.COMPLETED)
                session.commit()
                return True
            except Exception as exc:
                repo.set_job_status(job, JobStatus.FAILED, error_code=exc.__class__.__name__, error_message=str(exc))
                session.commit()
                return True

    def read_artifact(self, job_id: str, artifact_type: ArtifactType) -> Optional[tuple[bytes, str]]:
        detail = self.get_job(job_id)
        if detail is None:
            return None
        for artifact in detail.artifacts:
            if artifact.artifact_type == artifact_type:
                return self.storage.read_bytes(artifact.uri), artifact.mime_type
        return None

    def _run_reconstruction(self, repo: JobRepository, job, correlation_id: str) -> str:
        existing = self._artifact_uri(job.artifacts, ArtifactType.RECONSTRUCTED_DICOM)
        if existing:
            if job.status == JobStatus.DETECTING.value:
                return existing
        repo.set_job_status(job, JobStatus.RECONSTRUCTING)
        request = ReconstructionRequest(
            input_uri=job.input_uri,
            input_format=job.input_format,
            output_prefix=f"jobs/{job.job_id}/reconstruction",
            correlation_id=correlation_id,
        )
        response = self.inference_client.reconstruct(request)
        repo.upsert_artifact(
            job,
            ArtifactManifest(
                artifact_type=ArtifactType.RECONSTRUCTED_DICOM,
                uri=response.reconstructed_dicom_uri,
                mime_type="application/dicom",
                producer_stage="reconstruction-service",
            ),
        )
        repo.set_job_status(job, JobStatus.DETECTING)
        return response.reconstructed_dicom_uri

    def _run_detection(self, repo: JobRepository, job, correlation_id: str, reconstructed_uri: str) -> None:
        if self._artifact_uri(job.artifacts, ArtifactType.ANNOTATED_DICOM) and self._artifact_uri(
            job.artifacts,
            ArtifactType.FINDINGS_JSON,
        ):
            return
        repo.set_job_status(job, JobStatus.DETECTING)
        request = DetectionRequest(
            input_uri=reconstructed_uri,
            output_prefix=f"jobs/{job.job_id}/detection",
            correlation_id=correlation_id,
            job_id=job.job_id,
        )
        response = self.inference_client.detect(request)
        repo.upsert_artifact(
            job,
            ArtifactManifest(
                artifact_type=ArtifactType.ANNOTATED_DICOM,
                uri=response.annotated_dicom_uri,
                mime_type="application/dicom",
                producer_stage="detection-service",
            ),
        )
        repo.upsert_artifact(
            job,
            ArtifactManifest(
                artifact_type=ArtifactType.FINDINGS_JSON,
                uri=response.findings_json_uri,
                mime_type="application/json",
                producer_stage="detection-service",
            ),
        )

    @staticmethod
    def _artifact_uri(artifacts, artifact_type: ArtifactType) -> Optional[str]:
        for artifact in artifacts:
            if artifact.artifact_type == artifact_type.value:
                return artifact.uri
        return None

    @staticmethod
    def _normalize_input_format(filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix == ".h5":
            return "h5"
        if suffix in {".dcm", ".dicom"}:
            return "dicom"
        raise ValueError("Unsupported input format. Supported extensions: .h5, .dcm, .dicom")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        base = Path(filename).name
        return re.sub(r"[^A-Za-z0-9._-]", "_", base)

