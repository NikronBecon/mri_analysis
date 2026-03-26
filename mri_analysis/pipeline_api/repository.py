from __future__ import annotations

from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from mri_analysis.pipeline_api.models import ArtifactModel, JobModel
from mri_analysis.shared.schemas import ArtifactManifest, ArtifactType, JobDetail, JobStatus


class JobRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_job(self, job_id: str, status: JobStatus, input_format: Optional[str] = None) -> JobModel:
        job = JobModel(job_id=job_id, status=status.value, input_format=input_format)
        self.session.add(job)
        self.session.flush()
        self.session.refresh(job)
        return job

    def get_job(self, job_id: str) -> Optional[JobModel]:
        statement = select(JobModel).where(JobModel.job_id == job_id).options(selectinload(JobModel.artifacts))
        return self.session.execute(statement).scalar_one_or_none()

    def get_next_runnable_job(self) -> Optional[JobModel]:
        statement = (
            select(JobModel)
            .where(JobModel.status.in_([JobStatus.QUEUED.value, JobStatus.RECONSTRUCTING.value, JobStatus.DETECTING.value]))
            .order_by(JobModel.created_at.asc())
            .limit(1)
            .options(selectinload(JobModel.artifacts))
        )
        return self.session.execute(statement).scalar_one_or_none()

    def set_job_status(
        self,
        job: JobModel,
        status: JobStatus,
        *,
        input_uri: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> JobModel:
        job.status = status.value
        if input_uri is not None:
            job.input_uri = input_uri
        job.error_code = error_code
        job.error_message = error_message
        self.session.add(job)
        self.session.flush()
        self.session.refresh(job)
        return job

    def upsert_artifact(self, job: JobModel, manifest: ArtifactManifest) -> ArtifactModel:
        statement = select(ArtifactModel).where(
            ArtifactModel.job_id == job.job_id,
            ArtifactModel.artifact_type == manifest.artifact_type.value,
        )
        artifact = self.session.execute(statement).scalar_one_or_none()
        if artifact is None:
            artifact = ArtifactModel(job_id=job.job_id, artifact_type=manifest.artifact_type.value)
        artifact.uri = manifest.uri
        artifact.mime_type = manifest.mime_type
        artifact.producer_stage = manifest.producer_stage
        self.session.add(artifact)
        self.session.flush()
        return artifact

    def to_detail(self, job: JobModel) -> JobDetail:
        return JobDetail(
            job_id=job.job_id,
            status=JobStatus(job.status),
            input_format=job.input_format,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error_code=job.error_code,
            error_message=job.error_message,
            artifacts=[
                ArtifactManifest(
                    artifact_type=ArtifactType(artifact.artifact_type),
                    uri=artifact.uri,
                    mime_type=artifact.mime_type,
                    producer_stage=artifact.producer_stage,
                )
                for artifact in job.artifacts
            ],
        )

