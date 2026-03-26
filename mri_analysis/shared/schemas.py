from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    RECONSTRUCTING = "reconstructing"
    DETECTING = "detecting"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactType(str, Enum):
    INPUT = "input"
    RECONSTRUCTED_DICOM = "reconstructed_dicom"
    ANNOTATED_DICOM = "annotated_dicom"
    FINDINGS_JSON = "findings_json"


class ArtifactManifest(BaseModel):
    artifact_type: ArtifactType
    uri: str
    mime_type: str
    producer_stage: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobDetail(JobResponse):
    input_format: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    artifacts: List[ArtifactManifest] = Field(default_factory=list)


class ResultArtifact(BaseModel):
    artifact_type: ArtifactType
    uri: str
    download_url: str
    mime_type: str


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    artifacts: List[ResultArtifact]


class ReconstructionRequest(BaseModel):
    input_uri: str
    input_format: str
    output_prefix: str
    correlation_id: str


class ReconstructionResponse(BaseModel):
    reconstructed_dicom_uri: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class FindingLocation(BaseModel):
    description: str


class Finding(BaseModel):
    pathology_type: str
    confidence: float
    location: FindingLocation
    explanation: str


class DetectionRequest(BaseModel):
    input_uri: str
    output_prefix: str
    correlation_id: str
    job_id: Optional[str] = None


class DetectionResponse(BaseModel):
    annotated_dicom_uri: str
    findings_json_uri: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class FindingsDocument(BaseModel):
    job_id: Optional[str] = None
    study_id: str
    source_file: str
    findings: List[Finding]
    artifacts: Dict[str, str]

