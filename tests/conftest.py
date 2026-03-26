from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mri_analysis.detection_service.main import create_app as create_detection_app
from mri_analysis.pipeline_api.clients import InferenceClient
from mri_analysis.pipeline_api.database import Base, create_session_factory, get_engine
from mri_analysis.pipeline_api.service import PipelineService
from mri_analysis.reconstruction_service.main import create_app as create_reconstruction_app
from mri_analysis.shared.schemas import (
    DetectionRequest,
    DetectionResponse,
    ReconstructionRequest,
    ReconstructionResponse,
)
from mri_analysis.shared.storage import LocalStorageClient


class AppInferenceClient(InferenceClient):
    def __init__(self, reconstruction_client: TestClient, detection_client: TestClient) -> None:
        self.reconstruction_client = reconstruction_client
        self.detection_client = detection_client

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResponse:
        response = self.reconstruction_client.post("/infer", json=request.model_dump())
        response.raise_for_status()
        return ReconstructionResponse.model_validate(response.json())

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        response = self.detection_client.post("/infer", json=request.model_dump())
        response.raise_for_status()
        return DetectionResponse.model_validate(response.json())


class FailingInferenceClient(InferenceClient):
    def __init__(self, stage: str) -> None:
        self.stage = stage

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResponse:
        del request
        if self.stage == "reconstruct":
            raise TimeoutError("Reconstruction timed out")
        raise AssertionError("Unexpected reconstruct call")

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        del request
        if self.stage == "detect":
            raise TimeoutError("Detection timed out")
        raise AssertionError("Unexpected detect call")


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorageClient:
    client = LocalStorageClient(root=tmp_path / "storage")
    client.ensure_bucket()
    return client


@pytest.fixture
def session_factory(tmp_path: Path):
    db_path = tmp_path / "test.db"
    factory = create_session_factory(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=get_engine(factory))
    return factory


@pytest.fixture
def reconstruction_client(storage: LocalStorageClient) -> TestClient:
    app = create_reconstruction_app(storage=storage)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def detection_client(storage: LocalStorageClient) -> TestClient:
    app = create_detection_app(storage=storage)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def pipeline_service(session_factory, storage, reconstruction_client, detection_client) -> PipelineService:
    return PipelineService(
        session_factory=session_factory,
        storage=storage,
        inference_client=AppInferenceClient(reconstruction_client, detection_client),
    )

