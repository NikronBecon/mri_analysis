from __future__ import annotations

from typing import Protocol

import httpx

from mri_analysis.shared.schemas import (
    DetectionRequest,
    DetectionResponse,
    ReconstructionRequest,
    ReconstructionResponse,
)


class InferenceClient(Protocol):
    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResponse:
        ...

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        ...


class HttpInferenceClient:
    def __init__(self, reconstruction_url: str, detection_url: str, timeout_seconds: float) -> None:
        self.reconstruction_url = reconstruction_url.rstrip("/")
        self.detection_url = detection_url.rstrip("/")
        self.timeout = timeout_seconds

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResponse:
        response = httpx.post(
            f"{self.reconstruction_url}/infer",
            json=request.model_dump(),
            headers={"x-correlation-id": request.correlation_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return ReconstructionResponse.model_validate(response.json())

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        response = httpx.post(
            f"{self.detection_url}/infer",
            json=request.model_dump(),
            headers={"x-correlation-id": request.correlation_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return DetectionResponse.model_validate(response.json())

