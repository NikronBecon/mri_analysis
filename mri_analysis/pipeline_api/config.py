from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineSettings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./mri_analysis.db")
    worker_poll_interval_seconds: float = float(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "1.0"))
    reconstruction_service_url: str = os.getenv("RECONSTRUCTION_SERVICE_URL", "http://reconstruction-service:8000")
    detection_service_url: str = os.getenv("DETECTION_SERVICE_URL", "http://detection-service:8000")
    inference_timeout_seconds: float = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "60"))


def get_settings() -> PipelineSettings:
    return PipelineSettings()

