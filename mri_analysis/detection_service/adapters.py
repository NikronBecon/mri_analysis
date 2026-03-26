from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Tuple

from mri_analysis.shared.schemas import DetectionRequest, Finding, FindingLocation, FindingsDocument


class DetectionAdapter(Protocol):
    def run(self, input_path: Path, workdir: Path, request: DetectionRequest) -> Tuple[Path, Path]:
        ...


@dataclass
class StubDetectionAdapter:
    def run(self, input_path: Path, workdir: Path, request: DetectionRequest) -> Tuple[Path, Path]:
        annotated_path = workdir / "annotated.dicom"
        findings_path = workdir / "findings.json"

        payload = input_path.read_bytes()
        annotated_path.write_bytes(
            b"ANNOTATED-STUB-DICOM\n"
            + f"correlation_id={request.correlation_id}\n".encode("utf-8")
            + payload
        )

        findings = FindingsDocument(
            job_id=request.job_id,
            study_id=request.job_id or request.correlation_id,
            source_file=input_path.name,
            findings=[
                Finding(
                    pathology_type="stub_lesion",
                    confidence=0.93,
                    location=FindingLocation(description="left temporal lobe"),
                    explanation="Stub output for pipeline integration until a real pathology model is plugged in.",
                )
            ],
            artifacts={},
        )
        findings_path.write_text(json.dumps(findings.model_dump(), indent=2), encoding="utf-8")
        return annotated_path, findings_path

