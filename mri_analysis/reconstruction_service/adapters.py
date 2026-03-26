from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mri_analysis.shared.schemas import ReconstructionRequest


class ReconstructionAdapter(Protocol):
    def run(self, input_path: Path, workdir: Path, request: ReconstructionRequest) -> Path:
        ...


@dataclass
class StubReconstructionAdapter:
    def run(self, input_path: Path, workdir: Path, request: ReconstructionRequest) -> Path:
        output_path = workdir / "reconstructed.dicom"
        payload = input_path.read_bytes()
        content = (
            b"STUB-DICOM\n"
            + f"correlation_id={request.correlation_id}\n".encode("utf-8")
            + f"source_format={request.input_format}\n".encode("utf-8")
            + payload
        )
        output_path.write_bytes(content)
        return output_path

