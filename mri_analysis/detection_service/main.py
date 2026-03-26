from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request

from mri_analysis.detection_service.adapters import DetectionAdapter, StubDetectionAdapter
from mri_analysis.shared.schemas import DetectionRequest, DetectionResponse
from mri_analysis.shared.storage import StorageClient, build_storage_from_env


def create_app(
    storage: Optional[StorageClient] = None,
    adapter: Optional[DetectionAdapter] = None,
) -> FastAPI:
    storage = storage or build_storage_from_env()
    adapter = adapter or StubDetectionAdapter()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.storage = storage
        app.state.adapter = adapter
        app.state.ready = True
        yield

    app = FastAPI(title="MRI Pathology Detection Service", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="Service is not ready")
        return {"status": "ready"}

    @app.post("/infer", response_model=DetectionResponse)
    async def infer(request: Request, payload: DetectionRequest) -> DetectionResponse:
        with tempfile.TemporaryDirectory(prefix="detection-") as temp_dir:
            workdir = Path(temp_dir)
            source_path = workdir / "reconstructed.dicom"
            request.app.state.storage.download_file(payload.input_uri, source_path)
            annotated_path, findings_path = request.app.state.adapter.run(source_path, workdir, payload)
            annotated_uri = request.app.state.storage.upload_file(
                annotated_path,
                key=f"{payload.output_prefix}/annotated.dicom",
                content_type="application/dicom",
            )
            findings_payload = json.loads(findings_path.read_text(encoding="utf-8"))
            findings_payload["artifacts"] = {
                "annotated_dicom_uri": annotated_uri,
                "findings_json_uri": f"s3://{request.app.state.storage.bucket}/{payload.output_prefix}/findings.json",
            }
            findings_path.write_text(json.dumps(findings_payload, indent=2), encoding="utf-8")
            findings_uri = request.app.state.storage.upload_file(
                findings_path,
                key=f"{payload.output_prefix}/findings.json",
                content_type="application/json",
            )
        return DetectionResponse(
            annotated_dicom_uri=annotated_uri,
            findings_json_uri=findings_uri,
            metadata={"adapter": "stub", "correlation_id": payload.correlation_id},
        )

    return app
