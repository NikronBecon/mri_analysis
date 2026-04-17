from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request

from mri_analysis.detection_service.adapters import DetectionAdapter, build_detection_adapter_from_env
from mri_analysis.shared.schemas import DetectionRequest, DetectionResponse
from mri_analysis.shared.storage import StorageClient, build_storage_from_env, guess_mime_type


def create_app(
    storage: Optional[StorageClient] = None,
    adapter: Optional[DetectionAdapter] = None,
) -> FastAPI:
    storage = storage or build_storage_from_env()
    adapter = adapter or build_detection_adapter_from_env()

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
            source_name = Path(urlparse(payload.input_uri).path).name or "reconstructed.dicom"
            source_path = workdir / source_name
            request.app.state.storage.download_file(payload.input_uri, source_path)
            annotated_path, findings_path = request.app.state.adapter.run(source_path, workdir, payload)
            annotated_key = f"{payload.output_prefix}/{annotated_path.name}"
            annotated_content_type = (
                "application/dicom"
                if annotated_path.suffix in {".dcm", ".dicom"}
                else guess_mime_type(annotated_path.name)
            )
            annotated_uri = request.app.state.storage.upload_file(
                annotated_path,
                key=annotated_key,
                content_type=annotated_content_type,
            )
            findings_payload = json.loads(findings_path.read_text(encoding="utf-8"))
            findings_payload["artifacts"] = {
                "annotated_dicom_uri": annotated_uri,
                "annotated_output_uri": annotated_uri,
                "annotated_output_filename": annotated_path.name,
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
            metadata={
                "adapter": getattr(request.app.state.adapter, "name", request.app.state.adapter.__class__.__name__),
                "correlation_id": payload.correlation_id,
            },
        )

    return app
