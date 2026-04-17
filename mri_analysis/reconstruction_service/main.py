from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request

from mri_analysis.reconstruction_service.adapters import (
    ReconstructionAdapter,
    build_reconstruction_adapter_from_env,
)
from mri_analysis.shared.schemas import ReconstructionRequest, ReconstructionResponse
from mri_analysis.shared.storage import StorageClient, build_storage_from_env


def create_app(
    storage: Optional[StorageClient] = None,
    adapter: Optional[ReconstructionAdapter] = None,
) -> FastAPI:
    storage = storage or build_storage_from_env()
    adapter = adapter or build_reconstruction_adapter_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.storage = storage
        app.state.adapter = adapter
        app.state.ready = True
        yield

    app = FastAPI(title="MRI Reconstruction Service", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="Service is not ready")
        return {"status": "ready"}

    @app.post("/infer", response_model=ReconstructionResponse)
    async def infer(request: Request, payload: ReconstructionRequest) -> ReconstructionResponse:
        with tempfile.TemporaryDirectory(prefix="reconstruction-") as temp_dir:
            workdir = Path(temp_dir)
            source_path = workdir / f"input.{payload.input_format}"
            request.app.state.storage.download_file(payload.input_uri, source_path)
            output_path = request.app.state.adapter.run(source_path, workdir, payload)
            output_uri = request.app.state.storage.upload_file(
                output_path,
                key=f"{payload.output_prefix}/reconstructed.dicom",
                content_type="application/dicom",
            )
        return ReconstructionResponse(
            reconstructed_dicom_uri=output_uri,
            metadata={"adapter": getattr(request.app.state.adapter, "name", "unknown"), "correlation_id": payload.correlation_id},
        )

    return app
