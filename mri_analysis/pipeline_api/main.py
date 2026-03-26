from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from mri_analysis.pipeline_api.clients import HttpInferenceClient
from mri_analysis.pipeline_api.config import PipelineSettings, get_settings
from mri_analysis.pipeline_api.database import Base, create_session_factory, get_engine
from mri_analysis.pipeline_api.service import PipelineService
from mri_analysis.pipeline_api.worker import JobWorker
from mri_analysis.shared.schemas import ArtifactType, JobResponse
from mri_analysis.shared.storage import build_storage_from_env


def create_app(
    settings: Optional[PipelineSettings] = None,
    *,
    pipeline_service: Optional[PipelineService] = None,
    start_worker: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    if pipeline_service is None:
        session_factory = create_session_factory(settings.database_url)
        Base.metadata.create_all(bind=get_engine(session_factory))
        pipeline_service = PipelineService(
            session_factory=session_factory,
            storage=build_storage_from_env(),
            inference_client=HttpInferenceClient(
                reconstruction_url=settings.reconstruction_service_url,
                detection_url=settings.detection_service_url,
                timeout_seconds=settings.inference_timeout_seconds,
            ),
        )
    worker = JobWorker(pipeline_service, settings.worker_poll_interval_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline_service = pipeline_service
        app.state.ready = True
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            if start_worker:
                worker.stop()

    app = FastAPI(title="MRI Analysis Pipeline API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="Service is not ready")
        return {"status": "ready"}

    @app.post("/v1/jobs", response_model=JobResponse, status_code=202)
    async def create_job_endpoint(
        request: Request,
        file: UploadFile = File(...),
        x_correlation_id: Optional[str] = Header(default=None),
    ) -> JobResponse:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            job_id, status = request.app.state.pipeline_service.create_job(
                filename=file.filename or "input.bin",
                content=payload,
                correlation_id=x_correlation_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JobResponse(job_id=job_id, status=status)

    @app.get("/v1/jobs/{job_id}")
    async def get_job_endpoint(request: Request, job_id: str):
        detail = request.app.state.pipeline_service.get_job(job_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return detail

    @app.get("/v1/jobs/{job_id}/results")
    async def get_results_endpoint(request: Request, job_id: str):
        try:
            result = request.app.state.pipeline_service.get_results(job_id, str(request.base_url).rstrip("/"))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return result

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_type}")
    async def get_artifact_endpoint(request: Request, job_id: str, artifact_type: ArtifactType):
        artifact = request.app.state.pipeline_service.read_artifact(job_id, artifact_type)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        payload, mime_type = artifact
        return Response(content=payload, media_type=mime_type)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app
