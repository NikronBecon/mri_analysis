# MRI Analysis

MRI Analysis v1 is an async MRI inference pipeline with three application services:

- `pipeline-api`: public upload and job/status API
- `reconstruction-service`: black-box MRI reconstruction adapter
- `detection-service`: black-box pathology detection adapter

All file exchange happens through object storage URIs, not shared host paths. In v1 the model side is implemented with stub adapters so the orchestration and service contracts are ready before real model code lands.

## Architecture

See [docs/architecture.md](/Users/nikron/Desktop/mri_analysis/docs/architecture.md) for the component and data-flow view.

Public endpoints:

- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/results`
- `GET /v1/jobs/{job_id}/artifacts/{artifact_type}`

Internal endpoints:

- `POST /infer` on `reconstruction-service`
- `POST /infer` on `detection-service`

Job states:

- `uploaded`
- `queued`
- `reconstructing`
- `detecting`
- `completed`
- `failed`

## Local Python Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
uvicorn mri_analysis.pipeline_api.main:create_app --factory --reload
```

For service-by-service local work, run:

```bash
uvicorn mri_analysis.reconstruction_service.main:create_app --factory --port 8001 --reload
uvicorn mri_analysis.detection_service.main:create_app --factory --port 8002 --reload
```

When running outside Compose, set storage/database env vars first. For test-only local runs you can use:

```bash
export STORAGE_MODE=local
export STORAGE_ROOT=.data/storage
export DATABASE_URL=sqlite:///./mri_analysis.db
```

## Docker Compose

CPU baseline:

```bash
cp .env.example .env
docker compose up --build -d
```

GPU-enabled inference containers:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Default service ports:

- `pipeline-api`: `http://localhost:8000`
- `reconstruction-service`: `http://localhost:8001`
- `detection-service`: `http://localhost:8002`
- `minio`: `http://localhost:9000`
- `minio console`: `http://localhost:9001`

Quick health check:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

End-to-end smoke test:

```bash
chmod +x scripts/docker_smoke_test.sh
./scripts/docker_smoke_test.sh
```

Useful Docker commands:

```bash
docker compose ps
docker compose logs -f pipeline-api reconstruction-service detection-service
docker compose down -v
```

## Example Flow

Create a job:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -F "file=@sample.h5"
```

Check status:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
```

Get results:

```bash
curl http://localhost:8000/v1/jobs/<job_id>/results
```

Download findings JSON:

```bash
curl http://localhost:8000/v1/jobs/<job_id>/artifacts/findings_json
```

## Tests

```bash
pytest
```

The test suite runs with local filesystem-backed storage and SQLite, while inference containers are exercised through the same FastAPI contracts used in Compose.
