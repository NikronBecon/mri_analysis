# MRI Analysis

MRI Analysis is a containerized MRI processing pipeline with three services:

- `pipeline-api`: public API for upload, job tracking, and artifact download
- `reconstruction-service`: reconstructs MRI input into DICOM output
- `detection-service`: analyzes reconstructed DICOM and produces annotated DICOM plus findings JSON

The system is built around Docker, PostgreSQL, and MinIO. Services exchange artifacts through object storage URIs, which keeps them isolated and easy to orchestrate.

## What Is Included

- Async end-to-end pipeline with job lifecycle management
- Real reconstruction adapter based on MONAI `BasicUNet`
- Stub detection adapter for reproducible end-to-end runs
- Optional SegResNet detection adapter that can be enabled with an external MONAI bundle
- Docker Compose setup, smoke test script, and automated tests

## Current Status

- Reconstruction is connected to a real checkpoint stored in `demo_checkpoint/unet_mri_reconstruction.pt`.
- Detection runs in stub mode by default, so the repository works without external weights.
- If you have an external SegResNet bundle, the detection service can be switched to it through environment variables.

## Repository Layout

```text
mri_analysis/
  pipeline_api/           API, worker, persistence, orchestration
  reconstruction_service/ reconstruction adapters and HTTP service
  detection_service/      detection adapters and HTTP service
  shared/                 schemas and storage helpers
demo_checkpoint/          bundled reconstruction checkpoint
docs/                     architecture notes
scripts/                  smoke tests and helper scripts
tests/                    API and integration tests
```

## Architecture

High-level architecture is documented in [docs/architecture.md](docs/architecture.md).

Public API:

- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/results`
- `GET /v1/jobs/{job_id}/artifacts/{artifact_type}`

Internal services:

- `POST /infer` on `reconstruction-service`
- `POST /infer` on `detection-service`

## Quick Start

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Start the stack:

```bash
docker compose up --build -d
```

3. Check service health:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

Default ports:

- `pipeline-api`: `http://localhost:8000`
- `reconstruction-service`: `http://localhost:8001`
- `detection-service`: `http://localhost:8002`
- `minio`: `http://localhost:9000`
- `minio console`: `http://localhost:9001`

By default, Compose uses the real reconstruction adapter and the stub detection adapter.

## Example Usage

Create a job:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -F "file=@sample.dcm"
```

Check status:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
```

Get the result manifest:

```bash
curl http://localhost:8000/v1/jobs/<job_id>/results
```

Download findings JSON:

```bash
curl http://localhost:8000/v1/jobs/<job_id>/artifacts/findings_json
```

## Smoke Test

```bash
chmod +x scripts/docker_smoke_test.sh
./scripts/docker_smoke_test.sh
```

The script submits a generated DICOM file to `pipeline-api`, waits for completion, and prints the result manifest and findings JSON.

## Local Development

Create a virtual environment and install the base dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
```

Run the API locally:

```bash
uvicorn mri_analysis.pipeline_api.main:create_app --factory --reload
```

For local filesystem-backed development:

```bash
export STORAGE_MODE=local
export STORAGE_ROOT=.data/storage
export DATABASE_URL=sqlite:///./mri_analysis.db
```

## Model Configuration

Enable the real reconstruction adapter:

```bash
pip install ".[dev,reconstruction]"
export RECONSTRUCTION_ADAPTER=monai_unet
export RECONSTRUCTION_MODEL_PATH=$(pwd)/demo_checkpoint/unet_mri_reconstruction.pt
```

Optional external detection bundle:

```bash
mkdir -p model_bundles/brats_mri_segmentation/models
# place model.pt into model_bundles/brats_mri_segmentation/models/model.pt
pip install ".[dev,detection]"
export DETECTION_ADAPTER=segresnet
export DETECTION_BUNDLE_DIR=$(pwd)/model_bundles/brats_mri_segmentation
```

## Tests

```bash
pytest
```

## Notes

- The repository intentionally does not include large detection weights or training datasets.
- The optional SegResNet path is for external bundle integration, not for the default repository run.
- The public interface is HTTP upload based. If strict file-path container entrypoints are required, a thin wrapper still needs to be added.
