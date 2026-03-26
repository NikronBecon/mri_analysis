# MRI Analysis v1 Architecture

## Components
- `pipeline-api`: public HTTP ingress, upload validation, job state machine, artifact manifest API, and background worker.
- `reconstruction-service`: internal black-box inference service that converts raw `.h5` or DICOM input into reconstructed DICOM.
- `detection-service`: internal black-box inference service that enriches reconstructed DICOM with annotations and findings JSON.
- `postgres`: job and artifact metadata store.
- `minio`: object storage for all input and output artifacts.

## Data Flow
1. Client uploads a single `.h5`, `.dcm`, or `.dicom` file to `POST /v1/jobs`.
2. `pipeline-api` stores the file in object storage, creates a job row, and marks the job as `queued`.
3. Background worker claims the job, calls `reconstruction-service`, and stores the reconstructed DICOM URI as an artifact.
4. Worker calls `detection-service` with the reconstructed DICOM URI and stores annotated DICOM plus findings JSON URIs.
5. Job is marked `completed` and becomes available through status/results endpoints.

## Contracts
- Public API:
  - `POST /v1/jobs`
  - `GET /v1/jobs/{job_id}`
  - `GET /v1/jobs/{job_id}/results`
  - `GET /v1/jobs/{job_id}/artifacts/{artifact_type}`
- Internal inference API:
  - `POST /infer` on `reconstruction-service`
  - `POST /infer` on `detection-service`

All internal requests use storage URIs rather than host file paths, which keeps container boundaries stable and allows notebook-backed adapters later.

## Model Boundary
Each inference service owns an adapter interface plus a stub implementation. The HTTP contract stays stable while the underlying adapter can later wrap a notebook, script, or packaged model without changing the pipeline API or orchestration logic.

## Runtime Notes
- Compose baseline is CPU-first.
- `docker-compose.gpu.yml` adds `gpus: all` to inference containers for hosts with NVIDIA runtime.
- The worker is intentionally inside `pipeline-api` for v1 to avoid a queue dependency while keeping the state machine explicit in the database.

