# MRI Analysis Architecture

## Components
- `pipeline-api`: public HTTP API, job state machine, artifact manifest API, and background worker.
- `reconstruction-service`: internal inference service that accepts `.h5` or DICOM input and returns reconstructed DICOM.
- `detection-service`: internal inference service that accepts reconstructed DICOM and returns annotated DICOM plus findings JSON.
- `postgres`: stores job state and artifact metadata.
- `minio`: stores uploaded inputs and generated outputs.

## Data Flow
1. Client uploads a single `.h5`, `.dcm`, or `.dicom` file to `POST /v1/jobs`.
2. `pipeline-api` saves the file to object storage and creates a job with status `queued`.
3. Background worker calls `reconstruction-service`.
4. Reconstructed DICOM is stored as an artifact and passed to `detection-service`.
5. Detection outputs are saved as artifacts and the job moves to `completed`.

## Service Contracts
- Public API:
  - `POST /v1/jobs`
  - `GET /v1/jobs/{job_id}`
  - `GET /v1/jobs/{job_id}/results`
  - `GET /v1/jobs/{job_id}/artifacts/{artifact_type}`
- Internal services:
  - `POST /infer` on `reconstruction-service`
  - `POST /infer` on `detection-service`

All services exchange storage URIs rather than shared host paths. This keeps the containers isolated and makes the pipeline easier to deploy.

## Model Strategy
- `reconstruction-service` supports a stub adapter and a real MONAI `BasicUNet` adapter backed by `demo_checkpoint/unet_mri_reconstruction.pt`.
- `detection-service` supports a stub adapter by default.
- An optional SegResNet adapter exists for `detection-service`, but it requires an external MONAI bundle that is not versioned in this repository.

## Runtime Notes
- Default Compose setup is CPU-first.
- `docker-compose.gpu.yml` adds GPU runtime options for compatible hosts.
- The worker runs inside `pipeline-api` to keep the system simple for v1 while preserving an explicit job state machine.
