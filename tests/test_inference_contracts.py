from __future__ import annotations

import json


def test_reconstruction_contract(reconstruction_client, storage):
    input_uri = storage.upload_bytes("inputs/source.h5", b"kspace", "application/octet-stream")
    response = reconstruction_client.post(
        "/infer",
        json={
            "input_uri": input_uri,
            "input_format": "h5",
            "output_prefix": "jobs/job-1/reconstruction",
            "correlation_id": "job-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reconstructed_dicom_uri"] == "s3://mri-analysis/jobs/job-1/reconstruction/reconstructed.dicom"
    stored = storage.read_bytes(payload["reconstructed_dicom_uri"])
    assert stored.startswith(b"STUB-DICOM")


def test_detection_contract(detection_client, storage):
    input_uri = storage.upload_bytes("inputs/reconstructed.dicom", b"dicom-bytes", "application/dicom")
    response = detection_client.post(
        "/infer",
        json={
            "input_uri": input_uri,
            "output_prefix": "jobs/job-2/detection",
            "correlation_id": "job-2",
            "job_id": "job-2",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    findings = json.loads(storage.read_bytes(payload["findings_json_uri"]).decode("utf-8"))
    assert payload["annotated_dicom_uri"] == "s3://mri-analysis/jobs/job-2/detection/annotated.dicom"
    assert findings["job_id"] == "job-2"
    assert findings["artifacts"]["annotated_dicom_uri"] == payload["annotated_dicom_uri"]
    assert findings["findings"][0]["pathology_type"] == "stub_lesion"
