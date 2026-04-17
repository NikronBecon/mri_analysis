#!/usr/bin/env bash

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
echo "Submitting generated sample.dcm to ${API_URL}/v1/jobs"
JOB_ID="$(
  docker compose exec -T reconstruction-service python - <<'PY' \
    | curl -fsS -X POST "${API_URL}/v1/jobs" -F "file=@-;filename=sample.dcm;type=application/dicom" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
import io
import sys

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, PYDICOM_IMPLEMENTATION_UID, generate_uid

buffer = io.BytesIO()
meta = FileMetaDataset()
meta.MediaStorageSOPClassUID = MRImageStorage
meta.MediaStorageSOPInstanceUID = generate_uid()
meta.TransferSyntaxUID = ExplicitVRLittleEndian
meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

dataset = FileDataset("sample.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
dataset.is_little_endian = True
dataset.is_implicit_VR = False
dataset.SOPClassUID = meta.MediaStorageSOPClassUID
dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
dataset.StudyInstanceUID = generate_uid()
dataset.SeriesInstanceUID = generate_uid()
dataset.Modality = "MR"
dataset.PatientName = "Smoke^Test"
dataset.PatientID = "smoke-test"
dataset.Rows = 16
dataset.Columns = 16
dataset.NumberOfFrames = "3"
dataset.SamplesPerPixel = 1
dataset.PhotometricInterpretation = "MONOCHROME2"
dataset.BitsAllocated = 16
dataset.BitsStored = 16
dataset.HighBit = 15
dataset.PixelRepresentation = 0
dataset.PixelSpacing = [1.0, 1.0]
dataset.SliceThickness = 1.0
dataset.ImageType = ["ORIGINAL", "PRIMARY"]
dataset.PixelData = (np.arange(16 * 16 * 3, dtype=np.uint16).reshape(3, 16, 16)).tobytes()
dataset.save_as(buffer, write_like_original=False)
sys.stdout.buffer.write(buffer.getvalue())
PY
)"

echo "Job created: ${JOB_ID}"

for _ in $(seq 1 60); do
  STATUS="$(
    curl -fsS "${API_URL}/v1/jobs/${JOB_ID}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])'
  )"
  echo "Current status: ${STATUS}"

  if [[ "${STATUS}" == "completed" ]]; then
    break
  fi

  if [[ "${STATUS}" == "failed" ]]; then
    echo "Job failed:"
    curl -fsS "${API_URL}/v1/jobs/${JOB_ID}"
    exit 1
  fi

  sleep 1
done

echo
echo "Results:"
curl -fsS "${API_URL}/v1/jobs/${JOB_ID}/results"
echo
echo
echo "Findings JSON:"
curl -fsS "${API_URL}/v1/jobs/${JOB_ID}/artifacts/findings_json"
echo
