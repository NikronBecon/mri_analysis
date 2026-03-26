#!/usr/bin/env bash

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
TMP_DIR="$(mktemp -d)"
INPUT_FILE="${TMP_DIR}/sample.h5"

cleanup() {
  rm -rf "${TMP_DIR}"
}

trap cleanup EXIT

printf 'stub-mri-input\n' > "${INPUT_FILE}"

echo "Submitting ${INPUT_FILE} to ${API_URL}/v1/jobs"
JOB_ID="$(
  curl -fsS -X POST "${API_URL}/v1/jobs" \
    -F "file=@${INPUT_FILE};type=application/octet-stream" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
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
