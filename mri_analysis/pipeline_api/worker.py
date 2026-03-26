from __future__ import annotations

import threading
import time

from mri_analysis.pipeline_api.service import PipelineService


class JobWorker:
    def __init__(self, pipeline_service: PipelineService, poll_interval_seconds: float) -> None:
        self.pipeline_service = pipeline_service
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pipeline-worker")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self.poll_interval_seconds + 1)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            processed = self.pipeline_service.process_next_job()
            if not processed:
                self._stop_event.wait(self.poll_interval_seconds)
            else:
                time.sleep(0.1)

