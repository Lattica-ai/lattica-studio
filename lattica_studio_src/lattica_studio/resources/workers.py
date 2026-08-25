from contextlib import contextmanager

import time
from typing import Optional, Iterator

from lattica_query.api.app import AppAPI
from lattica_query.logging import (
    Logging,
    STUDIO_THEME,
    log_info,
    log_status,
)

from ..exceptions import WorkerStartupTimeoutError
from ..types import (
    ActiveWorkerGroup,
    JsonDict,
    ModelId,
    WorkerSessionId,
    WorkerStatus,
)


class WorkersAPI:
    def __init__(self, http: AppAPI):
        self._http = http

    def get(
        self,
        model_id: ModelId,
        session_id: WorkerSessionId,
    ) -> WorkerStatus:
        """Return the current status of a worker."""
        return self._http.send_http_request(
            "api/worker/poll_worker_status",
            req_params={
                "modelId": model_id,
                "workerSessionId": session_id,
            },
        )

    def active(
        self,
        model_id: ModelId,
    ) -> list[ActiveWorkerGroup]:
        """Return active worker groups for a model."""
        response = self._http.send_http_request(
            "api/worker/get_active_workers",
            req_params={
                "modelId": model_id,
            },
        )

        return response.get("activeWorkers", [])

    def stop(
        self,
        model_id: Optional[ModelId] = None,
        session_id: Optional[WorkerSessionId] = None,
    ) -> WorkerStatus:
        """
        Stop a worker session.

        If only model_id is provided, the backend may stop all workers
        associated with that model.
        """
        return self._http.send_http_request(
            "api/worker/stop_worker",
            req_params={
                "modelId": model_id,
                "workerSessionId": session_id,
            },
        )

    def start(
        self,
        model_id: ModelId,
        *,
        poll_interval: float = 5,
        timeout: float = 600,
    ) -> WorkerStatus:
        """
        Start a worker and wait until it becomes ready.
        """
        with Logging(
            "starting worker",
            theme=STUDIO_THEME,
        ):
            worker = self._start(model_id)

            start_time = time.monotonic()

            while worker.get("status") != "UP":
                if time.monotonic() - start_time >= timeout:
                    raise WorkerStartupTimeoutError(
                        f"Worker for model {model_id} did not become ready "
                        f"within {timeout:g} seconds."
                    )

                status = worker.get("status", "unknown")
                log_status(
                    f"worker status: {status.lower()}"
                )

                time.sleep(poll_interval)

                session_id = worker.get("workerSessionId")

                if session_id is None:
                    raise RuntimeError(
                        "Worker response does not contain workerSessionId."
                    )

                worker = self.get(
                    model_id,
                    session_id,
                )

            log_status("worker ready")
            log_info(f"worker status: {worker}")

            return worker

    def get_or_start(
        self,
        model_id: ModelId,
    ) -> WorkerStatus:
        """
        Return an existing ready worker.

        Non-ready active workers are stopped before a new worker is started.
        """
        active_worker_groups = self.active(model_id)

        for group in active_worker_groups:
            for worker in group.get("workers", []):
                if worker.get("status") == "UP":
                    return worker

                session_id = worker.get("workerSessionId")

                if session_id is not None:
                    self.stop(
                        model_id=model_id,
                        session_id=session_id,
                    )

        return self.start(model_id)

    def list_sessions(
        self,
        *,
        model_id: Optional[ModelId] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[JsonDict]:
        """List worker sessions."""
        params = {}

        if model_id is not None:
            params["modelId"] = model_id

        if from_date is not None:
            params["fromDate"] = from_date

        if to_date is not None:
            params["toDate"] = to_date

        response = self._http.send_http_request(
            "api/worker/list_worker_sessions",
            req_params=params,
        )

        return response.get("workerSessions", [])

    def _start(
        self,
        model_id: ModelId,
    ) -> WorkerStatus:
        return self._http.send_http_request(
            "api/worker/start_worker",
            req_params={
                "modelId": model_id,
            },
        )

    from contextlib import contextmanager
    from collections.abc import Iterator

    @contextmanager
    def running(
            self,
            model_id: ModelId,
            *,
            stop_on_exit: bool = False,
    ) -> Iterator[WorkerStatus]:
        worker = self.get_or_start(model_id)

        try:
            yield worker
        finally:
            if stop_on_exit:
                session_id = worker.get("workerSessionId")

                if session_id is not None:
                    self.stop(
                        model_id=model_id,
                        session_id=session_id,
                    )
