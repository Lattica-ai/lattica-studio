from contextlib import contextmanager, nullcontext
from collections.abc import Iterable

import time
from typing import Optional, Iterator

from lattica_query.api.app import AppAPI
from lattica_query.logging import (
    Logging,
    STUDIO_THEME,
    current_animation,
    log_info,
    log_status,
)

from ..display import display_table
from ..exceptions import InvalidResourceResponseError, WorkerStartupTimeoutError
from ..types import (
    ModelId,
    Worker,
    WorkerSessionId,
)


class WorkersAPI:
    def __init__(self, http: AppAPI):
        self._http = http

    @staticmethod
    def _phase(name: str):
        """Join an active operation so its details stay under one log entry."""
        if current_animation() is not None:
            return nullcontext()
        return Logging(name, theme=STUDIO_THEME)

    def get(
        self,
        model_id: ModelId,
        session_id: WorkerSessionId,
    ) -> Worker:
        """Return the current status of a worker."""
        response = self._http.send_http_request(
            "api/worker/poll_worker_status",
            req_params={
                "modelId": model_id,
                "workerSessionId": session_id,
            },
        )
        return self._worker(response)

    def active(
        self,
        model_id: ModelId,
    ) -> list[Worker]:
        """Return active workers for a model."""
        with self._phase("checking active workers"):
            log_info(f"model: {model_id}")
            response = self._http.send_http_request(
                "api/worker/get_active_workers",
                req_params={
                    "modelId": model_id,
                },
            )

            if not isinstance(response, dict):
                raise InvalidResourceResponseError("Active worker response is malformed")
            groups = response.get("activeWorkers", [])
            if not isinstance(groups, list) or not all(
                isinstance(group, dict) and isinstance(group.get("workers", []), list)
                for group in groups
            ):
                raise InvalidResourceResponseError("Active worker response is malformed")
            workers = [
                self._worker(worker)
                for group in groups
                if isinstance(group, dict)
                for worker in group.get("workers", [])
            ]
            log_info(f"active workers: {len(workers)}")
            return workers

    def stop(
        self,
        model_id: Optional[ModelId] = None,
        session_id: Optional[WorkerSessionId] = None,
    ) -> Worker:
        """
        Stop a worker session.

        If only model_id is provided, the backend may stop all workers
        associated with that model.
        """
        with self._phase("stopping worker"):
            if model_id is not None:
                log_info(f"model: {model_id}")
            if session_id is not None:
                log_info(f"worker session: {session_id}")
            response = self._http.send_http_request(
                "api/worker/stop_worker",
                req_params={
                    "modelId": model_id,
                    "workerSessionId": session_id,
                },
            )
            worker = self._worker(response)
            log_info(f"status: {worker.status or 'unknown'}")
            return worker

    def start(
        self,
        model_id: ModelId,
        *,
        poll_interval: float = 5,
        timeout: float = 600,
    ) -> Worker:
        """
        Start a worker and wait until it becomes ready.
        """
        with self._phase("starting worker"):
            log_info(f"model: {model_id}")
            worker = self._start(model_id)
            session_id = worker.session_id
            if session_id is not None:
                log_info(f"worker session: {session_id}")

            start_time = time.monotonic()

            while not worker.is_ready:
                if time.monotonic() - start_time >= timeout:
                    raise WorkerStartupTimeoutError(
                        f"Worker for model {model_id} did not become ready "
                        f"within {timeout:g} seconds."
                    )

                status = worker.status or "unknown"
                log_status(
                    f"worker status: {status.lower()}"
                )

                time.sleep(poll_interval)

                session_id = worker.session_id

                if session_id is None:
                    raise InvalidResourceResponseError(
                        "Worker response does not contain workerSessionId."
                    )

                worker = self.get(
                    model_id,
                    session_id,
                )

            log_status("worker ready")
            log_info(f"status: {worker.status or 'unknown'}")

            return worker

    def get_or_start(
        self,
        model_id: ModelId,
    ) -> Worker:
        """
        Return an existing ready worker.

        Non-ready active workers are stopped before a new worker is started.
        """
        with self._phase("ensuring worker is running"):
            log_info(f"model: {model_id}")
            for worker in self.active(model_id):
                status = worker.status or "unknown"
                session_id = worker.session_id
                if worker.is_ready:
                    log_info("decision: reuse ready worker")
                    log_info(f"worker session: {session_id or 'unknown'}")
                    log_info(f"status: {status}")
                    return worker

                if session_id is not None:
                    log_info("decision: replace non-ready worker")
                    log_info(f"worker session: {session_id}")
                    log_info(f"status: {status}")
                    self.stop(model_id=model_id, session_id=session_id)
                else:
                    log_info("decision: non-ready worker cannot be stopped; session id missing")
                    log_info(f"status: {status}")

            log_info("decision: no ready worker available; start a new worker")
            return self.start(model_id)

    def list_sessions(
        self,
        *,
        model_id: Optional[ModelId] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[Worker]:
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

        if not isinstance(response, dict):
            raise InvalidResourceResponseError("Worker session list response is malformed")
        sessions = response.get("workerSessions", [])
        if not isinstance(sessions, list):
            raise InvalidResourceResponseError("Worker session list response is malformed")
        return [self._worker(session) for session in sessions]

    @staticmethod
    def display(workers: Iterable[Worker]) -> None:
        """Print an easy-to-scan table of workers."""
        display_table(
            ("WORKER SESSION", "MODEL ID", "STATUS", "STARTED"),
            (
                (worker.session_id, worker.model_id, worker.status, worker.started_at)
                for worker in workers
            ),
            empty_message="No workers found.",
        )

    @staticmethod
    def _worker(response) -> Worker:
        if not isinstance(response, dict):
            raise InvalidResourceResponseError("Worker response is malformed")
        return Worker.from_api(response)

    def _start(
        self,
        model_id: ModelId,
    ) -> Worker:
        response = self._http.send_http_request(
            "api/worker/start_worker",
            req_params={
                "modelId": model_id,
            },
        )
        return self._worker(response)

    @contextmanager
    def running(
            self,
            model_id: ModelId,
            *,
            stop_on_exit: bool = False,
    ) -> Iterator[Worker]:
        with Logging("preparing worker context", theme=STUDIO_THEME):
            log_info(f"stop on exit: {stop_on_exit}")
            worker = self.get_or_start(model_id)
            session_id = worker.session_id

        try:
            yield worker
        finally:
            with Logging("closing worker context", theme=STUDIO_THEME):
                log_info(f"model: {model_id}")
                log_info(f"worker session: {session_id or 'unknown'}")
                if stop_on_exit and session_id is not None:
                    log_info("decision: stop worker")
                    self.stop(model_id=model_id, session_id=session_id)
                elif stop_on_exit:
                    log_info("decision: stop skipped; worker session id missing")
                else:
                    log_info("decision: leave worker running")
