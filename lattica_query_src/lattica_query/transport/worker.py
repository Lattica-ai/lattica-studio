import json
import time
from dataclasses import dataclass
from typing import TypeAlias

import requests

from ..errors import (
    PollingTimeoutError,
    ProtocolError,
    TransportTimeoutError,
    WorkerExecutionError,
)
from ..logging import log_status
from .http import DEFAULT_REQUEST_TIMEOUT, HTTPTransport, RequestTimeout
from .settings import ClientInfo, TransportConfig

"""
IMPLEMENTATION OF API CALLS TO A REMOTE WORKER
every API call should adhere to the following:
api params
    action      the name of the api action
    params      a json struct

response structure containing either:
    result          the payload execution result
    executionId     the ID of the api call execution when it takes long to
                    return a result
    error           the execution error
"""

WorkerResponse: TypeAlias = bytes | None


@dataclass(frozen=True, slots=True)
class _PendingResult:
    execution_id: str


@dataclass(frozen=True, slots=True)
class WorkerRequestTiming:
    action: str
    duration: float


@dataclass(frozen=True, slots=True)
class WorkerTiming:
    """Timing for a complete worker action, including polling requests."""

    total: float = 0.0
    requests: tuple[WorkerRequestTiming, ...] = ()
    server_header: str = ""

    @property
    def polling(self) -> float:
        return max(0.0, self.total - sum(request.duration for request in self.requests))


class WorkerTransport:
    def __init__(
        self,
        query_token: str,
        *,
        client_info: ClientInfo,
        session: requests.Session | None = None,
        request_timeout: RequestTimeout = DEFAULT_REQUEST_TIMEOUT,
        polling_interval: float = 1.0,
        polling_timeout: float = 3600.0,
        config: TransportConfig | None = None,
    ):
        if isinstance(polling_interval, bool) or not isinstance(polling_interval, (int, float)) or polling_interval <= 0:
            raise ValueError("polling_interval must be greater than zero")
        if isinstance(polling_timeout, bool) or not isinstance(polling_timeout, (int, float)) or polling_timeout <= 0:
            raise ValueError("polling_timeout must be greater than zero")
        self.query_token = query_token
        self.polling_interval = polling_interval
        self.polling_timeout = polling_timeout
        self.config = config or TransportConfig.from_environment()
        self.client_info = client_info
        self.timing = WorkerTiming()
        self._request_timings: list[WorkerRequestTiming] = []
        self._server_timing_header = ""
        self._transport = HTTPTransport(session=session, timeout=request_timeout)
        self.request_timeout = self._transport.timeout

    def close(self) -> None:
        self._transport.close()

    def invoke(
        self,
        action_name: str,
        parameters: dict | None = None,
        payload: bytes | None = None,
    ) -> WorkerResponse:
        started = time.perf_counter()
        self._request_timings = []
        self._server_timing_header = ""
        result = self._send_request_once(action_name, parameters, payload)
        if isinstance(result, _PendingResult):
            result = self._poll_for_result(action_name, result.execution_id)
        self.timing = WorkerTiming(
            total=time.perf_counter() - started,
            requests=tuple(self._request_timings),
            server_header=self._server_timing_header,
        )
        return result

    def _send_request_once(
        self,
        action_name: str,
        parameters: dict | None = None,
        payload: bytes | None = None,
        request_timeout: RequestTimeout | None = None,
    ) -> WorkerResponse | _PendingResult:
        start = time.perf_counter()
        api_call_payload = {
            "params": parameters if parameters else {},
        }

        req_data = {
            "api_call": api_call_payload,
            "client_info": {
                "module": self.client_info.module,
                "version": self.client_info.version,
            },
        }
        req_data.update(self.config.metadata)

        # Construct the full URL by appending the action name to the base URL.
        base_url = self.config.worker_url
        full_url = f"{base_url}/{action_name}"

        metadata_header_value = json.dumps(req_data, separators=(",", ":"))

        headers = {
            "Authorization": f"Bearer {self.query_token}",
            "Content-Type": "application/octet-stream",
            "X-Metadata": metadata_header_value,
        }

        response = self._transport.request(
            "POST",
            full_url,
            action_name,
            headers=headers,
            data=payload,
            timeout=request_timeout if request_timeout is not None else self.request_timeout,
        )
        duration = time.perf_counter() - start
        self._request_timings.append(WorkerRequestTiming(action_name, duration))

        if not response.ok:
            self._transport.raise_response_error(response, action_name)

        content_type = response.headers.get("Content-Type", "").partition(";")[0].strip().lower()

        if content_type == "application/octet-stream":
            self._record_server_timing(response)
            if not response.content:
                raise ProtocolError(action_name, "server returned an empty binary response")
            return response.content
        elif content_type == "application/json":
            response_json = self._transport.decode_json(response, action_name)

            status = str(response_json.get("status", "")).upper()
            log_status(f"{action_name} • {status.lower()}")

            if status == "RUNNING":
                execution_id = response_json.get("executionId")
                if execution_id is None or execution_id == "":
                    raise ProtocolError(action_name, "RUNNING response is missing 'executionId'")
                return _PendingResult(str(execution_id))

            if status == "ERROR":
                raise WorkerExecutionError(action_name, str(response_json.get("error") or "unknown worker error"))

            if status not in {"COMPLETED", "SUCCESS"}:
                displayed_status = status or "missing"
                raise ProtocolError(action_name, f"unknown worker status: {displayed_status}")

            self._record_server_timing(response)
            return None
        else:
            displayed_type = content_type or "missing"
            raise ProtocolError(action_name, f"unsupported Content-Type: {displayed_type}")

    def _record_server_timing(self, response: requests.Response) -> None:
        self._server_timing_header = response.headers.get("Server-Timing", "")

    def _poll_for_result(self, action_name: str, execution_id: str) -> WorkerResponse:
        """Poll an existing worker action until it completes or reaches its deadline."""
        deadline = time.monotonic() + self.polling_timeout
        log_status("waiting for worker")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PollingTimeoutError(action_name, self.polling_timeout)
            time.sleep(min(self.polling_interval, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PollingTimeoutError(action_name, self.polling_timeout)
            try:
                result = self._send_request_once(
                    "get_action_result",
                    parameters={"executionId": execution_id},
                    request_timeout=self._bounded_request_timeout(remaining),
                )
            except TransportTimeoutError as exc:
                if time.monotonic() >= deadline:
                    raise PollingTimeoutError(action_name, self.polling_timeout) from exc
                continue
            if isinstance(result, _PendingResult):
                execution_id = result.execution_id
                continue
            return result

    def _bounded_request_timeout(self, remaining: float) -> RequestTimeout:
        if isinstance(self.request_timeout, tuple):
            connect_timeout, read_timeout = self.request_timeout
            return min(connect_timeout, remaining), min(read_timeout, remaining)
        return min(self.request_timeout, remaining)
