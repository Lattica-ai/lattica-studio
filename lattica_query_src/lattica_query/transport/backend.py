import json
import os
import time
from typing import Self, TypeAlias

import requests

from ..errors import ProtocolError
from ..logging import (
    LatticaTheme,
    OperationLog,
    current_animation,
    log_info,
    log_status,
)
from .http import DEFAULT_REQUEST_TIMEOUT, HTTPTransport, RequestTimeout
from .settings import ClientInfo, TransportConfig

AppResponse: TypeAlias = str | dict
def _format_size(file_size: int) -> str:
    if file_size >= 1024 ** 2:
        return f"{file_size / 1024 ** 2:.1f} MB"
    if file_size >= 1024:
        return f"{file_size / 1024:.1f} KB"
    return f"{file_size} B"


class _UploadProgressFile:
    """File wrapper that reports upload progress while requests streams data."""

    def __init__(self, file_obj, total_bytes: int, update_interval_sec: float = 0.2):
        self._file = file_obj
        self._total_bytes = total_bytes
        self._uploaded_bytes = 0
        self._update_interval_sec = update_interval_sec
        self._last_update_ts = 0.0
        self._last_emitted_bytes = -1
        self._total_text = _format_size(total_bytes)

    def __len__(self) -> int:
        return self._total_bytes

    def read(self, size: int = -1) -> bytes:
        chunk = self._file.read(size)
        if not chunk:
            self._emit(force=True)
            return chunk

        self._uploaded_bytes += len(chunk)
        self._emit(force=False)
        return chunk

    def _emit(self, *, force: bool) -> None:
        now = time.perf_counter()
        should_emit = force or (now - self._last_update_ts >= self._update_interval_sec)
        if not should_emit or self._uploaded_bytes == self._last_emitted_bytes:
            return

        percent = 100.0 if self._total_bytes == 0 else min(100.0, (self._uploaded_bytes / self._total_bytes) * 100.0)
        uploaded_text = _format_size(self._uploaded_bytes)
        log_status(f"uploading • {uploaded_text}/{self._total_text} ({percent:.0f}%)")
        self._last_update_ts = now
        self._last_emitted_bytes = self._uploaded_bytes

    def __getattr__(self, item):
        return getattr(self._file, item)


class BackendAPI:
    def __init__(
            self,
            session_token: str | None = None,
            *,
            client_info: ClientInfo,
            theme: LatticaTheme,
            session: requests.Session | None = None,
            request_timeout: RequestTimeout = DEFAULT_REQUEST_TIMEOUT,
            config: TransportConfig | None = None,
        ):
        self.session_token = session_token
        self.client_info = client_info
        self.config = config or TransportConfig.from_environment()
        self.theme = theme
        self._transport = HTTPTransport(session=session, timeout=request_timeout)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _format_request_name(endpoint: str) -> str:
        name = endpoint.rsplit("/", 1)[-1]
        return name.replace("_", " ")

    def call(self, endpoint: str, parameters: dict | None = None) -> AppResponse:
        parameters = parameters if parameters else {}

        # Add client module and version information
        client_info = {
            "client_info": {
                "module": self.client_info.module,
                "version": self.client_info.version
            }
        }
        # Merge client info with request params
        request_body = {**parameters, **client_info}

        headers = {
            'Content-Type': 'application/json',
        }

        if self.session_token:
            headers['Authorization'] = f'Bearer {self.session_token}'

        # If we're already inside a higher-level operation,
        # report this request as a sub-status.
        active_animation = current_animation()
        if active_animation is not None:
            return self._execute_call(endpoint, request_body, headers)
        # Otherwise this HTTP request becomes its own phase.
        with OperationLog(self._format_request_name(endpoint), theme=self.theme):
            return self._execute_call(endpoint, request_body, headers)

    def _execute_call(
            self,
            endpoint: str,
            request_body: dict,
            headers: dict,
    ) -> AppResponse:

        log_status(f"requesting {self._format_request_name(endpoint)}")
        start = time.perf_counter()
        response = self._transport.request(
            "POST",
            f'{self.config.backend_url}/{endpoint}',
            endpoint,
            headers=headers,
            json=request_body
        )
        duration = time.perf_counter() - start
        log_status(f"received response • {duration * 1000:.0f} ms")

        if not response.ok:
            self._transport.raise_response_error(response, endpoint)

        return self._transport.decode_result(response, endpoint)

    def upload_binary(
            self,
            endpoint: str,
            parameters: dict | None = None,
            file_path: str | None = None,
    ) -> dict:
        if file_path is None:
            raise ValueError("file_path must be provided")

        parameters = parameters if parameters else {}

        # Add client module and version information
        client_info = {
            "client_module": self.client_info.module,
            "client_version": self.client_info.version
        }
        # Merge client info with request params
        request_metadata = {**parameters, **client_info}

        url = f'{self.config.backend_url}/{endpoint}'
        metadata = json.dumps(request_metadata, separators=(",", ":"))

        file_size = os.path.getsize(file_path)
        headers = {
            "Authorization": f"Bearer {self.session_token}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(file_size),
            "Accept": "application/json",
            "X-Metadata": metadata,
        }

        with open(file_path, "rb") as stream:
            response = self._transport.request(
                "POST", url, endpoint, headers=headers, data=stream
            )

        if not response.ok:
            self._transport.raise_response_error(response, endpoint)

        return self._transport.decode_result(response, endpoint, require_result=False)


    def is_worker_up(self) -> bool:
        """Whether the token's model has a worker up."""
        endpoint = 'api/token/get_token_info'
        response = self.call(endpoint, parameters={'token': self.session_token})
        worker = response.get('worker') if isinstance(response, dict) else None
        if not isinstance(worker, dict) or 'status' not in worker:
            raise ProtocolError(endpoint, "token info response is missing 'worker.status'")
        return worker['status'] == 'UP'

    def upload_file_and_alert(
            self,
            file_name: str,
            endpoint: str,
            upload_params: dict | None = None,
            alert_params: dict | None = None,
    ) -> None:
        file_key = self._upload_file(file_name, endpoint=endpoint, params=upload_params)
        log_status("registering upload")
        response = self._alert_upload_complete(file_key, params=alert_params)
        log_info(f"upload status: {response}")

    def _upload_file(self, file_path: str, endpoint: str, params: dict | None = None) -> str:
        log_status("requesting upload URL")
        upload_url, file_key = self._request_upload_target(endpoint, params)
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as file:
            self._put_upload(upload_url, file, file_size)
        return file_key

    def _request_upload_target(self, endpoint: str, params: dict | None) -> tuple[str, str]:
        response = self.call(endpoint, params)
        if not isinstance(response, dict):
            raise ProtocolError(endpoint, "upload URL response must be an object")
        upload_url = response.get('s3Url')
        file_key = response.get('s3Key')
        if not isinstance(upload_url, str) or not upload_url:
            raise ProtocolError(endpoint, "upload URL response is missing 's3Url'")
        if not isinstance(file_key, str) or not file_key:
            raise ProtocolError(endpoint, "upload URL response is missing 's3Key'")
        return upload_url, file_key

    def _put_upload(self, upload_url: str, stream, file_size: int) -> None:
        size_text = _format_size(file_size)
        log_status(f"uploading • {size_text}")
        if file_size == 0:
            # To avoid: header (Transfer-Encoding: chunked) that S3 does not support for pre-signed PUT uploads.
            res = self._transport.request("PUT", upload_url, "upload file", data=b'')
        else:
            res = self._transport.request(
                "PUT", upload_url, "upload file", data=_UploadProgressFile(stream, file_size)
            )
        if not res.ok:
            self._transport.raise_response_error(res, "upload file")
        log_status("upload complete")

    def _alert_upload_complete(self, key: str, params: dict | None = None) -> str:
        response = self.call(
            'api/files/upload_complete',
            parameters={'s3Key': key, 'initContextParams': params}
        )
        if not isinstance(response, str):
            raise ProtocolError("api/files/upload_complete", "result must be a string")
        return response
