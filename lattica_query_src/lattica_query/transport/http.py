from typing import Any, Self, TypeAlias

import requests

from ..errors import (
    AuthenticationError,
    ClientVersionError,
    HTTPResponseError,
    ProtocolError,
    TransportError,
    TransportTimeoutError,
)

RequestTimeout: TypeAlias = float | tuple[float, float]
DEFAULT_REQUEST_TIMEOUT: tuple[float, float] = (10.0, 300.0)
_MAX_ERROR_DETAIL_LENGTH = 500


def validate_request_timeout(request_timeout: RequestTimeout) -> RequestTimeout:
    values = request_timeout if isinstance(request_timeout, tuple) else (request_timeout,)
    if len(values) not in (1, 2) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in values
    ):
        raise ValueError("request_timeout must be a positive number or a pair of positive numbers")
    return request_timeout


class HTTPTransport:
    """Shared request execution and response validation for Lattica clients."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout: RequestTimeout = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.timeout = validate_request_timeout(timeout)
        self.session = session if session is not None else requests.Session()
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def request(
        self,
        method: str,
        url: str,
        action: str,
        *,
        timeout: RequestTimeout | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        try:
            return self.session.request(
                method,
                url,
                timeout=timeout if timeout is not None else self.timeout,
                **kwargs,
            )
        except requests.Timeout as exc:
            raise TransportTimeoutError(action, "request timed out") from exc
        except requests.RequestException as exc:
            raise TransportError(action, f"request failed ({type(exc).__name__})") from exc

    @staticmethod
    def decode_json(response: requests.Response, action: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except (ValueError, requests.RequestException) as exc:
            raise ProtocolError(action, "server returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ProtocolError(action, "JSON response must be an object")
        return payload

    @classmethod
    def decode_result(
        cls,
        response: requests.Response,
        action: str,
        *,
        require_result: bool = True,
    ) -> Any:
        """Decode the backend envelope and return its result."""
        payload = cls.decode_json(response, action)
        if payload.get("error_code") == "CLIENT_VERSION_INCOMPATIBLE":
            raise ClientVersionError(
                str(payload.get("error", "incompatible client version")),
                payload.get("min_version"),
            )
        if "error" in payload:
            raise HTTPResponseError(action, response.status_code, str(payload["error"]))
        if "result" in payload:
            return payload["result"]
        if require_result:
            raise ProtocolError(action, "JSON response is missing 'result'")
        return payload

    @classmethod
    def raise_response_error(cls, response: requests.Response, action: str) -> None:
        payload: Any = None
        try:
            payload = response.json()
        except (ValueError, requests.RequestException):
            pass
        if isinstance(payload, dict) and payload.get("error_code") == "CLIENT_VERSION_INCOMPATIBLE":
            raise ClientVersionError(
                str(payload.get("error", "incompatible client version")),
                payload.get("min_version"),
            )
        if isinstance(payload, dict):
            detail = payload.get("error") or payload.get("message") or payload.get("detail")
        else:
            detail = None
        detail = str(
            detail or response.text.strip() or response.reason or "request rejected"
        )[:_MAX_ERROR_DETAIL_LENGTH]
        error_type = AuthenticationError if response.status_code in (401, 403) else HTTPResponseError
        raise error_type(action, response.status_code, detail)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
