"""Public exceptions raised by the Lattica clients."""


class LatticaClientError(Exception):
    """Base class for client failures."""


class StorageError(LatticaClientError, ValueError):
    """Local client state could not be stored or loaded."""


class SerializationError(LatticaClientError, ValueError):
    """Serialized query data is malformed or unsupported."""


class InvalidCredentialError(LatticaClientError, ValueError):
    """A query credential is malformed or lacks required claims."""


class TransportError(LatticaClientError):
    def __init__(self, action: str, message: str):
        self.action = action
        super().__init__(f"{action}: {message}")


class TransportTimeoutError(TransportError):
    pass


class HTTPResponseError(LatticaClientError):
    def __init__(self, action: str, status_code: int, detail: str):
        self.action = action
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{action} failed with HTTP {status_code}: {detail}")


class AuthenticationError(HTTPResponseError):
    pass


class ProtocolError(LatticaClientError):
    def __init__(self, action: str, message: str):
        self.action = action
        super().__init__(f"{action}: {message}")


class WorkerExecutionError(LatticaClientError):
    def __init__(self, action: str, message: str):
        self.action = action
        super().__init__(f"{action} failed: {message}")


class PollingTimeoutError(LatticaClientError):
    def __init__(self, action: str, timeout: float):
        self.action = action
        self.timeout = timeout
        super().__init__(f"{action} did not finish within {timeout:g} seconds")


class ClientVersionError(LatticaClientError):
    def __init__(self, message: str, min_version: str | None = None):
        self.message = message
        self.min_version = min_version
        super().__init__(self.get_user_message())

    def get_user_message(self) -> str:
        base_msg = "Your client is outdated and incompatible with the server."
        if self.min_version:
            return f"{base_msg} Please upgrade to version {self.min_version} or higher."
        return f"{base_msg} Please upgrade to the latest version."
