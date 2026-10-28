class LatticaStudioError(RuntimeError):
    """Base exception for errors raised by lattica_studio."""


class CompilationError(LatticaStudioError):
    """Raised when remote model compilation fails."""


class CompilationTimeoutError(LatticaStudioError):
    """Raised when remote model compilation times out."""


class WorkerStartupTimeoutError(LatticaStudioError):
    """Raised when a worker does not become ready in time."""


class ResourceNotFoundError(LatticaStudioError):
    """Raised when a requested Studio resource does not exist."""


class InvalidResourceResponseError(LatticaStudioError):
    """Raised when a backend resource payload is incomplete or malformed."""
