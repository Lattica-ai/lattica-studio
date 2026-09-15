import os
import tempfile
from pathlib import Path
from urllib.parse import quote

_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700


def encode_path_component(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("path component must be a non-empty string")
    component = quote(value, safe="-_.")
    if component in {".", ".."}:
        return component.replace(".", "%2E")
    return component


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fchmod(descriptor, _PRIVATE_DIRECTORY_MODE)
    finally:
        os.close(descriptor)


def atomic_private_write(path: Path, content: str | bytes) -> None:
    """Atomically replace a text or binary file with owner-only permissions."""
    ensure_private_directory(path.parent)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    cleanup_path: Path | None = temporary_path
    try:
        os.fchmod(fd, _PRIVATE_FILE_MODE)
        if isinstance(content, bytes):
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        cleanup_path = None

        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)
