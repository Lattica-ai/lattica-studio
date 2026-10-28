import importlib.metadata
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class ClientInfo:
    """Name and installed version sent with backend requests."""

    module: str
    version: str

    @classmethod
    def from_package(cls, package: str) -> "ClientInfo":
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"No '{package}' package found") from exc
        return cls(module=package, version=version)


def _normalize_url(url: str, name: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{name} must be a non-empty URL")
    normalized = url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTP(S) URL")
    return normalized


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """Immutable endpoints and metadata used by one client."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    backend_url: str = "https://api.lattica.ai"
    worker_url: str | None = None

    def __post_init__(self) -> None:
        backend_url = _normalize_url(self.backend_url, "backend_url")
        worker_url = _normalize_url(
            self.worker_url or f"{backend_url}/api/do_action",
            "worker_url",
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "backend_url", backend_url)
        object.__setattr__(self, "worker_url", worker_url)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_environment(cls) -> "TransportConfig":
        return cls(backend_url=os.getenv("LATTICA_BE_URL", "https://api.lattica.ai"))
