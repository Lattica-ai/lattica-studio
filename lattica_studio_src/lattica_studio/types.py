from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias


ModelId: TypeAlias = str
WorkerSessionId: TypeAlias = str
Token: TypeAlias = str
JsonDict: TypeAlias = dict[str, Any]


def _value(data: dict[str, Any], *names: str) -> Any:
    return next((data[name] for name in names if data.get(name) is not None), None)


def _short(value: str | None) -> str | None:
    if value is None or len(value) <= 20:
        return value
    return f"{value[:8]}…{value[-4:]}"


@dataclass(frozen=True, slots=True)
class Model:
    id: str | None = None
    name: str | None = None
    status: str | None = None
    visibility: str | None = None
    instance_type: str | None = None
    num_devices: int | None = None
    is_compiled: bool | None = None
    compilation_error: str | None = None
    description: str | None = None
    auto_restart: bool | None = None
    input_type: str | None = None
    output_type: str | None = None
    required_resources: dict[str, Any] | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Model":
        return cls(
            id=_value(data, "modelId", "id"),
            name=_value(data, "modelName", "name"),
            status=data.get("status"),
            visibility=data.get("visibility"),
            instance_type=_value(data, "instanceTypeId", "instanceType", "instance_type"),
            num_devices=_value(data, "numDevices", "num_devices"),
            is_compiled=_value(data, "is_compiled", "isCompiled"),
            compilation_error=_value(data, "compilation_error", "compilationError"),
            description=data.get("description"),
            auto_restart=_value(data, "autoRestart", "auto_restart"),
            input_type=_value(data, "inputType", "input_type"),
            output_type=_value(data, "outputType", "output_type"),
            required_resources=_value(data, "requiredResources", "required_resources"),
        )

    def __repr__(self) -> str:
        return (
            f"Model(name={self.name!r}, id={_short(self.id)!r}, "
            f"status={self.status!r}, compiled={self.is_compiled!r})"
        )


ModelInfo = Model


@dataclass(frozen=True, slots=True)
class Worker:
    session_id: str | None = None
    model_id: str | None = None
    status: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Worker":
        return cls(
            session_id=_value(data, "workerSessionId", "sessionId", "session_id"),
            model_id=_value(data, "modelId", "model_id"),
            status=data.get("status"),
            started_at=_value(data, "startedAt", "startDate", "started_at"),
            stopped_at=_value(data, "stoppedAt", "endDate", "stopped_at"),
        )

    @property
    def is_ready(self) -> bool:
        return self.status == "UP"

    def __repr__(self) -> str:
        return (
            f"Worker(session_id={_short(self.session_id)!r}, "
            f"model_id={_short(self.model_id)!r}, status={self.status!r})"
        )


WorkerStatus = Worker


@dataclass(frozen=True, slots=True)
class TokenInfo:
    id: str | None = None
    name: str | None = None
    status: str | None = None
    expiration: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    model_status: str | None = None
    worker_status: str | None = None
    evaluation_key_created_at: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "TokenInfo":
        return cls(
            id=_value(data, "tokenId", "id"),
            name=_value(data, "tokenName", "name"),
            status=_value(data, "tokenStatus", "status"),
            expiration=_value(data, "tokenExpiration", "expirationDate", "expiration"),
            model_id=_value(data, "modelId", "model_id"),
            model_name=_value(data, "modelName", "model_name"),
            model_status=_value(data, "modelStatus", "model_status"),
            worker_status=_value(data, "workerStatus", "worker_status"),
            evaluation_key_created_at=_value(
                data,
                "evalKeyCreatedAt",
                "evaluationKeyCreatedAt",
                "evaluation_key_created_at",
            ),
        )

    def __repr__(self) -> str:
        return (
            f"TokenInfo(name={self.name!r}, id={_short(self.id)!r}, "
            f"status={self.status!r}, model={self.model_name!r})"
        )


class InstanceType(str, Enum):
    """Available model deployment instance types."""

    G4DN_XLARGE = "G4DN_XLARGE"
    G5_2XLARGE = "G5_2XLARGE"
    G6E_2XLARGE = "G6E_2XLARGE"
    G7E_2XLARGE = "G7E_2XLARGE"
    G7E_12XLARGE = "G7E_12XLARGE"
    TPU_MEDIUM = "TPU_MEDIUM"
    CPU_C7G_XLARGE = "CPU_C7G_XLARGE"
    CPU_C7G_2XLARGE = "CPU_C7G_2XLARGE"


class SchemeType(str, Enum):
    LATTICA = "LATTICA"
    SUNSCREEN = "SUNSCREEN"
