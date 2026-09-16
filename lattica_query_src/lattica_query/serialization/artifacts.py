from dataclasses import dataclass
from typing import Self, TypeVar

from google.protobuf.message import DecodeError, Message

from ..errors import SerializationError
from .generated import hom_op_pb2

_MessageT = TypeVar("_MessageT", bound=Message)


class ArtifactDecodingError(SerializationError):
    """A serialized query artifact is malformed or incomplete."""


def _parse(message: _MessageT, data: bytes, artifact: str) -> _MessageT:
    if not isinstance(data, bytes) or not data:
        raise ArtifactDecodingError(f"{artifact} must be non-empty bytes")
    try:
        message.ParseFromString(data)
    except DecodeError as exc:
        raise ArtifactDecodingError(f"Invalid serialized {artifact}") from exc
    return message


def _require_bytes(value: bytes, field: str, artifact: str) -> bytes:
    if not value:
        raise ArtifactDecodingError(f"{artifact} is missing {field}")
    return value


@dataclass(frozen=True, slots=True)
class QueryInitialization:
    """Context and model artifacts returned during key initialization."""
    context: bytes
    client_model: bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        message = _parse(hom_op_pb2.ClientData(), data, "client data")
        return cls(
            context=_require_bytes(message.serialized_extended_context, "context", "client data"),
            client_model=_require_bytes(message.serialized_model, "model", "client data"),
        )


@dataclass(frozen=True, slots=True)
class ClientModel:
    preprocess_block: bytes
    postprocess_block: bytes
    preprocessing_data: bytes
    external_axis: int | None
    as_complex: bool

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        message = _parse(hom_op_pb2.ClientModel(), data, "client model")
        return cls(
            preprocess_block=message.preprocess_block.SerializeToString(),
            postprocess_block=message.postprocess_block.SerializeToString(),
            preprocessing_data=message.preprocessing_data,
            external_axis=message.pt_axis_external if message.HasField("pt_axis_external") else None,
            as_complex=message.as_complex,
        )


@dataclass(frozen=True, slots=True)
class SecretKey:
    crt_basis: bytes
    coefs_basis: bytes

    def __post_init__(self) -> None:
        if not self.crt_basis or not self.coefs_basis:
            raise ArtifactDecodingError("secret key is missing a CRT or coefficients basis")

    @classmethod
    def _from_proto(cls, message: hom_op_pb2.SecretKey) -> Self:
        return cls(
            crt_basis=message.sk.SerializeToString(),
            coefs_basis=message.sk_coefs.SerializeToString(),
        )


@dataclass(frozen=True, slots=True)
class ExtendedContext:
    context: bytes
    ring_switch_context: bytes | None

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        message = _parse(hom_op_pb2.ExtendedContext(), data, "extended context")
        return cls(
            context=_require_bytes(message.serialized_context, "context", "extended context"),
            ring_switch_context=(
                _require_bytes(
                    message.serialized_ring_switch_context,
                    "ring switch context",
                    "extended context",
                )
                if message.HasField("serialized_ring_switch_context")
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ExtendedSecretKey:
    secret_key: SecretKey
    ring_switch_secret_key: SecretKey | None

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        message = _parse(hom_op_pb2.ExtendedSecretKey(), data, "extended secret key")
        if not message.HasField("secret_key"):
            raise ArtifactDecodingError("extended secret key is missing secret_key")
        return cls(
            secret_key=SecretKey._from_proto(message.secret_key),
            ring_switch_secret_key=(
                SecretKey._from_proto(message.ring_switch_secret_key)
                if message.HasField("ring_switch_secret_key")
                else None
            ),
        )
