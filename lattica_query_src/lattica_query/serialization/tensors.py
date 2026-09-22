import math
import warnings

import torch
from google.protobuf.message import DecodeError

from ..errors import SerializationError
from .generated import generic_pb2


class TensorDecodingError(SerializationError):
    """A serialized tensor is malformed or unsupported."""


_PROTO_DTYPE_BY_TORCH = {
    torch.int32: generic_pb2.DataType.INT32,
    torch.int64: generic_pb2.DataType.INT64,
    torch.float32: generic_pb2.DataType.FLOAT,
    torch.float64: generic_pb2.DataType.DOUBLE,
    torch.complex64: generic_pb2.DataType.COMPLEX_FLOAT,
    torch.complex128: generic_pb2.DataType.COMPLEX_DOUBLE,
    torch.bool: generic_pb2.DataType.BOOL,
}
_TORCH_DTYPE_BY_PROTO = {value: key for key, value in _PROTO_DTYPE_BY_TORCH.items()}


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    try:
        proto_dtype = _PROTO_DTYPE_BY_TORCH[tensor.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}") from exc

    contiguous = tensor.detach().to(device="cpu").contiguous()
    message = generic_pb2.TensorHolder(dtype=proto_dtype, data=contiguous.numpy().tobytes())
    message.sizes.extend(contiguous.shape)
    message.strides.extend(contiguous.stride())
    return message.SerializeToString()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    if not isinstance(data, bytes) or not data:
        raise TensorDecodingError("serialized tensor must be non-empty bytes")
    message = generic_pb2.TensorHolder()
    try:
        message.ParseFromString(data)
    except DecodeError as exc:
        raise TensorDecodingError("Invalid serialized tensor") from exc

    try:
        dtype = _TORCH_DTYPE_BY_PROTO[message.dtype]
    except KeyError as exc:
        raise TensorDecodingError(f"Unsupported serialized tensor dtype: {message.dtype}") from exc

    shape = tuple(message.sizes)
    strides = tuple(message.strides)
    if any(size < 0 for size in shape):
        raise TensorDecodingError("Serialized tensor shape contains a negative dimension")
    if strides and len(strides) != len(shape):
        raise TensorDecodingError("Serialized tensor shape and strides have different ranks")

    element_size = torch.empty((), dtype=dtype).element_size()
    element_count = math.prod(shape)
    expected_bytes = element_count * element_size
    if len(message.data) != expected_bytes:
        raise TensorDecodingError(
            f"Serialized tensor contains {len(message.data)} bytes; expected {expected_bytes}"
        )
    if element_count == 0:
        return torch.empty(shape, dtype=dtype)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable",
            category=UserWarning,
        )
        return torch.frombuffer(message.data, dtype=dtype).reshape(shape).clone()
