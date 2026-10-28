"""Tensor ownership for calls into the in-process FHE core."""

import struct
from itertools import count

import torch

IN_PROCESS_TENSOR_REF_MARKER = b"LATTICA_TENSOR_REF:"
_store_ids = count(1)


class InProcessTensorStore:
    def __init__(self) -> None:
        self.groups: dict[int, list[torch.Tensor]] = {}
        self._write_group: int | None = None

    def create_group(self) -> int:
        store_id = next(_store_ids)
        self.groups[store_id] = []
        return store_id

    def set_group(self, store_id: int, tensors: list[torch.Tensor]) -> None:
        existing = self.groups.get(store_id)
        if existing is None or existing:
            raise ValueError("Conflicting in-process tensor stores")
        existing.extend(tensors)

    def add(self, tensor: torch.Tensor) -> bytes:
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("In-process tensor transport requires strided CPU tensors")
        if self._write_group is None:
            self._write_group = self.create_group()
        tensors = self.groups[self._write_group]
        reference = IN_PROCESS_TENSOR_REF_MARKER + struct.pack(
            "<QQ", self._write_group, len(tensors)
        )
        tensors.append(tensor)
        return reference

    def merge(self, other: "InProcessTensorStore") -> None:
        for store_id, tensors in other.groups.items():
            existing = self.groups.get(store_id)
            if existing is not None and existing is not tensors:
                raise ValueError("Conflicting in-process tensor stores")
            self.groups[store_id] = tensors

    def resolve(
        self,
        reference: bytes,
        dtype: torch.dtype,
        sizes: tuple[int, ...],
        strides: tuple[int, ...],
    ) -> torch.Tensor | None:
        if not reference.startswith(IN_PROCESS_TENSOR_REF_MARKER):
            return None
        if len(reference) != len(IN_PROCESS_TENSOR_REF_MARKER) + 16:
            raise ValueError("Invalid in-process tensor reference")
        store_id, index = struct.unpack(
            "<QQ", reference[len(IN_PROCESS_TENSOR_REF_MARKER) :]
        )
        tensors = self.groups.get(store_id)
        if tensors is None or index >= len(tensors):
            raise ValueError("In-process tensor reference is out of range")
        tensor = tensors[index]
        if (
            tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.dtype != dtype
            or tuple(tensor.shape) != sizes
            or tensor.stride() != strides
        ):
            raise ValueError("In-process tensor metadata does not match its tensor")
        return tensor


class InProcessTransportedBytes(bytes):
    def __new__(
        cls,
        value: bytes,
        tensor_store: InProcessTensorStore,
        materialize_outputs: bool = False,
    ):
        result = super().__new__(cls, value)
        result.tensor_store = tensor_store
        result.materialize_outputs = materialize_outputs
        return result


def activate_in_process_transport(value: bytes) -> InProcessTransportedBytes:
    return InProcessTransportedBytes(value, InProcessTensorStore())


def materialize_native_outputs(value: InProcessTransportedBytes) -> InProcessTransportedBytes:
    return InProcessTransportedBytes(value, value.tensor_store, materialize_outputs=True)
