"""See `base_classes/README.md` for usage details."""

import torch
from torch import Tensor
from typing import ClassVar
from lattica_build.serialization.hom_op_pb2 import HomOpType


class HomOp:
    OP_TYPE: ClassVar[HomOpType] = None
    data:    torch.Tensor | tuple[torch.Tensor, ...] | None = None

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        from lattica_build.base_classes.hom_op_tracer import Tracer

        return Tracer.current().call(self, *inputs)

    def is_leaf_op(self) -> bool:
        return self.OP_TYPE is not None

    def serialize(self, tensors, *inputs, **kwargs):
        from lattica_build.base_classes.hom_op_tracer import Tracer

        with Tracer(tensors, **kwargs) as tracer:
            serialized_op, out_val = tracer.serialize_op(self, *inputs)
        tracer.finalize()
        return serialized_op, out_val

    def infer_output_shape(self, *inputs, n_slots=None):
        if not self.is_leaf_op():
            raise RuntimeError("infer_output_shape should only be called directly on leaf ops")
        return inputs[0]

    def infer_output_level_and_scale(self, *inputs, hom_params=None):
        if not self.is_leaf_op():
            raise RuntimeError("infer_output_level_and_scale should only be called directly on leaf ops")
        return inputs[0]


    # =============== methods related to set_data ================== #

    def _set_data_here(self, *data: "Tensor") -> None:
        """Set the data for this op."""
        if not self.is_leaf_op():
            raise RuntimeError("data should only be set directly on base ops")

        dtype = torch.float64
        converted_data = tuple(
            item.to(dtype=dtype)
            if torch.is_tensor(item)
            else torch.tensor(item, dtype=dtype)
            for item in data
        )
        if len(converted_data) == 1:
            self.data = converted_data[0]
            if hasattr(self, "dims"):
                self.data = self.data.reshape(self.dims)
        else:
            self.data = converted_data

    def _get_child(self, name: str) -> "HomOp":
        """Get the child op with the given name."""
        try:
            child = getattr(self, name)
        except AttributeError as error:
            raise KeyError(
                f"Unknown child op {name!r}."
                f"Available: {[n for n, v in self.__dict__.items() if isinstance(v, HomOp)]}"
        ) from error
        if not isinstance(child, HomOp):
            raise KeyError(f"{name!r} is not a HomOp.")
        return child

    def set_data(self, *data,  name: str | int | tuple[int, ...] | None = None) -> None:
        """Parse a path and set data on the corresponding op."""

        if name is None or (self.is_leaf_op() and name in (0, "0")):
            self._set_data_here(*data)
            return

        if isinstance(name, int):
            name = str(name)

        elif isinstance(name, tuple):
            if not all(isinstance(index, int) for index in name):
                raise TypeError("set_data tuple path must contain only ints.")

            name = ".".join(map(str, name))

        elif not isinstance(name, str):
            raise TypeError(
                "set_data accepts a string, int, tuple[int, ...], or None."
            )

        child_name, separator, remaining_path = name.partition(".")

        child = self._get_child(child_name)
        child.set_data(*data, name=remaining_path if separator else None)

    # =============== enf of methods related to set_data ================== #


class ClientOp(HomOp):
    """Marker base for operations that may only run on clear client data."""
