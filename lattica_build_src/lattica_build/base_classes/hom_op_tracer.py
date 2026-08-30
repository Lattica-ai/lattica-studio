"""See `base_classes/README.md` for usage details."""

import copy
from typing import Any
import numpy as np
import torch
import contextvars
import inspect

from lattica_build.base_classes.hom_op import ClientOp

_current_tracer = contextvars.ContextVar("current_hom_tracer", default=None)

class Tracer:
    def __init__(self, tensors, **kwargs):
        self.recorded_ops = []
        self.tensors = tensors
        self.hom_params = kwargs.get("hom_params", None)
        self.client_mode = kwargs.get("client_mode", False)
        self.primary_input_name = kwargs.get("primary_input_name", None)
        self.custom_input_values = kwargs.get("custom_input_values", {})

    @classmethod
    def current(cls):
        tracer = _current_tracer.get()
        if tracer is None:
            raise RuntimeError("HomOp called outside tracing context")
        return tracer

    def __enter__(self):
        self._token = _current_tracer.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _current_tracer.reset(self._token)

    def call(self, op, *inputs):
        """Add an HomOp invocation to the current graph."""
        op_ir, inferred_output = self.serialize_op(op, *inputs)
        output = self._allocate_value(inferred_output)
        op_ir["inputs"] = [value.id for value in inputs]
        op_ir["output"] = self.serialize_hom_val(output)
        self.recorded_ops.append(op_ir)
        return output

    def _allocate_value(self, inferred_output):
        return inferred_output.make_copy(id=f"v{len(self.recorded_ops)}", custom_input_ref=None)

    def serialize_op(self, op, *inputs):
        if op.is_leaf_op():
            return self._serialize_leaf_op(op, *inputs)
        return self._serialize_nonleaf_op(op, *inputs)

    def _serialize_leaf_op(self, op, *inputs):
        if not self.client_mode and isinstance(op, ClientOp):
            raise ValueError(f"Op {op} is client-only and cannot be used for homomorphic compute.")

        inputs = list(inputs)

        for i, input in enumerate(inputs):
            if not input.is_custom:
                continue

            source_input = self.custom_input_values.get(input.custom_input_ref)
            if source_input is not None:
                # Reuse the state established by the custom input's first occurrence,
                # while preserving this occurrence's ID in the serialized graph.
                inputs[i] = source_input.make_copy(id=input.id)
                continue

            if len(inputs) > 1:
                # find another input of this op and make the initial state of the custom input identical.
                # If there's no other input, initialize on the maximal state.
                for other_input in inputs:
                    if other_input.id != input.id:
                        input = input.make_copy(
                            active_rows=copy.deepcopy(other_input.active_rows),
                            active_cols=copy.deepcopy(other_input.active_cols),
                        )
                        break

            inputs[i] = input
            self.custom_input_values[input.custom_input_ref] = input.make_copy()

        inferred_output = op.infer_output_shape(*inputs, internal_n=self.hom_params.internal_n)
        if (not self.client_mode):
            inferred_output = op.infer_output_level_and_scale(inferred_output, *inputs[1:], hom_params=self.hom_params)

        op_ir = {
            "op": int(op.OP_TYPE),
            "op_attrs": _get_hom_op_attributes(op),
            "custom_inputs": {
                val.id: val.custom_input_ref for val in inputs if val.is_custom
            }
        }

        data = getattr(op, "data", None)
        if data is not None:
            refs = []
            data_tensors = (data,) if torch.is_tensor(data) else data
            for tensor in data_tensors:
                ref = str(len(self.tensors))
                self.tensors[ref] = tensor.contiguous()
                refs.append(ref)
            op_ir["data_ref"] = refs

        return op_ir, inferred_output

    def _serialize_nonleaf_op(self, op, *inputs):
        signature = inspect.signature(op.forward)
        body_input_names = list(signature.parameters.keys())
        body_input_values = self._bind_inputs_to_names(body_input_names, *inputs)

        with Tracer(
                tensors=self.tensors,
                hom_params=self.hom_params,
                client_mode=self.client_mode,
                primary_input_name=self.primary_input_name,
                custom_input_values=self.custom_input_values,
        ) as body_tracer:
            body_output = op.forward(*body_input_values)
        body_tracer.finalize()

        op_ir = {
            "op": None,
            "custom_inputs": {
                val.id: val.custom_input_ref for val in body_input_values if val.is_custom
            },
            "body_inputs": body_input_names,
            "body_output": self.serialize_hom_val(body_output),
            "child_ops": body_tracer.recorded_ops,
        }

        return op_ir, body_output

    @staticmethod
    def serialize_hom_val(hom_val):
        return {
            "id": hom_val.id,
            "tensor_shape": hom_val.tensor_shape,
            "n_axis": hom_val.n_axis,
            "active_rows": hom_val.active_rows if hom_val.active_rows is not None else None,
            "active_cols": hom_val.active_cols if hom_val.active_cols is not None else None,
            "pt_scale": f"{hom_val.pt_scale:.10f}",
        }

    @staticmethod
    def _bind_inputs_to_names(input_names, *inputs):
        if len(input_names) != len(inputs):
            raise ValueError(f"Expected {len(input_names)} inputs, but received {len(inputs)}.")

        return [
            input_value.make_copy(id=input_name)
            for input_name, input_value in zip(input_names, inputs)
        ]

    def finalize(self):
        """Run after tracing children to detect which inputs are last uses of their values."""
        seen = set()

        for recorded_op in reversed(self.recorded_ops):
            input_last_uses = []

            # Reverse inputs order as well, because e.g. mul(x, x) contains
            # two ordered uses of the same value.
            for value_id in reversed(recorded_op["inputs"]):
                is_last_use = value_id not in seen
                input_last_uses.append(is_last_use)
                seen.add(value_id)

            recorded_op["input_last_uses"] = list(reversed(input_last_uses))


def _to_json_serializable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, torch.Tensor):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _to_json_serializable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_json_serializable(item) for item in v]
    return v

def _get_hom_op_attributes(op):
    attrs = {}
    for name, value in op.__dict__.items():
        if not (
                name.startswith("_") or
                name == 'data'       or
                value is None        or
                callable(value)
        ):
            attrs[name] = _to_json_serializable(value)
    return attrs
