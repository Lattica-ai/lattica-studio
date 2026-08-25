"""See `base_classes/README.md` for usage details."""

from dataclasses import dataclass
from typing import Any, Tuple, Union, Sequence, Optional

import torch
from decimal import Decimal
TensorShape = Union[Tuple[int, ...], torch.Size, Sequence[int]]
ConstValue  = Union[torch.Tensor, int]

# ===== TODO: use functions from pt_shape_utils.py after circular dependency is resolved
def to_pos_axis(axis: int, shape: TensorShape) -> int:
    if axis >= 0:
        return axis
    else:
        return axis + len(shape)

def to_neg_axis(axis: int, shape: TensorShape) -> int:
    if axis < 0:
        return axis
    return axis - len(shape)


def is_n_axis_broadcasted(pt_shape: TensorShape, tensor_shape: TensorShape, n_axis: int) -> bool:
    n_axis = to_neg_axis(n_axis, pt_shape)
    return (
        -n_axis > len(tensor_shape)
        or tensor_shape[n_axis] != pt_shape[n_axis]
    )
# ===================================================== #

@dataclass(kw_only=True)
class HomValue:
    id:              str | None = None
    # Homomorphic shape
    tensor_shape:  TensorShape
    n_axis: int | None = None
    # homomorphic levels consumption
    active_rows: torch.Tensor | None = None
    active_cols: torch.Tensor | None = None
    pt_scale: float | int | Decimal = 1
    # If this value is a custom input, this holds its name, otherwise it is None.
    custom_input_ref: str | None = None

    def make_copy(self, **kwargs) -> 'HomValue':
        """Create a copy of this HomValue with optional modifications."""
        new_attrs = {
            'id': self.id,
            'tensor_shape': self.tensor_shape,
            'n_axis': self.n_axis,
            'active_rows': self.active_rows,
            'active_cols': self.active_cols,
            'pt_scale': self.pt_scale,
            'custom_input_ref': self.custom_input_ref,
        }
        new_attrs.update(kwargs)
        return HomValue(**new_attrs)

    def __post_init__(self):
        self.tensor_shape = tuple(self.tensor_shape)
        if self.n_axis is not None:
            self.n_axis = to_pos_axis(self.n_axis, self.tensor_shape)

    @property
    def is_custom(self) -> bool:
        return self.custom_input_ref is not None

    def __add__(self, other: 'HomValue') -> 'HomValue':
        return _homvalue_add(self, other)

    def __radd__(self, other: Any) -> 'HomValue':
        return _homvalue_add(self, other)

    def __sub__(self, other: Any) -> 'HomValue':
        return _homvalue_sub(self, other)

    def __rsub__(self, other: Any) -> 'HomValue':
        return _homvalue_rsub(self, other)

    def __neg__(self) -> 'HomValue':
        return _homvalue_mul(self, -1)

    def __mul__(self, other: Any) -> 'HomValue':
        return _homvalue_mul(self, other)

    def __rmul__(self, other: Any) -> 'HomValue':
        return _homvalue_mul(self, other)

    def __pow__(self, exponent: int, modulo: Optional[int] = None) -> 'HomValue':
        if modulo is not None:
            raise TypeError("HomValue exponentiation does not support a modulo argument.")

        if exponent != 2:
            raise TypeError("HomValue exponentiation only supports exponent 2.")

        return _auto_square(self)

    def __rpow__(self, other: Any) -> 'HomValue':
        raise TypeError("HomValue exponentiation only supports HomValue ** 2.")

    def __truediv__(self, other: Any) -> 'HomValue':
        raise TypeError("HomValue division is not supported.")

    def __rtruediv__(self, other: Any) -> 'HomValue':
        raise TypeError("HomValue division is not supported.")

    def __getitem__(self, key) -> 'HomValue':
        return _homvalue_getitem(self, key)


def _as_arithmetic_constant_tensor(value: ConstValue) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as e:
        raise TypeError(
            "HomValue arithmetic only supports numeric constants excluding bool and complex."
        ) from e

    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError("HomValue arithmetic only supports numeric constants excluding bool and complex.")

    return tensor


def _has_fractional_values(tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point():
        return False

    return bool(torch.any(tensor != torch.round(tensor)).item())


def _negate_constant(value: ConstValue) -> torch.Tensor:
    tensor = _as_arithmetic_constant_tensor(value)
    if tensor.is_floating_point():
        return -tensor
    return -tensor.to(dtype=torch.int64)


def _auto_const_add(x: HomValue, constant: Any) -> HomValue:
    from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
    tensor = _as_arithmetic_constant_tensor(constant)
    op = HomConstAdd(dims=tuple(tensor.shape))
    op.set_data(tensor)
    return op(x)


def _auto_add(x: HomValue, y: HomValue, is_sub: bool = False) -> HomValue:
    from lattica_build.operators.arithmetic.h_add import HomAdd
    return HomAdd(is_sub=is_sub)(x, y)



def _auto_const_mul(x: HomValue, other: Any) -> HomValue:
    from lattica_build.operators.arithmetic.h_const_mul import HomConstMul

    tensor = _as_arithmetic_constant_tensor(other)
    const_shape = tuple(tensor.shape)

    n_axis_broadcasted = (
        x.n_axis is not None
        and is_n_axis_broadcasted(
            pt_shape=x.tensor_shape,
            tensor_shape=const_shape,
            n_axis=x.n_axis,
        )
    )

    if n_axis_broadcasted and not _has_fractional_values(tensor):
        op = HomConstMul(dims=const_shape, with_modswitch=False, pt_scale=1)
    else:
        op = HomConstMul(dims=const_shape)

    op.set_data(tensor)
    return op(x)


def _auto_mul(x: HomValue, y: HomValue) -> HomValue:
    from lattica_build.operators.arithmetic.h_mul import HomMul
    return HomMul()(x, y)


def _auto_square(x: HomValue) -> HomValue:
    from lattica_build.operators.arithmetic.h_mul import HomMul
    return HomMul()(x, x)


def _homvalue_add(x: HomValue, other: Any) -> HomValue:
    if isinstance(other, HomValue):
        return _auto_add(x, other)

    return _auto_const_add(x, other)


def _homvalue_sub(x: HomValue, other: Any) -> HomValue:
    if isinstance(other, HomValue):
        return _auto_add(x, other, is_sub=True)

    return _auto_const_add(x, _negate_constant(other))


def _homvalue_rsub(x: HomValue, other: Any) -> HomValue:
    return _auto_const_add(-x, other)


def _homvalue_mul(x: HomValue, other: Any) -> HomValue:
    if isinstance(other, HomValue):
        if x == other:
            return _auto_square(x)
        return _auto_mul(x, other)

    return _auto_const_mul(x, other)


def _homvalue_getitem(x: HomValue, key) -> HomValue:
    from lattica_build.operators.shape.h_slice import HomSlice
    from lattica_build.params.shape_tracing import resolve_single_axis_index
    dim, entry = resolve_single_axis_index(key, len(x.tensor_shape))
    return HomSlice(dim, entry)(x)
