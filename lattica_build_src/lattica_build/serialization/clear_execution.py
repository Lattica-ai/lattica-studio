"""Execute serialized build graphs with borrowed, externally owned weights.

The graph contains JSON metadata only. A resolver supplies leaf data for the
current call; neither the graph nor the operator registry retains weight tensors.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from functools import lru_cache
import inspect
from typing import Any

import torch

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.operators.arithmetic.h_add import HomAdd
from lattica_build.operators.arithmetic.h_axis_sum import HomAxisSum
from lattica_build.operators.arithmetic.h_const_add import HomConstAdd
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.arithmetic.h_mul import HomMul
from lattica_build.operators.client_ops import Clamp, Softmax
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.fhe.h_mod_switch import HomModSwitch
from lattica_build.operators.fhe.h_ring_switch import HomRingSwitch
from lattica_build.operators.ml.h_conv import HomConv
from lattica_build.operators.ml.h_mat_mul import HomMatMul
from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.operators.shape.h_squeeze import HomSqueeze
from lattica_build.operators.shape.h_unsqueeze import HomUnsqueeze
from lattica_build.operators.slots.h_expand import HomExpand
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.operators.slots.h_running_sum import HomRunningSum
from lattica_build.operators.slots.h_sum_slots import HomSumSlots

_BUILD_LEAVES = {cls.OP_TYPE: cls for cls in (
    HomAdd, HomAxisSum, HomConstAdd, HomConstMul, HomMul, Clamp, Softmax,
    Bootstrap, HomModSwitch, HomRingSwitch, HomConv, HomMatMul, HomPolyEvalBase,
    HomReshape, HomSlice, HomSqueeze, HomUnsqueeze, HomExpand, HomRotateSum,
    HomRunningSum, HomSumSlots,
)}


@lru_cache(maxsize=None)
def _constructor(op_type: int) -> tuple[type[HomOp], tuple[str, ...]]:
    cls = _BUILD_LEAVES[op_type]
    names = tuple(name for name, parameter in inspect.signature(cls).parameters.items()
                  if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY))
    return cls, names


def run_clear_section(graph: dict[str, Any], inputs: Sequence[torch.Tensor],
                      resolve_data: Callable[[tuple[int, ...]], Any],
                      path: tuple[int, ...] = ()) -> torch.Tensor:
    """Evaluate a traced section with build operators, borrowing leaf weights.

    ``path`` consists of child indices, allowing callers to resolve the same leaf
    in a backend graph without storing a second tensor table. Composite scopes
    are independent, and intermediates are released after their final use.
    """
    if graph['op'] is not None:
        cls, names = _constructor(graph['op'])
        attrs = graph['op_attrs']
        kwargs = {name: attrs[name] for name in names if name in attrs}
        if cls is HomSlice:
            kwargs['key'] = (attrs['start'] if attrs['drop_dim'] else
                             slice(attrs['start'], attrs.get('end'), attrs['step']))
        op = cls(**kwargs)
        if graph.get('data_ref') is not None:
            op.data = resolve_data(path)
        return op.forward_clear(*inputs)

    env = dict(zip(graph['body_inputs'], inputs, strict=True))
    output_id = graph['body_output']['id']
    uses = Counter(name for child in graph['child_ops'] for name in child['inputs'])
    uses[output_id] += 1
    for index, child in enumerate(graph['child_ops']):
        values = [env[name] for name in child['inputs']]
        env[child['output']['id']] = run_clear_section(child, values, resolve_data, (*path, index))
        del values
        for name in child['inputs']:
            uses[name] -= 1
            if uses[name] == 0:
                del env[name]
    return env[output_id]
