"""Execute serialized build graphs with borrowed, externally owned weights.

The graph contains JSON metadata only. A resolver supplies leaf data for the
current call; neither the graph nor the operator registry retains weight tensors.
Each build operator owns its plaintext computation, including any flattened
layouts required by the homomorphic graph. This executor only dispatches leaves
and resolves their borrowed weights.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
import inspect
from typing import Any

import torch

from lattica_build.base_classes.hom_op import HomOp, clear_options
from lattica_build.base_classes.hom_pipeline import PipeSec
from lattica_build.base_classes.hom_value import HomValue
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
from lattica_build.operators.ml.h_mux_conv import HomMuxConv
from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.shape.h_batch_permute import HomBatchPermute
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.operators.shape.h_squeeze import HomSqueeze
from lattica_build.operators.shape.h_unsqueeze import HomUnsqueeze
from lattica_build.operators.slots.h_expand import HomExpand
from lattica_build.operators.slots.h_conj import HomConj
from lattica_build.operators.slots.h_real_imag import HomCombineRealImag, HomSplitRealImag
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.operators.slots.h_rotate_sum_elementwise import HomRotateSumElementwise
from lattica_build.operators.slots.h_running_sum import HomRunningSum
from lattica_build.operators.slots.h_sum_slots import HomSumSlots
from lattica_build.operators.slots.ring_utils import SlotOp

_BUILD_LEAVES = {cls.OP_TYPE: cls for cls in (
    HomAdd, HomAxisSum, HomConstAdd, HomConstMul, HomMul, Clamp, Softmax,
    Bootstrap, HomModSwitch, HomRingSwitch, HomConv, HomMatMul, HomPolyEvalBase,
    HomReshape, HomSlice, HomSqueeze, HomUnsqueeze, HomExpand, HomRotateSum,
    HomRunningSum, HomSumSlots, HomConj, HomMuxConv, HomRotateSumElementwise,
    HomCombineRealImag, HomSplitRealImag, HomBatchPermute,
)}


@lru_cache(maxsize=None)
def _constructor(op_type: int) -> tuple[type[HomOp], tuple[str, ...]]:
    cls = _BUILD_LEAVES[op_type]
    names = tuple(name for name, parameter in inspect.signature(cls).parameters.items()
                  if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY))
    return cls, names


def run_clear_section(graph: dict[str, Any], inputs: Sequence[torch.Tensor],
                      resolve_data: Callable[[tuple[int, ...]], Any],
                      path: tuple[int, ...] = (),
                      input_values: Sequence[dict[str, Any]] | None = None,
                      batched: bool = False) -> torch.Tensor:
    """Evaluate a traced section with build operators, borrowing leaf weights.

    ``path`` consists of child indices, allowing callers to resolve the same leaf
    in a backend graph without storing a second tensor table. Composite scopes
    are independent, and intermediates are released after their final use.

    ``input_values`` are the traced values of ``inputs`` (serialized HomValues), by
    default the section root's. Each slot operator gets the traced value of its first input and
    its ring size (``op.input_value``, ``op.internal_n``). The result of every section, the root
    and each child, is checked against its traced shape.

    With ``batched``, the values of a batch (their traced ``batch_size`` set) hold its plaintexts on their first axis.
    Each operator runs on each plaintext, with the inputs outside the batch as they are, except `HomBatchPermute`,
    which takes the whole batch.
    """
    if input_values is None:
        input_values = graph.get('input_values')
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
        if input_values and isinstance(op, SlotOp):
            traced = input_values[0]
            op.input_value = HomValue(tensor_shape=tuple(traced['tensor_shape']), n_axis=traced.get('n_axis'),
                                      n_slots=traced.get('n_slots'), n_batch=traced['n_batch'])
            op.internal_n = graph.get('internal_n')
        carried = [value['batch_size'] is not None for value in input_values] if batched else []
        # Let the operator know the ring size.
        with clear_options(internal_n=graph.get('internal_n')):
            if isinstance(op, HomBatchPermute) or not any(carried):
                result = op.forward_clear(*inputs)
            else:
                size = next(x.shape[0] for x, flag in zip(inputs, carried) if flag)
                result = torch.stack([op.forward_clear(*(x[b] if flag else x for x, flag in zip(inputs, carried)))
                                      for b in range(size)])
            return _check_shape(result, graph.get('output'), path, batched)

    env = dict(zip(graph['body_inputs'], inputs, strict=True))
    values_env = dict(zip(graph['body_inputs'], input_values, strict=True)) if input_values is not None else None
    output_id = graph['body_output']['id']
    uses = Counter(name for child in graph['child_ops'] for name in child['inputs'])
    uses[output_id] += 1
    for index, child in enumerate(graph['child_ops']):
        values = [env[name] for name in child['inputs']]
        traced = [values_env[name] for name in child['inputs']] if values_env is not None else None
        env[child['output']['id']] = run_clear_section(child, values, resolve_data, (*path, index), traced, batched)
        if values_env is not None:
            values_env[child['output']['id']] = child['output']
        del values, traced
        for name in child['inputs']:
            uses[name] -= 1
            if uses[name] == 0:
                del env[name]
                if values_env is not None:
                    del values_env[name]
    return _check_shape(env[output_id], graph['body_output'], path, batched)


def _check_shape(result: torch.Tensor, traced: Mapping[str, Any] | None, path: tuple[int, ...],
                 batched: bool) -> torch.Tensor:
    """The result of the section at ``path``, checked against the shape of its traced value ``traced`` (a
    serialized HomValue) when one is recorded. With ``batched``, a value of a batch has the batch's axis first."""
    if traced is None:
        return result
    shape = tuple(result.shape[1:] if batched and traced['batch_size'] is not None else result.shape)
    if shape != tuple(traced['tensor_shape']):
        raise ValueError(f'Clear execution of the section at {path} gives shape {shape}; '
                         f'the traced shape is {tuple(traced["tensor_shape"])}')
    return result


def run_clear_graph(graph: Mapping[str, Any], primary: torch.Tensor, custom: Mapping[str, torch.Tensor],
                    resolve_data: Callable[[PipeSec, tuple[int, ...]], Any], as_complex: bool) -> torch.Tensor:
    """Evaluate the traced sections of a pipeline (its serialized ``pipeline_sections``) on clear tensors.

    ``resolve_data(section, path)`` returns the data of the leaf at ``path`` (child indices, as in
    ``run_clear_section``) in the pipeline section ``section``.

    A pipeline built for a batch takes its plaintexts on the first axis of ``primary``. The client sections run on
    each plaintext, and the homomorphic section on the whole batch.
    """
    batched = graph[PipeSec.HOM]['input_values'][0]['batch_size'] is not None
    result = primary
    with torch.no_grad():
        for section in (PipeSec.CLIENT_PRE, PipeSec.HOM, PipeSec.CLIENT_POST):
            if section not in graph:
                continue
            data = lambda path, s=section: resolve_data(s, path)
            if section != PipeSec.HOM:
                result = (torch.stack([run_clear_section(graph[section], [x], data) for x in result]) if batched
                          else run_clear_section(graph[section], [result], data))
                continue
            names = graph[section]['inputs']
            shape = tuple(graph['input_shape_to_hom_section'])
            if tuple(result.shape[batched:]) != shape:
                raise ValueError(f"Clear input '{names[0]}' after preprocessing "
                                 f"has shape {tuple(result.shape[batched:])}; expected {shape}.")
            result = run_clear_section(graph[section], [result, *(custom[name] for name in names[1:])], data,
                                       batched=batched)
            if not as_complex:
                result = result.real
    return result
