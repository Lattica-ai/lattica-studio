"""See `base_classes/README.md` for usage details."""

import copy
import inspect
import json
import os
import zipfile
from pathlib import Path
import base64

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_value import HomValue, TensorShape
from torch import Tensor
from dataclasses import field
from typing import Optional, Dict, Tuple, Sequence, Union, BinaryIO

from lattica_build.params.level_and_scale_tracing import ModulusChain, init_active_rows_cols
from lattica_build.params.shape_tracing import resolve_n_axis
from lattica_build.serialization.hom_op_pb2 import HomOpType
from dataclasses import dataclass
import enum

from safetensors.torch import save as safetensors_save

from lattica_build.operators.composite.module_list import ModuleListHomOp
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.params.bootstrapping_params import BootstrappingParams
from lattica_build.params.params import HomParams, DecompositionType


class PipeSec(enum.StrEnum):
    CLIENT_PRE  = enum.auto()
    HOM         = enum.auto()
    CLIENT_POST = enum.auto()

_GRAPH_FILENAME   = "hom_pipeline.json"
_TENSORS_FILENAME = "hom_pipeline.safetensors"

_DEFAULT_VERIFICATION_ACCURACY = 1 / 2 ** 10

@dataclass(kw_only=True)
class HomomorphicPipeline:
    client_pre:  Optional[Sequence[HomOp]] = None
    hom:         HomOp
    client_post: Optional[Sequence[HomOp]] = None

    client_preprocessing_data: bytes = b""  # Additional raw data to be sent to the query client
    as_complex:                bool  = False
    input_shape: TensorShape | dict  = field(default_factory=tuple)

    # Optional per-input plaintext scales for additional encrypted inputs.
    # Keys must match names from input_shape when input_shape is a dict.
    custom_scales: Dict[str, int]    = field(default_factory=dict)
    # Optional per-input n_slots for additional encrypted inputs. The primary input
    # takes its n_slots from hom_params.n_slots.
    # Keys must match names from input_shape when input_shape is a dict.
    custom_n_slots: Dict[str, int]   = field(default_factory=dict)
    n_axis: Optional[int]  = None

    # Verification runs by default (the compiler derives an expected output from the clear
    # pipeline when none is supplied). Set True only to explicitly opt out of verification.
    skip_verification: bool = True # TODO: fix verification flow and set to default False
    verification_data: dict  = field(default_factory=dict)


    def __post_init__(self):
        if self.client_pre is not None:
            self.client_pre =SequentialHomOp(*self.client_pre)
        if self.client_post is not None:
            self.client_post =SequentialHomOp(*self.client_post)

        hom_input_names = list(inspect.signature(self.hom.forward).parameters)
        self.primary_input_name = hom_input_names[0]

        if isinstance(self.input_shape, dict):
            unknown_input_names = [name for name in self.input_shape if name not in hom_input_names]
            if unknown_input_names:
                raise ValueError(f"input_shape contains unknown inputs: {unknown_input_names}")

            missing_input_names = [name for name in hom_input_names if name not in self.input_shape]
            if missing_input_names:
                raise ValueError(f"input_shape is missing required inputs: {missing_input_names}")
        else:
            self.input_shape = {self.primary_input_name: self.input_shape}

        if self.primary_input_name in self.custom_scales:
            raise ValueError("custom_scales cannot override the primary pipeline input")

        unknown_scale_names = [name for name in self.custom_scales if name not in self.input_shape]
        if unknown_scale_names:
            raise ValueError(f"custom_scales contains unknown inputs: {unknown_scale_names}")

        if self.primary_input_name in self.custom_n_slots:
            raise ValueError(
                "custom_n_slots cannot override the primary pipeline input; "
                "set HomParams.n_slots instead")

        unknown_n_slots_names = [name for name in self.custom_n_slots if name not in self.input_shape]
        if unknown_n_slots_names:
            raise ValueError(f"custom_n_slots contains unknown inputs: {unknown_n_slots_names}")

    def _get_pipe_section(self, section: PipeSec) -> Optional[HomOp]:
        match section:
            case PipeSec.CLIENT_PRE:
                if self.client_pre is None:
                    raise ValueError("client_pre section is None")
                return self.client_pre
            case PipeSec.HOM:
                if self.hom is None:
                    raise ValueError("hom section is None")
                return self.hom
            case PipeSec.CLIENT_POST:
                if self.client_post is None:
                    raise ValueError("client_post section is None")
                return self.client_post

    def set_data(self, name: str | int | tuple[int, ...] | None, *data: Tensor, section: PipeSec = PipeSec.HOM) -> None:
        self._get_pipe_section(section).set_data(*data, name=name)

    def add_client_preprocessing_data(self, binary_data: bytes) -> None:
        self.client_preprocessing_data = binary_data

    def _serialize_pipeline_sections(self, tensors, hom_params):
        """
        Serialize the pipeline into its execution sections.

        For each pipeline section X, the method X.serialize(...) receives as input
        one or more HomValue objects in accordance with its X.forward(...) signature.
        It returns the serialized section (a json string) along with the output HomVal
        of this section.

        :param tensors: Tensor registry used to collect tensors referenced by the pipeline.
        :param hom_params: Homomorphic encryption parameters used during serialization (e.g. internal_n).
        """

        enc_params = hom_params.ring_switch_params or hom_params
        active_rows, active_cols = init_active_rows_cols(hom_params)
        hom_input_names = inspect.signature(self.hom.forward).parameters
        hom_inputs = [
            HomValue(
                id=name,
                tensor_shape=self.input_shape[name],
                n_axis=resolve_n_axis(
                    tensor_shape=self.input_shape[name],
                    n_axis=self.n_axis,
                    internal_n=(enc_params if name == self.primary_input_name else hom_params).internal_n,
                    n_slots=self.custom_n_slots.get(name, hom_params.n_slots),
                ),
                active_rows=copy.deepcopy(active_rows),
                active_cols=copy.deepcopy(active_cols),
                pt_scale=self.custom_scales.get(name, hom_params.pt_scale),
                custom_input_ref=None if name == self.primary_input_name else name,
                n_slots=self.custom_n_slots.get(name, hom_params.n_slots),
            ) for name in hom_input_names
        ]

        ser_sections = {}

        if self.client_pre is not None:
            first_input = hom_inputs[0]
            # Preprocessing is performed on the first (main) input only,
            # and without n_axis.
            first_input.n_axis = None
            ser_sections[PipeSec.CLIENT_PRE], first_input = self.client_pre.serialize(
                tensors, first_input, hom_params=enc_params, client_mode=True
            )
            # Resolve the n_axis to be used for the homomorphic
            # section based on the output of the client_pre section.
            first_input.n_axis = resolve_n_axis(
                    tensor_shape=first_input.tensor_shape,
                    n_axis=self.n_axis,
                    internal_n=enc_params.internal_n,
                    n_slots=first_input.n_slots
                )
            hom_inputs[0] = first_input

        # Save the tensor_shape produced by client_pre to determine the
        # shape of the input ciphertexts. This is sent to the query client
        # so that it can encrypt the input data correctly.
        ser_sections['input_shape_to_hom_section'] = hom_inputs[0].tensor_shape

        ser_sections[PipeSec.HOM], hom_out_val = self.hom.serialize(
            tensors, *hom_inputs, hom_params=hom_params, primary_input_name=self.primary_input_name
        )

        if self.client_post is not None:
            hom_out_val = copy.deepcopy(hom_out_val)
            hom_out_val.n_axis = None
            ser_sections[PipeSec.CLIENT_POST], _ = self.client_post.serialize(
                tensors, hom_out_val, hom_params=hom_params, client_mode=True
            )

        return ser_sections


    def _serialize_verification_data(self, tensors):
        res = {}
        if 'accuracy' in self.verification_data.keys():
            accuracy = self.verification_data['accuracy']
            if not isinstance(accuracy, (float, int)):
                raise ValueError("verification_data['accuracy'] must be a float or int")
        else:
            accuracy = _DEFAULT_VERIFICATION_ACCURACY
        res['accuracy'] = float(accuracy)
        for input_name in self.input_shape.keys():
            if input_name in self.verification_data.keys():
                input_tensor = self.verification_data[input_name]
                tensors[f'ver_input_{input_name}'] = input_tensor
        if 'expected_output' in self.verification_data.keys():
            output_tensor = self.verification_data['expected_output']
            tensors[f'ver_output'] = output_tensor
        return res

    def _serialize_modulus_chain(self, hom_params: HomParams) -> dict:
        """Serialize modulus-chain metadata into JSON-friendly lists."""
        mod_chain = getattr(hom_params, "mod_chain", None)
        if mod_chain is None:
            raise ValueError("hom_params.mod_chain is not initialized")
        return {
            "full_q_list_precision": [list(row) for row in mod_chain.full_q_list],
            "factors_per_row": [list(map(int, row)) for row in mod_chain.factors_per_row],
            "section_rows": {name: list(rows) for name, rows in mod_chain.section_rows.items()},
            "num_rows": int(mod_chain.num_rows),
            "max_num_cols": int(mod_chain.max_num_cols),
            "full_active_rows": mod_chain.full_active_rows.tolist(),
            "full_active_cols": mod_chain.full_active_cols.tolist(),
        }

    def serialize(self, hom_params) -> Tuple[bytes, bytes]:
        leaves = self.get_leaf_ops(sections=(PipeSec.HOM,))
        first_op = leaves[0] if leaves else None
        ring_switch_input = first_op is not None and first_op.OP_TYPE == HomOpType.RingSwitch
        if ring_switch_input:
            hom_params.ring_switch_params = HomParams(
                full_q_list_precision=((BootstrappingParams.get_q_base_precision(),),),
                n=2 ** first_op.log_n_subring,
                pt_scale=hom_params.pt_scale,
                decomposition_type=DecompositionType.BV,
            )
            hom_params.num_init_rows = 0
        else:
            hom_params.ring_switch_params = None

        needs_boot = ring_switch_input or any(op.OP_TYPE == HomOpType.Bootstrap for op in leaves)
        hom_params.boot_params = (
            BootstrappingParams(hom_params.bootstrapping_variant, hom_params.sk_hw)
            if needs_boot else None
        )
        hom_params.mod_chain = ModulusChain(hom_params)

        tensors = {}
        attributes_dict = {
            "as_complex":                    self.as_complex,
            "custom_scales":                 self.custom_scales,
            "custom_n_slots":                self.custom_n_slots,
            "primary_input_name":            self.primary_input_name,
            "input_shape":                   self.input_shape,
            "n_axis":                        self.n_axis,
            "client_preprocessing_data_b64": base64.b64encode(self.client_preprocessing_data).decode("ascii"),
            "skip_verification":             self.skip_verification,
            "verification_data":             self._serialize_verification_data(tensors),
            "modulus_chain":                 self._serialize_modulus_chain(hom_params),
            "pipeline_sections":             self._serialize_pipeline_sections(tensors, hom_params)
        }
        graph_bytes = json.dumps(attributes_dict).encode("utf-8")
        tensors_bytes = safetensors_save(tensors)

        return graph_bytes, tensors_bytes

    def save(self, path: Union[str, os.PathLike], hom_params):
        """Serialize pipeline into a zip file containing JSON graph + safetensors"""
        graph_bytes, tensor_bytes = self.serialize(hom_params)
        path = Path(path)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_GRAPH_FILENAME, graph_bytes)
            zf.writestr(_TENSORS_FILENAME, tensor_bytes)

    @classmethod
    def load(cls, source: Union[str, os.PathLike, BinaryIO]) -> 'HomomorphicPipeline':
        with zipfile.ZipFile(source, "r") as zf:
            graph_bytes = zf.read(_GRAPH_FILENAME)
            tensor_bytes = zf.read(_TENSORS_FILENAME)
        return graph_bytes, tensor_bytes

    def get_leaf_ops(
            self,
            sections: Sequence[PipeSec] = tuple(PipeSec),
    ) -> list[HomOp]:
        section_ops = {
            PipeSec.CLIENT_PRE:  self.client_pre,
            PipeSec.HOM:         self.hom,
            PipeSec.CLIENT_POST: self.client_post,
        }

        leaves: list[HomOp] = []
        seen: set[int] = set()

        def collect(op: Optional[HomOp]) -> None:
            if op is None or id(op) in seen:
                return
            seen.add(id(op))

            if op.is_leaf_op():
                leaves.append(op)
                return

            children = op.ops if isinstance(op, ModuleListHomOp) else (
                value for value in vars(op).values() if isinstance(value, HomOp)
            )
            for child in children:
                collect(child)

        for section in sections:
            collect(section_ops[section])
        return leaves
