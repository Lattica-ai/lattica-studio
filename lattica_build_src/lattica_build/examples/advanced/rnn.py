"""See `examples/README.md` for usage details."""
from __future__ import annotations
import random
from functools import cache
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import torch
from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.ml.h_linear import HomLinear
from lattica_build.operators.ml.h_mat_mul import HomMatMul
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.operators.shape.h_unsqueeze import HomUnsqueeze
from lattica_build.params.params import HomParams

INPUT_SIZE = 128
HIDDEN_SIZE = 128
N_CLASSES = 2
SEQ_LEN = 128
TANH_DEGREE = 3
LEVELS_PER_STEP = 3  # levels one timestep consumes: hh_matrix 1, tanh (degree 3) 2
WEIGHTS_DIR = Path(__file__).with_name('data') / 'rnn'
EMBEDDING_URL = 'https://lattica-public.s3.us-east-1.amazonaws.com/models_data/RNN/embedding_batch.npy'


@cache
def _load_weights() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the pre-trained RNN and classifier weights from the packaged checkpoints."""
    return tuple(
        torch.load(WEIGHTS_DIR / f'trained_{name}.pt', weights_only=True, map_location='cpu')
        for name in ('rnn_ih', 'rnn_hh', 'fc_weight', 'fc_bias')
    )


@cache
def _load_embedding() -> np.ndarray:
    """Download the batch of embedded example sequences once per process."""
    with urlopen(EMBEDDING_URL) as response:
        return np.load(BytesIO(response.read()))


class _RNN(HomOp):

    def __init__(self, seq_len: int, batch_ih_matmul: bool, boot_every: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.batch_ih_matmul = batch_ih_matmul
        self.boot_every = boot_every
        self.unsqueeze = HomUnsqueeze(1)
        # batched: contract (seq_len, 1, INPUT_SIZE) on axis 2; per timestep: contract (INPUT_SIZE,) on axis 1
        self.ih_matrix = HomMatMul((HIDDEN_SIZE, INPUT_SIZE), mul_axis=2 if batch_ih_matmul else 1)
        self.hh_matrix = HomMatMul((HIDDEN_SIZE, HIDDEN_SIZE), mul_axis=1)
        self.tanh = HomPolyEval(lambda x: np.tanh(2 * x), degree=TANH_DEGREE)  # fit on the rescaled variable x -> x/2 because range is [-2, 2]
        self.fc = HomLinear(dims=(N_CLASSES, HIDDEN_SIZE), bias=True, mul_axis=-1)
        self.boot = Bootstrap()

    def forward(self, x: HomValue) -> HomValue:
        if self.batch_ih_matmul:
            # (seq_len, 1, INPUT_SIZE)
            x = self.unsqueeze(x)

            # (seq_len, HIDDEN_SIZE)
            x = self.ih_matrix(x)

        for i in range(self.seq_len):
            # (HIDDEN_SIZE,)
            x_i = x[i] if self.batch_ih_matmul else self.ih_matrix(x[i])

            if i == 0:
                hidden = x_i
            else:
                hidden = self.hh_matrix(hidden)
                hidden = hidden + x_i

            hidden = self.tanh(hidden)

            if (i + 1) % self.boot_every == 0:
                hidden = self.boot(hidden)

        return self.fc(hidden)


class Pipeline(PipelineWrapper):

    def __init__(self, seq_len: int = SEQ_LEN, batch_ih_matmul: bool = False) -> None:
        # batch_ih_matmul: apply ih_matrix to all timesteps in one matmul instead of once per timestep.
        # The circuit runs faster, but that matmul's peak GPU memory grows linearly with seq_len * n,
        # so enable it only for short sequences that fit in memory:
        self.seq_len = seq_len
        self.batch_ih_matmul = batch_ih_matmul

    def build_pipeline(self) -> HomomorphicPipeline:
        """Build a single-layer RNN with a polynomial tanh, followed by a linear classifier."""
        w_ih, w_hh, fc_weight, fc_bias = _load_weights()
        # A bootstrap restores one level per prime of the q-list rows; refresh once they run out.
        levels = sum(len(row) for row in self.build_params().full_q_list_precision)
        pipeline = HomomorphicPipeline(
            hom=_RNN(self.seq_len, self.batch_ih_matmul, boot_every=levels // LEVELS_PER_STEP),
            input_shape=(self.seq_len, INPUT_SIZE),
            n_axis=1,
        )

        # swallow the rescaling before polynomial evaluation into the matrices to save a level
        pipeline.set_data("ih_matrix", w_ih / 2)
        pipeline.set_data("hh_matrix", w_hh / 2)
        pipeline.set_data("fc", fc_weight, fc_bias)
        return pipeline

    def build_params(self) -> HomParams:
        return HomParams(
            full_q_list_precision=6 * ((60, 30,),),
            n=2 ** 16,
            pt_scale=2 ** 30,
            sk_hw=192,
            num_special_primes=7,
            n_slots=INPUT_SIZE,
        )

    def _set_example_pt(self) -> torch.Tensor:
        """Pick a random sequence from the downloaded embedding batch."""
        embedding = _load_embedding()
        idx = random.randrange(embedding.shape[0])
        # TODO: currently we don't support batched computation
        return torch.tensor(embedding[idx, :self.seq_len, :])

    def compute_expected(self, example_pt: torch.Tensor) -> torch.Tensor:
        """Run the plaintext RNN, with a cubic tanh approximation instead of the fitted one."""
        w_ih, w_hh, fc_weight, fc_bias = _load_weights()
        hidden = torch.zeros(HIDDEN_SIZE)

        for t in range(self.seq_len):
            hidden = w_hh @ hidden + w_ih @ example_pt[t]

            # approximation taken from https://github.com/FHE-Applications/FHE-Applications/blob/master/dev/CKKS-App/LSTM/fhe_inference.cpp
            hidden = -0.10484599 * hidden ** 3 + 0.86501289 * hidden

        return fc_weight @ hidden + fc_bias

    def verify_results(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        # `expected` is the runtime's clear execution of this pipeline; check against the
        # independent reference instead, and only its binary classification rule: logit signs.
        reference = self.compute_expected(self.exmpl_pt)
        assert torch.equal(torch.sign(reference), torch.sign(actual)), (
            f"logit signs differ: reference {reference}, actual {actual}"
        )
