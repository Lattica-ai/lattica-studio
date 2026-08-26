"""See `examples/README.md` for usage details."""

from __future__ import annotations

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.comparison.h_compare import HomCompare
from lattica_build.params.params import DecompositionType, HomParams


class BranchRejoinCompare(HomOp):
    """Compute compare(x**2, x - 0.2) to demonstrate branch/rejoin topology."""

    def __init__(self) -> None:
        super().__init__()
        self.compare = HomCompare(x_accuracy=11, y_accuracy=11)

    def forward(self, x: HomValue) -> HomValue:
        return self.compare(x**2, x - 0.2)


def build_pipeline() -> HomomorphicPipeline:
    """Construct a branching homomorphic pipeline."""
    return HomomorphicPipeline(hom=BranchRejoinCompare(), input_shape=(4,))


def build_params() -> HomParams:
    """Create a compact params set suitable for a local branching quickstart build."""
    return HomParams(
        n=2**16,
        full_q_list_precision=((60, 30),) * 10,
        pt_scale=2**30,
        decomposition_type=DecompositionType.HYBRID,
        num_special_primes=2,
    )
