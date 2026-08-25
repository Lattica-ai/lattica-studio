"""See `operators/composite/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.operators.composite.module_list import ModuleListHomOp


class SequentialHomOp(ModuleListHomOp):
    """Composite op that applies children in order to a single input value.

    Args:
        *ops: Ordered child operators applied to the same running value.
    """

    def __init__(self, *ops: HomOp):
        super().__init__(ops)

    def forward(self, x):
        """Run all child operators sequentially."""
        for op in self:
            x = op(x)
        return x
