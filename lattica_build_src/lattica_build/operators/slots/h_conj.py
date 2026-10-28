"""See `operators/slots/README.md` for usage details."""

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.serialization.hom_op_pb2 import HomOpType


class HomConj(HomOp):
    """Complex conjugation of every slot."""

    OP_TYPE = HomOpType.Conj

    def forward_clear(self, input):
        return input.conj()
