"""See `operators/composite/README.md` for usage details."""

from collections.abc import Iterable, Iterator
from typing import overload

from lattica_build.base_classes.hom_op import HomOp


class ModuleListHomOp(HomOp):
    """Container of `HomOp` nodes used to build composite graph structures.

    Args:
        ops: Initial iterable of child operators.
    """

    def __init__(self, ops: Iterable[HomOp] = ()):
        super().__init__()
        self.ops = list(ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self) -> Iterator[HomOp]:
        return iter(self.ops)

    @overload
    def __getitem__(self, index: int) -> HomOp: ...

    @overload
    def __getitem__(self, index: slice) -> "ModuleListHomOp": ...

    def __getitem__(self, index: int | slice) -> "HomOp | ModuleListHomOp":
        if isinstance(index, slice):
            return ModuleListHomOp(self.ops[index])
        return self.ops[index]

    def __setitem__(self, index: int, op: HomOp) -> None:
        self.ops[index] = op

    def append(self, op: HomOp) -> "ModuleListHomOp":
        self.ops.append(op)
        return self

    def extend(self, ops: Iterable[HomOp]) -> "ModuleListHomOp":
        self.ops.extend(ops)
        return self

    def insert(self, index: int, op: HomOp) -> None:
        self.ops.insert(index, op)

    def forward(self, *args, **kwargs):
        """Container has no runtime computation; subclasses define execution."""
        raise NotImplementedError(
            "ModuleListHomOp is only a container and has no forward operation."
        )

    def _get_child(self, name: str) -> HomOp:
        """Resolve numeric child names used by recursive data-setting paths."""
        try:
            return self.ops[int(name)]
        except (ValueError, IndexError) as error:
            raise KeyError(
                f"Unknown op {name!r}. Available: {list(range(len(self.ops)))}"
            ) from error
