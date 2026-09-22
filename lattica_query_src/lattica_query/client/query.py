from typing import TYPE_CHECKING

from ..logging import QUERY_THEME, OperationLog
from ..performance import parse_server_timing, summarize_timing
from .artifacts import QueryKey
from .executor import EncryptedQueryExecutor

if TYPE_CHECKING:
    import torch

    from .worker import WorkerGateway


class QueryAPI:
    """Run clear and encrypted queries."""

    def __init__(self, worker: "WorkerGateway") -> None:
        self._worker = worker
        self._executor = EncryptedQueryExecutor(worker)

    def clear(self, plaintext: "torch.Tensor") -> "torch.Tensor":
        with OperationLog("running non-encrypted query", theme=QUERY_THEME):
            return self._worker.execute_clear(plaintext)

    def encrypted(
        self,
        plaintext: "torch.Tensor",
        *,
        key: QueryKey,
        show_timing: bool = False,
    ) -> "torch.Tensor":
        """Encrypt, execute, decrypt, and decode one query."""
        with OperationLog("running encrypted query", theme=QUERY_THEME):
            execution = self._executor.execute(plaintext, key)
            if show_timing:
                summarize_timing(execution.timings, "CLIENT")
                server_timing = self._worker.last_timing.server_header
                if server_timing:
                    summarize_timing(parse_server_timing(server_timing), "SERVER")
            return execution.value
