from typing import TYPE_CHECKING

from ..logging import QUERY_THEME, OperationLog
from ..performance import parse_server_timing, summarize_timing
from .artifacts import QueryKey
from .executor import EncryptedQueryExecutor

if TYPE_CHECKING:
    import torch

    from .worker import WorkerGateway


class QueryAPI:
    """Run clear and encrypted queries.

    ``encrypted()`` runs one query end to end. ``encrypt()``, ``execute()`` and
    ``decrypt()`` are its three steps, for callers that run or time them apart.
    """

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

    def encrypt(self, plaintext: "torch.Tensor", *, key: QueryKey) -> bytes:
        """Preprocess and encrypt one query locally; ``execute()`` runs the ciphertext."""
        with OperationLog("encrypting query", theme=QUERY_THEME):
            return self._executor.encrypt(plaintext, key)

    def execute(self, ciphertext: bytes) -> bytes:
        """Run the model on a ciphertext from ``encrypt()`` and return the encrypted result."""
        if not isinstance(ciphertext, bytes) or not ciphertext:
            raise ValueError("ciphertext must be non-empty bytes")
        with OperationLog("running encrypted query on the worker", theme=QUERY_THEME):
            return self._worker.execute_encrypted(ciphertext)

    def decrypt(self, ciphertext: bytes, *, key: QueryKey) -> "torch.Tensor":
        """Decrypt and postprocess an encrypted result from ``execute()`` locally."""
        with OperationLog("decrypting result", theme=QUERY_THEME):
            return self._executor.decrypt(ciphertext, key)

    @property
    def last_server_timing(self) -> dict[str, float]:
        """The server's timing of the last worker call: seconds per step, empty if it reported none."""
        return parse_server_timing(self._worker.last_timing.server_header)
