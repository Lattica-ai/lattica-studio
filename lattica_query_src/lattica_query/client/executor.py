import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import lattica_fhe_core as fhe_core

from ..logging import log_status
from ..serialization.artifacts import ClientModel, ExtendedContext, ExtendedSecretKey
from ..serialization.tensors import deserialize_tensor, serialize_tensor
from .artifacts import QueryKey

if TYPE_CHECKING:
    from .worker import WorkerGateway


@dataclass(frozen=True, slots=True)
class EncryptedExecution:
    value: Any
    timings: dict[str, float]


class EncryptedQueryExecutor:
    """Execute the client-side portions of one encrypted query."""

    def __init__(self, worker: "WorkerGateway") -> None:
        self._worker = worker

    def execute(self, plaintext: Any, key: QueryKey) -> EncryptedExecution:
        timings: dict[str, float] = {}

        log_status("preparing client-side data")
        started = time.perf_counter()
        serialized_plaintext = serialize_tensor(plaintext)
        model = ClientModel.from_bytes(key.client_model)
        context = ExtendedContext.from_bytes(key.context)
        secret_keys = ExtendedSecretKey.from_bytes(key.secret_key)
        input_context, input_secret_key = self._select_input_key(context, secret_keys)
        timings["serialization"] = time.perf_counter() - started

        started = time.perf_counter()
        serialized_plaintext = fhe_core.apply_client_block(
            model.preprocess_block,
            input_context,
            serialized_plaintext,
        )
        timings["preprocessing"] = time.perf_counter() - started

        log_status("encrypting data for transmission")
        started = time.perf_counter()
        ciphertext = fhe_core.enc(
            input_context,
            (input_secret_key.crt_basis, input_secret_key.coefs_basis),
            serialized_plaintext,
            True,
            model.external_axis,
            None,
            None,
        )
        timings["encryption"] = time.perf_counter() - started

        log_status("sending encrypted data to worker for processing")
        started = time.perf_counter()
        encrypted_result = self._worker.execute_encrypted(ciphertext)
        timings["worker"] = time.perf_counter() - started

        log_status("decrypting result from worker")
        started = time.perf_counter()
        serialized_plaintext = fhe_core.dec(
            context.context,
            secret_keys.secret_key.coefs_basis,
            encrypted_result,
            model.as_complex,
        )
        timings["decryption"] = time.perf_counter() - started

        log_status("postprocessing result on client")
        started = time.perf_counter()
        serialized_plaintext = fhe_core.apply_client_block(
            model.postprocess_block,
            context.context,
            serialized_plaintext,
        )
        timings["postprocessing"] = time.perf_counter() - started

        started = time.perf_counter()
        result = deserialize_tensor(serialized_plaintext)
        timings["serialization"] += time.perf_counter() - started
        return EncryptedExecution(result, timings)

    @staticmethod
    def _select_input_key(context: ExtendedContext, secret_keys: ExtendedSecretKey):
        has_ring_context = context.ring_switch_context is not None
        has_ring_key = secret_keys.ring_switch_secret_key is not None
        if has_ring_context != has_ring_key:
            raise ValueError(
                "Invalid query key: ring switch context and secret key must either both be present or both be absent"
            )
        if has_ring_context:
            assert context.ring_switch_context is not None
            assert secret_keys.ring_switch_secret_key is not None
            return context.ring_switch_context, secret_keys.ring_switch_secret_key
        return context.context, secret_keys.secret_key
