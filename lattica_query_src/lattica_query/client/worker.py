from typing import Any

import requests

from ..errors import ProtocolError
from ..logging import log_info
from ..serialization.tensors import deserialize_tensor, serialize_tensor
from ..transport.settings import ClientInfo, TransportConfig
from ..transport.worker import (
    WorkerResponse,
    WorkerTiming,
    WorkerTransport,
)


class WorkerGateway:
    """Typed query operations supported by a remote worker."""

    def __init__(
        self,
        query_token: str,
        *,
        client_info: ClientInfo,
        session: requests.Session,
        config: TransportConfig | None = None,
    ) -> None:
        self._transport = WorkerTransport(
            query_token,
            client_info=client_info,
            session=session,
            config=config,
        )

    @property
    def last_timing(self) -> WorkerTiming:
        return self._transport.timing

    def fetch_initialization(self) -> bytes:
        return self._require_binary_result(
            "get_user_init_data",
            self._transport.invoke("get_user_init_data"),
        )

    def preprocess_evaluation_key(self) -> None:
        self._transport.invoke("preprocess_pk")

    def load_encrypted_data(self) -> None:
        self._transport.invoke("load_custom_encrypted_data")

    def execute_encrypted(self, ciphertext: bytes) -> bytes:
        log_info(f"ciphertext {len(ciphertext) / 1024 ** 2:.1f} MB")
        return self._require_binary_result(
            "apply_hom_pipeline",
            self._transport.invoke(
                "apply_hom_pipeline",
                payload=ciphertext,
            ),
        )

    def execute_clear(self, plaintext: Any) -> Any:
        result = self._require_binary_result(
            "apply_clear",
            self._transport.invoke(
                "apply_clear",
                payload=serialize_tensor(plaintext),
            ),
        )
        return deserialize_tensor(result)

    @staticmethod
    def _require_binary_result(action_name: str, result: WorkerResponse) -> bytes:
        if not isinstance(result, bytes) or not result:
            raise ProtocolError(action_name, "worker completed without returning a binary result")
        return result
