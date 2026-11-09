import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import lattica_fhe_core as fhe_core

from ..logging import QUERY_THEME, OperationLog, log_size_info
from ..serialization.artifacts import ExtendedContext, ExtendedSecretKey
from ..serialization.tensors import serialize_tensor
from .artifacts import QueryKey

if TYPE_CHECKING:
    import torch

    from ..transport.backend import BackendAPI
    from .worker import WorkerGateway


class EncryptedDataAPI:
    """Encrypt and upload encrypted data used by a query pipeline."""

    def __init__(self, http: "BackendAPI", worker: "WorkerGateway") -> None:
        self._http = http
        self._worker = worker

    def encrypt(
        self,
        data: Mapping[str, "torch.Tensor"],
        path: str | Path,
        *,
        key: QueryKey,
    ) -> Path:
        """Encrypt a model's custom inputs locally into the archive ``upload()`` takes.

        Each tensor is encrypted for the model's custom input of the same name, at
        the level, scale and packing the model's context records for that input.
        """
        path = Path(path)
        context = ExtendedContext.from_bytes(key.context).context
        secret_key = ExtendedSecretKey.from_bytes(key.secret_key).secret_key
        with OperationLog("encrypting custom data", theme=QUERY_THEME):
            with zipfile.ZipFile(path, "w") as archive:
                for name, tensor in data.items():
                    ciphertext = fhe_core.enc(
                        context,
                        (secret_key.crt_basis, secret_key.coefs_basis),
                        serialize_tensor(tensor),
                        custom_state_name=name,
                    )
                    archive.writestr(f"{name}.bin", ciphertext)
            log_size_info("encrypted data", path.stat().st_size)
        return path

    def upload(self, path: str | Path) -> None:
        path = Path(path)
        with OperationLog("uploading encrypted data", theme=QUERY_THEME):
            self._upload_file(path)
        with OperationLog("loading encrypted data", theme=QUERY_THEME):
            self._worker.load_encrypted_data()

    def _upload_file(self, path: Path) -> None:
        self._http.upload_file_and_alert(
            str(path),
            endpoint="api/token/get_custom_data_upload_url",
        )
