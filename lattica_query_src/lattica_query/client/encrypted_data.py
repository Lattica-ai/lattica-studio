from pathlib import Path
from typing import TYPE_CHECKING

from ..logging import QUERY_THEME, OperationLog

if TYPE_CHECKING:
    from ..transport.backend import BackendAPI
    from .worker import WorkerGateway


class EncryptedDataAPI:
    """Upload encrypted data used by a query pipeline."""

    def __init__(self, http: "BackendAPI", worker: "WorkerGateway") -> None:
        self._http = http
        self._worker = worker

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
