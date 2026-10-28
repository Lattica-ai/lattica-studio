from pathlib import Path
from typing import TYPE_CHECKING

import lattica_fhe_core as fhe_core

from ..errors import LatticaClientError
from ..logging import QUERY_THEME, OperationLog, log_info, log_size_info, log_status
from ..serialization.artifacts import QueryInitialization
from ..storage.key_store import (
    StoredKey,
    get_key_path,
    key_bundle_exists,
    key_lock,
    load_key_bundle,
    mark_key_uploaded,
    save_key_bundle,
)
from .artifacts import QueryKey
from .credentials import QueryToken, TokenIdentity

if TYPE_CHECKING:
    from ..transport.backend import BackendAPI
    from .worker import WorkerGateway


class KeyTokenMismatchError(LatticaClientError, ValueError):
    """A key bundle belongs to a different query token."""

    def __init__(self, path: Path, expected: TokenIdentity, actual: TokenIdentity) -> None:
        super().__init__(
            f"Key bundle {path} belongs to token name={actual.name!r} id={actual.id!r}; "
            f"expected name={expected.name!r} id={expected.id!r}"
        )


class KeysAPI:
    """Generate, store, load, and register query keys."""

    def __init__(self, token: QueryToken, http: "BackendAPI", worker: "WorkerGateway") -> None:
        self._token = token
        self._http = http
        self._worker = worker

    def ensure(self, path: str | Path | None = None) -> QueryKey:
        """Load an existing key or generate one, completing interrupted registration."""
        identity = self._token.require_identity()
        resolved_path = Path(path) if path is not None else get_key_path(identity)
        with key_lock(resolved_path):
            if not key_bundle_exists(resolved_path):
                return self._generate(resolved_path)
            record = self._load_bundle(resolved_path)
            if not record.evaluation_key_uploaded:
                record = self._register(record)
            return record.key

    def generate(
        self,
        path: str | Path | None = None,
    ) -> QueryKey:
        """Generate a new key, replacing any local bundle at the destination."""
        identity = self._token.require_identity()
        resolved_path = Path(path) if path is not None else get_key_path(identity)
        with key_lock(resolved_path):
            return self._generate(resolved_path)

    def _generate(self, resolved_path: Path) -> QueryKey:
        identity = self._token.require_identity()
        init_data = self._fetch_client_data()
        with OperationLog("generating FHE keys", theme=QUERY_THEME):
            secret_key, evaluation_key = fhe_core.generate_key(
                init_data.client_model,
                init_data.context,
            )
            key = QueryKey(
                init_data.context,
                secret_key,
                init_data.client_model,
            )
            record = save_key_bundle(identity, key, evaluation_key, resolved_path)

        return self._register(record).key

    def load(self, path: str | Path | None = None) -> QueryKey:
        """Load a local key without contacting the backend."""
        identity = self._token.require_identity()
        resolved_path = Path(path) if path is not None else get_key_path(identity)
        with key_lock(resolved_path):
            return self._load_bundle(resolved_path).key

    def _load_bundle(self, resolved_path: Path) -> StoredKey:
        identity = self._token.require_identity()
        record = load_key_bundle(resolved_path)
        if record.token.id != identity.id:
            raise KeyTokenMismatchError(resolved_path, identity, record.token)
        return record

    def _register(self, record: StoredKey) -> StoredKey:
        with OperationLog("registering evaluation key", theme=QUERY_THEME):
            self._upload_evaluation_key(record.evaluation_key_path)
        return mark_key_uploaded(record)

    def _fetch_client_data(self) -> QueryInitialization:
        with OperationLog("retrieving initialization data", theme=QUERY_THEME):
            return QueryInitialization.from_bytes(self._worker.fetch_initialization())

    def _upload_evaluation_key(self, evaluation_key_path: Path) -> None:
        evaluation_key_size = evaluation_key_path.stat().st_size
        if evaluation_key_size == 0:
            raise ValueError("evaluation key must be non-empty")
        log_size_info("evaluation key", evaluation_key_size)
        log_status("uploading evaluation key")
        self._http.upload_file_and_alert(
            str(evaluation_key_path),
            endpoint="api/token/get_pk_upload_url",
        )
        if not self._http.is_worker_up():
            log_info("no worker running; the key will be loaded when a worker starts")
            return
        log_status("preprocessing evaluation key")
        self._worker.preprocess_evaluation_key()
