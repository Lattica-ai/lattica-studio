import requests

from ..logging import QUERY_THEME
from ..transport.backend import BackendAPI
from ..transport.settings import ClientInfo, TransportConfig
from .credentials import QueryToken
from .encrypted_data import EncryptedDataAPI
from .keys import KeysAPI
from .query import QueryAPI
from .worker import WorkerGateway


class QueryClient:
    """Client for executing Lattica homomorphic-encryption queries.

    The client owns its HTTP session and closes it through :meth:`close` or
    the context-manager protocol. One instance should not be used by multiple
    threads concurrently because query timing and HTTP session state are shared.
    """

    def __init__(self, query_token: QueryToken, *, config: TransportConfig | None = None) -> None:
        if not isinstance(query_token, QueryToken):
            raise TypeError("query_token must be a QueryToken")
        self._session = requests.Session()
        self._closed = False
        try:
            client_info = ClientInfo.from_package("lattica_query")
            self._http = BackendAPI(
                query_token.value,
                client_info=client_info,
                theme=QUERY_THEME,
                session=self._session,
                config=config,
            )
            self._worker = WorkerGateway(
                query_token.value,
                client_info=client_info,
                session=self._session,
                config=config,
            )
            self.token = query_token
            self.keys = KeysAPI(self.token, self._http, self._worker)
            self.query = QueryAPI(self._worker)
            self.encrypted_data = EncryptedDataAPI(self._http, self._worker)
        except Exception:
            self._session.close()
            raise

    def close(self) -> None:
        """Release HTTP connection pools owned by this client."""
        if self._closed:
            return
        self._closed = True
        self._session.close()

    def __enter__(self):
        """Return this open client from a context-manager block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
