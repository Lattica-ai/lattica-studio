from typing import Self

from lattica_query.logging import STUDIO_THEME
from lattica_query.transport.backend import BackendAPI
from lattica_query.transport.settings import ClientInfo, TransportConfig

from .deployment import DeploymentAPI
from .resources import (
    AccountAPI,
    FinanceAPI,
    ModelsAPI,
    TokensAPI,
    WorkersAPI,
)


class LatticaStudio:
    """Manage account resources and deployments through the Lattica API.

    The client owns its HTTP session. Use it as a context manager or call
    :meth:`close` when it is no longer needed.
    """

    def __init__(self, account_license: str, *, config: TransportConfig | None = None):
        self._http = BackendAPI(
            account_license,
            client_info=ClientInfo.from_package("lattica_studio"),
            theme=STUDIO_THEME,
            config=config,
        )
        self._closed = False

        try:
            self.models = ModelsAPI(self._http)
            self.workers = WorkersAPI(self._http)
            self.tokens = TokensAPI(self._http)
            self.account = AccountAPI(self._http)
            self.finance = FinanceAPI(self._http)

            self._deployment = DeploymentAPI(
                http=self._http,
                models=self.models,
                workers=self.workers,
            )
        except Exception:
            self._http.close()
            self._closed = True
            raise

    def close(self) -> None:
        """Release the HTTP connection pool owned by this client."""
        if self._closed:
            return
        self._closed = True
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def deploy_pipeline(self, *args, **kwargs):
        return self._deployment.deploy_pipeline(*args, **kwargs)

    def deploy(self, *args, **kwargs):
        """Deploy and compile an existing local build artifact."""
        return self._deployment.deploy(*args, **kwargs)
