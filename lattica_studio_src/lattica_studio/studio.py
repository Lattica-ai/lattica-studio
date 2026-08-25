from lattica_query.api.app import AppAPI

from .deployment import DeploymentAPI
from .resources import (
    AccountAPI,
    FinanceAPI,
    ModelsAPI,
    TokensAPI,
    WorkersAPI,
)


class LatticaStudio:
    def __init__(self, account_license: str):
        self._http = AppAPI(
            account_license,
            module_name="lattica_studio",
        )

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

    def deploy_pipeline(self, *args, **kwargs):
        return self._deployment.deploy_pipeline(*args, **kwargs)