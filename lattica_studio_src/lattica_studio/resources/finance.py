from lattica_query.transport.backend import BackendAPI

from ..types import JsonDict


class FinanceAPI:
    def __init__(self, http: BackendAPI):
        self._http = http

    def get_credits(self) -> str:
        """Return the remaining account credit quota."""
        return self._http.call(
            "api/finance/get_account_credits",
        )

    def list_transactions(self) -> list[JsonDict]:
        """Return payment transaction history."""
        response = self._http.call(
            "api/finance/get_transaction_history",
        )

        return response.get("payments", [])
