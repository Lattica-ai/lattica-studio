from lattica_query.api.app import AppAPI

from ..types import JsonDict


class FinanceAPI:
    def __init__(self, http: AppAPI):
        self._http = http

    def get_credits(self) -> str:
        """Return the remaining account credit quota."""
        return self._http.send_http_request(
            "api/finance/get_account_credits",
        )

    def list_transactions(self) -> list[JsonDict]:
        """Return payment transaction history."""
        response = self._http.send_http_request(
            "api/finance/get_transaction_history",
        )

        return response.get("payments", [])