from lattica_query.transport.backend import BackendAPI

from ..types import JsonDict


class AccountAPI:
    def __init__(self, http: BackendAPI):
        self._http = http

    def get(self) -> JsonDict:
        """Retrieve information about the current account."""
        response = self._http.call(
            "api/account/get_account_info",
        )

        return {
            "accountId": response.get("accountId"),
            "createdAt": response.get("createdAt"),
            "email": response.get("email"),
            "companyName": response.get("companyName"),
            "contactName": response.get("contactName"),
            "phoneNumber": response.get("phoneNumber"),
            "credits": response.get("credits"),
            "authExpDate": response.get("authExpDate"),
        }

    def update(
        self,
        *,
        company_name: str | None = None,
        contact_name: str | None = None,
        email: str | None = None,
        phone_number: str | None = None,
    ) -> str:
        """Update account information."""
        params = {}

        if company_name is not None:
            params["companyName"] = company_name

        if contact_name is not None:
            params["contactName"] = contact_name

        if email is not None:
            params["email"] = email

        if phone_number is not None:
            params["phoneNumber"] = phone_number

        response = self._http.call(
            "api/account/update_account_info",
            parameters=params,
        )

        return response["message"]
