from typing import Optional

from lattica_query.api.app import AppAPI

from ..types import JsonDict


class AccountAPI:
    def __init__(self, http: AppAPI):
        self._http = http

    def get(self) -> JsonDict:
        """Retrieve information about the current account."""
        response = self._http.send_http_request(
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
        company_name: Optional[str] = None,
        contact_name: Optional[str] = None,
        email: Optional[str] = None,
        phone_number: Optional[str] = None,
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

        response = self._http.send_http_request(
            "api/account/update_account_info",
            req_params=params,
        )

        return response["message"]