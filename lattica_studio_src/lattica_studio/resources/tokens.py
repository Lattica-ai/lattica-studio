from typing import Optional

from lattica_query.api.app import (
    AppAPI,
    generate_random_token_name,
)
from lattica_query.storage.tokens import save_query_token, load_query_token

from ..types import JsonDict, ModelId, Token, TokenInfo


class TokensAPI:
    def __init__(self, http: AppAPI):
        self._http = http

    def create(
            self,
            model_id: ModelId,
            *,
            name: str | None = None,
            save_as: str | None = None,
    ) -> Token:
        if name is None:
            name = generate_random_token_name(10)

        response = self._http.send_http_request(
            "api/token/generate_token",
            req_params={
                "modelId": model_id,
                "tokenName": name,
            },
        )

        token = response.get("token")

        if token is None:
            raise ValueError(
                "The server response does not contain a token."
            )

        if save_as is not None:
            save_query_token(save_as, token)

        return token

    def load(self, name: str) -> Token:
        """Load a locally saved query token."""
        return load_query_token(name)

    def delete(self, token_id: str) -> str:
        """Delete a token."""
        response = self._http.send_http_request(
            "api/token/delete_token",
            req_params={
                "tokenId": token_id,
            },
        )

        return response["message"]

    def assign(
        self,
        token_id: str,
        model_id: ModelId,
    ) -> JsonDict:
        """Assign a token to a model."""
        response = self._http.send_http_request(
            "api/token/assign_token_to_model",
            req_params={
                "tokenId": token_id,
                "modelIdToAssign": model_id,
            },
        )

        return {
            "message": response["message"],
            "warning": response.get("warning"),
        }

    def unassign(
        self,
        token_id: str,
        model_id: ModelId,
    ) -> JsonDict:
        """Unassign a token from a model."""
        response = self._http.send_http_request(
            "api/token/unassign_token_from_model",
            req_params={
                "tokenId": token_id,
                "modelId": model_id,
            },
        )

        return {
            "message": response["message"],
            "warning": response.get("warning"),
        }

    def update(
        self,
        token_id: str,
        *,
        name: Optional[str] = None,
        note: Optional[str] = None,
        status: Optional[str] = None,
    ) -> str:
        """Update token information."""
        params = {
            "tokenId": token_id,
        }

        if name is not None:
            params["tokenName"] = name

        if note is not None:
            params["tokenNote"] = note

        if status is not None:
            params["status"] = status

        response = self._http.send_http_request(
            "api/token/update_token_info",
            req_params=params,
        )

        return response["message"]

    def get(self, token: Token) -> TokenInfo:
        """Return information associated with a token."""
        response = self._http.send_http_request(
            "api/token/get_token_info",
            req_params={
                "token": token,
            },
        )

        token_data = response.get("token") or {}
        model_data = response.get("model") or {}
        worker_data = response.get("worker") or {}
        evaluation_key_data = response.get("evaluationKey") or {}

        return {
            "tokenStatus": token_data.get("status"),
            "tokenName": token_data.get("tokenName"),
            "tokenExpiration": token_data.get("expirationDate"),
            "modelName": model_data.get("modelName"),
            "modelStatus": model_data.get("status"),
            "workerStatus": worker_data.get("status"),
            "evalKeyCreatedAt": evaluation_key_data.get("createdAt"),
        }

    def list(
        self,
        *,
        status: Optional[str] = None,
        model_id: Optional[ModelId] = None,
        issue_date: Optional[str] = None,
    ) -> list[JsonDict]:
        """List tokens, optionally applying server-side filters."""
        params = {}

        if status is not None:
            params["status"] = status

        if model_id is not None:
            params["modelId"] = model_id

        if issue_date is not None:
            params["issueDate"] = issue_date

        response = self._http.send_http_request(
            "api/token/list_tokens",
            req_params=params,
        )

        return response.get("tokens", [])