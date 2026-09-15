import secrets
import string
from collections.abc import Iterable

from lattica_query import QueryToken, TokenIdentity
from lattica_query.storage.token_store import load_query_token, save_query_token
from lattica_query.transport.backend import BackendAPI

from ..display import display_table
from ..exceptions import InvalidResourceResponseError
from ..types import JsonDict, ModelId, TokenInfo


def _random_token_name(length: int = 10) -> str:
    alphabet = string.ascii_letters + string.digits + "-_"
    return "".join(secrets.choice(alphabet) for _ in range(length))


class TokensAPI:
    def __init__(self, http: BackendAPI):
        self._http = http

    def create(
            self,
            model_id: ModelId,
            *,
            name: str | None = None,
            save: bool = False,
    ) -> QueryToken:
        if name is None:
            name = _random_token_name()

        response = self._http.call(
            "api/token/generate_token",
            parameters={
                "modelId": model_id,
                "tokenName": name,
            },
        )

        if not isinstance(response, dict):
            raise InvalidResourceResponseError("Token creation response is malformed")
        token = response.get("token")
        token_id = response.get("tokenId")

        if not isinstance(token, str) or not token:
            raise InvalidResourceResponseError(
                "The server response does not contain a token."
            )
        if not isinstance(token_id, str) or not token_id:
            raise InvalidResourceResponseError(
                "The server response does not contain a token ID."
            )

        query_token = QueryToken(
            value=token,
            identity=TokenIdentity(id=token_id, name=name),
        )

        if save:
            save_query_token(query_token)

        return query_token

    def load(self, name: str | None = None) -> QueryToken:
        """Load the newest saved query token, optionally restricted by name."""
        return load_query_token(name)

    def delete(self, token_id: str) -> str:
        """Delete a token."""
        response = self._http.call(
            "api/token/delete_token",
            parameters={
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
        response = self._http.call(
            "api/token/assign_token_to_model",
            parameters={
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
        response = self._http.call(
            "api/token/unassign_token_from_model",
            parameters={
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
        name: str | None = None,
        note: str | None = None,
        status: str | None = None,
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

        response = self._http.call(
            "api/token/update_token_info",
            parameters=params,
        )

        return response["message"]

    def get(self, token: QueryToken) -> TokenInfo:
        """Return information associated with a token."""
        response = self._http.call(
            "api/token/get_token_info",
            parameters={
                "token": token.value,
            },
        )

        if not isinstance(response, dict):
            raise InvalidResourceResponseError("Token information response is malformed")
        token_data = response.get("token") or {}
        model_data = response.get("model") or {}
        worker_data = response.get("worker") or {}
        evaluation_key_data = response.get("evaluationKey") or {}
        if not all(
            isinstance(data, dict)
            for data in (token_data, model_data, worker_data, evaluation_key_data)
        ):
            raise InvalidResourceResponseError("Token information response is malformed")

        return TokenInfo(
            id=token_data.get("tokenId"),
            status=token_data.get("status"),
            name=token_data.get("tokenName"),
            expiration=token_data.get("expirationDate"),
            model_id=model_data.get("modelId"),
            model_name=model_data.get("modelName"),
            model_status=model_data.get("status"),
            worker_status=worker_data.get("status"),
            evaluation_key_created_at=evaluation_key_data.get("createdAt"),
        )

    def list(
        self,
        *,
        status: str | None = None,
        model_id: ModelId | None = None,
        issue_date: str | None = None,
    ) -> list[TokenInfo]:
        """List tokens, optionally applying server-side filters."""
        params = {}

        if status is not None:
            params["status"] = status

        if model_id is not None:
            params["modelId"] = model_id

        if issue_date is not None:
            params["issueDate"] = issue_date

        response = self._http.call(
            "api/token/list_tokens",
            parameters=params,
        )

        if not isinstance(response, dict):
            raise InvalidResourceResponseError("Token list response is malformed")
        tokens = response.get("tokens", [])
        if not isinstance(tokens, list) or not all(isinstance(token, dict) for token in tokens):
            raise InvalidResourceResponseError("Token list response is malformed")
        return [TokenInfo.from_api(token) for token in tokens]

    @staticmethod
    def display(tokens: Iterable[TokenInfo]) -> None:
        """Print an easy-to-scan table of query tokens."""
        display_table(
            ("NAME", "TOKEN ID", "STATUS", "MODEL", "EXPIRES"),
            (
                (token.name, token.id, token.status, token.model_name, token.expiration)
                for token in tokens
            ),
            empty_message="No tokens found.",
        )
