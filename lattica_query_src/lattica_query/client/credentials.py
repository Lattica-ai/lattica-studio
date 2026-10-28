import base64
import binascii
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ..errors import InvalidCredentialError


@dataclass(frozen=True, slots=True)
class TokenIdentity:
    """The non-secret identity of a query token."""

    id: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("token id must be a non-empty string")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("token name must be a non-empty string when provided")

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenIdentity":
        try:
            return cls(id=data["id"], name=data.get("name"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid token identity data") from exc


@dataclass(frozen=True, slots=True)
class QueryToken:
    """A secret query credential and its non-secret identity.

    When identity is omitted, the stable token ID is decoded locally from the
    credential. A token name is only required when persisting the credential.
    """

    value: str = field(repr=False)
    identity: TokenIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("token value must be a non-empty string")
        if self.identity is not None and not isinstance(self.identity, TokenIdentity):
            raise TypeError("identity must be a TokenIdentity")
        if self.identity is None:
            object.__setattr__(self, "identity", TokenIdentity(_decode_token_id(self.value)))

    def require_identity(self) -> TokenIdentity:
        if self.identity is None:
            raise ValueError("query token identity has not been resolved")
        return self.identity

    def to_dict(self) -> dict[str, str | dict[str, str | None]]:
        return {
            "value": self.value,
            "identity": self.require_identity().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryToken":
        try:
            return cls(
                value=data["value"],
                identity=TokenIdentity.from_dict(data["identity"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid query token data") from exc


def _decode_token_id(value: str) -> str:
    try:
        payload = value.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        token_id = claims["tokenId"]
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
    ) as exc:
        raise InvalidCredentialError("query credential does not contain a token ID") from exc
    if not isinstance(token_id, str) or not token_id.strip():
        raise InvalidCredentialError("query credential does not contain a token ID")
    return token_id
