import json
import os
from collections.abc import Iterable
from pathlib import Path

from ..client.credentials import QueryToken
from ..errors import StorageError
from ..logging import log_info
from ._files import (
    atomic_private_write,
    encode_path_component,
    ensure_private_directory,
)


class TokenStorageError(StorageError):
    """A stored query token is invalid."""


def get_query_token_path(query_token: QueryToken) -> Path:
    if not isinstance(query_token, QueryToken):
        raise TypeError("query_token must be a QueryToken")
    identity = query_token.require_identity()
    if identity.name is None:
        raise ValueError("query token must have a name before it can be saved")
    return _tokens_root() / encode_path_component(identity.name) / f"{encode_path_component(identity.id)}.json"


def save_query_token(query_token: QueryToken) -> Path:
    token_path = get_query_token_path(query_token)
    identity = query_token.require_identity()
    ensure_private_directory(_storage_home())
    ensure_private_directory(_tokens_root())
    ensure_private_directory(token_path.parent)
    log_info(f"saving query token name={identity.name!r} id={identity.id!r}")
    log_info(f"token path: {token_path}")
    atomic_private_write(token_path, json.dumps(query_token.to_dict(), separators=(",", ":")))
    return token_path


def load_query_token(name: str | None = None) -> QueryToken:
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("token name must be a non-empty string")

    if name is None:
        log_info("selecting latest saved query token")
        candidates = _tokens_root().glob("*/*.json")
    else:
        name = name.strip()
        log_info(f"selecting latest saved query token with name={name!r}")
        candidates = (_tokens_root() / encode_path_component(name)).glob("*.json")

    token_path = _newest(candidates, name)
    log_info(f"token path: {token_path}")
    try:
        data = json.loads(token_path.read_text(encoding="utf-8"))
        query_token = QueryToken.from_dict(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TokenStorageError(f"Invalid stored query token: {token_path}") from exc

    identity = query_token.require_identity()
    log_info(f"loaded query token name={identity.name!r} id={identity.id!r}")
    return query_token


def _tokens_root() -> Path:
    return _storage_home() / "tokens"


def _storage_home() -> Path:
    return Path(os.getenv('LATTICA_HOME', '~/.lattica')).expanduser()


def _newest(candidates: Iterable[Path], name: str | None) -> Path:
    newest: tuple[int, str, Path] | None = None
    for path in candidates:
        try:
            candidate = (path.stat().st_mtime_ns, path.as_posix(), path)
        except FileNotFoundError:
            # A concurrent cleanup may remove a token between globbing and stat.
            continue
        except OSError as exc:
            raise TokenStorageError(f"Unable to inspect stored query token: {path}") from exc
        if newest is None or candidate[:2] > newest[:2]:
            newest = candidate

    if newest is None:
        selection = f" named {name!r}" if name is not None else ""
        raise FileNotFoundError(f"No saved query tokens{selection}")
    return newest[2]
