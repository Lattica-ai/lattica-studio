import fcntl
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from ..client.artifacts import QueryKey
from ..client.credentials import TokenIdentity
from ..errors import StorageError
from ..logging import QUERY_THEME, OperationLog, log_info
from ._files import (
    atomic_private_write,
    encode_path_component,
    ensure_private_directory,
)

_KEY_BUNDLE_VERSION = 4
_CURRENT = "current"
_REVISIONS = "bundles"
_METADATA = "metadata.json"
_CONTEXT = "context.bin"
_SECRET_KEY = "secret-key.bin"
_CLIENT_MODEL = "client-model.bin"
_EVALUATION_KEY = "evaluation-key.bin"


class KeyStorageError(StorageError):
    """A local key bundle is malformed."""


@dataclass(frozen=True, slots=True)
class StoredKey:
    """A query key and the location of its lazily loaded evaluation key."""

    token: TokenIdentity
    key: QueryKey
    evaluation_key_path: Path
    evaluation_key_uploaded: bool

    def uploaded(self) -> "StoredKey":
        return replace(self, evaluation_key_uploaded=True)


def get_key_path(token: TokenIdentity) -> Path:
    if not isinstance(token, TokenIdentity):
        raise TypeError("token must be a TokenIdentity")
    return _storage_home() / "keys" / encode_path_component(token.id)


def key_bundle_exists(path: str | Path) -> bool:
    return (Path(path) / _CURRENT).is_file()


@contextmanager
def key_lock(path: str | Path) -> Iterator[None]:
    key_path = Path(path)
    if key_path.parent == _storage_home() / "keys":
        ensure_private_directory(_storage_home())
    ensure_private_directory(key_path.parent)
    ensure_private_directory(key_path)
    lock_path = key_path / ".lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def save_key_bundle(
    token: TokenIdentity,
    key: QueryKey,
    evaluation_key: bytes,
    path: str | Path,
) -> StoredKey:
    if not isinstance(evaluation_key, bytes) or not evaluation_key:
        raise ValueError("evaluation_key must be non-empty bytes")

    key_path = Path(path)
    revisions_path = key_path / _REVISIONS
    ensure_private_directory(revisions_path)
    revision = uuid.uuid4().hex
    staging_path = revisions_path / f".{revision}.tmp"
    bundle_path = revisions_path / revision
    ensure_private_directory(staging_path)

    record = StoredKey(
        token=token,
        key=key,
        evaluation_key_path=bundle_path / _EVALUATION_KEY,
        evaluation_key_uploaded=False,
    )
    try:
        atomic_private_write(staging_path / _CONTEXT, key.context)
        atomic_private_write(staging_path / _SECRET_KEY, key.secret_key)
        atomic_private_write(staging_path / _CLIENT_MODEL, key.client_model)
        atomic_private_write(staging_path / _EVALUATION_KEY, evaluation_key)
        _write_metadata(record, staging_path)
        os.replace(staging_path, bundle_path)
        atomic_private_write(key_path / _CURRENT, revision)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        shutil.rmtree(bundle_path, ignore_errors=True)
        raise

    _remove_old_revisions(revisions_path, revision)
    log_info(f"saving key for token name={token.name!r} id={token.id!r}")
    log_info(f"key path: {key_path}")
    return record


def mark_key_uploaded(record: StoredKey) -> StoredKey:
    uploaded = record.uploaded()
    _write_metadata(uploaded, record.evaluation_key_path.parent)
    return uploaded


def load_key_bundle(path: str | Path) -> StoredKey:
    key_path = Path(path)
    with OperationLog("loading local key artifacts", theme=QUERY_THEME):
        log_info(f"key path: {key_path}")
        bundle_path = _resolve_current_bundle(key_path)
        metadata = _load_metadata(bundle_path)
        try:
            token = TokenIdentity.from_dict(metadata["token"])
            uploaded = metadata["evaluation_key_uploaded"]
            if not isinstance(uploaded, bool):
                raise TypeError
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyStorageError(f"Invalid key bundle {key_path}: invalid metadata") from exc

        evaluation_key_path = bundle_path / _EVALUATION_KEY
        _validate_artifact(evaluation_key_path, key_path)
        record = StoredKey(
            token=token,
            key=QueryKey(
                context=_read_artifact(bundle_path, _CONTEXT, key_path),
                secret_key=_read_artifact(bundle_path, _SECRET_KEY, key_path),
                client_model=_read_artifact(bundle_path, _CLIENT_MODEL, key_path),
            ),
            evaluation_key_path=evaluation_key_path,
            evaluation_key_uploaded=uploaded,
        )
        log_info(f"loaded key for token name={token.name!r} id={token.id!r}")
        return record


def _storage_home() -> Path:
    return Path(os.getenv("LATTICA_HOME", "~/.lattica")).expanduser()


def _resolve_current_bundle(key_path: Path) -> Path:
    current_path = key_path / _CURRENT
    try:
        revision = current_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise FileNotFoundError(f"Key bundle not found: {key_path}") from exc
    if len(revision) != 32 or any(character not in "0123456789abcdef" for character in revision):
        raise KeyStorageError(f"Invalid key bundle {key_path}: invalid current revision")
    bundle_path = key_path / _REVISIONS / revision
    if not bundle_path.is_dir():
        raise KeyStorageError(f"Invalid key bundle {key_path}: current revision is missing")
    return bundle_path


def _write_metadata(record: StoredKey, bundle_path: Path) -> None:
    metadata = {
        "version": _KEY_BUNDLE_VERSION,
        "token": record.token.to_dict(),
        "evaluation_key_uploaded": record.evaluation_key_uploaded,
    }
    atomic_private_write(
        bundle_path / _METADATA,
        json.dumps(metadata, separators=(",", ":")),
    )


def _load_metadata(bundle_path: Path) -> dict:
    metadata_path = bundle_path / _METADATA
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KeyStorageError(f"Invalid key bundle {bundle_path}: unable to read metadata") from exc
    if not isinstance(metadata, dict):
        raise KeyStorageError(f"Invalid key bundle {bundle_path}: metadata must be an object")
    if metadata.get("version") != _KEY_BUNDLE_VERSION:
        raise KeyStorageError(
            f"Unsupported key bundle version {metadata.get('version')!r} in {bundle_path}"
        )
    return metadata


def _read_artifact(bundle_path: Path, name: str, key_path: Path) -> bytes:
    artifact_path = bundle_path / name
    _validate_artifact(artifact_path, key_path)
    try:
        return artifact_path.read_bytes()
    except OSError as exc:
        raise KeyStorageError(f"Invalid key bundle {key_path}: unable to read {name}") from exc


def _validate_artifact(artifact_path: Path, key_path: Path) -> None:
    try:
        size = artifact_path.stat().st_size
    except OSError as exc:
        raise KeyStorageError(
            f"Invalid key bundle {key_path}: unable to read {artifact_path.name}"
        ) from exc
    if size == 0:
        raise KeyStorageError(f"Invalid key bundle {key_path}: {artifact_path.name} is empty")


def _remove_old_revisions(revisions_path: Path, current_revision: str) -> None:
    for candidate in revisions_path.iterdir():
        if candidate.name != current_revision:
            shutil.rmtree(candidate, ignore_errors=True)
