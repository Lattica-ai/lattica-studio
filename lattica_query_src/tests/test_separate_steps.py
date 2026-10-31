"""The encrypted query, the custom-data encryption and the key generation as separate steps."""
import zipfile
from types import SimpleNamespace

import pytest
import torch

from lattica_query import QueryInitialization, QueryToken, TokenIdentity
from lattica_query.client import encrypted_data as encrypted_data_module
from lattica_query.client import executor as executor_module
from lattica_query.client import keys as keys_module
from lattica_query.client.artifacts import QueryKey
from lattica_query.client.encrypted_data import EncryptedDataAPI
from lattica_query.client.keys import KeysAPI
from lattica_query.client.query import QueryAPI
from lattica_query.serialization.tensors import deserialize_tensor
from lattica_query.storage.key_store import load_key_bundle

KEY = QueryKey(context=b"context", secret_key=b"secret", client_model=b"model")


def _fake_key_artifacts(monkeypatch, module):
    """The parsed key: no ring switch, so the query is encrypted with the primary context and secret key."""
    context = SimpleNamespace(context=b"primary context", ring_switch_context=None)
    secret_key = SimpleNamespace(crt_basis=b"crt", coefs_basis=b"coefs")
    secret_keys = SimpleNamespace(secret_key=secret_key, ring_switch_secret_key=None)
    monkeypatch.setattr(module.ExtendedContext, "from_bytes", staticmethod(lambda _: context))
    monkeypatch.setattr(module.ExtendedSecretKey, "from_bytes", staticmethod(lambda _: secret_keys))


@pytest.fixture
def fake_query_core(monkeypatch):
    """fhe_core for queries: encryption serializes the plaintext into the ciphertext, decryption reads it back."""
    model = SimpleNamespace(
        preprocess_block=b"preprocess", postprocess_block=b"postprocess", external_axis=None, as_complex=False,
    )
    monkeypatch.setattr(executor_module, "_USE_IN_PROCESS_TENSOR_TRANSPORT", False)
    monkeypatch.setattr(executor_module.ClientModel, "from_bytes", staticmethod(lambda _: model))
    _fake_key_artifacts(monkeypatch, executor_module)
    monkeypatch.setattr(executor_module.fhe_core, "apply_client_block", lambda _block, _context, pt: pt)
    monkeypatch.setattr(executor_module.fhe_core, "enc", lambda _context, _sk, pt, *_args: b"ct:" + pt)
    monkeypatch.setattr(executor_module.fhe_core, "dec", lambda _context, _sk, ct, _as_complex: ct[len(b"ct:"):])


class FakeQueryWorker:
    """Runs the model as the identity on the ciphertext and reports a server timing."""

    def __init__(self):
        self.ciphertexts = []
        self.last_timing = SimpleNamespace(server_header="")

    def execute_encrypted(self, ciphertext):
        self.ciphertexts.append(ciphertext)
        self.last_timing = SimpleNamespace(server_header="instance;dur=50, worker;dur=30")
        return ciphertext


def test_query_steps_compose_to_encrypted(fake_query_core):
    worker = FakeQueryWorker()
    query = QueryAPI(worker)
    plaintext = torch.arange(6, dtype=torch.float64)

    ciphertext = query.encrypt(plaintext, key=KEY)
    assert not worker.ciphertexts  # encrypting is local
    encrypted_result = query.execute(ciphertext)
    torch.testing.assert_close(query.decrypt(encrypted_result, key=KEY), plaintext)

    assert query.last_server_timing == pytest.approx({"instance": 0.02, "worker": 0.03})
    torch.testing.assert_close(query.encrypted(plaintext, key=KEY), plaintext)
    assert worker.ciphertexts == [ciphertext, ciphertext]


def test_execute_rejects_an_empty_ciphertext():
    with pytest.raises(ValueError):
        QueryAPI(FakeQueryWorker()).execute(b"")


def test_encrypt_custom_data_writes_one_entry_per_input(monkeypatch, tmp_path):
    _fake_key_artifacts(monkeypatch, encrypted_data_module)
    calls = []

    def enc(context, secret_key, pt, *, custom_state_name):
        # the packing of each input comes from the context
        calls.append((context, secret_key, custom_state_name))
        return b"ct:" + pt

    monkeypatch.setattr(encrypted_data_module.fhe_core, "enc", enc)
    data = {"db": torch.ones(2, 3), "payloads": torch.arange(4.0)}

    path = EncryptedDataAPI(http=None, worker=None).encrypt(data, tmp_path / "custom.zip", key=KEY)

    assert calls == [
        (b"primary context", (b"crt", b"coefs"), "db"),
        (b"primary context", (b"crt", b"coefs"), "payloads"),
    ]
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["db.bin", "payloads.bin"]
        for name, tensor in data.items():
            torch.testing.assert_close(deserialize_tensor(archive.read(f"{name}.bin")[len(b"ct:"):]), tensor)


class FakeKeysBackend:
    """The backend and the worker of key registration; fetching the initialization is not expected."""

    def __init__(self):
        self.uploads = []
        self.preprocessed = 0

    def upload_file_and_alert(self, path, *, endpoint):
        self.uploads.append(endpoint)

    def is_worker_up(self):
        return True

    def preprocess_evaluation_key(self):
        self.preprocessed += 1

    def fetch_initialization(self):
        raise AssertionError("the initialization was given")


def test_generate_without_registration_then_ensure_registers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        keys_module.fhe_core, "generate_key", lambda client_model, context: (b"secret:" + context, b"evaluation"),
    )
    backend = FakeKeysBackend()
    keys = KeysAPI(QueryToken("credential", identity=TokenIdentity("token-id")), backend, backend)
    path = tmp_path / "key"

    key = keys.generate(path, initialization=QueryInitialization(b"context", b"model"), register=False)
    assert key == QueryKey(context=b"context", secret_key=b"secret:context", client_model=b"model")
    assert not backend.uploads
    assert not load_key_bundle(path).evaluation_key_uploaded

    assert keys.ensure(path) == key
    assert backend.uploads == ["api/token/get_pk_upload_url"] and backend.preprocessed == 1
    assert load_key_bundle(path).evaluation_key_uploaded

    keys.ensure(path)  # already registered
    assert len(backend.uploads) == 1
