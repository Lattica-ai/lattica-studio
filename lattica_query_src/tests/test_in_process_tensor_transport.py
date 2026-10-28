from types import SimpleNamespace

import pytest
import torch

from lattica_query.client.artifacts import QueryKey
from lattica_query.client.executor import EncryptedQueryExecutor
from lattica_query.serialization.in_process_tensor_transport import (
    InProcessTransportedBytes,
)
from lattica_query.serialization.tensors import deserialize_tensor, serialize_tensor


def test_tensor_sidecar_roundtrip():
    tensor = torch.arange(12).reshape(3, 4).t()
    encoded = serialize_tensor(tensor, in_process=True)

    assert isinstance(encoded, InProcessTransportedBytes)
    assert len(encoded) < tensor.nbytes
    torch.testing.assert_close(deserialize_tensor(encoded), tensor)


@pytest.mark.parametrize("use_transport", [True, False])
def test_executor_materializes_only_the_worker_payload(monkeypatch, use_transport):
    expected = torch.arange(8, dtype=torch.float64)
    model = SimpleNamespace(
        preprocess_block=b"preprocess",
        postprocess_block=b"postprocess",
        external_axis=None,
        as_complex=False,
    )
    context = SimpleNamespace(context=b"context", ring_switch_context=None)
    secret_key = SimpleNamespace(crt_basis=b"crt", coefs_basis=b"coefs")
    secret_keys = SimpleNamespace(secret_key=secret_key, ring_switch_secret_key=None)

    from lattica_query.client import executor as executor_module

    monkeypatch.setattr(
        executor_module, "_USE_IN_PROCESS_TENSOR_TRANSPORT", use_transport
    )
    monkeypatch.setattr(executor_module.ClientModel, "from_bytes", staticmethod(lambda _: model))
    monkeypatch.setattr(executor_module.ExtendedContext, "from_bytes", staticmethod(lambda _: context))
    monkeypatch.setattr(
        executor_module.ExtendedSecretKey,
        "from_bytes",
        staticmethod(lambda _: secret_keys),
    )

    apply_calls = []

    def apply_client_block(block, serialized_context, plaintext):
        apply_calls.append((block, serialized_context, plaintext))
        assert isinstance(plaintext, InProcessTransportedBytes) == use_transport
        return plaintext

    def enc(_context, _secret_key, plaintext, *_args):
        assert isinstance(plaintext, InProcessTransportedBytes) == use_transport
        if use_transport:
            assert plaintext.materialize_outputs
        return b"wire ciphertext"

    def dec(serialized_context, _secret_key, ciphertext, _as_complex):
        assert isinstance(serialized_context, InProcessTransportedBytes) == use_transport
        if use_transport:
            assert not serialized_context.tensor_store.groups
        assert ciphertext == b"wire result"
        return serialize_tensor(expected, in_process=use_transport)

    monkeypatch.setattr(executor_module.fhe_core, "apply_client_block", apply_client_block)
    monkeypatch.setattr(executor_module.fhe_core, "enc", enc)
    monkeypatch.setattr(executor_module.fhe_core, "dec", dec)

    worker = SimpleNamespace(
        execute_encrypted=lambda ciphertext: (
            b"wire result" if ciphertext == b"wire ciphertext" else None
        )
    )
    key = QueryKey(context=b"context", secret_key=b"secret", client_model=b"model")
    result = EncryptedQueryExecutor(worker).execute(expected, key)

    torch.testing.assert_close(result.value, expected)
    assert apply_calls[0][0] == b"preprocess"
    assert apply_calls[1][0] == b"postprocess"
