"""The downloaded initialization data has no lookup tables; fhe_core rebuilds them."""
from types import SimpleNamespace

from lattica_query.client import keys as keys_module
from lattica_query.client.keys import KeysAPI
from lattica_query.serialization.generated import hom_op_pb2

FULL = hom_op_pb2.ClientData(serialized_extended_context=b"context", serialized_model=b"model").SerializeToString()
COMPACT = b"compact"


def test_downloaded_data_is_expanded(monkeypatch):
    calls = []

    def expand(data, include_keygen):
        calls.append((data, include_keygen))
        return FULL

    # Replace expand_client_data with a fake that records its arguments.
    monkeypatch.setattr(keys_module.fhe_core, "expand_client_data", expand, raising=False)
    worker = SimpleNamespace(fetch_initialization=lambda: COMPACT)

    initialization = KeysAPI(SimpleNamespace(), http=None, worker=worker).fetch_initialization()

    assert calls == [(COMPACT, True)]
    assert (initialization.context, initialization.client_model) == (b"context", b"model")
