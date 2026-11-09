"""Fetching the initialization data: compact when fhe_core can rebuild the lookup tables, full otherwise."""
from types import SimpleNamespace

from lattica_query.client import keys as keys_module
from lattica_query.client.keys import KeysAPI
from lattica_query.client.worker import WorkerGateway
from lattica_query.serialization.generated import hom_op_pb2

FULL = hom_op_pb2.ClientData(serialized_extended_context=b"context", serialized_model=b"model").SerializeToString()
COMPACT = b"compact"


class FakeWorker:
    """Hands out the compact or the full client data and remembers which was asked for."""

    def __init__(self):
        self.compact = None

    def fetch_initialization(self, compact=False):
        self.compact = compact
        return COMPACT if compact else FULL


def _keys(worker):
    return KeysAPI(SimpleNamespace(), http=None, worker=worker)


def test_compact_data_is_expanded_when_fhe_core_can(monkeypatch):
    calls = []

    def expand(data, include_keygen):
        calls.append((data, include_keygen))
        return FULL

    monkeypatch.setattr(keys_module.fhe_core, "expand_client_data", expand, raising=False)
    worker = FakeWorker()

    initialization = _keys(worker).fetch_initialization()

    assert worker.compact is True
    assert calls == [(COMPACT, True)]
    assert (initialization.context, initialization.client_model) == (b"context", b"model")


def test_full_data_is_taken_when_fhe_core_cannot_expand(monkeypatch):
    monkeypatch.delattr(keys_module.fhe_core, "expand_client_data", raising=False)
    worker = FakeWorker()

    initialization = _keys(worker).fetch_initialization()

    assert worker.compact is False
    assert initialization.client_model == b"model"


def test_gateway_asks_for_the_compact_data_only_when_told():
    invocations = []
    gateway = object.__new__(WorkerGateway)
    gateway._transport = SimpleNamespace(invoke=lambda action, parameters=None: invocations.append((action, parameters)) or b"data")

    gateway.fetch_initialization(compact=True)
    gateway.fetch_initialization()

    assert invocations == [("get_user_init_data", {"compact": True}), ("get_user_init_data", None)]
