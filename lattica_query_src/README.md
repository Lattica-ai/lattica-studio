# Lattica Query Client

The Lattica Query Client executes inference against Lattica workers. Encrypted
queries encrypt the input locally, process it remotely as ciphertext, and
decrypt the response locally before returning a clear tensor.

## Installation

```bash
pip install lattica-query
```

## Query API

```python
from lattica_query import QueryClient, QueryToken

token = QueryToken("<credential>")
with QueryClient(token) as client:
    key = client.keys.ensure()
    result = client.query.encrypted(plaintext, key=key)
```

The client exposes three focused namespaces:

- `client.keys.ensure()` loads a local key or generates and registers one.
  `generate()` always replaces the current key, while `load()` only reads local
  state and never contacts the service.
- `client.query.encrypted()` performs client preprocessing, encryption, worker
  execution, decryption, and postprocessing. `clear()` executes a clear query.
- `client.encrypted_data.upload(path)` uploads a custom encrypted-data artifact
  and asks the worker to load it.

`QueryClient` owns one HTTP session. Use it as a context manager or call
`close()`; repeated calls to `close()` are safe. Do not execute concurrent
queries through one client instance because its session and latest timing state
are shared.

## Credentials

`QueryToken` keeps the secret credential out of its representation. If you omit
`TokenIdentity`, the client decodes the stable token ID locally from the
credential; constructing a client does not make a token-information request.
The optional token name is a display and storage label supplied by Studio.

Stored tokens are organized by encoded token name and ID under
`~/.lattica/tokens`. A name is required to save a token. Credential files use
owner-only permissions.

## Key storage and recovery

Keys are stored under `~/.lattica/keys/<encoded-token-id>`. Each replacement is
written as a complete revision and then committed atomically, so an interrupted
generation leaves the previous revision usable. Operations for one token are
serialized with a file lock.

The evaluation key remains on disk and is streamed during registration. It is
not loaded when an already registered query key is reused. The persisted
`evaluation_key_uploaded` state means that upload and backend registration
completed; worker preprocessing may happen later if no worker was running.
When an upload is interrupted, the key stays pending and the next
`client.keys.ensure()` resumes registration from the stored file.

## Timing and output

Pass `show_timing=True` to render client and server timing after an encrypted
query:

```python
result = client.query.encrypted(plaintext, key=key, show_timing=True)
```

Interactive terminals receive colored animation. Redirected output is plain
text. Output can be suppressed for a block:

```python
from lattica_query import OutputConfig, output_context

with output_context(OutputConfig(enabled=False)):
    result = client.query.encrypted(plaintext, key=key)
```

## Configuration and errors

Production endpoints are used by default. Tests and local tools can inject an
immutable per-client configuration:

```python
from lattica_query import QueryClient, TransportConfig

config = TransportConfig(backend_url="http://localhost:3050")
with QueryClient(token, config=config) as client:
    result = client.query.encrypted(plaintext, key=key)
```

All client failures derive from `LatticaClientError`. The root package also
exports the stable categories `AuthenticationError`, `ClientVersionError`,
`InvalidCredentialError`, `SerializationError`, `StorageError`, and
`TransportError`. More specific diagnostic exceptions live in
`lattica_query.errors`.

See the full documentation at [platformdocs.lattica.ai](https://platformdocs.lattica.ai/).

## License

Proprietary - © Lattica AI. See `LICENSE.md` for details.
