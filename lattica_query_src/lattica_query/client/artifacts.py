from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QueryKey:
    """Serialized client artifacts required to execute encrypted queries."""

    context: bytes = field(repr=False)
    secret_key: bytes = field(repr=False)
    client_model: bytes = field(repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("context", self.context),
            ("secret_key", self.secret_key),
            ("client_model", self.client_model),
        ):
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"{name} must be non-empty bytes")
