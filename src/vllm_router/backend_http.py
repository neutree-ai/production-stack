"""Backend HTTP configuration and observable request phase state."""

import math
import time
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class BackendHttpConfig:
    max_connections_per_backend: int = 100
    pool_wait_timeout_seconds: float = 60
    connect_timeout_seconds: float = 10
    response_header_timeout_seconds: float = 0
    response_read_timeout_seconds: float = 600
    stream_idle_timeout_seconds: float = 600

    def __post_init__(self):
        for field in fields(self):
            if getattr(self, field.name) is None:
                object.__setattr__(self, field.name, field.default)
        limit = self.max_connections_per_backend
        if type(limit) is not int or not 0 <= limit <= 65535:
            raise ValueError(
                "max_connections_per_backend must be an integer in 0..65535 (0 disables the limit)"
            )
        for field in fields(self):
            if field.name == "max_connections_per_backend":
                continue
            value = getattr(self, field.name)
            if (
                type(value) not in (int, float)
                or not 0 <= value <= 604800
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{field.name} must be finite seconds in 0..604800 (0 disables the timeout)"
                )

    @classmethod
    def from_args(cls, args):
        return cls(
            **{
                field.name: getattr(args, field.name, field.default)
                for field in fields(cls)
            }
        )


class BackendTimeout(TimeoutError):
    def __init__(self, phase):
        self.phase = phase
        super().__init__(f"Backend {phase} timeout")


@dataclass
class ForwardContext:
    server: str
    workspace: str = ""
    endpoint: str = ""
    phase: str = "pool_wait"
    outcome: str = "cancelled"
    status_code: int | None = None
    header_wait_started: float | None = None
    header_wait_elapsed: float | None = None
    pool_wait_elapsed: float = 0
    connection_in_use: bool = False
    started: float = 0

    def __post_init__(self):
        self.started = time.monotonic()


@dataclass(frozen=True)
class BackendEndpoint:
    """A discovery selection; networking belongs to the shared HTTP wrapper."""

    _base_url: str
    workspace: str = ""
    endpoint: str = ""
