"""Transport metrics: per-backend forwarding phases, separate from engine queues."""

import time
from dataclasses import dataclass, field

from prometheus_client import Counter, Gauge, Histogram

_LABELS = ["workspace", "endpoint", "server"]
backend_requests_in_flight = Gauge(
    "router:backend_requests_in_flight",
    "Selected backend requests, including connection pool wait and response transfer",
    _LABELS,
)
connection_pool_waiting = Gauge(
    "router:connection_pool_waiting",
    "Requests waiting for a backend connection slot",
    _LABELS,
)
connections_in_use = Gauge(
    "router:connections_in_use",
    "Backend connector reservations, including connections being established; excludes idle keep-alives",
    _LABELS,
)
backend_response_headers_seconds = Histogram(
    "router:backend_response_headers_seconds",
    "Time from completing request upload to parsing response headers; not TTFT",
    _LABELS,
    buckets=(
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
        30,
        60,
        120,
        300,
        600,
        1800,
        3600,
    ),
)
backend_requests_total = Counter(
    "router:backend_requests_total",
    "Mutually exclusive forwarding outcomes; completed includes fully transferred HTTP error responses",
    _LABELS + ["outcome", "phase", "status_class"],
)


@dataclass
class _BackendState:
    active: int = 0
    last_seen: float = 0
    terminals: set = field(default_factory=set)


_states = {}


def observe_forwarding(event, context):
    key = (context.workspace or "", context.endpoint or "", context.server)
    state = _states.get(key)
    if state is None:
        state = _states[key] = _BackendState()
        for metric in (
            backend_requests_in_flight,
            connection_pool_waiting,
            connections_in_use,
            backend_response_headers_seconds,
        ):
            metric.labels(*key)
    state.last_seen = time.monotonic()
    if event == "request_started":
        state.active += 1
        backend_requests_in_flight.labels(*key).inc()
    elif event == "pool_wait_started":
        connection_pool_waiting.labels(*key).inc()
    elif event == "pool_wait_finished":
        connection_pool_waiting.labels(*key).dec()
    elif event == "connection_acquired":
        connections_in_use.labels(*key).inc()
    elif event == "connection_released":
        connections_in_use.labels(*key).dec()
    elif event == "headers_received" and context.header_wait_elapsed is not None:
        backend_response_headers_seconds.labels(*key).observe(
            context.header_wait_elapsed
        )
    elif event == "request_finished":
        state.active -= 1
        backend_requests_in_flight.labels(*key).dec()
        if (
            context.header_wait_started is not None
            and context.header_wait_elapsed is None
        ):
            backend_response_headers_seconds.labels(*key).observe(
                time.monotonic() - context.header_wait_started
            )
        status_class = (
            f"{context.status_code // 100}xx" if context.status_code else "none"
        )
        terminal = (context.outcome, context.phase, status_class)
        state.terminals.add(terminal)
        backend_requests_total.labels(*key, *terminal).inc()


def prune_forwarding_metrics(endpoints, *, now=None):
    """Keep active requests and live backends; expire retired labels after five minutes."""
    current = {(ep.workspace or "", ep.endpoint or "", ep.url) for ep in endpoints}
    now = time.monotonic() if now is None else now
    for key, state in list(_states.items()):
        if state.active or key in current or now - state.last_seen < 300:
            continue
        for metric in (
            backend_requests_in_flight,
            connection_pool_waiting,
            connections_in_use,
            backend_response_headers_seconds,
        ):
            metric.remove(*key)
        for terminal in state.terminals:
            backend_requests_total.remove(*key, *terminal)
        del _states[key]
