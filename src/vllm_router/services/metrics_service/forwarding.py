"""Backend connection-pool waiting, observed only through native aiohttp hooks."""

import time

import aiohttp
from prometheus_client import Gauge

from vllm_router.log import init_logger

logger = init_logger(__name__)
connection_pool_waiting = Gauge(
    "router:connection_pool_waiting",
    "Backend HTTP calls waiting for a connection slot",
    ["workspace", "endpoint", "server", "replica"],
)
_series = {}


def trace_context(workspace, endpoint, server, replica=""):
    return {
        "workspace": workspace or "",
        "endpoint": endpoint or "",
        "server": server,
        "replica": replica or "",
    }


async def _request_started(session, context, params):
    if context.trace_request_ctx is None:
        return
    try:
        data = context.trace_request_ctx
        context.queue_key = (
            data["workspace"],
            data["endpoint"],
            data["server"],
            data["replica"],
        )
        context.queued = False
        state = _series.setdefault(context.queue_key, [0, 0])
        state[0] = time.monotonic()
        connection_pool_waiting.labels(*context.queue_key)
    except Exception:
        logger.exception("Failed to initialize backend pool metrics")


async def _queue_started(session, context, params):
    key = getattr(context, "queue_key", None)
    if key is None:
        return
    try:
        _series[key][0] = time.monotonic()
        _series[key][1] += 1
        context.queued = True
        connection_pool_waiting.labels(*key).inc()
    except Exception:
        logger.exception("Failed to record backend pool wait")


async def _queue_finished(session, context, params):
    if not getattr(context, "queued", False):
        return
    try:
        context.queued = False
        key = context.queue_key
        _series[key][0] = time.monotonic()
        _series[key][1] -= 1
        connection_pool_waiting.labels(*key).dec()
    except Exception:
        logger.exception("Failed to finish backend pool wait")


def create_trace_config():
    trace = aiohttp.TraceConfig()
    trace.on_request_start.append(_request_started)
    trace.on_connection_queued_start.append(_queue_started)
    trace.on_connection_queued_end.append(_queue_finished)
    trace.on_request_exception.append(_queue_finished)
    return trace


def prune_forwarding_metrics(endpoints, *, now=None):
    current = {
        (ep.workspace or "", ep.endpoint or "", ep.url, ep.pod_name or "")
        for ep in endpoints
    }
    now = time.monotonic() if now is None else now
    for key, (last_seen, waiting) in list(_series.items()):
        if waiting or key in current or now - last_seen < 300:
            continue
        connection_pool_waiting.remove(*key)
        del _series[key]
