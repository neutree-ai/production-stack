import asyncio
import time

import pytest
from aiohttp import web
from prometheus_client import REGISTRY

from tests.test_backend_http import serve, wrapper
from vllm_router.backend_http import BackendTimeout
from vllm_router.services.metrics_service.forwarding import (
    observe_forwarding,
    prune_forwarding_metrics,
)


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels)


@pytest.mark.asyncio
async def test_forwarding_metrics_balance_pool_timeout_and_completion():
    async def handler(request):
        return web.Response(status=422, text="invalid input")

    async with (
        serve(handler) as url,
        wrapper(
            max_connections_per_backend=1, pool_wait_timeout_seconds=0.02
        ) as client,
    ):
        client.observer = observe_forwarding
        labels = dict(workspace="metrics-ws", endpoint="metrics-ep", server=url)
        async with client.forward(
            url,
            method="GET",
            url=url,
            workspace=labels["workspace"],
            endpoint=labels["endpoint"],
        ) as first:
            assert sample("router:backend_requests_in_flight", **labels) == 1
            assert sample("router:connections_in_use", **labels) == 1
            with pytest.raises(BackendTimeout):
                async with client.forward(
                    url,
                    method="GET",
                    url=url,
                    workspace=labels["workspace"],
                    endpoint=labels["endpoint"],
                ):
                    pass
            assert sample("router:connection_pool_waiting", **labels) == 0
            assert (
                sample(
                    "router:backend_requests_total",
                    **labels,
                    outcome="timeout",
                    phase="pool_wait",
                    status_class="none",
                )
                == 1
            )
            assert await first.read() == b"invalid input"
        for name in [
            "router:backend_requests_in_flight",
            "router:connections_in_use",
            "router:connection_pool_waiting",
        ]:
            assert sample(name, **labels) == 0
        assert (
            sample(
                "router:backend_requests_total",
                **labels,
                outcome="completed",
                phase="response_read",
                status_class="4xx",
            )
            == 1
        )
        assert sample("router:backend_response_headers_seconds_count", **labels) == 1
        prune_forwarding_metrics([], now=time.monotonic() + 301)
        assert sample("router:backend_requests_in_flight", **labels) is None
        assert sample("router:backend_response_headers_seconds_count", **labels) is None


@pytest.mark.asyncio
async def test_cancelled_request_has_one_terminal_and_active_metrics_are_not_pruned():
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        entered.set()
        await release.wait()
        return web.Response(text="ok")

    async with serve(handler) as url, wrapper() as client:
        client.observer = observe_forwarding
        labels = dict(workspace="cancel-ws", endpoint="cancel-ep", server=url)

        async def send():
            async with client.forward(
                url,
                method="GET",
                url=url,
                workspace=labels["workspace"],
                endpoint=labels["endpoint"],
            ) as response:
                await response.read()

        task = asyncio.create_task(send())
        await asyncio.wait_for(entered.wait(), 1)
        prune_forwarding_metrics([], now=time.monotonic() + 301)
        assert sample("router:backend_requests_in_flight", **labels) == 1
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        assert sample("router:backend_requests_in_flight", **labels) == 0
        assert sample("router:connections_in_use", **labels) == 0
        assert (
            sample(
                "router:backend_requests_total",
                **labels,
                outcome="cancelled",
                phase="response_header",
                status_class="none",
            )
            == 1
        )
        prune_forwarding_metrics([], now=time.monotonic() + 301)


@pytest.mark.asyncio
async def test_header_timeout_also_observes_header_wait_duration():
    async def handler(request):
        await asyncio.sleep(0.06)
        return web.Response(text="late")

    async with (
        serve(handler) as url,
        wrapper(response_header_timeout_seconds=0.02) as client,
    ):
        client.observer = observe_forwarding
        labels = dict(workspace="header-ws", endpoint="header-ep", server=url)
        with pytest.raises(BackendTimeout):
            async with client.forward(
                url,
                method="GET",
                url=url,
                workspace=labels["workspace"],
                endpoint=labels["endpoint"],
            ):
                pass
        assert sample("router:backend_response_headers_seconds_count", **labels) == 1
        assert sample("router:backend_response_headers_seconds_sum", **labels) >= 0.02
        prune_forwarding_metrics([], now=time.monotonic() + 301)
