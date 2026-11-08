import asyncio
import time

import aiohttp
import pytest
from aiohttp import web

from vllm_router.services.metrics_service.forwarding import (
    connection_pool_waiting,
    create_trace_config,
    prune_forwarding_metrics,
    trace_context,
)


def waiting(server):
    return next(
        (
            sample.value
            for family in connection_pool_waiting.collect()
            for sample in family.samples
            if sample.labels == {"workspace": "ws", "endpoint": "ep", "server": server}
        ),
        None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", ["acquired", "cancelled", "timeout"])
async def test_backend_pool_wait_is_balanced_and_active_series_are_preserved(exit_path):
    # Long inference responses hold the only connection while another call queues.
    release = asyncio.Event()
    queued = asyncio.Event()

    async def inference(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await release.wait()
        await response.write(b"result")
        return response

    app = web.Application()
    app.router.add_get("/", inference)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    trace = create_trace_config()

    async def on_queued(session, context, params):
        queued.set()

    trace.on_connection_queued_start.append(on_queued)
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit_per_host=1),
            timeout=aiohttp.ClientTimeout(total=None, connect=0.2),
            trace_configs=[trace],
        ) as client:
            first = await client.get(
                server, trace_request_ctx=trace_context("ws", "ep", server)
            )
            second = asyncio.create_task(
                client.get(server, trace_request_ctx=trace_context("ws", "ep", server))
            )
            await asyncio.wait_for(queued.wait(), 2)
            assert waiting(server) == 1
            prune_forwarding_metrics([], now=time.monotonic() + 301)
            assert waiting(server) == 1
            if exit_path == "acquired":
                release.set()
                assert await first.read() == b"result"
                response = await second
                assert await response.read() == b"result"
            elif exit_path == "cancelled":
                second.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await second
            else:
                with pytest.raises(aiohttp.ConnectionTimeoutError):
                    await second
            assert waiting(server) == 0
            release.set()
            await first.read()
            prune_forwarding_metrics([], now=time.monotonic() + 301)
            assert waiting(server) is None
            response = await client.get(server)
            await response.read()
            assert waiting(server) is None
    finally:
        release.set()
        await runner.cleanup()
