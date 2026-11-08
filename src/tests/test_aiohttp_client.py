import asyncio

import aiohttp
import pytest
from aiohttp import web

from vllm_router.aiohttp_client import AiohttpClientWrapper


@pytest.mark.asyncio
async def test_start_configures_global_and_per_host_connection_limits() -> None:
    wrapper = AiohttpClientWrapper()
    timeout = aiohttp.ClientTimeout(
        total=None, connect=60, sock_connect=10, sock_read=600
    )
    wrapper.start(timeout=timeout)
    client = wrapper()
    connector = client.connector

    try:
        assert connector.limit == 0
        assert connector.limit_per_host == 100
        assert client.timeout is timeout
    finally:
        await wrapper.stop()

    assert client.closed
    assert connector.closed
    assert wrapper.async_client is None


@pytest.fixture
async def backend_url():
    async def handle(request):
        if request.path == "/headers":
            await asyncio.sleep(0.12)
            return web.Response(text="done")
        response = web.StreamResponse()
        await response.prepare(request)
        try:
            for _ in range(6):
                await response.write(b"chunk")
                await asyncio.sleep(0.12 if request.path == "/body" else 0.02)
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    app = web.Application()
    app.router.add_get("/{phase}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_read_idle_timeout_covers_headers_and_body(backend_url, phase):
    wrapper = AiohttpClientWrapper()
    wrapper.start(timeout=aiohttp.ClientTimeout(total=None, sock_read=0.04))
    try:
        with pytest.raises(aiohttp.SocketTimeoutError):
            async with wrapper().get(f"{backend_url}/{phase}") as response:
                await response.read()
    finally:
        await wrapper.stop()


async def test_active_stream_can_outlast_connect_and_read_budgets(backend_url):
    wrapper = AiohttpClientWrapper()
    wrapper.start(
        timeout=aiohttp.ClientTimeout(total=None, connect=0.04, sock_read=0.05)
    )
    try:
        async with wrapper().get(f"{backend_url}/stream") as response:
            chunks = [chunk async for chunk in response.content.iter_any()]
        assert b"".join(chunks) == b"chunk" * 6
    finally:
        await wrapper.stop()


async def test_connect_timeout_also_bounds_connection_pool_wait(backend_url):
    wrapper = AiohttpClientWrapper()
    wrapper.start(
        timeout=aiohttp.ClientTimeout(total=None, connect=0.04),
        max_connections_per_backend=1,
    )
    try:
        async with wrapper().get(f"{backend_url}/body"):
            with pytest.raises(aiohttp.ConnectionTimeoutError):
                await wrapper().get(f"{backend_url}/stream")
    finally:
        await wrapper.stop()


async def test_lifespan_starts_shared_client_before_discovery_and_closes_it_last(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from fastapi import FastAPI

    import vllm_router.app as app_module

    app = FastAPI()
    app.state.aiohttp_client_wrapper = AiohttpClientWrapper()
    app.state.aiohttp_timeout = aiohttp.ClientTimeout(total=None, sock_read=0.5)
    app.state.max_connections_per_backend = 0
    app.state.engine_stats_scraper = None
    calls = []

    def set_event_loop(loop):
        assert loop is asyncio.get_running_loop() is app.state.event_loop
        calls.append("loop")

    async def initialize_client_sessions():
        assert not app.state.aiohttp_client_wrapper().closed
        calls.append("initialize")

    def close():
        assert not app.state.aiohttp_client_wrapper().closed
        calls.append("close")

    discovery = SimpleNamespace(
        set_event_loop=set_event_loop,
        initialize_client_sessions=initialize_client_sessions,
        close=close,
    )
    monkeypatch.setattr(app_module, "get_service_discovery", lambda: discovery)
    monkeypatch.setattr(app_module, "get_dynamic_config_watcher", lambda: None)
    monkeypatch.setattr(app_module, "initialize_cpu_sampling", Mock())
    monkeypatch.setattr(app_module, "cleanup_routing_logic", Mock())
    async with app_module.lifespan(app):
        client = app.state.aiohttp_client_wrapper()
        assert client.connector.limit_per_host == 0
        assert client.timeout is app.state.aiohttp_timeout
    assert calls == ["loop", "initialize", "close"]
    assert client.closed
