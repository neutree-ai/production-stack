import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web

from vllm_router.aiohttp_client import AiohttpClientWrapper
from vllm_router.backend_http import BackendHttpConfig, BackendTimeout


@asynccontextmanager
async def serve(handler):
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def wrapper(**kwargs):
    client = AiohttpClientWrapper()
    client.configure(BackendHttpConfig(**kwargs))
    client.start()
    try:
        yield client
    finally:
        await client.stop()


def test_config_defaults_zero_and_explicit_ranges():
    config = BackendHttpConfig()
    assert config.max_connections_per_backend == 100
    assert config.pool_wait_timeout_seconds == 60
    assert config.connect_timeout_seconds == 10
    assert config.response_header_timeout_seconds == 0
    assert config.response_read_timeout_seconds == 600
    assert config.stream_idle_timeout_seconds == 600
    assert BackendHttpConfig.from_args(SimpleNamespace()) == config
    for name in config.__dataclass_fields__:
        assert getattr(BackendHttpConfig(**{name: 0}), name) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_connections_per_backend", -1),
        ("max_connections_per_backend", 65536),
        ("max_connections_per_backend", 1.5),
        ("max_connections_per_backend", True),
        ("connect_timeout_seconds", float("nan")),
        ("connect_timeout_seconds", float("inf")),
        ("response_read_timeout_seconds", -1),
        ("response_header_timeout_seconds", 604801),
        ("stream_idle_timeout_seconds", "30"),
        ("pool_wait_timeout_seconds", False),
    ],
)
def test_invalid_config_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        BackendHttpConfig(**{field: value})


@pytest.mark.asyncio
async def test_pool_wait_is_bounded_without_blocking_another_backend():
    async def handler(request):
        return web.Response(text="ok")

    async with (
        serve(handler) as first,
        serve(handler) as second,
        wrapper(
            max_connections_per_backend=1,
            pool_wait_timeout_seconds=0.03,
        ) as client,
    ):
        async with client.forward(first, method="GET", url=first) as response:
            with pytest.raises(BackendTimeout) as error:
                async with client.forward(first, method="GET", url=first):
                    pass
            assert error.value.phase == "pool_wait"
            async with client.forward(second, method="GET", url=second) as other:
                assert await other.read() == b"ok"
            assert await response.read() == b"ok"
        async with client.forward(first, method="GET", url=first) as response:
            assert await response.read() == b"ok"
        assert not client._pools


@pytest.mark.asyncio
async def test_header_wait_has_its_own_deadline_and_releases_pool():
    async def handler(request):
        await asyncio.sleep(0.06)
        return web.Response(text="ok")

    async with (
        serve(handler) as url,
        wrapper(response_header_timeout_seconds=0.02) as client,
    ):
        with pytest.raises(BackendTimeout) as error:
            async with client.forward(url, method="GET", url=url):
                pass
        assert error.value.phase == "response_header"
        assert not client._pools


@pytest.mark.asyncio
async def test_short_connect_timeout_does_not_limit_header_wait():
    async def handler(request):
        await asyncio.sleep(0.04)
        return web.Response(text="ok")

    async with (
        serve(handler) as url,
        wrapper(
            connect_timeout_seconds=0.01, response_header_timeout_seconds=0
        ) as client,
    ):
        async with client.forward(url, method="GET", url=url) as response:
            assert await response.read() == b"ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream,phase", [(False, "response_read"), (True, "stream_idle")]
)
async def test_body_idle_deadline_starts_after_headers(stream, phase):
    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first")
        await asyncio.sleep(0.08)
        return response

    config = {
        "response_read_timeout_seconds": 0.02,
        "stream_idle_timeout_seconds": 0.02,
    }
    async with serve(handler) as url, wrapper(**config) as client:
        async with client.forward(
            url, method="GET", url=url, stream=stream
        ) as response:
            chunks = response.iter_chunks()
            assert await anext(chunks) == b"first"
            with pytest.raises(BackendTimeout) as error:
                await anext(chunks)
            assert error.value.phase == phase
        assert not client._pools


@pytest.mark.asyncio
async def test_active_stream_can_exceed_all_per_phase_timeouts():
    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        for _ in range(6):
            await response.write(b"x")
            await asyncio.sleep(0.02)
        return response

    async with (
        serve(handler) as url,
        wrapper(
            pool_wait_timeout_seconds=0.03,
            connect_timeout_seconds=0.03,
            response_header_timeout_seconds=0.03,
            response_read_timeout_seconds=0.01,
            stream_idle_timeout_seconds=0.06,
        ) as client,
    ):
        async with client.forward(url, method="GET", url=url, stream=True) as response:
            assert await response.read() == b"xxxxxx"


@pytest.mark.asyncio
async def test_cancelled_pool_waiter_does_not_release_active_slot():
    async def handler(request):
        return web.Response(text="ok")

    async with serve(handler) as url, wrapper(max_connections_per_backend=1) as client:
        async with client.forward(url, method="GET", url=url):

            async def wait():
                async with client.forward(url, method="GET", url=url):
                    pass

            task = asyncio.create_task(wait())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.03):
                    async with client.forward(url, method="GET", url=url):
                        pass
        assert not client._pools


@pytest.mark.asyncio
async def test_runtime_config_change_requires_restart_without_mutation():
    async with wrapper() as client:
        original = client.config
        with pytest.raises(ValueError, match="restart"):
            client.configure(BackendHttpConfig(max_connections_per_backend=2))
        assert client.config == original


@pytest.mark.asyncio
async def test_dns_and_socket_connect_phase_is_bounded(monkeypatch):
    import aiohttp

    class SlowResolver(aiohttp.abc.AbstractResolver):
        async def resolve(self, *args, **kwargs):
            await asyncio.sleep(1)
            return []

        async def close(self):
            pass

    original = aiohttp.TCPConnector
    monkeypatch.setattr(
        aiohttp,
        "TCPConnector",
        lambda **kwargs: original(resolver=SlowResolver(), **kwargs),
    )
    async with wrapper(connect_timeout_seconds=0.02) as client:
        with pytest.raises(BackendTimeout) as error:
            async with client.forward(
                "http://test.invalid", method="GET", url="http://test.invalid"
            ):
                pass
        assert error.value.phase == "connect"
        assert not client._pools


@pytest.mark.asyncio
async def test_zero_read_timeout_allows_slow_body_and_lifecycle_is_balanced():
    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await asyncio.sleep(0.04)
        await response.write(b"ok")
        return response

    async with (
        serve(handler) as url,
        wrapper(
            max_connections_per_backend=0, response_read_timeout_seconds=0
        ) as client,
    ):
        observed = []
        client.observer = lambda event, context: observed.append(
            (event, context.outcome, context.phase)
        )
        async with client.forward(url, method="GET", url=url) as response:
            assert await response.read() == b"ok"
        events = [event for event, _, _ in observed]
        assert (
            events.count("connection_acquired")
            == events.count("connection_released")
            == 1
        )
        assert observed[-1][0:2] == ("request_finished", "completed")
        assert not client._pools


def test_cli_json_config_and_dynamic_rejection_keep_last_valid(tmp_path, monkeypatch):
    import json
    import sys
    from unittest.mock import Mock

    from vllm_router.dynamic_config import DynamicConfigWatcher, DynamicRouterConfig
    from vllm_router.parsers.parser import parse_args

    path = tmp_path / "router.json"
    raw = {
        "service_discovery": "static",
        "routing_logic": "default",
        "static_backends": "http://engine:8000",
        "static_models": "m",
        "max_connections_per_backend": 7,
        "pool_wait_timeout_seconds": 0.25,
        "connect_timeout_seconds": 0,
        "response_header_timeout_seconds": 0,
        "response_read_timeout_seconds": 0,
        "stream_idle_timeout_seconds": 0.5,
    }
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(sys, "argv", ["router", "--dynamic-config-json", str(path)])
    args = parse_args()
    assert BackendHttpConfig.from_args(args).max_connections_per_backend == 7
    assert BackendHttpConfig.from_args(
        DynamicRouterConfig.from_args(args)
    ) == BackendHttpConfig.from_args(args)
    config = DynamicRouterConfig.from_json(path)
    assert config.pool_wait_timeout_seconds == 0.25
    raw["response_read_timeout_seconds"] = -1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="response_read_timeout_seconds"):
        DynamicRouterConfig.from_json(path)
    assert config.response_read_timeout_seconds == 0
    watcher = object.__new__(DynamicConfigWatcher)
    client = AiohttpClientWrapper()
    client.async_client = SimpleNamespace()
    original = client.config
    watcher.app = SimpleNamespace(state=SimpleNamespace(aiohttp_client_wrapper=client))
    watcher.reconfigure_service_discovery = Mock()
    with pytest.raises(ValueError, match="restart"):
        watcher.reconfigure_all(config)
    watcher.reconfigure_service_discovery.assert_not_called()
    assert client.config == original


@pytest.mark.asyncio
async def test_headers_timer_does_not_limit_slow_body_after_headers():
    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await asyncio.sleep(0.06)
        await response.write(b"ok")
        return response

    async with (
        serve(handler) as url,
        wrapper(
            response_header_timeout_seconds=0.02, response_read_timeout_seconds=0
        ) as client,
    ):
        async with client.forward(url, method="GET", url=url) as response:
            assert await response.read() == b"ok"


@pytest.mark.asyncio
async def test_legacy_raw_request_cannot_create_unbounded_internal_pool_wait():
    release = asyncio.Event()

    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first")
        await release.wait()
        return response

    async with (
        serve(handler) as url,
        wrapper(
            max_connections_per_backend=1, pool_wait_timeout_seconds=0.02
        ) as client,
    ):
        events = []
        client.observer = lambda event, context: events.append(event)
        raw = await client().get(url)
        try:
            with pytest.raises(BackendTimeout) as error:
                async with client.forward(url, method="GET", url=url):
                    pass
            assert error.value.phase == "pool_wait"
            assert (
                events.count("pool_wait_started")
                == events.count("pool_wait_finished")
                == 2
            )
            assert "connection_acquired" not in events
            assert "connection_released" not in events
            assert not client._pools
        finally:
            release.set()
            raw.close()


def test_initial_json_rejects_invalid_backend_value_before_argparse_coercion(
    tmp_path, monkeypatch
):
    import json
    import sys

    from vllm_router.parsers.parser import parse_args

    path = tmp_path / "router.json"
    path.write_text(json.dumps({"connect_timeout_seconds": "10"}))
    monkeypatch.setattr(sys, "argv", ["router", "--dynamic-config-json", str(path)])
    with pytest.raises(ValueError, match="connect_timeout_seconds"):
        parse_args()


@pytest.mark.parametrize("format", ["json", "yaml"])
def test_initial_file_null_means_default_but_zero_is_preserved(
    tmp_path, monkeypatch, format
):
    import json
    import sys

    import yaml

    from vllm_router.dynamic_config import DynamicRouterConfig
    from vllm_router.parsers.parser import parse_args

    raw = {
        "service_discovery": "static",
        "routing_logic": "default",
        "static_models": "m",
        "static_backends": "http://engine:8000",
        "max_connections_per_backend": None,
        "connect_timeout_seconds": None,
        "response_read_timeout_seconds": 0,
    }
    if format == "yaml":
        raw["static_models"] = {"m": {"static_backends": ["http://engine:8000"]}}
        raw.pop("static_backends")
    path = tmp_path / ("router." + format)
    path.write_text(json.dumps(raw) if format == "json" else yaml.safe_dump(raw))
    monkeypatch.setattr(
        sys, "argv", ["router", "--dynamic-config-" + format, str(path)]
    )
    args = parse_args()
    config = BackendHttpConfig.from_args(args)
    assert config.max_connections_per_backend == 100
    assert config.connect_timeout_seconds == 10
    assert config.response_read_timeout_seconds == 0
    loaded = (
        DynamicRouterConfig.from_json(path)
        if format == "json"
        else DynamicRouterConfig.from_yaml(path)
    )
    assert BackendHttpConfig.from_args(loaded) == config


@pytest.mark.asyncio
@pytest.mark.parametrize("header_delay,should_timeout", [(0, False), (0.06, True)])
async def test_response_header_budget_starts_after_async_upload_finishes(
    header_delay, should_timeout
):
    async def handler(request):
        assert await request.read() == b"firstlast"
        await asyncio.sleep(header_delay)
        return web.Response(text="ok")

    async def body():
        yield b"first"
        await asyncio.sleep(0.06)
        yield b"last"

    async with (
        serve(handler) as url,
        wrapper(response_header_timeout_seconds=0.02) as client,
    ):
        events = []
        client.observer = lambda event, context: events.append(
            (event, context.header_wait_elapsed)
        )
        if should_timeout:
            with pytest.raises(BackendTimeout) as error:
                async with client.forward(url, method="POST", url=url, data=body()):
                    pass
            assert error.value.phase == "response_header"
        else:
            async with client.forward(
                url, method="POST", url=url, data=body()
            ) as response:
                assert await response.read() == b"ok"
            elapsed = next(
                elapsed for event, elapsed in events if event == "headers_received"
            )
            assert elapsed < 0.04
        assert not client._pools


@pytest.mark.asyncio
async def test_early_response_does_not_rearm_header_timer_during_payload_cleanup():
    import aiohttp

    release = asyncio.Event()
    closed = []
    written = asyncio.Event()

    class LateBody(aiohttp.payload.Payload):
        def decode(self, encoding="utf-8", errors="strict"):
            return "firstlast"

        async def write(self, writer):
            try:
                await writer.write(b"first")
                await release.wait()
                await writer.write(b"last")
            finally:
                written.set()

        async def close(self):
            closed.append(True)
            release.set()
            await asyncio.sleep(0.05)

    async def handler(request):
        return web.Response(status=413, text="too large")

    async with (
        serve(handler) as url,
        wrapper(response_header_timeout_seconds=0.01) as client,
    ):
        events = []
        client.observer = lambda event, context: events.append(event)
        async with client.forward(
            url, method="POST", url=url, data=LateBody(None)
        ) as response:
            assert response.status == 413
            assert await response.read() == b"too large"
        assert closed == [True]
        await asyncio.wait_for(written.wait(), 1)
        assert events.count("request_finished") == 1
        assert (
            events.count("connection_acquired")
            == events.count("connection_released")
            == 1
        )
        assert not client._pools


@pytest.mark.asyncio
async def test_late_payload_completion_cannot_rearm_finished_request_deadline():
    import aiohttp

    from vllm_router.aiohttp_client import _RequestBody
    from vllm_router.backend_http import ForwardContext

    class Writer:
        async def write(self, data):
            pass

        async def write_eof(self):
            pass

    async with asyncio.timeout(None) as deadline:
        state = SimpleNamespace(
            context=ForwardContext("http://engine"),
            deadline=deadline,
            active=True,
            headers_received=False,
        )
        payload = _RequestBody(
            aiohttp.payload.BytesPayload(b"body"),
            state,
            BackendHttpConfig(response_header_timeout_seconds=0.01),
        )
        await payload.close()
        state.active = False
        await payload.write(Writer())
        assert deadline.when() is None
        assert state.context.header_wait_started is None
