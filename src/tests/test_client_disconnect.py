import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, FastAPI
from starlette.requests import ClientDisconnect, Request

from vllm_router.services.request_service import request as request_service
from vllm_router.services.request_service.request import (
    _anext_or_disconnect,
    _cancel_and_wait,
    route_general_request,
)


def _request(app, receive):
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/ws/ep/v1/chat/completions",
            "raw_path": b"/ws/ep/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("router", 80),
            "app": app,
        },
        receive=receive,
    )


@pytest.mark.asyncio
async def test_disconnect_before_backend_headers_closes_stream():
    disconnected = asyncio.Event()
    closed = asyncio.Event()
    app = FastAPI()

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def stream():
        try:
            await asyncio.Event().wait()
            yield {}, 200
        finally:
            closed.set()

    task = asyncio.create_task(_anext_or_disconnect(stream(), _request(app, receive)))
    await asyncio.sleep(0)
    disconnected.set()

    with pytest.raises(ClientDisconnect):
        await task
    assert closed.is_set()


@pytest.mark.asyncio
async def test_route_returns_499_and_closes_pending_backend_on_disconnect(monkeypatch):
    app = FastAPI()
    app.state.router = SimpleNamespace(
        extract_session_id=lambda *args: None,
    )
    app.state.engine_stats_scraper = None
    app.state.request_stats_monitor = SimpleNamespace(get_request_stats=lambda _: {})
    app.state.callbacks = None
    app.state.semantic_cache_available = False
    endpoint = SimpleNamespace(
        url="http://engine",
        model_names=["model"],
        sleep=False,
        workspace="ws",
        endpoint="ep",
        routing_logic=None,
        Id="engine-id",
        pod_name="engine-pod",
    )
    discovery = SimpleNamespace(
        aliases=None,
        get_endpoint_info=lambda: [endpoint],
        has_ever_seen_model=lambda _: True,
    )
    monkeypatch.setattr(request_service, "get_service_discovery", lambda: discovery)
    monkeypatch.setattr(
        request_service,
        "get_routing_logic_by_type",
        lambda _: SimpleNamespace(route_request=lambda *args: "http://engine"),
    )

    disconnected = asyncio.Event()
    closed = asyncio.Event()

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    request = _request(app, receive)
    request._body = b'{"model":"model"}'

    async def pending_backend(*args, **kwargs):
        try:
            await asyncio.Event().wait()
            yield {}, 200
        finally:
            closed.set()

    monkeypatch.setattr(request_service, "process_request", pending_backend)
    task = asyncio.create_task(
        route_general_request(
            request, "/v1/chat/completions", BackgroundTasks(), "ws", "ep"
        )
    )
    await asyncio.sleep(0)
    disconnected.set()

    response = await asyncio.wait_for(task, timeout=1)
    assert response.status_code == 499
    assert closed.is_set()


@pytest.mark.asyncio
async def test_cancel_and_wait_preserves_independent_caller_cancellation():
    task = asyncio.create_task(asyncio.sleep(1000))
    caller_cancelled = asyncio.Event()

    async def caller():
        try:
            await _cancel_and_wait(task)
        except asyncio.CancelledError:
            caller_cancelled.set()

    caller_task = asyncio.create_task(caller())
    await asyncio.sleep(0)
    caller_task.cancel()
    await asyncio.gather(caller_task, return_exceptions=True)

    assert caller_cancelled.is_set()
