import asyncio
import importlib
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException
from starlette.requests import Request

from vllm_router.services.metrics_service import (
    clear_endpoint_metrics,
    current_qps,
    num_incoming_requests_total,
    num_requests_waiting,
)
from vllm_router.services.request_service import request as request_service
from vllm_router.services.request_service.request import (
    process_request,
    route_general_request,
)
from vllm_router.stats.request_stats import RequestStatsMonitor, SingletonMeta


@pytest.fixture
def monitor():
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    stats_monitor = RequestStatsMonitor(sliding_window_size=10)
    yield stats_monitor
    SingletonMeta._instances.pop(RequestStatsMonitor, None)


def test_request_completed_before_first_token_releases_prefill_state(monitor):
    engine_url = "http://engine"
    request_id = "prefill-request"

    monitor.on_new_request(engine_url, request_id, timestamp=100)
    monitor.on_request_complete(engine_url, request_id, timestamp=101)

    stats = monitor.get_request_stats(current_time=101)[engine_url]
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.active_requests == 0
    assert (engine_url, request_id) not in monitor.request_start_time
    assert (engine_url, request_id) not in monitor.first_token_time


def test_request_completed_after_first_token_releases_decoding_state(monitor):
    engine_url = "http://engine"
    request_id = "decoding-request"

    monitor.on_new_request(engine_url, request_id, timestamp=100)
    monitor.on_request_response(engine_url, request_id, timestamp=101)
    monitor.on_request_complete(engine_url, request_id, timestamp=102)

    stats = monitor.get_request_stats(current_time=102)[engine_url]
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.active_requests == 0
    assert (engine_url, request_id) not in monitor.request_start_time
    assert (engine_url, request_id) not in monitor.first_token_time


def test_unsuccessful_completion_does_not_increment_finished_requests(monitor):
    engine_url = "http://engine"
    monitor.on_new_request(engine_url, "failed", timestamp=100)
    monitor.on_request_complete(engine_url, "failed", timestamp=101, success=False)
    monitor.on_new_request(engine_url, "successful", timestamp=102)
    monitor.on_request_complete(engine_url, "successful", timestamp=103)

    stats = monitor.get_request_stats(current_time=103)[engine_url]
    assert stats.finished_requests == 1
    assert stats.in_prefill_requests == 0
    assert stats.active_requests == 0


class _HangingContent:
    async def iter_any(self):
        yield b"first-token"
        await asyncio.Event().wait()


class _FakeBackendResponse:
    headers = {}
    status = 200
    content = _HangingContent()


class _FakeRequestContext:
    async def __aenter__(self):
        return _FakeBackendResponse()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _FailingRequestContext:
    async def __aenter__(self):
        raise RuntimeError("backend exception detail must not be logged")

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _FakeClient:
    def __init__(self, request_context):
        self.request_context = request_context

    def request(self, **kwargs):
        return self.request_context


def _create_request(monitor, request_context=None):
    app = FastAPI()
    app.state.request_stats_monitor = monitor
    app.state.aiohttp_client_wrapper = lambda: _FakeClient(
        request_context or _FakeRequestContext()
    )
    app.state.semantic_cache_available = False
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "app": app,
        }
    )


def test_invalid_rewritten_json_returns_bad_request(monkeypatch):
    app = FastAPI()
    app.state.router = None
    app.state.semantic_cache_available = False
    request_body = b'{"model": "model"}'

    async def receive():
        return {"type": "http.request", "body": request_body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "app": app,
        },
        receive=receive,
    )
    rewriter = SimpleNamespace(rewrite_request=lambda *args: b"not-json")
    monkeypatch.setattr(
        request_service, "is_request_rewriter_initialized", lambda: True
    )
    monkeypatch.setattr(request_service, "get_request_rewriter", lambda: rewriter)

    async def route_request():
        with pytest.raises(HTTPException) as exc_info:
            await route_general_request(
                request, "/v1/chat/completions", BackgroundTasks()
            )
        assert exc_info.value.status_code == 400

    asyncio.run(route_request())


def test_closing_stream_before_first_token_releases_request_stats(monitor):
    request = _create_request(monitor)

    async def close_stream():
        generator = process_request(
            request=request,
            body=b'{"stream": true}',
            backend_url="http://engine",
            request_id="cancelled-prefill-request",
            endpoint="/v1/chat/completions",
            background_tasks=None,
        )
        await anext(generator)
        await generator.aclose()

    asyncio.run(close_stream())

    stats = monitor.get_request_stats(current_time=102)["http://engine"]
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.active_requests == 0


def test_closing_stream_after_first_token_releases_request_stats(monitor):
    request = _create_request(monitor)

    async def close_stream():
        generator = process_request(
            request=request,
            body=b'{"stream": true}',
            backend_url="http://engine",
            request_id="cancelled-request",
            endpoint="/v1/chat/completions",
            background_tasks=None,
        )
        await anext(generator)
        await anext(generator)
        await generator.aclose()

    asyncio.run(close_stream())

    stats = monitor.get_request_stats(current_time=102)["http://engine"]
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.active_requests == 0


def test_cancelling_stream_releases_request_stats(monitor):
    request = _create_request(monitor)

    async def cancel_stream():
        generator = process_request(
            request=request,
            body=b'{"stream": true}',
            backend_url="http://engine",
            request_id="cancelled-task",
            endpoint="/v1/chat/completions",
            background_tasks=None,
        )
        await anext(generator)
        await anext(generator)
        pending_chunk = asyncio.create_task(anext(generator))
        await asyncio.sleep(0)
        pending_chunk.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_chunk

    asyncio.run(cancel_stream())

    stats = monitor.get_request_stats(current_time=102)["http://engine"]
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.active_requests == 0
    assert stats.finished_requests == 0


def test_backend_error_releases_stats_and_logs_safe_terminal_fields(
    monitor, monkeypatch
):
    request = _create_request(monitor, _FailingRequestContext())
    logged_messages = []

    def record_warning(message, *args):
        logged_messages.append(message % args)

    monkeypatch.setattr(request_service.logger, "warning", record_warning)

    async def send_request():
        generator = process_request(
            request=request,
            body=b'{"stream": true}',
            backend_url="http://engine",
            request_id="backend-error-request",
            endpoint="/v1/chat/completions",
            background_tasks=None,
        )
        with pytest.raises(RuntimeError, match="backend exception"):
            await anext(generator)

    asyncio.run(send_request())

    stats = monitor.get_request_stats(current_time=102)["http://engine"]
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.active_requests == 0
    assert stats.finished_requests == 0
    assert len(logged_messages) == 1
    log_output = logged_messages[0]
    assert "request_id=backend-error-request" in log_output
    assert "outcome=backend_error" in log_output
    assert "stage=prefill" in log_output
    assert "exception_type=RuntimeError" in log_output
    assert "backend exception detail" not in log_output


def test_clear_endpoint_metrics_removes_stale_labels_without_resetting_counter():
    clear_endpoint_metrics()
    labels = {"workspace": "workspace", "endpoint": "endpoint", "server": "stale"}
    current_qps.labels(**labels).set(1)
    num_requests_waiting.labels(**labels).set(2)
    num_incoming_requests_total.labels(workspace="workspace", endpoint="endpoint").inc()

    clear_endpoint_metrics()

    assert not current_qps._metrics
    assert not num_requests_waiting._metrics
    assert num_incoming_requests_total._metrics


def test_metrics_refresh_removes_labels_for_removed_endpoint(monkeypatch):
    metrics_module = importlib.import_module("vllm_router.routers.metrics_router")
    old_endpoint = SimpleNamespace(
        url="http://stale", workspace="workspace", endpoint="endpoint", healthy=True
    )
    current_endpoint = SimpleNamespace(
        url="http://current", workspace="workspace", endpoint="endpoint", healthy=True
    )
    endpoints = [old_endpoint]
    request_stat = SimpleNamespace(
        qps=1,
        avg_decoding_length=2,
        in_prefill_requests=3,
        in_decoding_requests=4,
        avg_latency=5,
        avg_itl=6,
        num_swapped_requests=7,
    )
    discovery = SimpleNamespace(get_endpoint_info=lambda: endpoints)
    monitor = SimpleNamespace(
        get_request_stats=lambda _: {
            endpoint.url: request_stat for endpoint in endpoints
        }
    )

    monkeypatch.setattr(metrics_module, "get_service_discovery", lambda: discovery)
    monkeypatch.setattr(metrics_module, "get_request_stats_monitor", lambda: monitor)
    monkeypatch.setattr(metrics_module, "get_engine_stats_scraper", lambda: None)
    monkeypatch.setattr(metrics_module.psutil, "cpu_percent", lambda interval: 0)
    monkeypatch.setattr(
        metrics_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=0)
    )
    monkeypatch.setattr(
        metrics_module.psutil, "disk_usage", lambda _: SimpleNamespace(percent=0)
    )

    asyncio.run(metrics_module.metrics())
    endpoints[:] = [current_endpoint]
    asyncio.run(metrics_module.metrics())

    assert ("workspace", "endpoint", "http://stale") not in current_qps._metrics
    assert ("workspace", "endpoint", "http://current") in current_qps._metrics
