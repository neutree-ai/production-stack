import asyncio
import importlib
import io
import json
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import FormData, UploadFile

from vllm_router.routers.routing_logic import PrefixAwareRouter, RoundRobinRouter
from vllm_router.services.metrics_service.forwarding import (
    connection_pool_waiting,
    create_trace_config,
)
from vllm_router.stats.request_stats import RequestStatsMonitor, SingletonMeta
from vllm_router.utils import SingletonABCMeta, is_model_healthy

proxy = importlib.import_module("vllm_router.services.request_service.request")


@pytest.fixture
def app(monkeypatch):
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    SingletonABCMeta._instances.pop(RoundRobinRouter, None)
    result = SimpleNamespace(
        state=SimpleNamespace(
            router=RoundRobinRouter(),
            request_stats_monitor=RequestStatsMonitor(sliding_window_size=10),
            engine_stats_scraper=None,
            semantic_cache_available=False,
            endpoints=[],
        )
    )
    monkeypatch.setattr(
        proxy,
        "get_service_discovery",
        lambda: SimpleNamespace(get_endpoint_info=lambda: result.state.endpoints),
    )
    monkeypatch.setattr(
        proxy, "get_routing_logic_by_type", lambda _: result.state.router
    )
    yield result
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    SingletonABCMeta._instances.pop(RoundRobinRouter, None)


def endpoint(url):
    return SimpleNamespace(
        url=url,
        model_names=["m"],
        workspace="ws",
        endpoint="ep",
        sleep=False,
        routing_logic=None,
        Id="engine",
        pod_name="inference-0",
    )


def request_for(app, body=b"", headers=None):
    headers = headers or {"content-type": "application/json"}
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "headers": [
                (key.encode(), value.encode()) for key, value in headers.items()
            ],
            "app": app,
        }
    )
    request._body = body
    return request


async def audio_request(app, fields=(), headers=None):
    form = aiohttp.FormData([("model", "m"), *fields])
    form.add_field("file", b"RIFF", filename="clip.wav", content_type="audio/wav")
    payload = form()
    return request_for(
        app,
        await payload.as_bytes(),
        {"content-type": payload.content_type, **(headers or {})},
    )


async def response_body(response):
    if isinstance(response, StreamingResponse):
        return b"".join([chunk async for chunk in response.body_iterator])
    return response.body


@pytest.mark.asyncio
async def test_json_input_api_preserves_payload_and_scoped_selection(app):
    received = []

    async def backend(request):
        received.append(await request.json())
        return web.Response(status=422, text="invalid input")

    backend_app = web.Application()
    backend_app.router.add_post("/pooling", backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        app.state.endpoints = [endpoint(str(server.make_url("/")).rstrip("/"))]
        app.state.aiohttp_client_wrapper = lambda: session
        app.state.router = PrefixAwareRouter()
        payload = {"model": "m", "input": ["hello"], "encoding_format": "float"}
        body = json.dumps(payload).encode()
        response = await proxy.route_general_request(
            request_for(app, body), "/pooling", None, "ws", "ep"
        )
        assert response.status_code == 422
        assert await response_body(response) == b"invalid input"
        assert response.headers["content-type"].startswith("text/plain")
        denied = await proxy.route_general_request(
            request_for(app, body), "/pooling", None, "other", "ep"
        )
        assert denied.status_code == 404
        assert received == [payload]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["json", "multipart"])
async def test_forwarding_metrics_identify_the_selected_inference_replica(
    app, protocol
):
    async def backend(request):
        await request.read()
        return web.Response(text="result")

    backend_app = web.Application()
    backend_app.router.add_post("/{path:.*}", backend)
    async with (
        TestServer(backend_app) as server,
        aiohttp.ClientSession(trace_configs=[create_trace_config()]) as session,
    ):
        base = str(server.make_url("/")).rstrip("/")
        app.state.endpoints = [endpoint(base)]
        app.state.aiohttp_client_wrapper = lambda: session
        if protocol == "multipart":
            response = await proxy.route_general_transcriptions(
                await audio_request(app), "/v1/audio/transcriptions", None, "ws", "ep"
            )
        else:
            response = await proxy.route_general_request(
                request_for(app, b'{"model":"m"}'), "/pooling", None, "ws", "ep"
            )
        assert response.status_code == 200
        assert await response_body(response) == b"result"
        assert [
            sample.value
            for metric in connection_pool_waiting.collect()
            for sample in metric.samples
            if sample.labels
            == {
                "workspace": "ws",
                "endpoint": "ep",
                "server": base,
                "replica": "inference-0",
            }
        ] == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"{", b"[]", b'{"model":""}'])
async def test_new_json_routes_reject_invalid_model_payload(app, body):
    response = await proxy.route_general_request(
        request_for(app, body), "/pooling", None, "ws", "ep"
    )
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,async_router",
    [
        ("/v1/audio/transcriptions", False),
        ("/v1/audio/translations", True),
    ],
)
async def test_audio_fields_raw_translation_and_text_error(app, path, async_router):
    if async_router:
        app.state.router = PrefixAwareRouter()
    route = (
        proxy.route_multipart_request
        if path.endswith("translations")
        else proxy.route_general_transcriptions
    )
    received = []

    async def backend(request):
        fields = await request.post()
        received.append(
            {
                "model": fields["model"],
                "timestamps": fields.getall("timestamp_granularities[]", []),
                "seed": fields.get("seed"),
                "prompt": fields.get("prompt"),
                "response_format": fields.get("response_format"),
                "temperature": fields.get("temperature"),
                "file": fields["file"].file.read(),
                "filename": fields["file"].filename,
                "language": fields.get("language"),
                "authorization": request.headers.get("authorization"),
            }
        )
        return web.Response(status=422, text="upstream text error")

    backend_app = web.Application()
    backend_app.router.add_post(path, backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        app.state.endpoints = [endpoint(str(server.make_url("/")).rstrip("/"))]
        app.state.aiohttp_client_wrapper = lambda: session
        fields = [
            ("prompt", "heard before"),
            ("response_format", "text"),
            ("temperature", "0.25"),
            ("seed", "7"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
        response = await route(
            await audio_request(app, fields, {"authorization": "Bearer client"}),
            path,
            None,
            "ws",
            "ep",
        )
        assert response.status_code == 422
        assert await response_body(response) == b"upstream text error"
        assert response.headers["content-type"].startswith("text/plain")
        denied = await route(
            await audio_request(app, fields), path, None, "other", "ep"
        )
        assert denied.status_code == 404
        assert received == [
            {
                "model": "m",
                "timestamps": (
                    ["word", "segment"] if path.endswith("translations") else []
                ),
                "seed": "7" if path.endswith("translations") else None,
                "prompt": "heard before",
                "response_format": "text",
                "temperature": "0.25",
                "file": b"RIFF",
                "filename": "clip.wav",
                "language": None if path.endswith("translations") else "en",
                "authorization": "Bearer client",
            }
        ]
        assert (
            app.state.request_stats_monitor.get_active_request_count(
                app.state.endpoints[0].url
            )
            == 0
        )


@pytest.mark.asyncio
async def test_audio_rejects_invalid_temperature_and_closes_upload(app):
    upload = UploadFile(io.BytesIO(b"RIFF"), filename="clip.wav")
    request = request_for(app)
    request._form = FormData([("file", upload), ("model", "m"), ("temperature", "nan")])
    response = await proxy.route_general_transcriptions(
        request, "/v1/audio/transcriptions", None
    )
    assert response.status_code == 400
    assert upload.file.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type,path",
    [
        ("pooling", "/pooling"),
        ("classification", "/classify"),
        ("translation", "/v1/audio/translations"),
    ],
)
async def test_static_health_probes_send_valid_input_or_audio(model_type, path):
    received = []

    async def backend(request):
        if model_type == "translation":
            fields = await request.post()
            assert fields["model"] == "m"
            assert fields["file"].file.read().startswith(b"RIFF")
        else:
            assert await request.json() == {"model": "m", "input": "Hello"}
        received.append(request.path)
        return web.json_response({"ok": True})

    backend_app = web.Application()
    backend_app.router.add_post(path, backend)
    async with TestServer(backend_app) as server:
        base = str(server.make_url("/")).rstrip("/")
        assert await asyncio.to_thread(is_model_healthy, base, "m", model_type)
    assert received == [path]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream,status", [(False, 200), (True, 200), (False, 503)])
async def test_audio_preserves_json_or_sse(app, stream, status):
    async def backend(request):
        fields = await request.post()
        assert fields.get("stream") == ("true" if stream else None)
        if stream:
            return web.Response(
                text="data: hello\n\n", content_type="text/event-stream", status=status
            )
        return web.json_response({"text": "hello"}, status=status)

    backend_app = web.Application()
    backend_app.router.add_post("/v1/audio/transcriptions", backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        app.state.endpoints = [endpoint(str(server.make_url("/")).rstrip("/"))]
        app.state.aiohttp_client_wrapper = lambda: session
        request = await audio_request(app, [("stream", "true")] if stream else [])
        response = await proxy.route_general_transcriptions(
            request, "/v1/audio/transcriptions", None, "ws", "ep"
        )
        assert response.status_code == status
        assert await response_body(response) == (
            b"data: hello\n\n" if stream else b'{"text":"hello"}'
        )
        assert response.headers["content-type"].startswith(
            "text/event-stream" if stream else "application/json"
        )
        assert (
            app.state.request_stats_monitor.get_active_request_count(
                app.state.endpoints[0].url
            )
            == 0
        )


@pytest.mark.asyncio
async def test_cancel_audio_while_waiting_for_response_clears_running_stats(app):
    arrived = asyncio.Event()
    release = asyncio.Event()

    async def backend(request):
        await request.read()
        arrived.set()
        await release.wait()
        return web.json_response({"text": "hello"})

    path = "/v1/audio/translations"
    backend_app = web.Application()
    backend_app.router.add_post(path, backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        url = str(server.make_url("/")).rstrip("/")
        app.state.endpoints = [endpoint(url)]
        app.state.aiohttp_client_wrapper = lambda: session
        request = await audio_request(app)
        task = asyncio.create_task(
            proxy.route_multipart_request(request, path, None, "ws", "ep")
        )
        try:
            await asyncio.wait_for(arrived.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

        monitor = app.state.request_stats_monitor
        assert monitor.get_active_request_count(url) == 0
        assert monitor.in_prefill_requests[url] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/pooling", "/v1/audio/transcriptions"])
async def test_native_read_timeout_before_response_returns_504(app, path):
    async def backend(request):
        await request.read()
        await asyncio.sleep(0.05)
        return web.json_response({"ok": True})

    backend_app = web.Application()
    backend_app.router.add_post(path, backend)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=0.01)
    async with (
        TestServer(backend_app) as server,
        aiohttp.ClientSession(timeout=timeout) as session,
    ):
        app.state.endpoints = [endpoint(str(server.make_url("/")).rstrip("/"))]
        app.state.aiohttp_client_wrapper = lambda: session
        if path == "/pooling":
            response = await proxy.route_general_request(
                request_for(app, b'{"model":"m","input":"hello"}'),
                path,
                None,
                "ws",
                "ep",
            )
        else:
            response = await proxy.route_general_transcriptions(
                await audio_request(app), path, None, "ws", "ep"
            )
        assert response.status_code == 504
        assert (
            app.state.request_stats_monitor.get_active_request_count(
                app.state.endpoints[0].url
            )
            == 0
        )
