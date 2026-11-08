import asyncio
import gzip
import importlib
import io
import json
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import BackgroundTasks, FastAPI, Request
from starlette.datastructures import FormData, Headers, UploadFile

from vllm_router.routers.main_router import main_router
from vllm_router.routers.routing_logic import (
    DisaggregatedPrefillRouter,
    PrefixAwareRouter,
    RoundRobinRouter,
)
from vllm_router.stats.request_stats import RequestStatsMonitor
from vllm_router.utils import SingletonMeta, is_model_healthy

proxy = importlib.import_module("vllm_router.services.request_service.request")


@pytest.fixture
def app(monkeypatch):
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    monitor = RequestStatsMonitor(sliding_window_size=10)
    result = FastAPI()
    result.include_router(main_router)
    result.state.router = RoundRobinRouter()
    result.state.request_stats_monitor = monitor
    result.state.engine_stats_scraper = None
    result.state.semantic_cache_available = False
    result.state.endpoints = []
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


def endpoint(url):
    return SimpleNamespace(
        url=url,
        model_names=["m"],
        workspace="ws",
        endpoint="ep",
        sleep=False,
        routing_logic=None,
        Id="engine",
    )


def request_for(app, *, body=b'{"model":"m","stream":true}', disconnect=None):
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await (disconnect or asyncio.Event()).wait()
        return {"type": "http.disconnect"}

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/ws/ep/v1/chat/completions",
            "raw_path": b"/ws/ep/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"x-request-id", b"req"),
                (b"authorization", b"Bearer client"),
            ],
            "client": ("127.0.0.1", 1),
            "server": ("router", 80),
            "app": app,
        },
        receive,
    )


async def post(
    app,
    path,
    *,
    payload=None,
    content=None,
    files=None,
    headers=None,
    first_chunk=None,
    disconnect=None,
):
    headers = dict(headers or {})
    if files is not None:
        boundary = "router-test-boundary"
        parts = []
        for name, value in files:
            filename, data, *content_type = value
            disposition = f'Content-Disposition: form-data; name="{name}"'
            if filename is not None:
                disposition += f'; filename="{filename}"'
            if content_type:
                disposition += f"\r\nContent-Type: {content_type[0]}"
            parts.append(
                f"--{boundary}\r\n{disposition}\r\n\r\n".encode()
                + (data.encode() if isinstance(data, str) else data)
                + b"\r\n"
            )
        content = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers["content-type"] = f"multipart/form-data; boundary={boundary}"
    elif payload is not None:
        content = json.dumps(payload).encode()
        headers["content-type"] = "application/json"
    request = request_for(app, body=content or b"", disconnect=disconnect)
    scope = dict(
        request.scope,
        path=path,
        raw_path=path.encode(),
        headers=[(key.encode(), value.encode()) for key, value in headers.items()],
    )
    messages = []

    async def send(message):
        messages.append(message)
        if first_chunk is not None and message.get("body"):
            first_chunk.set()

    await app(scope, request.receive, send)
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return SimpleNamespace(
        status_code=start["status"],
        text=body.decode(),
        headers=Headers(raw=start["headers"]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "passthrough"])
async def test_disconnect_waiting_for_headers_closes_backend(app, monkeypatch, kind):
    entered = asyncio.Event()
    closed = []

    class PendingResponse:
        async def __aenter__(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.append(True)

        async def __aexit__(self, *args):
            pass

    app.state.aiohttp_client_wrapper = lambda: SimpleNamespace(
        request=lambda **kw: PendingResponse()
    )
    app.state.endpoints = [endpoint("http://engine")]
    disconnected = asyncio.Event()
    request = request_for(app, disconnect=disconnected)
    if kind == "json":
        call = proxy.route_general_request(
            request, "/v1/chat/completions", BackgroundTasks(), "ws", "ep"
        )
    else:
        passthrough = importlib.import_module("vllm_router.routers.passthrough_router")
        monkeypatch.setattr(
            passthrough, "_passthrough_endpoints", lambda *args: app.state.endpoints
        )
        call = passthrough.route_passthrough_request("ws", "ep", "predict", request)
    pending = asyncio.create_task(call)
    await asyncio.wait_for(entered.wait(), 1)
    disconnected.set()
    response = await asyncio.wait_for(pending, 1)
    assert response.status_code == 499
    assert closed == [True]
    assert (
        app.state.request_stats_monitor.get_active_request_count("http://engine") == 0
    )


@pytest.mark.asyncio
async def test_json_send_failure_closes_suspended_backend(app):
    closed = []

    async def chunks():
        yield b"first"
        await asyncio.Event().wait()

    class ResponseContext:
        async def __aenter__(self):
            return SimpleNamespace(
                headers={"content-type": "text/event-stream"},
                status=200,
                content=SimpleNamespace(iter_any=chunks),
            )

        async def __aexit__(self, *args):
            closed.append(True)

    app.state.aiohttp_client_wrapper = lambda: SimpleNamespace(
        request=lambda **kw: ResponseContext()
    )
    app.state.endpoints = [endpoint("http://engine")]
    request = request_for(app)
    response = await proxy.route_general_request(
        request, "/v1/chat/completions", BackgroundTasks(), "ws", "ep"
    )

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    with pytest.raises(BaseExceptionGroup):
        await response(request.scope, request.receive, send)
    assert closed == [True]
    assert (
        app.state.request_stats_monitor.get_active_request_count("http://engine") == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["pooling", "classify"])
async def test_json_input_api_preserves_payload_and_scoped_selection(app, api):
    received = []

    async def backend(request):
        received.append(await request.json())
        return web.Response(status=422, text="invalid input")

    backend_app = web.Application()
    backend_app.router.add_post("/" + api, backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        app.state.endpoints = [endpoint(str(server.make_url("/")).rstrip("/"))]
        app.state.aiohttp_client_wrapper = lambda: session
        app.state.router = PrefixAwareRouter()
        payload = {"model": "m", "input": ["hello"], "encoding_format": "float"}
        response = await post(app, f"/ws/ep/{api}", payload=payload)
        assert response.status_code == 422
        assert response.text == "invalid input"
        assert response.headers["content-type"].startswith("text/plain")
        denied = await post(app, f"/other/ep/{api}", payload=payload)
        assert denied.status_code == 404
        assert received == [payload]


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"{", b"[]", b'"m"', b'{"model":""}'])
async def test_new_json_routes_reject_invalid_model_payload(app, body):
    response = await post(app, "/ws/ep/pooling", content=body)
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("async_router", [False, True])
@pytest.mark.parametrize(
    "path",
    [
        "/v1/audio/transcriptions",
        "/ws/ep/v1/audio/transcriptions",
        "/ws/ep/v1/audio/translations",
    ],
)
async def test_audio_fields_raw_translation_and_text_error(app, path, async_router):
    if async_router:
        app.state.router = PrefixAwareRouter()
    received = []

    async def backend(request):
        fields = await request.post()
        received.append(
            {
                "model": fields["model"],
                "timestamps": fields.getall("timestamp_granularities[]"),
                "seed": fields["seed"],
                "file": fields["file"].file.read(),
                "filename": fields["file"].filename,
                "language": fields.get("language"),
                "authorization": request.headers.get("authorization"),
            }
        )
        return web.Response(status=422, text="upstream text error")

    backend_app = web.Application()
    backend_app.router.add_post("/v1/audio/transcriptions", backend)
    backend_app.router.add_post("/v1/audio/translations", backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        app.state.endpoints = [endpoint(str(server.make_url("/")).rstrip("/"))]
        app.state.aiohttp_client_wrapper = lambda: session
        fields = [
            ("file", ("clip.wav", b"RIFF", "audio/wav")),
            ("model", (None, "m")),
            ("seed", (None, "7")),
            ("timestamp_granularities[]", (None, "word")),
            ("timestamp_granularities[]", (None, "segment")),
        ]
        response = await post(
            app, path, files=fields, headers={"authorization": "Bearer client"}
        )
        assert response.status_code == 422
        assert response.text == "upstream text error"
        assert response.headers["content-type"].startswith("text/plain")
        if path.startswith("/ws"):
            denied = await post(app, path.replace("/ws/", "/other/"), files=fields)
            assert denied.status_code == 404
        assert len(received) == 1
        assert received[0] == {
            "model": "m",
            "timestamps": ["word", "segment"],
            "seed": "7",
            "file": b"RIFF",
            "filename": "clip.wav",
            "language": None if path.endswith("translations") else "en",
            "authorization": "Bearer client",
        }
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
        request, "/v1/audio/transcriptions", BackgroundTasks()
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
@pytest.mark.parametrize("stream", [False, True])
async def test_audio_success_preserves_json_or_sse(app, stream):
    async def backend(request):
        fields = await request.post()
        assert fields.get("stream") == ("true" if stream else None)
        if stream:
            return web.Response(
                text="data: hello\n\n", content_type="text/event-stream"
            )
        return web.json_response({"text": "hello"})

    backend_app = web.Application()
    backend_app.router.add_post("/v1/audio/transcriptions", backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        app.state.endpoints = [endpoint(str(server.make_url("/")).rstrip("/"))]
        app.state.aiohttp_client_wrapper = lambda: session
        files = [("file", ("clip.wav", b"RIFF", "audio/wav")), ("model", (None, "m"))]
        if stream:
            files.append(("stream", (None, "true")))
        response = await post(app, "/ws/ep/v1/audio/transcriptions", files=files)
        assert response.status_code == 200
        assert response.text == ("data: hello\n\n" if stream else '{"text":"hello"}')
        assert (
            app.state.request_stats_monitor.get_active_request_count(
                app.state.endpoints[0].url
            )
            == 0
        )


@pytest.mark.asyncio
async def test_audio_response_start_failure_closes_response_and_upload(app):
    closed = []

    async def chunks():
        yield b"data: first\n\n"

    async def open_response(*args, **kwargs):
        return SimpleNamespace(
            status=200,
            headers={"content-type": "text/event-stream"},
            content=SimpleNamespace(iter_any=chunks),
            close=lambda: closed.append(True),
        )

    app.state.endpoints = [endpoint("http://engine")]
    app.state.aiohttp_client_wrapper = lambda: SimpleNamespace(post=open_response)
    request = request_for(app)
    upload = UploadFile(io.BytesIO(b"RIFF"), filename="clip.wav")
    request._form = FormData([("file", upload), ("model", "m"), ("stream", "true")])
    response = await proxy.route_general_transcriptions(
        request, "/v1/audio/transcriptions", BackgroundTasks()
    )

    async def send(message):
        raise OSError("client left before response headers")

    with pytest.raises(BaseExceptionGroup):
        await response(request.scope, request.receive, send)
    assert closed == [True]
    assert upload.file.closed
    assert (
        app.state.request_stats_monitor.get_active_request_count("http://engine") == 0
    )


@pytest.mark.asyncio
async def test_audio_disconnect_while_buffering_nonstream_response(app):
    reading = asyncio.Event()
    closed = []

    async def read():
        reading.set()
        await asyncio.Event().wait()

    async def open_response(*args, **kwargs):
        return SimpleNamespace(
            status=200,
            headers={"content-type": "text/plain"},
            read=read,
            close=lambda: closed.append(True),
        )

    app.state.endpoints = [endpoint("http://engine")]
    app.state.aiohttp_client_wrapper = lambda: SimpleNamespace(post=open_response)
    disconnect = asyncio.Event()
    request = request_for(app, disconnect=disconnect)
    upload = UploadFile(io.BytesIO(b"RIFF"), filename="clip.wav")
    request._form = FormData([("file", upload), ("model", "m")])
    pending = asyncio.create_task(
        proxy.route_general_transcriptions(
            request, "/v1/audio/transcriptions", BackgroundTasks()
        )
    )
    await asyncio.wait_for(reading.wait(), 1)
    disconnect.set()
    response = await asyncio.wait_for(pending, 1)
    assert response.status_code == 499
    assert closed == [True]
    assert upload.file.closed


@pytest.mark.asyncio
async def test_pd_captures_targets_while_discovery_updates(app):
    observed = []

    async def backend(request):
        observed.append((request.path, await request.json()))
        if request.path.startswith("/prefill"):
            app.state.decode_endpoint = endpoint("http://unavailable")
            return web.json_response({"prefill": "ready"})
        return web.Response(text="data: complete\n\n", content_type="text/event-stream")

    backend_app = web.Application()
    backend_app.router.add_post("/{role}/v1/chat/completions", backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        base = str(server.make_url("/")).rstrip("/")
        app.state.prefill_endpoint = endpoint(base + "/prefill")
        app.state.decode_endpoint = endpoint(base + "/decode")
        app.state.aiohttp_client_wrapper = lambda: session
        app.state.router = DisaggregatedPrefillRouter(["prefill"], ["decode"])
        response = await post(
            app,
            "/ws/ep/v1/chat/completions",
            payload={"model": "m", "max_tokens": 8, "stream": True},
        )
    assert response.status_code == 200
    assert response.text == "data: complete\n\n"
    assert [path for path, _ in observed] == [
        "/prefill/v1/chat/completions",
        "/decode/v1/chat/completions",
    ]
    assert observed[0][1]["max_tokens"] == 1
    assert observed[1][1]["max_tokens"] == 8


@pytest.mark.asyncio
async def test_pd_decode_reframes_automatically_decompressed_response(app):
    payload = b"data: decoded token\n\n" * 20

    async def backend(request):
        if request.path.startswith("/prefill"):
            return web.json_response({"prefill": "ready"})
        return web.Response(
            body=gzip.compress(payload),
            headers={
                "Content-Encoding": "gzip",
                "Content-Type": "text/event-stream",
                "Connection": "keep-alive",
                "X-Backend": "decode",
            },
        )

    backend_app = web.Application()
    backend_app.router.add_post("/{role}/v1/chat/completions", backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        base = str(server.make_url("/")).rstrip("/")
        app.state.prefill_endpoint = endpoint(base + "/prefill")
        app.state.decode_endpoint = endpoint(base + "/decode")
        app.state.aiohttp_client_wrapper = lambda: session
        app.state.router = DisaggregatedPrefillRouter(["prefill"], ["decode"])
        response = await post(
            app,
            "/ws/ep/v1/chat/completions",
            payload={"model": "m", "stream": True},
        )
    assert response.status_code == 200
    assert response.text.encode() == payload
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["x-backend"] == "decode"
    for header in (
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ):
        assert header not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["pooling", "classify"])
async def test_input_apis_enforce_scope_and_model_with_global_pd_router(
    app, monkeypatch, api
):
    observed = []

    async def backend(request):
        observed.append((request.path, await request.json()))
        return web.json_response({"result": "selected backend"})

    backend_app = web.Application()
    backend_app.router.add_post("/{path:.*}", backend)
    async with TestServer(backend_app) as server, aiohttp.ClientSession() as session:
        base = str(server.make_url("/")).rstrip("/")
        app.state.endpoints = [endpoint(base)]
        app.state.prefill_endpoint = endpoint(base + "/prefill")
        app.state.decode_endpoint = endpoint(base + "/decode")
        app.state.router = DisaggregatedPrefillRouter(["prefill"], ["decode"])
        app.state.aiohttp_client_wrapper = lambda: session
        monkeypatch.setattr(
            proxy, "get_routing_logic_by_type", lambda _: RoundRobinRouter()
        )
        invalid_scope = await post(
            app, f"/other/ep/{api}", payload={"model": "m", "input": "hello"}
        )
        invalid_model = await post(
            app, f"/ws/ep/{api}", payload={"model": "wrong", "input": "hello"}
        )
        valid_payload = {"model": "m", "input": "hello"}
        valid = await post(app, f"/ws/ep/{api}", payload=valid_payload)
    assert [invalid_scope.status_code, invalid_model.status_code] == [404, 404]
    assert valid.status_code == 200
    assert observed == [(f"/{api}", valid_payload)]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/ws/ep/pooling", "/ws/ep/v1/audio/transcriptions"])
async def test_native_read_timeout_before_response_returns_504(app, path):
    async def backend(request):
        await request.read()
        await asyncio.sleep(0.05)
        return web.json_response({"ok": True})

    backend_app = web.Application()
    backend_app.router.add_post("/pooling", backend)
    backend_app.router.add_post("/v1/audio/transcriptions", backend)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=0.01)
    async with (
        TestServer(backend_app) as server,
        aiohttp.ClientSession(timeout=timeout) as session,
    ):
        app.state.endpoints = [endpoint(str(server.make_url("/")).rstrip("/"))]
        app.state.aiohttp_client_wrapper = lambda: session
        if path.endswith("pooling"):
            response = await post(app, path, payload={"model": "m", "input": "hello"})
        else:
            response = await post(
                app,
                path,
                files=[
                    ("file", ("clip.wav", b"RIFF", "audio/wav")),
                    ("model", (None, "m")),
                ],
            )
        assert response.status_code == 504
        assert (
            app.state.request_stats_monitor.get_active_request_count(
                app.state.endpoints[0].url
            )
            == 0
        )


@pytest.mark.asyncio
async def test_translation_streams_before_eof_and_disconnect_releases_pool(app):
    release = asyncio.Event()
    first_chunk = asyncio.Event()
    disconnect = asyncio.Event()

    async def backend(request):
        fields = await request.post()
        assert fields["stream"] == "true"
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        await release.wait()
        return response

    async def health(request):
        return web.Response(text="ok")

    backend_app = web.Application()
    backend_app.router.add_post("/v1/audio/translations", backend)
    backend_app.router.add_get("/health", health)
    async with (
        TestServer(backend_app) as server,
        aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit_per_host=1)
        ) as session,
    ):
        base = str(server.make_url("/")).rstrip("/")
        app.state.endpoints = [endpoint(base)]
        app.state.aiohttp_client_wrapper = lambda: session
        files = [
            ("file", ("clip.wav", b"RIFF", "audio/wav")),
            ("model", (None, "m")),
            ("stream", (None, "true")),
        ]
        pending = asyncio.create_task(
            post(
                app,
                "/ws/ep/v1/audio/translations",
                files=files,
                first_chunk=first_chunk,
                disconnect=disconnect,
            )
        )
        try:
            await asyncio.wait_for(first_chunk.wait(), 0.5)
            disconnect.set()
            response = await asyncio.wait_for(pending, 1)
            assert response.status_code == 200
            assert response.text == "data: first\n\n"
            assert app.state.request_stats_monitor.get_active_request_count(base) == 0
            async with session.get(
                base + "/health", timeout=aiohttp.ClientTimeout(total=1)
            ) as health_response:
                assert health_response.status == 200
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
