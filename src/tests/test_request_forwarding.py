import asyncio
import io
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, FastAPI
from starlette.datastructures import FormData, Headers, UploadFile
from starlette.requests import Request

from vllm_router.routers import passthrough_router as passthrough_module
from vllm_router.services.request_service import request as proxy
from vllm_router.stats.request_stats import RequestStatsMonitor, SingletonMeta


@pytest.fixture
def state(monkeypatch):
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    monitor = RequestStatsMonitor(sliding_window_size=10)
    app = FastAPI()
    router = SimpleNamespace(
        route_request=lambda *args: "http://engine",
        extract_session_id=lambda *args: None,
    )
    app.state.router = router
    app.state.request_stats_monitor = monitor
    app.state.engine_stats_scraper = None
    app.state.semantic_cache_available = False
    backend = SimpleNamespace(
        url="http://engine",
        model_names=["m"],
        sleep=False,
        workspace="ws",
        endpoint="ep",
        routing_logic=None,
        passthrough=True,
    )
    discovery = SimpleNamespace(get_endpoint_info=lambda: [backend], aliases=None)
    monkeypatch.setattr(proxy, "get_service_discovery", lambda: discovery)
    monkeypatch.setattr(passthrough_module, "get_service_discovery", lambda: discovery)
    monkeypatch.setattr(proxy, "get_routing_logic_by_type", lambda _: router)
    monkeypatch.setattr(proxy, "is_request_rewriter_initialized", lambda: False)
    yield app.state
    SingletonMeta._instances.pop(RequestStatsMonitor, None)


def request_for(state, disconnect=None):
    async def receive():
        if disconnect is not None:
            await disconnect.wait()
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "POST",
            "path": "/ws/ep/v1/chat/completions",
            "scheme": "http",
            "server": ("router", 80),
            "query_string": b"",
            "headers": [
                (b"x-request-id", b"req-abc"),
                (b"authorization", b"Bearer client"),
            ],
            "app": SimpleNamespace(state=state),
        },
        receive=receive,
    )
    request._body = b'{"model":"m"}'
    return request


class Content:
    async def iter_any(self):
        yield b"hello"


class BackendContext:
    def __init__(self, *, blocked=False, headers=None, status=202):
        self.blocked = blocked
        self.entered = asyncio.Event()
        self.closed = False
        self.response = SimpleNamespace(
            headers=Headers(headers or {"content-type": "text/plain"}),
            status=status,
            content=Content(),
        )

    async def __aenter__(self):
        self.entered.set()
        try:
            if self.blocked:
                await asyncio.Event().wait()
            return self.response
        except BaseException:
            self.closed = True
            raise

    async def __aexit__(self, *args):
        self.closed = True


class Client:
    def __init__(self, context):
        self.context = context
        self.sent = None

    def request(self, **kwargs):
        self.sent = kwargs
        return self.context

    async def post(self, url, **kwargs):
        self.sent = dict(url=url, **kwargs)
        return self.context.response


@pytest.mark.asyncio
async def test_disconnect_before_headers_cancels_backend_and_releases_stats(state):
    context = BackendContext(blocked=True)
    state.aiohttp_client_wrapper = lambda: Client(context)
    disconnect = asyncio.Event()
    req = request_for(state, disconnect)
    task = asyncio.create_task(
        proxy.route_general_request(
            req, "/v1/chat/completions", BackgroundTasks(), "ws", "ep"
        )
    )
    await asyncio.wait_for(context.entered.wait(), 1)
    disconnect.set()
    try:
        # Bound hangs without timing AnyIO's first backend import as request latency.
        response = await asyncio.wait_for(task, 5)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert response.status_code == 499
    assert context.closed
    assert state.request_stats_monitor.get_active_request_count("http://engine") == 0
    assert state.request_stats_monitor.finished_requests.get("http://engine", 0) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("passthrough", [False, True])
async def test_response_headers_strip_duplicates_hop_headers_and_encoding(
    state, passthrough
):
    context = BackendContext(
        headers={
            "content-type": "text/plain",
            "x-request-id": "backend-id",
            "date": "Tue, 20 Oct 2026 21:00:37 GMT",
            "server": "backend",
            "content-encoding": "gzip",
            "content-length": "10",
            "connection": "x-backend-only",
            "x-backend-only": "private",
        }
    )
    state.aiohttp_client_wrapper = lambda: Client(context)
    req = request_for(state)
    if passthrough:
        response = await passthrough_module.route_passthrough_request(
            "ws", "ep", "predict", req
        )
    else:
        response = await proxy.route_general_request(
            req, "/v1/chat/completions", BackgroundTasks(), "ws", "ep"
        )
    try:
        names = [k.decode() for k, _ in response.raw_headers]
        assert names.count("x-request-id") == 1
        assert response.headers["x-request-id"] == "req-abc"
        assert response.headers["content-type"] == "text/plain"
        for name in [
            "date",
            "server",
            "connection",
            "content-encoding",
            "content-length",
            "x-backend-only",
        ]:
            assert name not in names
        assert response.status_code == 202
        assert b"".join([chunk async for chunk in response.body_iterator]) == b"hello"
        assert context.closed
    finally:
        await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_send_failure_closes_backend_suspended_at_yield(state):
    context = BackendContext()
    state.aiohttp_client_wrapper = lambda: Client(context)
    req = request_for(state)
    response = await proxy.route_general_request(
        req, "/v1/chat/completions", BackgroundTasks(), "ws", "ep"
    )

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client went away")

    with pytest.raises(Exception):
        await response(req.scope, req.receive, send)
    assert context.closed
    assert state.request_stats_monitor.get_active_request_count("http://engine") == 0


@pytest.mark.asyncio
async def test_cancel_wait_does_not_swallow_independent_caller_cancellation():
    task = asyncio.create_task(asyncio.sleep(1000))
    outcome = []

    async def caller():
        try:
            await proxy._cancel_and_wait(task)
        except asyncio.CancelledError:
            outcome.append("cancelled")

    caller_task = asyncio.create_task(caller())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    caller_task.cancel()
    await asyncio.gather(caller_task, return_exceptions=True)
    assert outcome == ["cancelled"]


@pytest.mark.asyncio
async def test_transcription_preserves_repeated_fields_and_upload_metadata(state):
    audio = UploadFile(
        io.BytesIO(b"RIFF"),
        filename="audio.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )
    fields = FormData(
        [
            ("file", audio),
            ("model", "m"),
            ("language", "fr"),
            ("temperature", "0.25"),
            ("stream", "false"),
            ("seed", "7"),
            ("top_p", "0.9"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
    )
    context = BackendContext(
        headers={"content-type": "text/plain", "x-request-id": "backend"}
    )
    client = Client(context)
    state.aiohttp_client_wrapper = lambda: client
    req = request_for(state)
    req._form = fields
    try:
        response = await proxy.route_general_transcriptions(
            req, "/v1/audio/transcriptions", BackgroundTasks()
        )
        assert response.status_code == 202
        assert b"".join([chunk async for chunk in response.body_iterator]) == b"hello"
        sent = [(opts["name"], value) for opts, _, value in client.sent["data"]._fields]
        assert ("seed", "7") in sent
        assert ("top_p", "0.9") in sent
        assert ("timestamp_granularities[]", "word") in sent
        assert ("timestamp_granularities[]", "segment") in sent
        assert ("stream", "false") in sent
        assert [name for name, _ in sent].count("model") == 1
        part = next(
            opts for opts, _, _ in client.sent["data"]._fields if opts["name"] == "file"
        )
        assert part["filename"] == "audio.wav"
        assert client.sent["headers"]["authorization"] == "Bearer client"
        assert context.closed
        assert audio.file.closed
    finally:
        await fields.close()


@pytest.mark.asyncio
async def test_cancel_wait_only_swallows_its_own_cancellation():
    task = asyncio.create_task(asyncio.sleep(1000))
    await proxy._cancel_and_wait(task)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_client_cancellation_during_header_listener_cleanup_closes_stream(
    monkeypatch,
):
    received = asyncio.Event()
    release_listener = asyncio.Event()
    closed = []

    class DisconnectRequest:
        async def receive(self):
            try:
                await asyncio.Event().wait()
            finally:
                received.set()
                await release_listener.wait()

    async def stream():
        try:
            yield {}, 200
            yield b"data"
        finally:
            closed.append(True)

    task = asyncio.create_task(
        proxy._anext_or_disconnect(stream(), DisconnectRequest())
    )
    await asyncio.wait_for(received.wait(), 1)
    task.cancel()
    release_listener.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]


@pytest.mark.asyncio
async def test_transcription_error_closes_file_without_backend_request(state):
    audio = UploadFile(io.BytesIO(b"RIFF"), filename="audio.wav")
    req = request_for(state)
    req._form = FormData(
        [("file", audio), ("model", "m"), ("temperature", "not-a-number")]
    )
    response = await proxy.route_general_transcriptions(
        req, "/v1/audio/transcriptions", BackgroundTasks()
    )
    assert response.status_code == 400
    assert audio.file.closed


@pytest.mark.asyncio
async def test_real_multipart_wire_preserves_fields_status_content_type_and_headers(
    state,
):
    import aiohttp
    from aiohttp import web

    observed = {}

    async def backend(request):
        observed["authorization"] = request.headers.get("authorization")
        parts = await request.post()
        observed["seed"] = parts["seed"]
        observed["timestamps"] = parts.getall("timestamp_granularities[]")
        observed["filename"] = parts["file"].filename
        observed["content_type"] = parts["file"].content_type
        observed["file"] = parts["file"].file.read()
        return web.Response(
            status=422,
            text="backend text error",
            headers={
                "x-request-id": "backend-id",
                "Connection": "x-hidden",
                "x-hidden": "hidden",
            },
        )

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", backend)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    state.router.route_request = lambda *args: url
    audio = UploadFile(
        io.BytesIO(b"RIFF"),
        filename="a.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )
    req = request_for(state)
    req._form = FormData(
        [
            ("model", "m"),
            ("file", audio),
            ("seed", "7"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
    )
    try:
        async with aiohttp.ClientSession() as client:
            state.aiohttp_client_wrapper = lambda: client
            response = await proxy.route_general_transcriptions(
                req, "/v1/audio/transcriptions", BackgroundTasks()
            )
            assert response.status_code == 422
            assert response.headers["content-type"].startswith("text/plain")
            assert response.headers["x-request-id"] == "req-abc"
            assert "x-hidden" not in response.headers
            assert (
                b"".join([chunk async for chunk in response.body_iterator])
                == b"backend text error"
            )
            assert observed == {
                "authorization": "Bearer client",
                "seed": "7",
                "timestamps": ["word", "segment"],
                "filename": "a.wav",
                "content_type": "audio/wav",
                "file": b"RIFF",
            }
            assert audio.file.closed
            assert state.request_stats_monitor.get_active_request_count(url) == 0
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_transcription_send_failure_closes_response_and_uploads(state):
    audio = UploadFile(io.BytesIO(b"RIFF"), filename="audio.wav")
    context = BackendContext()
    state.aiohttp_client_wrapper = lambda: Client(context)
    req = request_for(state)
    req._form = FormData([("file", audio), ("model", "m")])
    response = await proxy.route_general_transcriptions(
        req, "/v1/audio/transcriptions", BackgroundTasks()
    )

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client went away")

    with pytest.raises(Exception):
        await response(req.scope, req.receive, send)
    assert context.closed
    assert audio.file.closed
    assert state.request_stats_monitor.get_active_request_count("http://engine") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["gzip, br", "compress"])
async def test_passthrough_wire_preserves_undecoded_content_encoding(
    state, monkeypatch, encoding
):
    import aiohttp
    from aiohttp import web

    payload = b"encoded payload owned by the backend"

    async def backend(request):
        return web.Response(body=payload, headers={"Content-Encoding": encoding})

    app = web.Application()
    app.router.add_post("/predict", backend)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    monkeypatch.setattr(
        passthrough_module,
        "_passthrough_endpoints",
        lambda *_: [SimpleNamespace(url=url)],
    )
    try:
        async with aiohttp.ClientSession() as client:
            state.aiohttp_client_wrapper = lambda: client
            response = await passthrough_module.route_passthrough_request(
                "ws", "ep", "predict", request_for(state)
            )
            assert response.headers.get("content-encoding") == encoding
            assert (
                b"".join([chunk async for chunk in response.body_iterator]) == payload
            )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_asgi23_disconnect_shields_async_rolled_upload_close(state):
    import tempfile

    import anyio

    rolled = tempfile.SpooledTemporaryFile(max_size=1)
    rolled.write(b"RIFF")
    rolled.seek(0)
    audio = UploadFile(rolled, filename="audio.wav")
    close_finished = []

    class AsyncCloseForm(FormData):
        async def close(self):
            await anyio.sleep(0)
            await super().close()
            close_finished.append(True)

    class HangingContent:
        async def iter_any(self):
            yield b"first"
            await asyncio.Event().wait()

    context = BackendContext()
    context.response.content = HangingContent()
    state.aiohttp_client_wrapper = lambda: Client(context)
    body_sent = asyncio.Event()
    req = request_for(state, body_sent)
    req.scope["asgi"]["spec_version"] = "2.3"
    req._form = AsyncCloseForm([("file", audio), ("model", "m")])
    response = await proxy.route_general_transcriptions(
        req, "/v1/audio/transcriptions", BackgroundTasks()
    )

    async def send(message):
        if message["type"] == "http.response.body":
            body_sent.set()

    try:
        await asyncio.wait_for(response(req.scope, req.receive, send), 1)
        assert close_finished == [True]
        assert rolled.closed
        assert context.closed
        assert (
            state.request_stats_monitor.get_active_request_count("http://engine") == 0
        )
    finally:
        rolled.close()
