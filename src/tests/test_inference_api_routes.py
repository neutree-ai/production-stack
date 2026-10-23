import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from fastapi import FastAPI

from tests.test_backend_http import serve, wrapper
from vllm_router.routers.main_router import main_router
from vllm_router.services.request_service import request as proxy
from vllm_router.utils import ModelType, is_model_healthy

pytest_plugins = ["tests.test_request_forwarding"]


async def call(app, path, body, content_type="application/json"):
    sent = False
    messages = []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "scheme": "http",
            "query_string": b"",
            "headers": [(b"content-type", content_type.encode())],
            "client": ("127.0.0.1", 1234),
            "server": ("router", 80),
        },
        receive,
        send,
    )
    return next(
        m["status"] for m in messages if m["type"] == "http.response.start"
    ), b"".join(
        m.get("body", b"") for m in messages if m["type"] == "http.response.body"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix,prefixaware",
    [("/pooling", False), ("/classify", False), ("/classify", True)],
)
async def test_new_json_apis_forward_raw_payload_status_and_isolation(
    state, monkeypatch, suffix, prefixaware
):
    payload = {
        "model": "m",
        "input": ["hello"],
        "encoding_format": "float",
        "extra_parameter": True,
    }

    async def backend(request):
        assert request.path == suffix
        assert await request.json() == payload
        return web.json_response({"accepted": True}, status=422)

    app = FastAPI()
    app.state = state
    app.include_router(main_router)
    async with serve(backend) as url, wrapper() as client:
        state.aiohttp_client_wrapper = client
        ep = SimpleNamespace(
            url=url,
            model_names=["m"],
            sleep=False,
            workspace="ws",
            endpoint="ep",
            routing_logic=None,
        )
        monkeypatch.setattr(
            proxy,
            "get_service_discovery",
            lambda: SimpleNamespace(get_endpoint_info=lambda: [ep], aliases=None),
        )
        router = SimpleNamespace(
            route_request=lambda *a: url, extract_session_id=lambda *a: None
        )
        if prefixaware:
            from vllm_router.routers.routing_logic import PrefixAwareRouter

            router = PrefixAwareRouter()
        state.router = router
        monkeypatch.setattr(proxy, "get_routing_logic_by_type", lambda _: router)
        status, body = await call(app, "/ws/ep" + suffix, json.dumps(payload).encode())
        assert status == 422
        assert json.loads(body) == {"accepted": True}
        status, _ = await call(app, "/other/ep" + suffix, json.dumps(payload).encode())
        assert status == 404
        assert not client._pools


@pytest.mark.asyncio
async def test_audio_translation_uses_scoped_multipart_path(state, monkeypatch):
    async def backend(request):
        assert request.path == "/v1/audio/translations"
        form = await request.post()
        assert form["model"] == "m"
        assert form["seed"] == "42"
        assert form["file"].file.read() == b"audio"
        return web.Response(text="translated", content_type="text/plain", status=201)

    app = FastAPI()
    app.state = state
    app.include_router(main_router)
    boundary = "neu869"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nm\r\n--{boundary}\r\nContent-Disposition: form-data; name="seed"\r\n\r\n42\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\naudio\r\n--{boundary}--\r\n'
    ).encode()
    async with serve(backend) as url, wrapper() as client:
        state.aiohttp_client_wrapper = client
        ep = SimpleNamespace(
            url=url,
            model_names=["m"],
            sleep=False,
            workspace="ws",
            endpoint="ep",
            routing_logic=None,
        )
        monkeypatch.setattr(
            proxy,
            "get_service_discovery",
            lambda: SimpleNamespace(get_endpoint_info=lambda: [ep], aliases=None),
        )
        router = SimpleNamespace(
            route_request=lambda *a: url, extract_session_id=lambda *a: None
        )
        state.router = router
        monkeypatch.setattr(proxy, "get_routing_logic_by_type", lambda _: router)
        assert await call(
            app,
            "/ws/ep/v1/audio/translations",
            body,
            f"multipart/form-data; boundary={boundary}",
        ) == (201, b"translated")
        status, _ = await call(
            app,
            "/ws/wrong/v1/audio/translations",
            body,
            f"multipart/form-data; boundary={boundary}",
        )
        assert status == 404
        assert not client._pools


@pytest.mark.parametrize(
    "kind,path",
    [
        ("pooling", "/pooling"),
        ("classification", "/classify"),
        ("translation", "/v1/audio/translations"),
    ],
)
def test_static_health_checks_cover_new_api_types(monkeypatch, kind, path):
    from unittest.mock import Mock

    post = Mock(
        return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    )
    monkeypatch.setattr("vllm_router.utils.requests.post", post)
    assert ModelType.get_url(kind) == path
    assert is_model_healthy("http://backend", "m", kind)
    if kind == "translation":
        assert "file" in post.call_args.kwargs["files"]
    else:
        assert post.call_args.kwargs["json"]["input"] == "Hello"


def test_responses_api_remains_deferred_and_admin_routes_are_not_expanded():
    paths = {route.path for route in main_router.routes}
    assert not any("responses" in path for path in paths)
    assert not any(
        path.endswith(("/collective_rpc", "/start_profile", "/reset_prefix_cache"))
        for path in paths
    )
