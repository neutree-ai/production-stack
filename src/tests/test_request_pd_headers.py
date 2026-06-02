from types import SimpleNamespace

import pytest

from vllm_router.services.request_service.request import process_request


class RequestStatsMonitor:
    def __init__(self):
        self.new_request = None

    def on_new_request(self, *args, **kwargs):
        self.new_request = (args, kwargs)

    def on_request_response(self, *args, **kwargs):
        pass

    def on_request_complete(self, *args, **kwargs):
        pass


class FakeContent:
    async def iter_any(self):
        yield b'{"ok": true}'


class FakeBackendResponse:
    headers = {"content-type": "application/json"}
    status = 200
    content = FakeContent()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class FakeClient:
    def __init__(self):
        self.headers = None

    def request(self, **kwargs):
        self.headers = kwargs["headers"]
        return FakeBackendResponse()


@pytest.mark.asyncio
async def test_process_request_forwards_pd_route_headers_to_backend():
    client = FakeClient()
    stats_monitor = RequestStatsMonitor()
    request = SimpleNamespace(
        method="POST",
        headers={
            "host": "router.local",
            "authorization": "Bearer token",
            "x-neutree-pd-role-group": "client-rg",
            "X-Neutree-PD-Prefill-Index": "99",
            "X-Neutree-PD-Decode-Index": "99",
        },
        app=SimpleNamespace(
            state=SimpleNamespace(
                request_stats_monitor=stats_monitor,
                semantic_cache_available=False,
                aiohttp_client_wrapper=lambda: client,
            )
        ),
    )

    stream = process_request(
        request=request,
        body=b'{"stream": false}',
        backend_url="http://10.0.0.1:9000",
        request_id="req-1",
        endpoint="/v1/chat/completions",
        background_tasks=None,
        route_headers={
            "X-Neutree-PD-Role-Group": "rg-0",
            "X-Neutree-PD-Prefill-Index": "1",
            "X-Neutree-PD-Decode-Index": "0",
        },
        route_stats_metadata={
            "pd_prefill_unit_id": "rg-0:prefill:1:http://10.0.0.1:9000",
            "pd_decode_unit_id": "rg-0:decode:0:http://10.0.0.1:9000",
        },
    )

    headers, status = await anext(stream)
    chunk = await anext(stream)

    assert status == 200
    assert headers == {"content-type": "application/json"}
    assert chunk == b'{"ok": true}'
    assert "host" not in client.headers
    assert "x-neutree-pd-role-group" not in client.headers
    assert client.headers["authorization"] == "Bearer token"
    assert client.headers["X-Neutree-PD-Role-Group"] == "rg-0"
    assert client.headers["X-Neutree-PD-Prefill-Index"] == "1"
    assert client.headers["X-Neutree-PD-Decode-Index"] == "0"
    assert stats_monitor.new_request[1] == {
        "pd_prefill_unit_id": "rg-0:prefill:1:http://10.0.0.1:9000",
        "pd_decode_unit_id": "rg-0:decode:0:http://10.0.0.1:9000",
        "pd_track_both_units": True,
    }
