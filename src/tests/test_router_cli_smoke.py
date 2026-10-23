"""Run the actual Router CLI against a controllable HTTP engine, without GPUs."""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import unused_port

from tests.test_backend_http import serve


@asynccontextmanager
async def router_process(api, kubeconfig, log_path):
    port = unused_port()
    output = log_path.open("wb")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "vllm_router.app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--service-discovery",
        "k8s",
        "--k8s-namespace",
        "ns",
        "--k8s-label-selector",
        "app=inference",
        "--k8s-port",
        str(api[1]),
        "--session-key",
        "session_id",
        "--response-header-timeout-seconds",
        "0.1",
        "--stream-idle-timeout-seconds",
        "0.1",
        "--max-connections-per-backend",
        "1",
        stdout=output,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": "src", "KUBECONFIG": str(kubeconfig)},
    )
    url = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with asyncio.timeout(10):
                while True:
                    if process.returncode is not None:
                        pytest.fail(log_path.read_text())
                    try:
                        async with session.get(url + "/health") as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)
        yield url
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()
        output.close()


@asynccontextmanager
async def fake_kubernetes(engine, directory):
    import json
    from urllib.parse import urlparse

    engine_port = urlparse(engine).port
    release = asyncio.Event()
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "engine",
            "namespace": "ns",
            "uid": "engine-uid",
            "resourceVersion": "1",
            "labels": {
                "app": "inference",
                "workspace": "ws",
                "endpoint": "ep",
                "sleeping": "false",
            },
        },
        "status": {
            "podIP": "127.0.0.1",
            "containerStatuses": [
                {
                    "name": "vllm",
                    "ready": True,
                    "restartCount": 0,
                    "image": "fake",
                    "imageID": "fake",
                }
            ],
        },
    }

    async def api_handler(request):
        if request.query.get("watch", "").lower() == "true":
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(
                (json.dumps({"type": "ADDED", "object": pod}) + "\n").encode()
            )
            await release.wait()
            return response
        return web.json_response(
            {
                "apiVersion": "v1",
                "kind": "PodList",
                "metadata": {"resourceVersion": "1"},
                "items": [pod],
            }
        )

    async with serve(api_handler) as api_url:
        config = {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [{"name": "test", "cluster": {"server": api_url}}],
            "users": [{"name": "test", "user": {}}],
            "contexts": [
                {"name": "test", "context": {"cluster": "test", "user": "test"}}
            ],
            "current-context": "test",
        }
        (directory / "kubeconfig.json").write_text(json.dumps(config))
        try:
            yield api_url, engine_port
        finally:
            release.set()


@pytest.mark.asyncio
async def test_real_cli_new_apis_timeout_active_stream_and_metrics(tmp_path):
    async def backend(request):
        if request.path == "/v1/audio/translations":
            form = await request.post()
            assert form["model"] == "m"
            assert form.getall("timestamp_granularities[]") == ["word", "segment"]
            assert form["file"].file.read() == b"audio"
            return web.Response(text="translated", status=201)
        if request.path == "/v1/models":
            return web.json_response(
                {"data": [{"id": "m", "created": 0, "owned_by": "test"}]}
            )
        body = await request.json()
        if body.get("stream"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for _ in range(8):
                await response.write(b"data: token\n\n")
                await asyncio.sleep(0.025)
            return response
        if body.get("slow"):
            await asyncio.sleep(0.2)
        return web.json_response({"forwarded_path": request.path}, status=202)

    async with (
        serve(backend) as engine,
        fake_kubernetes(engine, tmp_path) as api,
        router_process(
            api, tmp_path / "kubeconfig.json", tmp_path / "router.log"
        ) as url,
        aiohttp.ClientSession() as session,
    ):
        for suffix in ["/pooling", "/classify"]:
            async with session.post(
                url + "/ws/ep" + suffix, json={"model": "m", "input": ["hi"]}
            ) as response:
                assert response.status == 202
                assert (await response.json())["forwarded_path"] == suffix
        form = aiohttp.FormData()
        form.add_field("model", "m")
        form.add_field("timestamp_granularities[]", "word")
        form.add_field("timestamp_granularities[]", "segment")
        form.add_field(
            "file", b"audio", filename="speech.wav", content_type="audio/wav"
        )
        async with session.post(
            url + "/ws/ep/v1/audio/translations", data=form
        ) as response:
            assert response.status == 201
            assert await response.text() == "translated"
        async with session.post(
            url + "/ws/ep/pooling", json={"model": "m", "input": "hi", "slow": True}
        ) as response:
            assert response.status == 504
        async with session.post(
            url + "/ws/ep/pooling", json={"model": "m", "input": "hi", "stream": True}
        ) as response:
            assert response.status == 200
            assert (await response.read()).count(b"data: token") == 8
        async with session.get(url + "/metrics") as response:
            metrics = await response.text()
            assert response.status == 200
            assert "router:backend_requests_total{" in metrics
            assert 'outcome="timeout"' in metrics
            assert 'phase="response_header"' in metrics
            assert "router:backend_response_headers_seconds_count{" in metrics
        async with session.post(
            url + "/ws/ep/v1/responses", json={"model": "m"}
        ) as response:
            assert response.status == 404
