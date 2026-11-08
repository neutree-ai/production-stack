import asyncio
import os
import signal
import socket
import sys

import aiohttp
import pytest
import yaml
from aiohttp import web


@pytest.mark.asyncio
async def test_cli_discovers_backend_and_serves_scoped_apis_with_native_pool_metrics(
    tmp_path,
):
    # Exercise real CLI startup and the Kubernetes SDK, without a live cluster.
    held = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def pods(request):
        if request.query.get("watch", "").lower() == "true":
            return web.Response()
        return web.json_response(
            {
                "kind": "PodList",
                "apiVersion": "v1",
                "metadata": {"resourceVersion": "1"},
                "items": [
                    {
                        "metadata": {
                            "name": "backend",
                            "labels": {"workspace": "ws", "endpoint": "ep"},
                        },
                        "spec": {"containers": [{"name": "engine"}]},
                        "status": {
                            "podIP": "127.0.0.1",
                            "containerStatuses": [
                                {
                                    "name": "engine",
                                    "image": "stub",
                                    "imageID": "stub",
                                    "ready": True,
                                    "restartCount": 0,
                                }
                            ],
                        },
                    }
                ],
            }
        )

    async def inference(request):
        if request.path == "/v1/models":
            return web.json_response({"data": [{"id": "m"}]})
        if "/audio/" in request.path:
            fields = await request.post()
            assert fields["model"] == "m"
            assert fields.getall("timestamp_granularities[]") == ["word", "segment"]
            seen.append(request.path)
            return web.Response(text="transcript")
        payload = await request.json()
        seen.append(request.path)
        if payload["input"] == "hold":
            held.set()
            await release.wait()
        return web.json_response({"path": request.path})

    async def pod_info(request):
        return web.json_response(
            {
                "metadata": {"name": "backend"},
                "spec": {"containers": [{"name": "engine"}]},
            }
        )

    stub = web.Application()
    stub.router.add_get("/api/v1/namespaces/default/pods", pods)
    stub.router.add_get("/api/v1/namespaces/default/pods/backend", pod_info)
    stub.router.add_route("*", "/{path:.*}", inference)
    runner = web.AppRunner(stub)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    backend_port = runner.addresses[0][1]
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Config",
                "clusters": [
                    {
                        "name": "stub",
                        "cluster": {"server": f"http://127.0.0.1:{backend_port}"},
                    }
                ],
                "contexts": [
                    {"name": "stub", "context": {"cluster": "stub", "user": "stub"}}
                ],
                "current-context": "stub",
                "users": [{"name": "stub", "user": {}}],
            }
        )
    )
    with socket.socket() as free_port:
        free_port.bind(("127.0.0.1", 0))
        router_port = free_port.getsockname()[1]
    log_path = tmp_path / "router.log"
    process = None
    try:
        with log_path.open("w") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "vllm_router.app",
                "--host",
                "127.0.0.1",
                "--port",
                str(router_port),
                "--service-discovery",
                "k8s",
                "--k8s-port",
                str(backend_port),
                "--session-key",
                "X-Session-ID",
                "--routing-logic",
                "default",
                "--max-connections-per-backend",
                "1",
                "--connect-timeout-seconds",
                "5",
                "--sock-read-timeout-seconds",
                "5",
                env={**os.environ, "KUBECONFIG": str(kubeconfig)},
                stdout=log,
                stderr=log,
            )
        base = f"http://127.0.0.1:{router_port}"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as client:
            for _ in range(200):
                if process.returncode is not None:
                    pytest.fail(log_path.read_text())
                try:
                    async with client.get(base + "/ws/ep/health") as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.05)
            else:
                pytest.fail(log_path.read_text())

            first = asyncio.create_task(
                client.post(
                    base + "/ws/ep/pooling", json={"model": "m", "input": "hold"}
                )
            )
            await asyncio.wait_for(held.wait(), 5)
            second = asyncio.create_task(
                client.post(
                    base + "/ws/ep/classify", json={"model": "m", "input": "hello"}
                )
            )
            for _ in range(100):
                async with client.get(base + "/metrics") as response:
                    text = await response.text()
                if (
                    'router:connection_pool_waiting{endpoint="ep",replica="backend",server='
                    in text
                    and ',workspace="ws"} 1.0' in text
                ):
                    break
                await asyncio.sleep(0.02)
            else:
                pytest.fail(
                    "Router did not expose a waiting call\n" + log_path.read_text()
                )
            release.set()
            for task, path in ((first, "/pooling"), (second, "/classify")):
                response = await task
                assert response.status == 200
                assert await response.json() == {"path": path}

            for path in ("transcriptions", "translations"):
                form = aiohttp.FormData()
                form.add_field("model", "m")
                form.add_field("timestamp_granularities[]", "word")
                form.add_field("timestamp_granularities[]", "segment")
                form.add_field(
                    "file", b"audio", filename="sample.wav", content_type="audio/wav"
                )
                async with client.post(
                    base + "/ws/ep/v1/audio/" + path, data=form
                ) as response:
                    assert response.status == 200
                    assert await response.text() == "transcript"
            assert seen == [
                "/pooling",
                "/classify",
                "/v1/audio/transcriptions",
                "/v1/audio/translations",
            ]
    finally:
        release.set()
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        await runner.cleanup()
    assert process.returncode in (0, -signal.SIGTERM), log_path.read_text()
