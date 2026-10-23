import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import FastAPI

from vllm_router.service_discovery import (
    K8sPodIPServiceDiscovery,
    K8sServiceNameServiceDiscovery,
    ServiceDiscoveryEventType,
)


def pod(name="p", ip="10.0.0.1", uid="u1", ready=True, **labels):
    return NS(
        metadata=NS(
            name=name,
            uid=uid,
            labels={
                "workspace": "ws",
                "endpoint": "ep",
                "passthrough": "true",
                **labels,
            },
            annotations={},
            deletion_timestamp=None,
        ),
        status=NS(pod_ip=ip, container_statuses=[NS(ready=ready)]),
    )


@pytest.fixture
def discovery(monkeypatch):
    monkeypatch.setattr(
        "vllm_router.service_discovery.config.load_incluster_config", lambda: None
    )
    d = K8sPodIPServiceDiscovery(FastAPI(), "ns", 8000)
    d._get_model_names = Mock(side_effect=AssertionError("passthrough must not probe"))
    d._check_engine_sleep_mode = Mock(return_value=False)
    yield d
    d.close()


def test_watcher_does_not_start_until_loop_ready(discovery):
    assert not discovery.watcher_thread.is_alive()
    assert discovery.prefill_model_labels is None
    assert not discovery.get_health()


@pytest.mark.parametrize(
    "replacement",
    [
        pod(ip="10.0.0.2"),
        pod(uid="u2"),
        pod(workspace="other"),
        pod(endpoint="other"),
        pod(routing_logic="static_hash"),
    ],
)
def test_ready_to_ready_replacement_updates_route_and_notifies_old_identity(
    discovery, replacement
):
    events = []
    discovery.register_callback(lambda *args: events.append(args))
    discovery._handle_resource(pod(), "ADDED")
    old = discovery.get_endpoint_info()[0]
    discovery._handle_resource(replacement, "MODIFIED")
    new = discovery.get_endpoint_info()[0]
    assert new is not old
    assert events[-2][0] == ServiceDiscoveryEventType.ENGINE_DELETED
    assert events[-2][2] is old
    assert events[-1][0] == ServiceDiscoveryEventType.ENGINE_ADDED
    assert new.workspace == replacement.metadata.labels["workspace"]


@pytest.mark.parametrize("replacement", [pod(ip=None), pod(ready=False)])
def test_lost_ip_or_readiness_removes_registered_route(discovery, replacement):
    discovery._handle_resource(pod(), "ADDED")
    discovery._handle_resource(replacement, "MODIFIED")
    assert discovery.get_endpoint_info() == []
    discovery._delete_engine("p")


def test_unchanged_relist_does_not_duplicate_callbacks_or_probe(discovery):
    callback = Mock()
    callback.__name__ = "callback"
    discovery.register_callback(callback)
    discovery._handle_resource(pod(), "ADDED")
    discovery._handle_resource(pod(), "MODIFIED")
    assert callback.call_count == 1
    discovery._get_model_names.assert_not_called()


def test_old_uid_delete_does_not_remove_replacement(discovery):
    discovery._handle_resource(pod(uid="new"), "ADDED")
    discovery._handle_resource(pod(uid="old"), "DELETED")
    assert len(discovery.get_endpoint_info()) == 1


def test_probe_failure_keeps_same_identity_but_never_old_tenant_or_ip(discovery):
    discovery._get_model_names = Mock(return_value=["m"])
    discovery._get_model_info = Mock(return_value={})
    initial = pod(passthrough="false")
    discovery._handle_resource(initial, "ADDED")
    old = discovery.get_endpoint_info()[0]
    discovery._get_model_names.return_value = []
    changed = pod(passthrough="false", model="different")
    discovery._handle_resource(changed, "MODIFIED")
    assert discovery.get_endpoint_info()[0] is old
    changed = pod(passthrough="false", workspace="new")
    discovery._handle_resource(changed, "MODIFIED")
    assert discovery.get_endpoint_info() == []


def test_probe_finishing_after_shutdown_cannot_publish_route(discovery):
    discovery._get_model_names = Mock(return_value=["m"])

    def late_probe(ip):
        discovery.close()
        return {}

    discovery._get_model_info = late_probe
    discovery._handle_resource(pod(passthrough="false"), "ADDED")
    assert discovery.get_endpoint_info() == []


@pytest.mark.parametrize(
    "sleep_label,expected", [(None, True), ("true", True), ("false", False)]
)
def test_missing_sleep_label_preserves_engine_probe(discovery, sleep_label, expected):
    discovery._get_model_names = Mock(return_value=["m"])
    discovery._get_model_info = Mock(return_value={})
    discovery._check_engine_sleep_mode = Mock(return_value=True)
    discovery._get_engine_sleep_status = Mock(return_value=True)
    resource = pod(passthrough="false")
    if sleep_label is not None:
        resource.metadata.labels["sleeping"] = sleep_label
    discovery._handle_resource(resource, "ADDED")
    assert discovery.get_endpoint_info()[0].sleep is expected
    assert discovery._get_engine_sleep_status.call_count == (
        1 if sleep_label is None else 0
    )


@pytest.mark.asyncio
async def test_pd_replacement_keeps_inflight_response_alive(discovery):
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(request):
        entered.set()
        await release.wait()
        return web.Response(text="completed")

    server_app = web.Application()
    server_app.router.add_get("/slow", slow)
    async with TestServer(server_app) as server:
        discovery.port = server.port
        discovery.prefill_model_labels = ["prefill"]
        discovery.decode_model_labels = ["decode"]
        discovery._handle_resource(pod(ip="127.0.0.1", model="prefill"), "ADDED")
        await discovery.initialize_client_sessions()
        old = discovery.app.state.prefill_client

        from vllm_router.aiohttp_client import AiohttpClientWrapper

        wrapper = AiohttpClientWrapper()
        wrapper.start()

        async def consume():
            async with wrapper.forward(
                str(old._base_url), method="GET", url=str(old._base_url) + "/slow"
            ) as response:
                return (await response.read()).decode()

        request = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), timeout=1)
        try:
            discovery._handle_resource(
                pod(ip="127.0.0.2", uid="u2", model="prefill"), "MODIFIED"
            )
            await discovery.initialize_client_sessions()
            assert discovery.app.state.prefill_client is not old
            assert not wrapper().closed
        finally:
            release.set()
        assert await asyncio.wait_for(request, timeout=1) == "completed"
        await discovery.close_client_sessions()
        assert discovery.app.state.prefill_client is None
        assert not wrapper().closed
        await wrapper.stop()


@pytest.mark.asyncio
async def test_identical_pd_backend_keeps_selection_and_shutdown_clears_it(discovery):
    discovery.prefill_model_labels = ["prefill"]
    discovery.decode_model_labels = ["decode"]
    resource = pod(model="prefill")
    discovery._handle_resource(resource, "ADDED")
    await discovery.initialize_client_sessions()
    session = discovery.app.state.prefill_client
    await discovery.initialize_client_sessions()
    assert discovery.app.state.prefill_client is session
    assert not hasattr(
        session, "get"
    )  # Discovery owns no sockets or retired session list.
    await discovery.close_client_sessions()
    assert discovery.app.state.prefill_client is None


def test_service_name_recovers_selector_change_without_reusing_old_snapshot(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm_router.service_discovery.config.load_incluster_config", lambda: None
    )
    d = K8sServiceNameServiceDiscovery(FastAPI(), "ns", 8000)
    d._check_service_ready = Mock(return_value=True)
    d._get_model_names = Mock(return_value=["m"])
    d._get_model_info = Mock(return_value={})
    d._check_engine_sleep_mode = Mock(return_value=False)
    svc = NS(
        metadata=NS(
            name="s", uid="u", labels={}, annotations={}, deletion_timestamp=None
        ),
        spec=NS(selector={"model": "one"}),
    )
    d._handle_resource(svc, "ADDED")
    first = d.get_endpoint_info()[0]
    svc.spec.selector = {"model": "two"}
    d._handle_resource(svc, "MODIFIED")
    assert d.get_endpoint_info()[0].model_label == "two"
    assert d.get_endpoint_info()[0] is not first
    d.close()


def test_service_name_new_uid_is_invalidated_before_readiness_query(monkeypatch):
    monkeypatch.setattr(
        "vllm_router.service_discovery.config.load_incluster_config", lambda: None
    )
    d = K8sServiceNameServiceDiscovery(FastAPI(), "ns", 8000)
    d._check_service_ready = Mock(return_value=True)
    d._get_model_names = Mock(return_value=["m"])
    d._get_model_info = Mock(return_value={})
    d._check_engine_sleep_mode = Mock(return_value=False)
    svc = NS(
        metadata=NS(
            name="s", uid="old", labels={}, annotations={}, deletion_timestamp=None
        ),
        spec=NS(selector={"model": "one"}),
    )
    try:
        d._handle_resource(svc, "ADDED")
        svc.metadata.uid = "new"
        d._check_service_ready.side_effect = RuntimeError("API unavailable")
        with pytest.raises(RuntimeError, match="API unavailable"):
            d._handle_resource(svc, "MODIFIED")
        assert d.get_endpoint_info() == []
    finally:
        d.close()
