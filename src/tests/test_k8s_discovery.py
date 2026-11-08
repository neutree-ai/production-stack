import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_router import service_discovery as discovery_module
from vllm_router.service_discovery import (
    K8sPodIPServiceDiscovery,
    K8sServiceNameServiceDiscovery,
    ModelInfo,
    ServiceDiscoveryEventType,
    StaticServiceDiscovery,
)

DISCOVERIES = [K8sPodIPServiceDiscovery, K8sServiceNameServiceDiscovery]


def resource(name, ip="10.0.0.1", workspace="ws", passthrough=True):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            deletion_timestamp=None,
            labels={
                "workspace": workspace,
                "endpoint": "service",
                "passthrough": str(passthrough).lower(),
                "model": "decode",
            },
        ),
        spec=SimpleNamespace(selector={"model": "decode"}),
        status=SimpleNamespace(
            pod_ip=ip, container_statuses=[SimpleNamespace(ready=True)]
        ),
    )


def page(items, token=None, version="42"):
    return SimpleNamespace(
        items=items,
        metadata=SimpleNamespace(_continue=token, resource_version=version),
    )


@pytest.fixture
def make_discovery(monkeypatch):
    monkeypatch.setattr(
        "vllm_router.service_discovery.config.load_incluster_config", Mock()
    )
    monkeypatch.setattr("vllm_router.service_discovery.threading.Thread.start", Mock())

    def make(cls, **kwargs):
        discovery = cls(
            SimpleNamespace(state=SimpleNamespace()), "ns", "8000", **kwargs
        )
        monkeypatch.setattr(discovery, "_get_model_names", lambda _: ["m"])
        monkeypatch.setattr(discovery, "_get_model_info", lambda _: {})
        monkeypatch.setattr(discovery, "_check_engine_sleep_mode", lambda _: False)
        if cls is K8sServiceNameServiceDiscovery:
            monkeypatch.setattr(discovery, "_check_service_ready", lambda *_: True)
        return discovery

    return make


@pytest.mark.parametrize("cls", DISCOVERIES)
def test_watcher_observes_initialized_pd_fields(make_discovery, monkeypatch, cls):
    observed = []

    def start(thread):
        discovery = thread._target.__self__
        observed.append((discovery.prefill_model_labels, discovery.decode_model_labels))

    monkeypatch.setattr("vllm_router.service_discovery.threading.Thread.start", start)
    make_discovery(
        cls, prefill_model_labels=["prefill"], decode_model_labels=["decode"]
    )
    assert observed == [(["prefill"], ["decode"])]


@pytest.mark.parametrize("cls", DISCOVERIES)
def test_paginated_relist_reconciles_before_resuming_watch(
    make_discovery, monkeypatch, cls
):
    discovery = make_discovery(cls)
    if cls is K8sPodIPServiceDiscovery:
        discovery._on_engine_update(
            "old", "10.0.0.8", "ADDED", True, [], None, "ws", "service", None, True
        )
    else:
        discovery._on_engine_update("old", "ADDED", True, ["m"], "decode")
    list_resources = Mock(
        side_effect=[page([resource("one")], "next"), page([resource("two")])]
    )
    name = (
        "list_namespaced_pod"
        if cls is K8sPodIPServiceDiscovery
        else "list_namespaced_service"
    )
    monkeypatch.setattr(discovery.k8s_api, name, list_resources)
    watch_options = []

    def stream(*args, **kwargs):
        watch_options.append(kwargs)
        discovery.running = False
        return iter(())

    monkeypatch.setattr(discovery.k8s_watcher, "stream", stream)
    discovery._watch_engines()
    assert set(discovery.available_engines) == {"one", "two"}
    assert list_resources.call_args_list[1].kwargs["_continue"] == "next"
    assert watch_options[0]["resource_version"] == "42"
    assert 0 < watch_options[0]["timeout_seconds"] <= 30
    assert list_resources.call_args_list[0].kwargs["_request_timeout"] == (3, 10)


@pytest.mark.parametrize("cls", DISCOVERIES)
def test_incomplete_list_keeps_cache_until_complete_retry(
    make_discovery, monkeypatch, cls
):
    discovery = make_discovery(cls)
    if cls is K8sPodIPServiceDiscovery:
        discovery._on_engine_update(
            "old", "10.0.0.8", "ADDED", True, [], None, "ws", "service", None, True
        )
    else:
        discovery._on_engine_update("old", "ADDED", True, ["m"], "decode")
    previous = discovery.get_endpoint_info()
    lists = Mock(
        side_effect=[
            page([resource("partial")], "next"),
            OSError("apiserver down"),
            page([resource("new")]),
        ]
    )
    name = (
        "list_namespaced_pod"
        if cls is K8sPodIPServiceDiscovery
        else "list_namespaced_service"
    )
    monkeypatch.setattr(discovery.k8s_api, name, lists)
    waits = []

    def wait(seconds):
        waits.append(seconds)
        assert discovery.get_endpoint_info() == previous
        return False

    discovery._stop_event.wait = wait

    def stream(*args, **kwargs):
        discovery.running = False
        return iter(())

    monkeypatch.setattr(discovery.k8s_watcher, "stream", stream)
    discovery._watch_engines()
    assert waits == [5]
    assert set(discovery.available_engines) == {"new"}


@pytest.mark.parametrize("cls", DISCOVERIES)
def test_shutdown_stops_fetching_more_list_pages(make_discovery, monkeypatch, cls):
    discovery = make_discovery(cls)

    def first_page(**kwargs):
        discovery.running = False
        discovery._stop_event.set()
        return page([resource("partial")], "next")

    calls = []

    def lists(**kwargs):
        calls.append(kwargs)
        return first_page(**kwargs) if len(calls) == 1 else page([resource("last")])

    name = (
        "list_namespaced_pod"
        if cls is K8sPodIPServiceDiscovery
        else "list_namespaced_service"
    )
    monkeypatch.setattr(discovery.k8s_api, name, lists)
    discovery._watch_engines()
    assert len(calls) == 1
    assert discovery.get_endpoint_info() == []


def test_ready_pod_identity_update_replaces_old_tenant_and_notifies(make_discovery):
    events = []
    discovery = make_discovery(
        K8sPodIPServiceDiscovery, event_callbacks=[lambda *args: events.append(args)]
    )
    discovery._on_engine_update(
        "pod", "10.0.0.1", "ADDED", True, ["m"], "decode", "old", "service", None, False
    )
    discovery._on_engine_update(
        "pod",
        "10.0.0.2",
        "MODIFIED",
        True,
        [],
        "decode",
        "new",
        "other",
        "roundrobin",
        True,
    )
    endpoint = discovery.get_endpoint_info()[0]
    assert (
        endpoint.url,
        endpoint.workspace,
        endpoint.endpoint,
        endpoint.passthrough,
    ) == ("http://10.0.0.2:8000", "new", "other", True)
    assert [event[0] for event in events] == [
        ServiceDiscoveryEventType.ENGINE_ADDED,
        ServiceDiscoveryEventType.ENGINE_DELETED,
        ServiceDiscoveryEventType.ENGINE_ADDED,
    ]
    discovery._on_engine_update(
        "pod", None, "MODIFIED", False, [], None, None, None, None, False
    )
    assert discovery.get_endpoint_info() == []


@pytest.mark.parametrize("cls", DISCOVERIES)
def test_sleep_and_wake_modified_events_refresh_endpoint_state(
    make_discovery, monkeypatch, cls
):
    discovery = make_discovery(cls)
    monkeypatch.setattr(discovery, "_check_engine_sleep_mode", lambda _: True)
    monkeypatch.setattr(
        discovery, "_get_engine_sleep_status", Mock(side_effect=[False, True, False])
    )
    initial = {"m": ModelInfo(id="m", object="model")}
    updated = {"m": ModelInfo(id="m", object="model", parent="base", is_adapter=True)}
    monkeypatch.setattr(
        discovery, "_get_model_info", Mock(side_effect=[initial, updated, updated])
    )
    backend = resource("backend", passthrough=False)
    handle = (
        discovery._handle_pod
        if cls is K8sPodIPServiceDiscovery
        else discovery._handle_service
    )
    handle(backend, "ADDED")
    assert discovery.get_endpoint_info()[0].sleep is False
    backend.metadata.labels["sleeping"] = "true"
    handle(backend, "MODIFIED")
    assert discovery.get_endpoint_info()[0].sleep is True
    assert discovery.get_endpoint_info()[0].model_info == updated
    backend.metadata.labels.pop("sleeping")
    handle(backend, "MODIFIED")
    assert discovery.get_endpoint_info()[0].sleep is False


@pytest.mark.parametrize("cls", [StaticServiceDiscovery, *DISCOVERIES])
def test_loop_ready_refreshes_targets_without_replacing_shared_session(
    make_discovery, cls
):
    async def run():
        if cls is StaticServiceDiscovery:
            discovery = cls(
                SimpleNamespace(state=SimpleNamespace()),
                ["http://old"],
                ["m"],
                model_labels=["decode"],
                prefill_model_labels=["prefill"],
                decode_model_labels=["decode"],
            )
        else:
            discovery = make_discovery(
                cls, prefill_model_labels=["prefill"], decode_model_labels=["decode"]
            )
            if cls is K8sPodIPServiceDiscovery:
                discovery._on_engine_update(
                    "pod",
                    "10.0.0.1",
                    "ADDED",
                    True,
                    [],
                    "decode",
                    "ws",
                    "service",
                    None,
                    True,
                )
            else:
                discovery._on_engine_update("pod", "ADDED", True, ["m"], "decode")
        shared = Mock()
        discovery.app.state.aiohttp_client_wrapper = lambda: shared
        discovery.set_event_loop(asyncio.get_running_loop())
        await discovery.initialize_client_sessions()
        before = discovery.app.state.decode_endpoint
        if cls is StaticServiceDiscovery:
            discovery.urls = ["http://new"]
        elif cls is K8sPodIPServiceDiscovery:
            discovery._on_engine_update(
                "pod",
                "10.0.0.2",
                "MODIFIED",
                True,
                [],
                "decode",
                "ws",
                "service",
                None,
                True,
            )
        else:
            discovery._on_engine_update("pod", "MODIFIED", False, [], None)
        await discovery.initialize_client_sessions()
        assert discovery.app.state.decode_endpoint != before
        assert before.url
        shared.close.assert_not_called()
        await asyncio.sleep(0)

    asyncio.run(run())


@pytest.mark.parametrize("loop_published", [True, False])
def test_reconfigure_refreshes_pd_targets_when_loop_is_published(
    monkeypatch, loop_published
):
    async def run():
        monkeypatch.setattr(discovery_module, "_global_service_discovery", None)
        app = SimpleNamespace(state=SimpleNamespace())
        if loop_published:
            app.state.event_loop = asyncio.get_running_loop()
        options = dict(
            app=app,
            models=["m"],
            model_labels=["decode"],
            prefill_model_labels=["prefill"],
            decode_model_labels=["decode"],
        )
        previous = discovery_module.initialize_service_discovery(
            discovery_module.ServiceDiscoveryType.STATIC, urls=["http://old"], **options
        )
        if loop_published:
            previous.set_event_loop(app.state.event_loop)
            await previous.initialize_client_sessions()
        updated = discovery_module.reconfigure_service_discovery(
            discovery_module.ServiceDiscoveryType.STATIC, urls=["http://new"], **options
        )
        if not loop_published:
            app.state.event_loop = asyncio.get_running_loop()
            discovery_module.get_service_discovery().set_event_loop(
                app.state.event_loop
            )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert app.state.decode_endpoint.url == "http://new"
        updated.close()

    asyncio.run(run())
