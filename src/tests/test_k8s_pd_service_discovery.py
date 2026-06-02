import threading
from types import SimpleNamespace

import pytest

from vllm_router.service_discovery import (
    EndpointInfo,
    K8sPodIPServiceDiscovery,
)


def make_pod(labels=None, annotations=None):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="endpoint-collocated-0",
            uid="pod-uid-0",
            labels=labels or {},
            annotations=annotations or {},
        ),
        status=SimpleNamespace(
            pod_ip="10.0.0.1",
            container_statuses=[],
        ),
    )


def test_endpoint_info_rejects_deprecated_pd_count_metadata():
    with pytest.raises(TypeError):
        EndpointInfo(
            url="http://10.0.0.1:9000",
            model_names=["llama"],
            Id="deprecated",
            added_timestamp=0,
            model_label="llama",
            sleep=False,
            prefill_count=1,
            decode_count=1,
        )


def test_k8s_service_discovery_reads_group_target_static_metadata_from_pod():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    pod = make_pod(
        labels={
            "neutree.ai/role-group-id": "ignored-rg",
        },
        annotations={
            "neutree.io/pd-deployment-type": "group",
            "neutree.io/pd-sidecar-port": "9000",
        },
    )

    assert discovery._get_pd_domain(pod) == "endpoint-collocated-0"
    assert discovery._get_pd_deployment_type(pod) == "group"
    assert discovery._get_pd_sidecar_port(pod) == 9000
    assert discovery._get_pd_metadata(pod, "pd") == ("endpoint-collocated-0", 9000)


def test_k8s_service_discovery_rejects_incomplete_pd_metadata():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    pod = make_pod(labels={"routing_logic": "pd"})

    assert discovery._get_pd_domain(pod) == "endpoint-collocated-0"
    assert discovery._get_pd_sidecar_port(pod) is None
    assert discovery._get_pd_metadata(pod, "pd") == ("endpoint-collocated-0", None)


def test_k8s_service_discovery_expands_group_topology_to_rank_endpoints():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    discovery.namespace = "default"
    discovery.available_engines = {}
    discovery.available_engines_lock = threading.Lock()
    discovery.known_models = set()
    discovery.known_models_lock = threading.Lock()
    discovery.port = 8000
    discovery.app = SimpleNamespace(state=SimpleNamespace(event_loop=None))
    discovery._trigger_callbacks = lambda *args: None
    discovery.initialize_client_sessions = lambda: None
    model_info_ports = []
    discovery._get_model_info = (
        lambda engine_ip, port=None: model_info_ports.append(port) or {}
    )
    discovery._check_engine_sleep_mode = lambda engine_name: False
    discovery._get_pd_metadata_for_engine = lambda engine_name, routing_logic: (
        "endpoint-collocated-0",
        9000,
    )
    discovery._get_pd_topology = lambda engine_ip, pd_sidecar_port: SimpleNamespace(
        group_id="endpoint-collocated-0",
        units=[
            SimpleNamespace(role="prefill", rank=0),
            SimpleNamespace(role="prefill", rank=1),
            SimpleNamespace(role="decode", rank=0),
        ],
    )

    discovery._add_engine(
        engine_name="endpoint-collocated-0",
        engine_ip="10.0.0.1",
        model_names=["llama"],
        model_label="llama",
        workspace="ws",
        endpoint="ep",
        routing_logic="pd",
    )

    endpoints = discovery.get_endpoint_info()

    assert len(endpoints) == 3
    assert {endpoint.role for endpoint in endpoints} == {"prefill", "decode"}
    assert {endpoint.url for endpoint in endpoints} == {"http://10.0.0.1:9000"}
    assert {
        (endpoint.role, endpoint.rank)
        for endpoint in endpoints
        if endpoint.role == "prefill"
    } == {("prefill", 0), ("prefill", 1)}
    assert {
        (endpoint.role, endpoint.rank)
        for endpoint in endpoints
        if endpoint.role == "decode"
    } == {("decode", 0)}
    assert {endpoint.domain for endpoint in endpoints} == {"endpoint-collocated-0"}
    assert model_info_ports == [9000]


def test_k8s_service_discovery_gets_pd_model_names_from_sidecar_port():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    discovery.running = True
    discovery.namespace = "default"
    discovery.label_selector = "release=router"
    discovery.watcher_timeout_seconds = 0
    discovery.port = 8000
    calls = []

    pod = make_pod(
        annotations={
            "neutree.io/pd-deployment-type": "group",
            "neutree.io/pd-sidecar-port": "9000",
        }
    )

    class Watcher:
        def stream(self, *args, **kwargs):
            discovery.running = False
            return [{"type": "ADDED", "object": pod}]

    discovery.k8s_watcher = Watcher()
    discovery.k8s_api = SimpleNamespace(list_namespaced_pod=lambda *args, **kwargs: [])
    discovery._is_pod_terminating = lambda pod: False
    discovery._check_pod_ready = lambda container_statuses: True
    discovery._get_model_names = lambda pod_ip, port=None: calls.append(
        ("models", pod_ip, port)
    ) or ["llama"]
    discovery._get_model_label = lambda pod: "llama"
    discovery._get_workspace = lambda pod: "ws"
    discovery._get_endpoint = lambda pod: "ep"
    discovery._get_routing_logic = lambda pod: "pd"
    discovery._on_engine_update = lambda *args: calls.append(("update", args))

    discovery._watch_engines()

    assert ("models", "10.0.0.1", 9000) in calls


def test_k8s_service_discovery_deletes_all_expanded_group_endpoints():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    discovery.available_engines = {
        "pod-0:prefill:0": EndpointInfo(
            url="http://10.0.0.1:9000",
            model_names=["llama"],
            Id="pod-0:prefill:0",
            added_timestamp=0,
            model_label="llama",
            sleep=False,
            pod_name="pod-0",
            routing_logic="pd",
            role="prefill",
            rank=0,
        ),
        "pod-0:decode:0": EndpointInfo(
            url="http://10.0.0.1:9000",
            model_names=["llama"],
            Id="pod-0:decode:0",
            added_timestamp=0,
            model_label="llama",
            sleep=False,
            pod_name="pod-0",
            routing_logic="pd",
            role="decode",
            rank=0,
        ),
    }
    discovery.available_engines_lock = threading.Lock()
    deleted = []
    discovery._trigger_callbacks = lambda event, name, endpoint: deleted.append(name)

    discovery._delete_engine("pod-0")

    assert discovery.available_engines == {}
    assert deleted == ["pod-0:prefill:0", "pod-0:decode:0"]


def test_k8s_service_discovery_ignores_ready_modified_pod_without_state_change():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    discovery.port = 8000
    discovery.available_engines = {
        "pod-0:prefill:0": EndpointInfo(
            url="http://10.0.0.1:9000",
            model_names=["llama"],
            Id="pod-0:prefill:0",
            added_timestamp=0,
            model_label="llama",
            sleep=False,
            pod_name="pod-0",
            workspace="ws",
            endpoint="ep",
            routing_logic="pd",
            domain="pod-0",
            role="prefill",
            rank=0,
        )
    }
    discovery.available_engines_lock = threading.Lock()
    calls = []

    def delete_engine(engine_name):
        calls.append(("delete", engine_name))
        for key, endpoint_info in list(discovery.available_engines.items()):
            if key == engine_name or endpoint_info.pod_name == engine_name:
                del discovery.available_engines[key]

    def add_engine(
        engine_name,
        engine_ip,
        model_names,
        model_label,
        workspace,
        endpoint,
        routing_logic,
    ):
        calls.append(("add", engine_name))

    discovery._delete_engine = delete_engine
    discovery._add_engine = add_engine

    discovery._on_engine_update(
        engine_name="pod-0",
        engine_ip="10.0.0.1",
        event="MODIFIED",
        is_pod_ready=True,
        model_names=["llama"],
        model_label="llama",
        workspace="ws",
        endpoint="ep",
        routing_logic="pd",
    )

    assert calls == []
