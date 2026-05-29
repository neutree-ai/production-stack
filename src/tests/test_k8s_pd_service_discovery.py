from types import SimpleNamespace
import threading

from vllm_router.service_discovery import EndpointInfo, K8sPodIPServiceDiscovery


def make_pod(labels=None, annotations=None):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="endpoint-collocated-0",
            labels=labels or {},
            annotations=annotations or {},
        )
    )


def test_k8s_service_discovery_reads_pd_metadata_from_pod_metadata():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    pod = make_pod(
        labels={
            "neutree.ai/role-group-id": "rg-0",
        },
        annotations={
            "neutree.ai/prefill-replicas": "2",
            "neutree.ai/decode-replicas": "3",
            "neutree.ai/pd-sidecar-port": "9000",
        },
    )

    assert discovery._get_role_group_id(pod) == "rg-0"
    assert discovery._get_pd_role_counts(pod) == (2, 3)
    assert discovery._get_pd_sidecar_port(pod) == 9000


def test_k8s_service_discovery_rejects_incomplete_pd_metadata():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    pod = make_pod(labels={"routing_logic": "pd"})

    assert discovery._get_role_group_id(pod) == "endpoint-collocated-0"
    assert discovery._get_pd_role_counts(pod) == (None, None)
    assert discovery._get_pd_sidecar_port(pod) is None
    assert discovery._get_pd_metadata(pod, "pd") == (
        "endpoint-collocated-0",
        None,
        None,
        None,
    )


def test_k8s_service_discovery_refreshes_ready_pod_when_pd_metadata_changes():
    discovery = object.__new__(K8sPodIPServiceDiscovery)
    discovery.port = 8000
    discovery.available_engines = {
        "pod-0": EndpointInfo(
            url="http://10.0.0.1:9000",
            model_names=["llama"],
            Id="pod-0",
            added_timestamp=0,
            model_label="llama",
            sleep=False,
            workspace="ws",
            endpoint="ep",
            routing_logic="pd",
            role_group_id="rg-0",
            prefill_count=1,
            decode_count=1,
        )
    }
    discovery.available_engines_lock = threading.Lock()
    calls = []

    def delete_engine(engine_name):
        calls.append(("delete", engine_name))
        del discovery.available_engines[engine_name]

    def add_engine(
        engine_name,
        engine_ip,
        model_names,
        model_label,
        workspace,
        endpoint,
        routing_logic,
        role_group_id,
        prefill_count,
        decode_count,
        pd_sidecar_port,
    ):
        calls.append(
            (
                "add",
                engine_name,
                role_group_id,
                prefill_count,
                decode_count,
                pd_sidecar_port,
            )
        )

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
        role_group_id="rg-0",
        prefill_count=2,
        decode_count=3,
        pd_sidecar_port=9000,
    )

    assert calls == [
        ("delete", "pod-0"),
        ("add", "pod-0", "rg-0", 2, 3, 9000),
    ]
