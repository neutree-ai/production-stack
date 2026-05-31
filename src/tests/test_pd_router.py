from typing import Dict

import pytest

from vllm_router.routers.routing_logic import (
    PDRouter,
    RoutingLogic,
    cleanup_routing_logic,
    get_routing_logic_by_type,
    initialize_routing_logic,
)
from vllm_router.service_discovery import EndpointInfo


class Request:
    def __init__(self, headers: Dict[str, str] | None = None):
        self.headers = headers or {}


def make_endpoint(
    url: str,
    domain: str,
    role: str | None = None,
    rank: int | None = None,
) -> EndpointInfo:
    return EndpointInfo(
        url=url,
        model_names=["llama"],
        Id=domain,
        added_timestamp=0,
        model_label="llama",
        sleep=False,
        pod_name=f"{domain}-pod",
        workspace="ws",
        endpoint="ep",
        routing_logic="pd",
        domain=domain,
        role=role,
        rank=rank,
    )


@pytest.mark.asyncio
async def test_pd_router_selects_prefill_and_decode_from_same_domain():
    router = PDRouter(virtual_nodes_per_replica=8, load_factor=10.0)
    endpoints = [
        make_endpoint("http://10.0.0.1:9000", "domain-0", role="prefill", rank=0),
        make_endpoint("http://10.0.0.1:9000", "domain-0", role="decode", rank=0),
        make_endpoint("http://10.0.0.2:9000", "domain-1", role="prefill", rank=0),
        make_endpoint("http://10.0.0.2:9000", "domain-1", role="decode", rank=0),
    ]

    decision = await router.route_request(
        endpoints,
        engine_stats={},
        request_stats={},
        request=Request(),
        request_json={"model": "llama", "prompt": "hello"},
    )

    assert decision is not None
    assert decision.url in {endpoint.url for endpoint in endpoints}
    assert decision.prefill.domain == decision.decode.domain
    assert decision.prefill.endpoint_info.role == "prefill"
    assert decision.decode.endpoint_info.role == "decode"
    assert decision.headers["X-Neutree-PD-Prefill-Index"] == str(decision.prefill.rank)
    assert decision.headers["X-Neutree-PD-Decode-Index"] == str(decision.decode.rank)
    assert decision.headers["X-Neutree-PD-Role-Group"] == decision.decode.domain


@pytest.mark.asyncio
async def test_pd_router_fails_closed_when_decode_group_has_no_prefill_unit():
    router = PDRouter(virtual_nodes_per_replica=1, load_factor=10.0)
    endpoints = [
        make_endpoint("http://10.0.0.1:9000", "domain-0", role="decode", rank=0),
        make_endpoint("http://10.0.0.2:9000", "domain-1", role="prefill", rank=0),
    ]

    decision = await router.route_request(
        endpoints,
        engine_stats={},
        request_stats={},
        request=Request(),
        request_json={"model": "llama", "prompt": "hello"},
    )

    assert decision is None


@pytest.mark.asyncio
async def test_pd_router_does_not_expand_unresolved_group_target():
    router = PDRouter(virtual_nodes_per_replica=1, load_factor=10.0)
    endpoints = [
        make_endpoint(
            "http://10.0.0.1:9000",
            "domain-0",
        ),
    ]

    decision = await router.route_request(
        endpoints,
        engine_stats={},
        request_stats={},
        request=Request(),
        request_json={"model": "llama", "prompt": "hello"},
    )

    assert decision is None


@pytest.mark.asyncio
async def test_pd_router_prefill_uses_chwbl_within_selected_domain(monkeypatch):
    cleanup_routing_logic()
    router = PDRouter(virtual_nodes_per_replica=8, load_factor=1.0)
    endpoints = [
        make_endpoint("http://10.0.0.1:9000", "domain-0", role="prefill", rank=0),
        make_endpoint("http://10.0.0.1:9000", "domain-0", role="prefill", rank=1),
        make_endpoint("http://10.0.0.1:9000", "domain-0", role="decode", rank=0),
    ]

    def fake_get_unit_load(state, unit):
        if unit.role == "prefill" and unit.rank == 0:
            return 100
        return 0

    monkeypatch.setattr(router, "_get_unit_load", fake_get_unit_load)

    decision = await router.route_request(
        endpoints,
        engine_stats={},
        request_stats={},
        request=Request(),
        request_json={"model": "llama", "prompt": "prefill-load-test-0"},
    )

    assert decision is not None
    assert decision.prefill.rank == 1
    assert decision.prefill.domain == decision.decode.domain

    cleanup_routing_logic()


def test_pd_routing_logic_is_registered_for_dynamic_endpoint_labels():
    cleanup_routing_logic()

    initialize_routing_logic(RoutingLogic.PD)

    assert isinstance(get_routing_logic_by_type("pd"), PDRouter)

    cleanup_routing_logic()
