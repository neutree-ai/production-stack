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
    role_group_id: str,
    prefill_count: int = 0,
    decode_count: int = 0,
    pd_role: str | None = None,
    pd_rank: int | None = None,
) -> EndpointInfo:
    return EndpointInfo(
        url=url,
        model_names=["llama"],
        Id=role_group_id,
        added_timestamp=0,
        model_label="llama",
        sleep=False,
        pod_name=f"{role_group_id}-pod",
        workspace="ws",
        endpoint="ep",
        routing_logic="pd",
        group_id=role_group_id,
        pd_role=pd_role,
        pd_rank=pd_rank,
        route_meta=(
            {f"{pd_role}_index": pd_rank}
            if pd_role in {"prefill", "decode"} and pd_rank is not None
            else {}
        ),
        role_group_id=role_group_id,
        prefill_count=prefill_count,
        decode_count=decode_count,
    )


@pytest.mark.asyncio
async def test_pd_router_selects_prefill_and_decode_from_same_role_group():
    router = PDRouter(virtual_nodes_per_replica=8, load_factor=10.0)
    endpoints = [
        make_endpoint("http://10.0.0.1:9000", "rg-0", pd_role="prefill", pd_rank=0),
        make_endpoint("http://10.0.0.1:9000", "rg-0", pd_role="decode", pd_rank=0),
        make_endpoint("http://10.0.0.2:9000", "rg-1", pd_role="prefill", pd_rank=0),
        make_endpoint("http://10.0.0.2:9000", "rg-1", pd_role="decode", pd_rank=0),
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
    assert decision.prefill.role_group_id == decision.decode.role_group_id
    assert decision.prefill.endpoint_info.pd_role == "prefill"
    assert decision.decode.endpoint_info.pd_role == "decode"
    assert decision.headers["X-Neutree-PD-Prefill-Index"] == str(decision.prefill.index)
    assert decision.headers["X-Neutree-PD-Decode-Index"] == str(decision.decode.index)
    assert decision.headers["X-Neutree-PD-Role-Group"] == decision.decode.role_group_id


@pytest.mark.asyncio
async def test_pd_router_fails_closed_when_decode_group_has_no_prefill_unit():
    router = PDRouter(virtual_nodes_per_replica=1, load_factor=10.0)
    endpoints = [
        make_endpoint("http://10.0.0.1:9000", "rg-0", pd_role="decode", pd_rank=0),
        make_endpoint("http://10.0.0.2:9000", "rg-1", pd_role="prefill", pd_rank=0),
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
async def test_pd_router_does_not_expand_unresolved_group_target_from_counts():
    router = PDRouter(virtual_nodes_per_replica=1, load_factor=10.0)
    endpoints = [
        make_endpoint(
            "http://10.0.0.1:9000",
            "rg-0",
            prefill_count=2,
            decode_count=2,
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
async def test_pd_router_prefill_uses_chwbl_within_selected_role_group(monkeypatch):
    cleanup_routing_logic()
    router = PDRouter(virtual_nodes_per_replica=8, load_factor=1.0)
    endpoints = [
        make_endpoint("http://10.0.0.1:9000", "rg-0", pd_role="prefill", pd_rank=0),
        make_endpoint("http://10.0.0.1:9000", "rg-0", pd_role="prefill", pd_rank=1),
        make_endpoint("http://10.0.0.1:9000", "rg-0", pd_role="decode", pd_rank=0),
    ]

    def fake_get_unit_load(state, unit):
        if unit.role == "prefill" and unit.index == 0:
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
    assert decision.prefill.index == 1
    assert decision.prefill.role_group_id == decision.decode.role_group_id

    cleanup_routing_logic()


def test_pd_routing_logic_is_registered_for_dynamic_endpoint_labels():
    cleanup_routing_logic()

    initialize_routing_logic(RoutingLogic.PD)

    assert isinstance(get_routing_logic_by_type("pd"), PDRouter)

    cleanup_routing_logic()
