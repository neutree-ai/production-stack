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
    prefill_count: int,
    decode_count: int,
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
        role_group_id=role_group_id,
        prefill_count=prefill_count,
        decode_count=decode_count,
    )


@pytest.mark.asyncio
async def test_pd_router_selects_prefill_and_decode_from_same_role_group():
    router = PDRouter(virtual_nodes_per_replica=8, load_factor=10.0)
    endpoints = [
        make_endpoint("http://10.0.0.1:9000", "rg-0", prefill_count=2, decode_count=2),
        make_endpoint("http://10.0.0.2:9000", "rg-1", prefill_count=2, decode_count=2),
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
    assert decision.headers["X-Neutree-PD-Prefill-Index"] == str(decision.prefill.index)
    assert decision.headers["X-Neutree-PD-Decode-Index"] == str(decision.decode.index)
    assert decision.headers["X-Neutree-PD-Role-Group"] == decision.decode.role_group_id


@pytest.mark.asyncio
async def test_pd_router_fails_closed_when_decode_group_has_no_prefill_unit():
    router = PDRouter(virtual_nodes_per_replica=1, load_factor=10.0)
    endpoints = [
        make_endpoint("http://10.0.0.1:9000", "rg-0", prefill_count=0, decode_count=1),
        make_endpoint("http://10.0.0.2:9000", "rg-1", prefill_count=1, decode_count=0),
    ]

    decision = await router.route_request(
        endpoints,
        engine_stats={},
        request_stats={},
        request=Request(),
        request_json={"model": "llama", "prompt": "hello"},
    )

    assert decision is None


def test_pd_routing_logic_is_registered_for_dynamic_endpoint_labels():
    cleanup_routing_logic()

    initialize_routing_logic(RoutingLogic.PD)

    assert isinstance(get_routing_logic_by_type("pd"), PDRouter)

    cleanup_routing_logic()
