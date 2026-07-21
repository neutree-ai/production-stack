# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Catch-all forwarding for non-OpenAI workloads.

The regular routes reach a backend by matching the ``model`` field in a JSON
body against models discovered from each pod's ``/v1/models``. A workload that
speaks neither cannot be reached at all -- it is rejected during service
discovery, long before routing.

This router forwards any method and any path to a backend chosen purely from
the ``(workspace, endpoint)`` pair already present in the URL, which service
discovery reads from pod labels rather than probing over HTTP. It applies only
to pods labelled ``passthrough=true``; every other pod keeps its existing
OpenAI-shaped contract, so opening arbitrary paths on an inference engine is
never a side effect of deploying this router.

Registration order matters: this router must be included *last* so that all
explicitly declared routes -- including the three-segment ones such as
``/v1/files/{file_id}`` -- win the match.
"""

import uuid
from typing import Dict, List, Tuple

import aiohttp
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_router.log import init_logger
from vllm_router.service_discovery import EndpointInfo, get_service_discovery
from vllm_router.services.metrics_service import num_incoming_requests_total
from vllm_router.services.request_service.request import _HOP_BY_HOP_HEADERS

passthrough_router = APIRouter()

logger = init_logger(__name__)

# HEAD is omitted deliberately: Starlette adds it whenever GET is present.
_PASSTHROUGH_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

# One long-lived aiohttp timeout for every proxied request. Immutable, so it is
# safe to share and pointless to rebuild per request.
_NO_TIMEOUT = aiohttp.ClientTimeout(total=None)

# Round-robin cursors, one per (workspace, endpoint). Passthrough workloads are
# opaque to the router, so none of the OpenAI-aware strategies (prefix cache,
# session affinity, KV-aware) have anything to work with here.
#
# No lock: handlers are async and this function never awaits, so it cannot be
# preempted between the read and the write.
_rr_counters: Dict[Tuple[str, str], int] = {}


def _select_backend(
    endpoints: List[EndpointInfo], key: Tuple[str, str]
) -> EndpointInfo:
    index = _rr_counters.get(key, 0)
    _rr_counters[key] = index + 1
    # Sorted, so the rotation survives service discovery handing back the pool
    # in a different order after a pod restart.
    return sorted(endpoints, key=lambda e: e.url)[index % len(endpoints)]


def _passthrough_endpoints(workspace: str, endpoint: str) -> List[EndpointInfo]:
    return [
        info
        for info in get_service_discovery().get_endpoint_info()
        if info.passthrough
        and info.workspace == workspace
        and info.endpoint == endpoint
    ]


async def _stream_upstream(
    request: Request,
    backend_url: str,
    upstream_path: str,
    body: bytes,
):
    """
    Forward one request upstream and yield (headers, status) then body chunks.

    Deliberately leaner than ``process_request``: no semantic cache, no
    callbacks, no body parsing. Those all assume an OpenAI-shaped JSON payload,
    which is exactly the assumption this path exists to drop.
    """
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS
    }

    async with request.app.state.aiohttp_client_wrapper().request(
        method=request.method,
        url=f"{backend_url}/{upstream_path}",
        headers=headers,
        params=request.query_params.multi_items(),
        data=body,
        timeout=_NO_TIMEOUT,
    ) as backend_response:
        yield backend_response.headers, backend_response.status
        async for chunk in backend_response.content.iter_any():
            yield chunk


@passthrough_router.api_route(
    "/{workspace}/{endpoint}/{upstream_path:path}",
    methods=_PASSTHROUGH_METHODS,
)
async def route_passthrough_request(
    workspace: str,
    endpoint: str,
    upstream_path: str,
    request: Request,
):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    endpoints = _passthrough_endpoints(workspace, endpoint)
    if not endpoints:
        # Deliberately indistinguishable from "endpoint does not exist": an
        # OpenAI endpoint reached on an undeclared path lands here too, and
        # should not learn that it exists but is not in passthrough mode.
        return JSONResponse(
            status_code=404,
            content={
                "error": (
                    f"No passthrough backend available for endpoint "
                    f"'{endpoint}' in workspace '{workspace}'."
                )
            },
            headers={"X-Request-Id": request_id},
        )

    num_incoming_requests_total.labels(workspace=workspace, endpoint=endpoint).inc()

    backend = _select_backend(endpoints, (workspace, endpoint))
    body = await request.body()

    logger.debug(
        f"Passthrough request {request_id} -> {backend.url}/{upstream_path} "
        f"({request.method})"
    )

    stream = _stream_upstream(request, backend.url, upstream_path, body)
    upstream_headers, status = await anext(stream)

    # Strip hop-by-hop headers on the way back too. content-length in
    # particular must go: the response is re-framed as a stream, so a length
    # copied from upstream would contradict the transfer encoding.
    response_headers = {
        k: v
        for k, v in upstream_headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }
    response_headers["X-Request-Id"] = request_id

    # No media_type=: Content-Type is already in response_headers, and
    # Response.init_headers only fills one in when the headers dict lacks it.
    return StreamingResponse(
        stream,
        status_code=status,
        headers=response_headers,
    )
