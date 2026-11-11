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

# --- Request Processing & Routing ---
import asyncio
import json
import math
import os
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile

from vllm_router.log import init_logger
from vllm_router.routers.routing_logic import (
    ConsistentHashRouter,
    DisaggregatedPrefillRouter,
    KvawareRouter,
    PrefixAwareRouter,
    RoutingLogic,
    SessionRouter,
    StaticHashRouter,
    get_routing_logic_by_type,
)
from vllm_router.service_discovery import get_service_discovery
from vllm_router.services.request_service.rewriter import (
    get_request_rewriter,
    is_request_rewriter_initialized,
)
from vllm_router.utils import replace_model_in_request_body, update_content_length

try:
    # Semantic cache integration
    from vllm_router.experimental.semantic_cache_integration import (
        store_in_semantic_cache,
    )

    semantic_cache_available = True
except ImportError:
    semantic_cache_available = False

from vllm_router.services.metrics_service import num_incoming_requests_total
from vllm_router.services.metrics_service.forwarding import trace_context

logger = init_logger(__name__)

_HOP_BY_HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "content-length",
    "upgrade",
    "te",  # codespell:ignore
    "trailer",
}


# TODO: (Brian) check if request is json beforehand
async def process_request(
    request: Request,
    body,
    backend_url,
    request_id,
    endpoint,
    background_tasks: BackgroundTasks,
    debug_request=None,
):
    """
    Process a request by sending it to the chosen backend.

    Args:
        request(Request): Request object.
        body: The content of the request to send to the backend.
        backend_url: The URL of the backend to send the request to.
        request_id: A unique identifier for the request.
        endpoint: The endpoint to send the request to on the backend.
        debug_request: The original request object from the client, used for
            optional debug logging.

    Yields:
        The response headers and status code, followed by the response content.

    Raises:
        HTTPError: If the backend returns a 4xx or 5xx status code.
    """
    # Check if this is a streaming request
    try:
        request_json = json.loads(body)
        is_streaming = request_json.get("stream", False)
    except json.JSONDecodeError:
        # If we can't parse the body as JSON, assume it's not streaming
        raise HTTPException(status=400, detail="Request body is not JSON parsable.")

    first_token = False
    total_len = 0
    start_time = time.time()
    request_stats_monitor = request.app.state.request_stats_monitor
    request_stats_monitor.on_new_request(backend_url, request_id, start_time)

    # sanitize the request headers
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS
    }

    # For non-streaming requests, collect the full response to cache it properly
    full_response = bytearray()

    success = False
    outcome = "client_disconnected"
    exception_type = "GeneratorExit"
    try:
        async with request.app.state.aiohttp_client_wrapper().request(
            method=request.method,
            url=backend_url + endpoint,
            headers=headers,
            data=body,
            trace_request_ctx=trace_context(
                *getattr(request.state, "backend_labels", ("", "", backend_url, ""))
            ),
        ) as backend_response:
            # Yield headers and status code first.
            yield backend_response.headers, backend_response.status
            # Stream response content.
            async for chunk in backend_response.content.iter_any():
                total_len += len(chunk)
                if not first_token:
                    first_token = True
                    request_stats_monitor.on_request_response(
                        backend_url, request_id, time.time()
                    )
                # For non-streaming requests, collect the full response
                if full_response is not None:
                    full_response.extend(chunk)
                yield chunk
        success = True
        outcome = "completed"
        exception_type = None
    except asyncio.CancelledError:
        outcome = "cancelled"
        exception_type = "CancelledError"
        raise
    except GeneratorExit:
        raise
    except Exception as exc:
        outcome = "backend_error"
        exception_type = type(exc).__name__
        raise
    finally:
        # Every exit releases active state; only a fully read response is completed.
        stage, active_after = request_stats_monitor.on_request_complete(
            backend_url, request_id, time.time(), success=success
        )
        if not success:
            logger.warning(
                "request_terminated request_id=%s backend=%s stage=%s outcome=%s "
                "exception_type=%s active_after=%s",
                request_id,
                backend_url,
                stage,
                outcome,
                exception_type,
                active_after,
            )

    # if debug_request:
    #    logger.debug(f"Finished the request with request id: {debug_request.headers.get('x-request-id', None)} at {time.time()}")
    # Store in semantic cache if applicable
    # Use the full response for non-streaming requests, or the last chunk for streaming
    if request.app.state.semantic_cache_available:
        cache_chunk = bytes(full_response) if not is_streaming else chunk
        await store_in_semantic_cache(
            endpoint=endpoint, method=request.method, body=body, chunk=cache_chunk
        )
    if background_tasks and getattr(request.app.state, "callbacks", None):
        background_tasks.add_task(
            request.app.state.callbacks.post_request, request, full_response
        )


async def route_general_request(
    request: Request,
    endpoint: str,
    background_tasks: BackgroundTasks,
    workspace: Optional[str] = None,
    endpoint_name: Optional[str] = None,
):
    """
    Route the incoming request to the backend server and stream the response back to the client.

    This function extracts the requested model from the request body and retrieves the
    corresponding endpoints. It uses routing logic to determine the best server URL to handle
    the request, then streams the request to that server. If the requested model is not available,
    it returns an error response.

    Args:
        request (Request): The incoming HTTP request.
        endpoint (str): The endpoint to which the request should be routed.

    Returns:
        StreamingResponse: A response object that streams data from the backend server to the client.
    """
    if isinstance(request.app.state.router, DisaggregatedPrefillRouter):
        response = await route_disaggregated_prefill_request(
            request, endpoint, background_tasks
        )
        return response
    in_router_time = time.time()
    # Same as vllm, Get request_id from X-Request-Id header if available
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request_body = await request.body()
    try:
        request_json = json.loads(request_body) if request_body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            status_code=400, content={"error": "Request body must be a JSON object."}
        )
    if not isinstance(request_json, dict):
        return JSONResponse(
            status_code=400, content={"error": "Request body must be a JSON object."}
        )

    if request.query_params:
        request_endpoint = request.query_params.get("id")
    else:
        request_endpoint = None

    if getattr(request.app.state, "callbacks", None) and (
        response_overwrite := request.app.state.callbacks.pre_request(
            request, request_body, request_json
        )
    ):
        response_overwrite.headers["X-Request-Id"] = request_id
        return response_overwrite

    requested_model = request_json.get("model", None)
    if not isinstance(requested_model, str) or not requested_model:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request: missing 'model' in request body."},
            headers={"X-Request-Id": request_id},
        )

    # Apply request rewriting if enabled
    if is_request_rewriter_initialized():
        rewriter = get_request_rewriter()
        rewritten_body = rewriter.rewrite_request(
            request_body, requested_model, endpoint
        )
        logger.info(f"Request for model {requested_model} was rewritten")
        request_body = rewritten_body
        # Update request_json if the body was rewritten
        try:
            request_json = json.loads(request_body)
        except json.JSONDecodeError:
            logger.warning("Failed to parse rewritten request body as JSON")
            raise HTTPException(
                status_code=400, detail="Request body is not JSON parsable."
            )

    service_discovery = get_service_discovery()
    endpoints = service_discovery.get_endpoint_info()

    aliases = getattr(service_discovery, "aliases", None)
    if aliases and requested_model in aliases.keys():
        requested_model = aliases[requested_model]
        request_body = replace_model_in_request_body(request_json, requested_model)
        update_content_length(request, request_body)

    # Check if model has ever been seen (even if currently scaled to zero)
    model_ever_existed = False
    if hasattr(service_discovery, "has_ever_seen_model"):
        model_ever_existed = service_discovery.has_ever_seen_model(requested_model)

    # --- Endpoint Filtering ---
    # 1. Initial filter by requested model and sleep status
    endpoints = list(
        filter(
            lambda x: requested_model in x.model_names and not x.sleep,
            endpoints,
        )
    )

    # 2. Further filter by workspace and endpoint name if provided
    if workspace and endpoint_name:
        endpoints = list(
            filter(
                lambda x: x.workspace == workspace and x.endpoint == endpoint_name,
                endpoints,
            )
        )

    # 3. Finally, filter by specific endpoint ID if provided in query params
    if request_endpoint:
        endpoints = list(
            filter(
                lambda x: x.Id == request_endpoint,
                endpoints,
            )
        )
    # Get engine stats and request stats
    engine_stats = {}
    if request.app.state.engine_stats_scraper is not None:
        engine_stats = request.app.state.engine_stats_scraper.engine_stats

    request_stats = request.app.state.request_stats_monitor.get_request_stats(
        time.time()
    )

    # Track all valid incoming requests
    num_incoming_requests_total.labels(
        workspace=workspace or "", endpoint=endpoint_name or ""
    ).inc()

    if not endpoints:
        if not model_ever_existed:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Model '{requested_model}' not found. Available models can be listed at /v1/models."
                },
                headers={"X-Request-Id": request_id},
            )
        else:
            # Model existed before but is now scaled to zero
            return JSONResponse(
                status_code=503,
                content={
                    "error": f"Model '{requested_model}' is temporarily unavailable. Please try again later."
                },
                headers={"X-Request-Id": request_id},
            )

    logger.debug(f"Routing request {request_id} for model: {requested_model}")
    if request_endpoint:
        server_url = endpoints[0].url
        logger.debug(
            f"Routing request {request_id} to engine with Id: {endpoints[0].Id}"
        )

    routing_logic = (
        endpoints[0].routing_logic
        if endpoints[0].routing_logic
        else RoutingLogic.ROUND_ROBIN
    )
    router = get_routing_logic_by_type(routing_logic)

    if isinstance(
        router,
        (
            KvawareRouter,
            PrefixAwareRouter,
            SessionRouter,
            ConsistentHashRouter,
            StaticHashRouter,
        ),
    ):
        server_url = await router.route_request(
            endpoints, engine_stats, request_stats, request, request_json
        )
    else:
        server_url = router.route_request(
            endpoints, engine_stats, request_stats, request
        )

    curr_time = time.time()
    # Extract actual session ID from request headers for logging
    session_key = getattr(request.app.state.router, "session_key", None)
    session_id = request.app.state.router.extract_session_id(request, request_json)
    session_id_display = session_id if session_id is not None else "None"

    # Debug logging to help troubleshoot session ID extraction
    logger.debug(f"Debug session extraction - Router type: {type(router).__name__}")
    logger.debug(f"Debug session extraction - Session key config: {session_key}")
    logger.debug(f"Debug session extraction - Extracted session ID: {session_id}")

    logger.info(
        f"Routing request {request_id} with session id {session_id_display} to {server_url} at {curr_time}, process time = {curr_time - in_router_time:.4f}"
    )
    selected = next(info for info in endpoints if info.url == server_url)
    # Discovery may replace this backend before the request's queue hooks run.
    request.state.backend_labels = (
        selected.workspace or "",
        selected.endpoint or "",
        server_url,
        selected.pod_name or "",
    )
    stream_generator = process_request(
        request,
        request_body,
        server_url,
        request_id,
        endpoint,
        background_tasks,
    )
    try:
        headers, status = await anext(stream_generator)
    except TimeoutError:
        # Downstream headers have not been sent, so an early timeout can be a 504.
        return JSONResponse(
            status_code=504,
            content={"error": "Backend request timed out."},
            headers={"X-Request-Id": request_id},
        )
    headers_dict = {key: value for key, value in headers.items()}
    headers_dict["X-Request-Id"] = request_id
    return StreamingResponse(
        stream_generator,
        status_code=status,
        headers=headers_dict,
        media_type="text/event-stream",
    )


async def send_request_to_prefiller(
    client: aiohttp.ClientSession, endpoint: str, req_data: dict, request_id: str
):
    """Send a request to a prefiller service."""
    req_data = req_data.copy()
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client.post(endpoint, json=req_data, headers=headers) as response:
        response.raise_for_status()
        return await response.json()


async def send_request_to_decode(
    client: aiohttp.ClientSession, endpoint: str, req_data: dict, request_id: str
):
    """Asynchronously stream the response from a service using a persistent client."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client.post(endpoint, json=req_data, headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.content.iter_any():
            yield chunk


async def route_disaggregated_prefill_request(
    request: Request,
    endpoint: str,
    background_tasks: BackgroundTasks,
):
    in_router_time = time.time()
    # Same as vllm, Get request_id from X-Request-Id header if available
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request_json = await request.json()

    orig_max_tokens = request_json.get("max_tokens", 0)
    request_json["max_tokens"] = 1
    st = time.time()
    try:
        await send_request_to_prefiller(
            request.app.state.prefill_client, endpoint, request_json, request_id
        )
        et = time.time()
        logger.info(f"{request_id} prefill time (TTFT): {et - st:.4f}")
        logger.info(
            f"Routing request {request_id} with session id None to {request.app.state.prefill_client._base_url} at {et}, process time = {et - in_router_time:.4f}"
        )
        request_json["max_tokens"] = orig_max_tokens
    except aiohttp.ClientResponseError as e:
        logger.error(f"HTTP error in prefiller: {e}", exc_info=True)
        return JSONResponse(
            status_code=e.status,
            content={
                "error": {
                    "message": f"Prefiller error: {e.message}",
                    "type": "prefiller_error",
                    "code": e.status,
                }
            },
            headers={"X-Request-Id": request_id},
        )
    except Exception as e:
        logger.error(f"Unexpected error in prefiller: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Prefiller error: {str(e)}",
                    "type": "prefiller_error",
                    "code": 500,
                }
            },
            headers={"X-Request-Id": request_id},
        )

    async def generate_stream():
        try:
            async for chunk in send_request_to_decode(
                request.app.state.decode_client, endpoint, request_json, request_id
            ):
                yield chunk
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error in decoder: {e}", exc_info=True)
            try:
                error_text = e.message
            except Exception:
                error_text = f"HTTP {e.status}"
            # Yield error as JSON response
            error_response = {
                "error": {
                    "message": f"Decoder error: {error_text}",
                    "type": "decoder_error",
                    "code": e.status,
                }
            }
            yield json.dumps(error_response).encode("utf-8")
        except Exception as e:
            logger.error(f"Unexpected error in decoder: {e}", exc_info=True)
            # Yield error as JSON response
            error_response = {
                "error": {
                    "message": f"Decoder error: {str(e)}",
                    "type": "decoder_error",
                    "code": 500,
                }
            }
            yield json.dumps(error_response).encode("utf-8")

    curr_time = time.time()
    logger.info(
        f"Routing request {request_id} with session id None to {request.app.state.decode_client._base_url} at {curr_time}, process time = {curr_time - et:.4f}"
    )

    return StreamingResponse(
        generate_stream(),
        media_type="application/json",
        headers={"X-Request-Id": request_id},
    )


async def route_sleep_wakeup_request(
    request: Request,
    endpoint: str,
    background_tasks: BackgroundTasks,
):
    in_router_time = time.time()
    # Same as vllm, Get request_id from X-Request-Id header if available
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    if request.query_params:
        request_endpoint = request.query_params.get("id")
    else:
        request_endpoint = None

    if request_endpoint is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request: missing target Engine Id."},
            headers={"X-Request-Id": request_id},
        )

    service_discovery = get_service_discovery()
    endpoints = service_discovery.get_endpoint_info()

    endpoints = list(filter(lambda x: x.Id == request_endpoint, endpoints))
    if not endpoints:
        return JSONResponse(
            status_code=400,
            content={"error": f"Engine with Id {request_endpoint} not found."},
        )
    logger.debug(f"Routing request {request_id} to engine with Id: {endpoints[0].Id}")

    server_url = endpoints[0].url
    curr_time = time.time()
    logger.info(
        f"Routing request {request_id} to {server_url} at {curr_time}, process time = {curr_time - in_router_time:.4f}"
    )

    headers = {
        "X-Request-Id": request_id,
    }

    if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
        logger.info("Using vllm server authentication")
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    url = server_url + endpoint

    async with aiohttp.ClientSession() as client:
        if endpoint == "/is_sleeping":
            async with client.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        else:
            request_body = await request.body()
            response_status = None
            if request_body:
                req_data = json.loads(request_body)
                async with client.post(url, json=req_data, headers=headers) as response:
                    response.raise_for_status()
                    response_status = response.status
            else:
                async with client.post(url, headers=headers) as response:
                    response.raise_for_status()
                    response_status = response.status

            pod_name = endpoints[0].pod_name
            if endpoint == "/sleep":
                service_discovery.add_sleep_label(pod_name)
            elif endpoint == "/wake_up":
                service_discovery.remove_sleep_label(pod_name)

            return JSONResponse(
                status_code=response_status,
                content={"status": "success"},
                headers={"X-Request-Id": request_id},
            )


# Merged upstream 7b1164d: shared audio forwarding; scope only changes selection.
async def route_general_transcriptions(
    request: Request,
    endpoint: str,
    background_tasks: BackgroundTasks,
    workspace: str | None = None,
    endpoint_name: str | None = None,
):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    form = await request.form()
    try:
        file = form.get("file")
        model = form.get("model")
        if not isinstance(file, UploadFile) or not isinstance(model, str) or not model:
            return JSONResponse(
                status_code=400,
                content={"error": "A file and non-empty model are required."},
                headers={"X-Request-Id": request_id},
            )
        try:
            temperature = float(form["temperature"]) if "temperature" in form else None
            if temperature is not None and not math.isfinite(temperature):
                raise ValueError()
            # Retain the fork's default; merged upstream defaults to no language.
            language = form.get("language", "en")
            stream = str(form.get("stream", "false")).lower() == "true"
            if not isinstance(language, str):
                raise TypeError()
        except (TypeError, ValueError):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid multipart/form-data request."},
                headers={"X-Request-Id": request_id},
            )
        payload_bytes = await file.read()
        form_data = aiohttp.FormData()
        form_data.add_field(
            "file",
            payload_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )
        data = {"model": model}
        language = language.strip()
        if language and language.lower() not in ("none", "null", "undefined"):
            data["language"] = language
        for key in ("prompt", "response_format"):
            value = form.get(key, "json" if key == "response_format" else None)
            if value:
                if not isinstance(value, str):
                    return JSONResponse(
                        status_code=400, content={"error": f"{key} must be text."}
                    )
                data[key] = value
        if temperature is not None:
            data["temperature"] = str(temperature)
        if stream:
            data["stream"] = "true"
        for key, value in data.items():
            form_data.add_field(key, value)
        return await proxy_multipart_request(
            form_data,
            model,
            endpoint,
            request,
            stream=stream,
            workspace=workspace,
            endpoint_name=endpoint_name,
        )
    finally:
        await form.close()


async def route_multipart_request(
    request: Request,
    endpoint: str,
    background_tasks: BackgroundTasks,
    workspace: str | None = None,
    endpoint_name: str | None = None,
):
    # Raw translation keeps the original multipart bytes and its boundary.
    body = await request.body()
    form = await request.form()
    try:
        model = form.get("model")
        if (
            not isinstance(form.get("file"), UploadFile)
            or not isinstance(model, str)
            or not model
        ):
            return JSONResponse(
                status_code=400,
                content={"error": "A file and non-empty model are required."},
            )
        return await proxy_multipart_request(
            body,
            model,
            endpoint,
            request,
            workspace=workspace,
            endpoint_name=endpoint_name,
        )
    finally:
        await form.close()


async def proxy_multipart_request(
    form_data: bytes | aiohttp.FormData,
    model: str,
    endpoint: str,
    request: Request,
    *,
    stream: bool = False,
    workspace: str | None = None,
    endpoint_name: str | None = None,
):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    monitor = request.app.state.request_stats_monitor
    endpoints = [
        info
        for info in get_service_discovery().get_endpoint_info()
        if model in info.model_names
        and not info.sleep
        and (workspace is None or info.workspace == workspace)
        and (endpoint_name is None or info.endpoint == endpoint_name)
    ]
    if not endpoints:
        return JSONResponse(
            status_code=404,
            content={"error": f"No backend for model {model}"},
            headers={"X-Request-Id": request_id},
        )
    router = (
        get_routing_logic_by_type(
            endpoints[0].routing_logic or RoutingLogic.ROUND_ROBIN
        )
        if workspace is not None
        else request.app.state.router
    )
    scraper = request.app.state.engine_stats_scraper
    engine_stats = scraper.engine_stats if scraper is not None else {}
    stats = monitor.get_request_stats(time.time())
    if isinstance(
        router,
        (
            KvawareRouter,
            PrefixAwareRouter,
            SessionRouter,
            ConsistentHashRouter,
            StaticHashRouter,
        ),
    ):
        chosen_url = await router.route_request(
            endpoints, engine_stats, stats, request, {}
        )
    else:
        chosen_url = router.route_request(endpoints, engine_stats, stats, request)
    selected = next(info for info in endpoints if info.url == chosen_url)
    try:
        # Rebuilt forms need aiohttp's new boundary; raw multipart keeps its header.
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS
            and key.lower() != "x-request-id"
            and (isinstance(form_data, bytes) or key.lower() != "content-type")
        }
        headers["X-Request-Id"] = request_id
        stats_id = str(uuid.uuid4())
        monitor.on_new_request(chosen_url, stats_id, time.time())
        try:
            backend_response = await request.app.state.aiohttp_client_wrapper().post(
                f"{chosen_url}{endpoint}",
                data=form_data,
                headers=headers,
                trace_request_ctx=trace_context(
                    selected.workspace or "",
                    selected.endpoint or "",
                    chosen_url,
                    selected.pod_name or "",
                ),
            )
        except asyncio.CancelledError:
            monitor.on_request_complete(
                chosen_url, stats_id, time.time(), success=False
            )
            raise
        except Exception:
            monitor.on_request_complete(
                chosen_url, stats_id, time.time(), success=False
            )
            raise
        resp_headers = {
            key: value
            for key, value in backend_response.headers.items()
            if key.lower()
            not in {
                "content-length",
                "content-encoding",
                "transfer-encoding",
                "connection",
                "server",
            }
        }
        resp_headers["X-Request-Id"] = request_id
        if stream:

            async def traced_stream():
                first_token = False
                success = False
                try:
                    async for chunk in backend_response.content.iter_any():
                        if not first_token:
                            first_token = True
                            monitor.on_request_response(
                                chosen_url, stats_id, time.time()
                            )
                        if chunk:
                            yield chunk
                    success = True
                finally:
                    backend_response.close()
                    monitor.on_request_complete(
                        chosen_url, stats_id, time.time(), success=success
                    )

            return StreamingResponse(
                traced_stream(),
                status_code=backend_response.status,
                headers=resp_headers,
                media_type=backend_response.headers.get(
                    "content-type", "text/event-stream"
                ),
            )
        success = False
        try:
            monitor.on_request_response(chosen_url, stats_id, time.time())
            media_type = (
                backend_response.headers.get("content-type", "")
                .partition(";")[0]
                .strip()
                .lower()
            )
            is_json = media_type == "application/json" or media_type.endswith("+json")
            content = await (
                backend_response.json() if is_json else backend_response.read()
            )
            success = True
            response_type = JSONResponse if is_json else Response
            return response_type(
                content=content,
                status_code=backend_response.status,
                headers=resp_headers,
            )
        except (aiohttp.ContentTypeError, json.JSONDecodeError) as parse_error:
            try:
                text_content = await backend_response.text()
            except aiohttp.ClientError:
                text_content = str(parse_error)
            return JSONResponse(
                status_code=502,
                content={
                    "error": f"Backend returned non-JSON response: {text_content}"
                },
                headers=resp_headers,
            )
        finally:
            backend_response.close()
            monitor.on_request_complete(
                chosen_url, stats_id, time.time(), success=success
            )
    except TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "Backend request timed out."},
            headers={"X-Request-Id": request_id},
        )
    except aiohttp.ClientError:
        return JSONResponse(
            status_code=503,
            content={"error": "Failed to connect to backend."},
            headers={"X-Request-Id": request_id},
        )
    except Exception as exc:
        logger.error(exc)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
