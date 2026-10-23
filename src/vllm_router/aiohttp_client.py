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
import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import aiohttp
import anyio
from yarl import URL

from vllm_router.backend_http import BackendHttpConfig, BackendTimeout, ForwardContext
from vllm_router.log import init_logger

logger = init_logger(__name__)


class _RequestBody(aiohttp.payload.Payload):
    """Observe body completion through aiohttp's public payload write contract."""

    def __init__(self, body, state, config):
        super().__init__(
            body,
            headers=body.headers,
            content_type=body.content_type,
            filename=body.filename,
            encoding=body.encoding,
        )
        self.body = body
        self.state = state
        self.config = config
        self._closed = False

    @property
    def size(self):
        return self.body.size

    @property
    def consumed(self):
        return self.body.consumed

    @property
    def autoclose(self):
        return self.body.autoclose

    def decode(self, encoding="utf-8", errors="strict"):
        return self.body.decode(encoding, errors)

    async def as_bytes(self, encoding="utf-8", errors="strict"):
        return await self.body.as_bytes(encoding, errors)

    async def write(self, writer):
        await self.write_with_length(writer, None)

    async def write_with_length(self, writer, content_length):
        await self.body.write_with_length(writer, content_length)
        # Flush the final chunk/compression bytes before the response-header clock.
        await writer.write_eof()
        if self.state.active and not self.state.headers_received:
            _start_header_wait(self.state, self.config)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        # aiohttp closes a sent payload after receiving headers, before resource cleanup.
        if self.state.active and not self.state.headers_received:
            _finish_header_wait(self.state)
        await self.body.close()


def _start_header_wait(state, config):
    if not state.active or state.headers_received or state.deadline.expired():
        return
    state.context.phase = "response_header"
    state.context.header_wait_started = time.monotonic()
    seconds = config.response_header_timeout_seconds
    state.deadline.reschedule(
        asyncio.get_running_loop().time() + seconds if seconds else None
    )


def _finish_header_wait(state):
    state.headers_received = True
    if not state.deadline.expired():
        state.deadline.reschedule(None)
    if state.context.header_wait_started is not None:
        state.context.header_wait_elapsed = (
            time.monotonic() - state.context.header_wait_started
        )


def _request_body(kwargs, state, config):
    body = kwargs.pop("data", None)
    json_body = kwargs.pop("json", None)
    if json_body is not None:
        if body is not None:
            raise ValueError("data and json cannot be supplied together")
        body = aiohttp.payload.JsonPayload(json_body)
    elif body is not None:
        if isinstance(body, aiohttp.FormData):
            body = body()
        try:
            body = aiohttp.payload.get_payload(body, disposition=None)
        except aiohttp.payload.LookupError:
            body = aiohttp.FormData(body)()
    if body is None:
        return None
    return _RequestBody(body, state, config)


class _BackendResponse:
    def __init__(self, response, config, stream, context):
        self.response = response
        self.config = config
        self.streaming = (
            stream
            or "text/event-stream" in response.headers.get("content-type", "").lower()
        )
        self.context = context
        self.headers = response.headers
        self.status = response.status
        self._chunks = response.content.iter_any().__aiter__()

    def iter_chunks(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        phase = "stream_idle" if self.streaming else "response_read"
        seconds = (
            self.config.stream_idle_timeout_seconds
            if self.streaming
            else self.config.response_read_timeout_seconds
        )
        self.context.phase = phase
        try:
            async with asyncio.timeout(seconds or None):
                return await anext(self._chunks)
        except StopAsyncIteration:
            self.context.outcome = "completed"
            raise
        except TimeoutError as exc:
            raise BackendTimeout(phase) from exc

    async def close(self):
        if hasattr(self._chunks, "aclose"):
            with anyio.CancelScope(shield=True):
                await self._chunks.aclose()

    async def read(self):
        return b"".join([chunk async for chunk in self.iter_chunks()])

    async def json(self):
        return json.loads(await self.read())

    def raise_for_status(self):
        self.response.raise_for_status()


@dataclass
class _BackendPool:
    semaphore: asyncio.Semaphore
    references: int = 0


class AiohttpClientWrapper:
    def __init__(self):
        self.async_client = None
        self.config = BackendHttpConfig()
        self._pools = {}
        self.observer = None

    def configure(self, config):
        if self.async_client is not None and config != self.config:
            raise ValueError(
                "Backend HTTP configuration changes require a Router restart"
            )
        self.config = config

    def _observe(self, event, context):
        if self.observer is not None:
            try:
                self.observer(event, context)
            except Exception:
                logger.exception("Backend request observer failed: %s", event)

    def start(self):
        """Instantiate the client from the FastAPI startup hook."""
        trace = aiohttp.TraceConfig()

        async def connection_start(session, trace_context, params):
            state = trace_context.trace_request_ctx
            if state is not None:
                state.context.phase = "connect"
                if not state.context.connection_in_use:
                    state.context.connection_in_use = True
                    self._observe("connection_acquired", state.context)
                seconds = self.config.connect_timeout_seconds
                state.deadline.reschedule(
                    asyncio.get_running_loop().time() + seconds if seconds else None
                )

        async def connection_end(session, trace_context, params):
            state = trace_context.trace_request_ctx
            if state is not None:
                state.deadline.reschedule(None)

        async def connection_reused(session, trace_context, params):
            state = trace_context.trace_request_ctx
            if state is not None and not state.context.connection_in_use:
                state.context.connection_in_use = True
                self._observe("connection_acquired", state.context)

        async def queued_start(session, trace_context, params):
            state = trace_context.trace_request_ctx
            if state is not None:
                state.queue_started = time.monotonic()
                state.context.phase = "pool_wait"
                self._observe("pool_wait_started", state.context)
                seconds = self.config.pool_wait_timeout_seconds
                remaining = max(0, seconds - state.context.pool_wait_elapsed)
                state.deadline.reschedule(
                    asyncio.get_running_loop().time() + remaining if seconds else None
                )

        async def queued_end(session, trace_context, params):
            state = trace_context.trace_request_ctx
            if state is not None and state.queue_started is not None:
                state.context.pool_wait_elapsed += (
                    time.monotonic() - state.queue_started
                )
                state.queue_started = None
                self._observe("pool_wait_finished", state.context)
                state.context.phase = "connect"
                state.deadline.reschedule(None)

        async def headers_sent(session, trace_context, params):
            state = trace_context.trace_request_ctx
            if state is not None and state.active and not state.headers_received:
                if state.body_expected:
                    state.context.phase = "request_body"
                    state.deadline.reschedule(None)
                else:
                    _start_header_wait(state, self.config)

        trace.on_connection_queued_start.append(queued_start)
        trace.on_connection_queued_end.append(queued_end)
        trace.on_connection_reuseconn.append(connection_reused)
        trace.on_connection_create_start.append(connection_start)
        trace.on_connection_create_end.append(connection_end)
        trace.on_request_headers_sent.append(headers_sent)
        connector = aiohttp.TCPConnector(
            limit=0, limit_per_host=self.config.max_connections_per_backend
        )
        self.async_client = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None),
            trace_configs=[trace],
        )
        logger.info("aiohttp ClientSession instantiated")

    async def stop(self):
        if self.async_client is not None:
            await self.async_client.close()
            self.async_client = None
        self._pools.clear()

    def __call__(self):
        assert self.async_client is not None
        return self.async_client

    @asynccontextmanager
    async def forward(
        self, backend_url, *, stream=False, workspace="", endpoint="", **kwargs
    ):
        context = ForwardContext(
            server=backend_url, workspace=workspace, endpoint=endpoint
        )
        key = str(URL(backend_url).origin())
        pool = None
        acquired = False
        response = None
        adapted_response = None
        trace_state = None
        request_body = None
        self._observe("request_started", context)
        try:
            if self.config.max_connections_per_backend:
                pool = self._pools.setdefault(
                    key,
                    _BackendPool(
                        asyncio.Semaphore(self.config.max_connections_per_backend)
                    ),
                )
                pool.references += 1
                self._observe("pool_wait_started", context)
                wait_started = time.monotonic()
                try:
                    async with asyncio.timeout(
                        self.config.pool_wait_timeout_seconds or None
                    ):
                        await pool.semaphore.acquire()
                    acquired = True
                except TimeoutError as exc:
                    raise BackendTimeout("pool_wait") from exc
                finally:
                    context.pool_wait_elapsed += time.monotonic() - wait_started
                    self._observe("pool_wait_finished", context)
            context.phase = "connect"
            kwargs["timeout"] = aiohttp.ClientTimeout(
                total=None, sock_connect=self.config.connect_timeout_seconds or None
            )
            kwargs["allow_redirects"] = False
            try:
                async with asyncio.timeout(None) as deadline:
                    trace_state = SimpleNamespace(
                        context=context,
                        deadline=deadline,
                        queue_started=None,
                        active=True,
                        headers_received=False,
                        body_expected=False,
                    )
                    kwargs["trace_request_ctx"] = trace_state
                    request_body = _request_body(kwargs, trace_state, self.config)
                    trace_state.body_expected = request_body is not None
                    if request_body is not None:
                        kwargs["data"] = request_body
                    response = await self().request(**kwargs)
                    if not trace_state.headers_received:
                        _finish_header_wait(trace_state)
                    trace_state.active = False
            except TimeoutError as exc:
                raise BackendTimeout(context.phase) from exc
            context.status_code = response.status
            self._observe("headers_received", context)
            adapted_response = _BackendResponse(response, self.config, stream, context)
            yield adapted_response
        except BackendTimeout:
            context.outcome = "timeout"
            raise
        except (asyncio.CancelledError, GeneratorExit):
            context.outcome = "cancelled"
            raise
        except Exception:
            context.outcome = "backend_error"
            raise
        finally:
            if trace_state is not None:
                trace_state.active = False
            try:
                if response is not None:
                    response.close()
                if adapted_response is not None:
                    await adapted_response.close()
                if request_body is not None:
                    with anyio.CancelScope(shield=True):
                        await request_body.close()
            finally:
                if trace_state is not None and trace_state.queue_started is not None:
                    context.pool_wait_elapsed += (
                        time.monotonic() - trace_state.queue_started
                    )
                    self._observe("pool_wait_finished", context)
                if context.connection_in_use:
                    context.connection_in_use = False
                    self._observe("connection_released", context)
                if pool is not None:
                    if acquired:
                        pool.semaphore.release()
                    pool.references -= 1
                    if not pool.references:
                        self._pools.pop(key, None)
                self._observe("request_finished", context)


@asynccontextmanager
async def backend_request(request, backend_url, *, stream=False, **kwargs):
    """Share phase controls across inference routes, preserving injected test clients."""
    wrapper = request.app.state.aiohttp_client_wrapper
    if isinstance(wrapper, AiohttpClientWrapper):
        labels = getattr(request.state, "backend_labels", {})
        async with wrapper.forward(
            backend_url, stream=stream, **labels, **kwargs
        ) as response:
            yield response
    else:
        config = getattr(request.app.state, "backend_http_config", BackendHttpConfig())
        kwargs["timeout"] = aiohttp.ClientTimeout(
            total=None, sock_connect=config.connect_timeout_seconds or None
        )
        async with wrapper().request(**kwargs) as response:
            context = ForwardContext(server=backend_url)
            adapted_response = _BackendResponse(response, config, stream, context)
            try:
                yield adapted_response
            finally:
                await adapted_response.close()
