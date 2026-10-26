import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from vllm_router.service_discovery import K8sPodIPServiceDiscovery


@pytest.fixture
def discovery(monkeypatch):
    monkeypatch.setattr(
        "vllm_router.service_discovery.config.load_incluster_config", lambda: None
    )
    monkeypatch.setattr("vllm_router.service_discovery.client.CoreV1Api", Mock())
    monkeypatch.setattr(K8sPodIPServiceDiscovery, "_watch_engines", lambda self: None)
    result = K8sPodIPServiceDiscovery(
        FastAPI(),
        "ns",
        8000,
        prefill_model_labels=["prefill"],
        decode_model_labels=["decode"],
    )
    yield result
    result.close()


def test_watcher_observes_initialized_pd_fields(monkeypatch):
    monkeypatch.setattr(
        "vllm_router.service_discovery.config.load_incluster_config", lambda: None
    )
    monkeypatch.setattr("vllm_router.service_discovery.client.CoreV1Api", Mock())
    seen = []

    def inspect(self):
        seen.append(
            (
                self.prefill_model_labels,
                self.decode_model_labels,
                self.event_loop_ready.is_set(),
            )
        )

    monkeypatch.setattr(K8sPodIPServiceDiscovery, "_watch_engines", inspect)
    result = K8sPodIPServiceDiscovery(
        FastAPI(), "ns", 8000, prefill_model_labels=["p"], decode_model_labels=["d"]
    )
    result.watcher_thread.join(timeout=1)
    assert seen == [(["p"], ["d"], False)]
    result.close()


def test_backend_discovered_before_loop_does_not_create_unawaited_coroutine(
    discovery, monkeypatch
):
    monkeypatch.setattr(
        "vllm_router.service_discovery.asyncio.run_coroutine_threadsafe",
        Mock(side_effect=AssertionError("no loop yet")),
    )
    discovery.initialize_client_sessions = AsyncMock()
    discovery._get_model_info = Mock(return_value={})
    discovery._add_engine("engine", "10.0.0.1", [], "prefill", "ws", "ep", None, True)
    discovery.initialize_client_sessions.assert_not_called()
    assert discovery.get_endpoint_info()[0].workspace == "ws"


@pytest.mark.asyncio
async def test_loop_ready_refreshes_backends_discovered_earlier(discovery):
    refreshed = asyncio.Event()

    async def refresh():
        refreshed.set()

    discovery.initialize_client_sessions = refresh
    loop = asyncio.get_running_loop()
    discovery.set_event_loop(loop)
    await asyncio.wait_for(refreshed.wait(), 1)
    assert discovery.event_loop_ready.is_set()
    assert discovery.event_loop is loop


@pytest.mark.asyncio
async def test_lifespan_attaches_loop_before_session_initialization(monkeypatch):
    from vllm_router.app import lifespan

    app = FastAPI()
    app.state.aiohttp_client_wrapper = Mock(start=Mock(), stop=AsyncMock())
    app.state.engine_stats_scraper = None
    observed = []

    class Discovery:
        def set_event_loop(self, loop):
            assert app.state.event_loop is loop
            observed.append("loop")

        async def initialize_client_sessions(self):
            assert app.state.event_loop is asyncio.get_running_loop()
            observed.append("sessions")

        def close(self):
            pass

    monkeypatch.setattr("vllm_router.app.get_service_discovery", lambda: Discovery())
    monkeypatch.setattr("vllm_router.app.get_dynamic_config_watcher", lambda: None)
    monkeypatch.setattr("vllm_router.app.cleanup_routing_logic", lambda: None)
    async with lifespan(app):
        assert observed == ["loop", "sessions"]


@pytest.mark.asyncio
async def test_burst_refreshes_do_not_overlap_session_replacement(
    discovery, monkeypatch
):
    from types import SimpleNamespace

    active_closes = 0
    overlapping_closes = []
    created = []

    class Session:
        closed = False

        async def close(self):
            nonlocal active_closes
            active_closes += 1
            overlapping_closes.append(active_closes)
            await asyncio.sleep(0)
            self.closed = True
            active_closes -= 1

    old = Session()
    discovery.app.state.prefill_client = old
    monkeypatch.setattr(
        discovery,
        "get_endpoint_info",
        lambda: [SimpleNamespace(model_label="prefill", url="http://engine")],
    )

    def create(**kwargs):
        session = Session()
        created.append(session)
        return session

    monkeypatch.setattr("vllm_router.service_discovery.aiohttp.ClientSession", create)
    await asyncio.gather(
        discovery.initialize_client_sessions(), discovery.initialize_client_sessions()
    )
    assert max(overlapping_closes) == 1
    assert all(
        session.closed or session is discovery.app.state.prefill_client
        for session in created
    )
    await discovery.app.state.prefill_client.close()
