import asyncio

import pytest
from fastapi import FastAPI

from vllm_router.app import lifespan


@pytest.mark.asyncio
async def test_lifespan_has_event_loop_before_starting_discovery_and_closes_watcher_first(
    monkeypatch,
):
    app = FastAPI()
    events = []

    class Discovery:
        def set_event_loop(self, loop):
            assert loop is app.state.event_loop
            events.append("watcher_started")

        async def initialize_client_sessions(self):
            assert app.state.event_loop is asyncio.get_running_loop()
            events.append("clients_initialized")

        def close(self):
            assert app.state.event_loop.is_running()
            events.append("watcher_closed")

        async def close_client_sessions(self):
            events.append("pd_closed")

    class Wrapper:
        def start(self):
            events.append("http_started")

        async def stop(self):
            events.append("http_closed")

    app.state.aiohttp_client_wrapper = Wrapper()
    app.state.engine_stats_scraper = None
    discovery = Discovery()
    monkeypatch.setattr("vllm_router.app.get_service_discovery", lambda: discovery)
    monkeypatch.setattr("vllm_router.app.get_dynamic_config_watcher", lambda: None)
    monkeypatch.setattr("vllm_router.app.cleanup_routing_logic", lambda: None)
    async with lifespan(app):
        assert events == ["http_started", "watcher_started", "clients_initialized"]
    assert events[-3:] == ["watcher_closed", "pd_closed", "http_closed"]
