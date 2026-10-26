import importlib
from types import SimpleNamespace

import pytest

from vllm_router.stats.request_stats import RequestStatsMonitor
from vllm_router.utils import SingletonMeta


@pytest.mark.asyncio
async def test_metrics_cpu_sampling_is_nonblocking(monkeypatch):
    module = importlib.import_module("vllm_router.routers.metrics_router")
    calls = []

    def cpu_percent(interval):
        calls.append(interval)
        return 12.5

    monkeypatch.setattr(module.psutil, "cpu_percent", cpu_percent)
    monkeypatch.setattr(
        module,
        "get_service_discovery",
        lambda: SimpleNamespace(get_endpoint_info=lambda: []),
    )
    monkeypatch.setattr(
        module,
        "get_request_stats_monitor",
        lambda: SimpleNamespace(get_request_stats=lambda now: {}),
    )
    response = await module.metrics()
    assert calls == [None]
    assert b"router_cpu_usage_percent 12.5" in response.body


def test_idle_backend_latency_expires_with_window():
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    try:
        monitor = RequestStatsMonitor(sliding_window_size=10)
        monitor.on_new_request("http://engine", "request", 100)
        monitor.on_request_complete("http://engine", "request", 102)
        assert monitor.get_request_stats(102)["http://engine"].avg_latency == 2
        stats = monitor.get_request_stats(120)["http://engine"]
        assert stats.avg_latency == -1
        assert stats.qps == 0
        assert stats.active_requests == 0
    finally:
        SingletonMeta._instances.pop(RequestStatsMonitor, None)
