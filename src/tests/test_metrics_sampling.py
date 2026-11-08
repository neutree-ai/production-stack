from types import SimpleNamespace

import pytest

from vllm_router.routers import metrics_router


@pytest.mark.asyncio
async def test_scraping_metrics_does_not_request_a_blocking_cpu_sample(monkeypatch):
    intervals = []
    monkeypatch.setattr(
        metrics_router.psutil,
        "cpu_percent",
        lambda interval: intervals.append(interval) or 0,
    )
    monkeypatch.setattr(
        metrics_router,
        "get_service_discovery",
        lambda: SimpleNamespace(get_endpoint_info=lambda: []),
    )
    monkeypatch.setattr(
        metrics_router,
        "get_request_stats_monitor",
        lambda: SimpleNamespace(get_request_stats=lambda now: {}),
    )
    metrics_router.initialize_cpu_sampling()
    response = await metrics_router.metrics()
    assert response.status_code == 200
    assert intervals == [None, None]
