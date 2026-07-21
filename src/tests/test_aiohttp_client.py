import pytest

from vllm_router.aiohttp_client import AiohttpClientWrapper


@pytest.mark.asyncio
async def test_start_configures_global_and_per_host_connection_limits() -> None:
    wrapper = AiohttpClientWrapper()
    wrapper.start()
    client = wrapper()
    connector = client.connector

    try:
        assert connector.limit == 0
        assert connector.limit_per_host == 100
    finally:
        await wrapper.stop()

    assert client.closed
    assert connector.closed
    assert wrapper.async_client is None
