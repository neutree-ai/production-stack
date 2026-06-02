from vllm_router.stats.request_stats import RequestStatsMonitor, SingletonMeta


def setup_function():
    SingletonMeta._instances.pop(RequestStatsMonitor, None)


def teardown_function():
    SingletonMeta._instances.pop(RequestStatsMonitor, None)


def test_request_stats_tracks_pd_unit_active_stage_transitions():
    monitor = RequestStatsMonitor(10.0)
    engine_url = "http://10.0.0.1:9000"
    prefill_unit = "domain-0:prefill:0:http://10.0.0.1:9000"
    decode_unit = "domain-0:decode:0:http://10.0.0.1:9000"

    monitor.on_new_request(
        engine_url,
        "req-1",
        1.0,
        pd_prefill_unit_id=prefill_unit,
        pd_decode_unit_id=decode_unit,
    )

    assert monitor.get_active_request_count(engine_url) == 1
    assert monitor.get_active_pd_unit_request_count(prefill_unit) == 1
    assert monitor.get_active_pd_unit_request_count(decode_unit) == 0

    monitor.on_request_response(engine_url, "req-1", 2.0)

    assert monitor.get_active_request_count(engine_url) == 1
    assert monitor.get_active_pd_unit_request_count(prefill_unit) == 0
    assert monitor.get_active_pd_unit_request_count(decode_unit) == 1

    monitor.on_request_complete(engine_url, "req-1", 3.0)

    assert monitor.get_active_request_count(engine_url) == 0
    assert monitor.get_active_pd_unit_request_count(prefill_unit) == 0
    assert monitor.get_active_pd_unit_request_count(decode_unit) == 0


def test_request_stats_tracks_both_pd_units_without_token_boundary():
    monitor = RequestStatsMonitor(10.0)
    engine_url = "http://10.0.0.1:9000"
    prefill_unit = "domain-0:prefill:0:http://10.0.0.1:9000"
    decode_unit = "domain-0:decode:0:http://10.0.0.1:9000"

    monitor.on_new_request(
        engine_url,
        "req-1",
        1.0,
        pd_prefill_unit_id=prefill_unit,
        pd_decode_unit_id=decode_unit,
        pd_track_both_units=True,
    )

    assert monitor.get_active_pd_unit_request_count(prefill_unit) == 1
    assert monitor.get_active_pd_unit_request_count(decode_unit) == 1

    monitor.on_request_response(engine_url, "req-1", 2.0)

    assert monitor.get_active_pd_unit_request_count(prefill_unit) == 1
    assert monitor.get_active_pd_unit_request_count(decode_unit) == 1

    monitor.on_request_complete(engine_url, "req-1", 3.0)

    assert monitor.get_active_pd_unit_request_count(prefill_unit) == 0
    assert monitor.get_active_pd_unit_request_count(decode_unit) == 0
