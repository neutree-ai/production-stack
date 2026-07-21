import pytest

from vllm_router.routers.passthrough_router import _rr_counters, _select_backend
from vllm_router.service_discovery import EndpointInfo, K8sPodIPServiceDiscovery


def _endpoint(url: str) -> EndpointInfo:
    return EndpointInfo(
        url=url,
        model_names=[],
        Id=url,
        added_timestamp=0,
        model_label=None,
        sleep=False,
        workspace="ws",
        endpoint="ep",
        passthrough=True,
    )


class TestAdmission:
    """
    A passthrough engine serves no /v1/models, so it must qualify for the pool
    on its routing key instead of on advertised models.
    """

    admit = staticmethod(K8sPodIPServiceDiscovery._is_admissible)

    def test_openai_engine_still_requires_models(self):
        assert self.admit(True, ["m"], False, "ws", "ep") is True
        assert self.admit(True, [], False, "ws", "ep") is False

    def test_passthrough_engine_admitted_without_models(self):
        assert self.admit(True, [], True, "ws", "ep") is True

    def test_passthrough_engine_needs_a_routing_key(self):
        # Without both labels it could never be selected; admitting it would
        # only put an unreachable entry in the pool.
        assert self.admit(True, [], True, None, "ep") is False
        assert self.admit(True, [], True, "ws", None) is False

    def test_unready_pod_never_admitted(self):
        assert self.admit(False, ["m"], False, "ws", "ep") is False
        assert self.admit(False, [], True, "ws", "ep") is False


class TestBackendSelection:
    def setup_method(self):
        _rr_counters.clear()

    def test_round_robins_within_a_key(self):
        endpoints = [_endpoint("a"), _endpoint("b"), _endpoint("c")]
        picked = [_select_backend(endpoints, ("ws", "ep")).url for _ in range(6)]
        assert picked == ["a", "b", "c", "a", "b", "c"]

    def test_keys_have_independent_cursors(self):
        endpoints = [_endpoint("a"), _endpoint("b")]
        assert _select_backend(endpoints, ("ws", "one")).url == "a"
        assert _select_backend(endpoints, ("ws", "two")).url == "a"
        assert _select_backend(endpoints, ("ws", "one")).url == "b"

    def test_survives_pool_shrinking_between_calls(self):
        three = [_endpoint("a"), _endpoint("b"), _endpoint("c")]
        for _ in range(5):
            _select_backend(three, ("ws", "ep"))
        # Replicas can disappear while the cursor keeps climbing.
        one = [_endpoint("a")]
        assert _select_backend(one, ("ws", "ep")).url == "a"


@pytest.fixture(scope="module")
def routes():
    from vllm_router.app import app

    return app.routes


class TestRouteOrdering:
    """
    The catch-all must not shadow any explicitly declared route. Ordering is
    positional in Starlette, so this is a property of include_router order in
    app.py and silently breaks if that order changes.
    """

    def test_catch_all_is_registered_last(self, routes):
        paths = [getattr(r, "path", None) for r in routes]
        assert paths[-1] == "/{workspace}/{endpoint}/{upstream_path:path}"

    @pytest.mark.parametrize(
        "path,expected",
        [
            (
                "/ws/ep/v1/chat/completions",
                "/{workspace}/{endpoint}/v1/chat/completions",
            ),
            ("/ws/ep/v1/models", "/{workspace}/{endpoint}/v1/models"),
            ("/ws/ep/health", "/{workspace}/{endpoint}/health"),
            ("/v1/files/abc", "/v1/files/{file_id}"),
            ("/v1/files/abc/content", "/v1/files/{file_id}/content"),
            ("/v1/batches/xyz", "/v1/batches/{batch_id}"),
            ("/metrics", "/metrics"),
            # Only genuinely unclaimed paths reach the catch-all.
            ("/ws/ep/predict", "/{workspace}/{endpoint}/{upstream_path:path}"),
            ("/ws/ep/a/b/c", "/{workspace}/{endpoint}/{upstream_path:path}"),
        ],
    )
    def test_first_matching_route(self, routes, path, expected):
        for route in routes:
            regex = getattr(route, "path_regex", None)
            if regex is not None and regex.fullmatch(path):
                assert route.path == expected
                return
        pytest.fail(f"no route matched {path}")
