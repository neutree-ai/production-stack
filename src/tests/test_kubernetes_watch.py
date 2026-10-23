import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from kubernetes.client.exceptions import ApiException

from vllm_router.kubernetes_watch import KubernetesWatch


def page(names, rv="opaque-rv", token=None):
    return NS(
        items=[NS(metadata=NS(name=name)) for name in names],
        metadata=NS(resource_version=rv, _continue=token),
    )


def runner(list_resources):
    owner = NS(
        namespace="ns",
        label_selector="app=inference",
        watcher_timeout_seconds=0,
        available_engines={"gone": object()},
        available_engines_lock=__import__("threading").Lock(),
        _delete_engine=Mock(),
        _handle_resource=Mock(),
    )
    return KubernetesWatch(owner, list_resources), owner


def test_list_pages_are_complete_before_reconciling_and_watch_uses_list_version():
    list_resources = Mock(side_effect=[page(["one"], token="next"), page(["two"])])
    watch, owner = runner(list_resources)
    assert watch.reconcile() == "opaque-rv"
    owner._delete_engine.assert_called_once_with("gone")
    assert [
        call.args[0].metadata.name for call in owner._handle_resource.call_args_list
    ] == ["one", "two"]
    assert list_resources.call_args_list[1].kwargs["_continue"] == "next"
    assert all(
        call.kwargs["_request_timeout"] == (3, 10)
        for call in list_resources.call_args_list
    )


def test_failed_list_page_does_not_delete_or_publish_partial_snapshot():
    watch, owner = runner(
        Mock(side_effect=[page(["one"], token="next"), ApiException(status=500)])
    )
    with pytest.raises(ApiException):
        watch.reconcile()
    owner._delete_engine.assert_not_called()
    owner._handle_resource.assert_not_called()
    assert watch.last_success is None


def test_watch_error_and_bookmark_are_not_passed_to_resource_handler():
    watch, owner = runner(Mock())
    watch.handle_event(
        {"type": "BOOKMARK", "raw_object": {"metadata": {"resourceVersion": "rv2"}}}
    )
    assert watch.resource_version == "rv2"
    with pytest.raises(ApiException) as exc:
        watch.handle_event(
            {"type": "ERROR", "raw_object": {"code": 410, "message": "expired"}}
        )
    assert exc.value.status == 410
    owner._handle_resource.assert_not_called()


def test_health_expires_without_success_even_when_thread_alive(monkeypatch):
    watch, _ = runner(Mock())
    watch.thread = NS(is_alive=lambda: True)
    monkeypatch.setattr("vllm_router.kubernetes_watch.time.monotonic", lambda: 100)
    assert not watch.healthy()
    watch.last_success = 99
    assert watch.healthy()
    watch.last_success = 30
    assert not watch.healthy()


def test_eof_relists_and_410_recovers_without_busy_retry(monkeypatch):
    watch, owner = runner(Mock(side_effect=[page([]), page(["new"]), page(["new"])]))
    calls = []

    class Stream:
        def stream(self, fn, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                yield {"type": "ERROR", "raw_object": {"code": 410}}
            else:
                watch.stop_event.set()
                return

    watch.watcher = Stream()
    monkeypatch.setattr(watch.stop_event, "wait", lambda delay: False)
    watch.run()
    assert len(calls) == 2
    assert calls[0]["resource_version"] == "opaque-rv"
    assert calls[0]["timeout_seconds"] == 30
    assert calls[0]["_request_timeout"] == (3, 35)
    assert watch.expirations == 1
    assert owner._handle_resource.call_count == 1


def test_real_kubernetes_watch_decodes_typed_resources_and_releases_response(
    monkeypatch,
):
    watch, owner = runner(Mock())

    class Response:
        closed = False
        released = False

        def stream(self, **kwargs):
            yield (
                json.dumps(
                    {
                        "type": "ADDED",
                        "object": {
                            "apiVersion": "v1",
                            "kind": "Pod",
                            "metadata": {"name": "p", "resourceVersion": "rv3"},
                        },
                    }
                )
                + "\n"
            ).encode()

        def close(self):
            self.closed = True

        def release_conn(self):
            self.released = True

    response = Response()
    watch.list_resources = lambda **kw: response if kw.get("watch") else page([])

    def handle(obj, event):
        assert obj.metadata.name == "p"
        assert event == "ADDED"
        watch.stop_event.set()

    owner._handle_resource = handle
    watch.run()
    assert response.closed and response.released
    assert watch.resource_version == "rv3"


def test_permissions_error_is_recorded_and_retry_wait_is_interruptible(monkeypatch):
    watch, _ = runner(Mock(side_effect=ApiException(status=403, reason="Forbidden")))
    waits = []

    def wait(delay):
        waits.append(delay)
        watch.stop_event.set()
        return True

    monkeypatch.setattr(watch.stop_event, "wait", wait)
    watch.run()
    assert "403" in watch.last_error
    assert watch.last_success is None
    assert len(waits) == 1 and 0 < waits[0] <= 0.5


def test_changing_page_version_does_not_apply_inconsistent_snapshot():
    watch, owner = runner(
        Mock(side_effect=[page(["one"], token="next"), page(["two"], rv="different")])
    )
    with pytest.raises(RuntimeError, match="resourceVersion"):
        watch.reconcile()
    owner._delete_engine.assert_not_called()
    owner._handle_resource.assert_not_called()


def test_close_aborts_active_watch_response_without_waiting_for_io():
    watch, _ = runner(Mock())
    response = Mock()
    watch._response = response
    watch.close()
    response.close.assert_called_once()
    assert watch.stop_event.is_set()
