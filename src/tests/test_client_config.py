import json
import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest
import yaml

from vllm_router.dynamic_config import DynamicConfigWatcher, DynamicRouterConfig
from vllm_router.parsers.parser import parse_args


def router_args(monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "router",
            "--service-discovery",
            "static",
            "--static-backends",
            "http://localhost:8000",
            "--static-models",
            "model",
            *extra,
        ],
    )
    return parse_args()


def test_native_client_config_defaults_and_cli_zero(monkeypatch):
    args = router_args(monkeypatch)
    assert (
        args.max_connections_per_backend,
        args.connect_timeout_seconds,
        args.sock_connect_timeout_seconds,
        args.sock_read_timeout_seconds,
    ) == (100, 60, 10, 600)

    args = router_args(
        monkeypatch,
        "--max-connections-per-backend",
        "0",
        "--connect-timeout-seconds",
        "0",
        "--sock-read-timeout-seconds",
        "0.25",
    )
    assert args.max_connections_per_backend == 0
    assert args.connect_timeout_seconds == 0
    assert args.sock_read_timeout_seconds == 0.25


def test_cli_timeout_rejects_nonfinite_value(monkeypatch):
    with pytest.raises(ValueError, match="connect_timeout_seconds"):
        router_args(monkeypatch, "--connect-timeout-seconds", "nan")


@pytest.mark.parametrize("file_type", ["json", "yaml"])
def test_client_config_file_null_fractional_and_cli_precedence(
    monkeypatch, tmp_path, file_type
):
    config = {
        "max_connections_per_backend": 32,
        "connect_timeout_seconds": None,
        "sock_connect_timeout_seconds": 0,
        "sock_read_timeout_seconds": 0.125,
    }
    path = tmp_path / f"router.{file_type}"
    path.write_text(
        json.dumps(config) if file_type == "json" else yaml.safe_dump(config)
    )
    args = router_args(
        monkeypatch,
        f"--dynamic-config-{file_type}",
        str(path),
        "--max-connections-per-backend",
        "8",
    )
    assert (
        args.max_connections_per_backend,
        args.connect_timeout_seconds,
        args.sock_connect_timeout_seconds,
        args.sock_read_timeout_seconds,
    ) == (8, 60, 0, 0.125)


@pytest.mark.parametrize(
    ("file_type", "field", "value"),
    [
        ("json", "max_connections_per_backend", True),
        ("json", "max_connections_per_backend", 1.5),
        ("json", "max_connections_per_backend", 65536),
        ("yaml", "connect_timeout_seconds", "30"),
        ("yaml", "connect_timeout_seconds", float("nan")),
        ("yaml", "sock_connect_timeout_seconds", -1),
        ("json", "sock_read_timeout_seconds", 604801),
    ],
)
def test_raw_client_config_rejects_invalid_values_before_argparse_conversion(
    monkeypatch, tmp_path, file_type, field, value
):
    path = tmp_path / f"router.{file_type}"
    config = {field: value}
    path.write_text(
        json.dumps(config) if file_type == "json" else yaml.safe_dump(config)
    )
    with pytest.raises(ValueError, match=field):
        router_args(monkeypatch, f"--dynamic-config-{file_type}", str(path))


@pytest.mark.parametrize("file_type", ["json", "yaml"])
def test_dynamic_client_config_roundtrip_keeps_zero_and_defaults(tmp_path, file_type):
    path = tmp_path / f"router.{file_type}"
    config = {
        "service_discovery": "k8s",
        "routing_logic": "default",
        "max_connections_per_backend": 0,
        "connect_timeout_seconds": None,
        "sock_connect_timeout_seconds": 0.5,
        "sock_read_timeout_seconds": 0,
    }
    path.write_text(
        json.dumps(config) if file_type == "json" else yaml.safe_dump(config)
    )
    result = getattr(DynamicRouterConfig, f"from_{file_type}")(str(path))
    serialized = json.loads(result.to_json_str())
    assert serialized["max_connections_per_backend"] == 0
    assert serialized["connect_timeout_seconds"] == 60
    assert serialized["sock_connect_timeout_seconds"] == 0.5
    assert serialized["sock_read_timeout_seconds"] == 0


def test_dynamic_client_change_is_rejected_before_discovery_reconfiguration():
    current = DynamicRouterConfig(service_discovery="static", routing_logic="default")
    watcher = object.__new__(DynamicConfigWatcher)
    watcher.current_config = current
    watcher.reconfigure_service_discovery = Mock()
    updated = replace(
        current, static_backends="http://new:8000", sock_read_timeout_seconds=5
    )
    with pytest.raises(ValueError, match="restart"):
        watcher.reconfigure_all(updated)
    watcher.reconfigure_service_discovery.assert_not_called()
    assert watcher.current_config is current


def test_dynamic_discovery_change_with_unchanged_client_config_still_applies():
    current = DynamicRouterConfig(service_discovery="static", routing_logic="default")
    watcher = object.__new__(DynamicConfigWatcher)
    watcher.current_config = current
    for method in (
        "reconfigure_service_discovery",
        "reconfigure_routing_logic",
        "reconfigure_batch_api",
        "reconfigure_stats",
        "reconfigure_callbacks",
    ):
        setattr(watcher, method, Mock())
    updated = replace(current, static_backends="http://new:8000")
    watcher.reconfigure_all(updated)
    watcher.reconfigure_service_discovery.assert_called_once_with(updated)


def test_cli_client_override_allows_other_file_updates_but_rejects_client_changes(
    monkeypatch, tmp_path
):
    from fastapi import FastAPI

    from vllm_router.utils import SingletonMeta

    path = tmp_path / "router.json"
    config = {
        "service_discovery": "k8s",
        "routing_logic": "default",
        "k8s_namespace": "original",
        "k8s_port": 8000,
        "max_connections_per_backend": 32,
    }
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "router",
            "--dynamic-config-json",
            str(path),
            "--max-connections-per-backend",
            "8",
        ],
    )
    effective_config = DynamicRouterConfig.from_args(parse_args())
    reconfigure = Mock()
    monkeypatch.setattr(
        "vllm_router.dynamic_config.reconfigure_service_discovery", reconfigure
    )
    monkeypatch.setattr(
        "vllm_router.dynamic_config.threading.Thread.start", lambda _: None
    )
    SingletonMeta._instances.pop(DynamicConfigWatcher, None)
    watcher = DynamicConfigWatcher(str(path), "JSON", 10, effective_config, FastAPI())
    monkeypatch.setattr(
        watcher, "_sleep_or_break", lambda: setattr(watcher, "running", False)
    )
    try:
        path.write_text(json.dumps({**config, "k8s_namespace": "updated"}))
        watcher._watch_worker()
        assert reconfigure.call_args.kwargs["namespace"] == "updated"
        assert watcher.get_current_config().max_connections_per_backend == 8

        reconfigure.reset_mock()
        path.write_text(
            json.dumps(
                {
                    **config,
                    "k8s_namespace": "blocked",
                    "max_connections_per_backend": 64,
                }
            )
        )
        watcher.running = True
        watcher._watch_worker()
        reconfigure.assert_not_called()
        assert watcher.get_current_config().k8s_namespace == "updated"
        assert watcher.get_current_config().max_connections_per_backend == 8
    finally:
        SingletonMeta._instances.pop(DynamicConfigWatcher, None)
