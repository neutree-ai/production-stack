import sys

import pytest

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
