"""Tests for the shared Celery factory and Redis URL handling."""

import pytest

from bestee_tasking import build_celery_app, redis_url_with_ssl_cert_reqs


def test_rediss_appends_cert_reqs() -> None:
    assert (
        redis_url_with_ssl_cert_reqs("rediss://h:25061/0")
        == "rediss://h:25061/0?ssl_cert_reqs=CERT_NONE"
    )


def test_rediss_with_query_uses_ampersand() -> None:
    assert (
        redis_url_with_ssl_cert_reqs("rediss://h:25061/0?foo=bar")
        == "rediss://h:25061/0?foo=bar&ssl_cert_reqs=CERT_NONE"
    )


def test_rediss_with_existing_cert_reqs_unchanged() -> None:
    url = "rediss://h:25061/0?ssl_cert_reqs=CERT_REQUIRED"
    assert redis_url_with_ssl_cert_reqs(url) == url


def test_plain_redis_unchanged() -> None:
    assert redis_url_with_ssl_cert_reqs("redis://h:6379/0") == "redis://h:6379/0"


def test_none_unchanged() -> None:
    assert redis_url_with_ssl_cert_reqs(None) is None


def test_build_app_pins_json_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DO_REDIS_CONNECTION", "redis://localhost:6379/0")
    app = build_celery_app("test")
    assert app.conf.task_serializer == "json"
    assert app.conf.result_serializer == "json"
    assert app.conf.accept_content == ["json"]


def test_build_app_applies_extra_conf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DO_REDIS_CONNECTION", "redis://localhost:6379/0")
    app = build_celery_app("test", include=["tasks"], worker_prefetch_multiplier=1)
    assert app.conf.worker_prefetch_multiplier == 1
