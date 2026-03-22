"""Tests for finance_ingest.fetch with mocked HTTP."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from finance_ingest.fetch import fetch_url


class _Resp:
    def __init__(self, status_code: int, url: str, text: str) -> None:
        self.status_code = status_code
        self.url = url
        self.text = text
        self.content = text.encode("utf-8") if text else b""


def _client_with_response(resp: _Resp) -> MagicMock:
    mc = MagicMock()
    mc.get.return_value = resp
    mc.__enter__.return_value = mc
    mc.__exit__.return_value = None
    return mc


@patch("finance_ingest.fetch.httpx.Client")
def test_fetch_ok(mock_client_cls) -> None:
    mock_client_cls.return_value = _client_with_response(
        _Resp(200, "https://ex.com/final", "<html>ok</html>")
    )
    fr = fetch_url("https://ex.com/")
    assert fr.status_code == 200
    assert fr.body == "<html>ok</html>"
    assert fr.error is None
    assert "final" in fr.url


@patch("finance_ingest.fetch.httpx.Client")
def test_fetch_http_error_sets_error(mock_client_cls) -> None:
    mock_client_cls.return_value = _client_with_response(
        _Resp(404, "https://ex.com/miss", "not found")
    )
    fr = fetch_url("https://ex.com/miss")
    assert fr.status_code == 404
    assert fr.body == "not found"
    assert fr.error == "HTTP 404"


@patch("finance_ingest.fetch.httpx.Client")
def test_fetch_request_error(mock_client_cls) -> None:
    def boom(*_a, **_k):
        raise httpx.ConnectError("nope", request=MagicMock())

    mc = MagicMock()
    mc.get.side_effect = boom
    mc.__enter__.return_value = mc
    mc.__exit__.return_value = None
    mock_client_cls.return_value = mc

    fr = fetch_url("https://unreachable.test/")
    assert fr.status_code is None
    assert fr.body is None
    assert fr.error is not None
