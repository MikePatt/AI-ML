"""Tests for finance_ingest.extract."""

from __future__ import annotations

from finance_ingest.extract import extract_title_and_text


def test_extract_title_and_strips_scripts() -> None:
    html = """
    <html><head><title>News &amp; Co</title></head>
    <body>
    <script>alert(1)</script>
    <p>First line.</p>
    <style>.x{}</style>
    <p>Second line.</p>
    </body></html>
    """
    title, text = extract_title_and_text(html)
    assert title == "News & Co"
    assert "alert" not in text
    assert "First line" in text
    assert "Second line" in text
