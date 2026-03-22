"""HTML to plain text using BeautifulSoup + lxml."""

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup


def extract_title_and_text(html: str) -> tuple[Optional[str], str]:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    body = soup.body or soup
    chunks: list[str] = []
    for el in body.stripped_strings:
        t = str(el).strip()
        if t:
            chunks.append(t)
    text = "\n".join(chunks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text
