"""Downloading the law's current text from the Verkhovna Rada's open-data API.

The API serves anonymous clients only when the User-Agent is exactly "OpenData"; any other value,
including one that names this app, gets 403. A refused request is redirected to
zakon.rada.gov.ua, which robots.txt closes to programs, so redirects are never followed. The
client names itself in a separate header instead.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from policy_assistant import __version__

TEXT_URL = "https://data.rada.gov.ua/laws/show/{nreg}.txt"


class LawDownloadError(RuntimeError):
    """The text could not be downloaded; the message says what to do instead."""


def fetch_law_text(nreg: str, *, timeout_s: float = 60, url: str | None = None) -> str:
    url = url or TEXT_URL.format(nreg=nreg)
    headers = {"User-Agent": "OpenData", "X-Client": f"policy-assistant/{__version__}"}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout_s, follow_redirects=False)
    except httpx.TransportError as error:
        raise LawDownloadError(f"cannot reach {url}: {type(error).__name__}: {error}") from None
    manual = "save the text yourself as described in the README ('If the download is refused')"
    if response.is_redirect or response.status_code == 403:
        raise LawDownloadError(f"{url} refused the request (HTTP {response.status_code}); {manual}")
    if response.status_code != 200:
        raise LawDownloadError(f"{url} answered HTTP {response.status_code}; {manual}")
    if not response.headers.get("content-type", "").startswith("text/plain"):
        raise LawDownloadError(
            f"{url} sent {response.headers.get('content-type')!r}, not plain text; {manual}"
        )
    text = _decode(response.content, response.charset_encoding)
    if "Стаття 1" not in text:
        raise LawDownloadError(f"{url} did not return the text of a law; {manual}")
    return text


def save_law_text(text: str, path: Path, *, url: str) -> dict:
    """Writes the text as UTF-8 and a note beside it with where and when it came from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    note = {
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    return note


def _decode(raw: bytes, charset: str | None) -> str:
    for encoding in (charset, "utf-8-sig", "cp1251"):
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise LawDownloadError("the downloaded text is neither UTF-8 nor Windows-1251")
