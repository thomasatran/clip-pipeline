"""Deepgram pre-recorded Nova-3 client.

Request shape checked against the docs on 2026-11-10:
POST /v1/listen?model=nova-3&language=en&smart_format=true&paragraphs=true
Authorization: Token <key>
Word start/end timestamps are in the response by default. See README.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.parse
from pathlib import Path

from clip.errors import ClipError

LISTEN_HOST = "api.deepgram.com"
LISTEN_PARAMS = (
    ("model", "nova-3"),
    ("language", "en"),
    ("smart_format", "true"),
    ("paragraphs", "true"),
)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def listen_url() -> str:
    return "https://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(LISTEN_PARAMS)


def request_headers(api_key: str, content_type: str) -> dict[str, str]:
    if not api_key or not api_key.strip():
        raise ClipError("DEEPGRAM_API_KEY is empty")
    return {
        "Authorization": f"Token {api_key.strip()}",
        "Content-Type": content_type,
    }


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }.get(suffix, "application/octet-stream")


def api_key_from_env() -> str | None:
    value = os.environ.get("DEEPGRAM_API_KEY")
    if value is None or not value.strip():
        return None
    return value.strip()


def transcribe_file(path: Path, api_key: str | None = None) -> dict:
    key = api_key if api_key is not None else api_key_from_env()
    if not key:
        raise ClipError(
            "DEEPGRAM_API_KEY is not set. Pass --transcript to run from a fixture transcript."
        )
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise ClipError("Deepgram pre-recorded uploads are limited to 2 GB")
    if size == 0:
        raise ClipError(f"nothing to transcribe: {path}")

    headers = request_headers(key, content_type_for(path))
    query = urllib.parse.urlencode(LISTEN_PARAMS)
    connection = http.client.HTTPSConnection(LISTEN_HOST, timeout=600)
    try:
        connection.putrequest("POST", f"/v1/listen?{query}")
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.putheader("Content-Length", str(size))
        connection.endheaders()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        response = connection.getresponse()
        body = response.read()
        status = response.status
    except urllib.error.URLError as exc:
        raise ClipError(f"Deepgram request failed: {exc.reason}") from exc
    finally:
        connection.close()

    if status >= 400:
        detail = body[:500].decode("utf-8", "replace")
        raise ClipError(f"Deepgram returned HTTP {status}: {detail}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ClipError("Deepgram returned a response that is not JSON") from exc
    if not isinstance(payload, dict):
        raise ClipError("Deepgram returned a response that is not a JSON object")
    return payload
