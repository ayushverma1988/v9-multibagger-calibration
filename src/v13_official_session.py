"""Verify same-session NSE originals and identify temporary source failures."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from v13_breadth_primary import LIMIT, rows_in_original, url, window


class SourceUnavailable(Exception):
    def __init__(self, attempts):
        super().__init__('No verified current NSE source available')
        self.attempts = attempts


def unavailable_reason(exc):
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        if status in {403, 404, 408, 410, 429, 500, 502, 503, 504}:
            return f'HTTP_{status}'
    if isinstance(exc, (requests.exceptions.SSLError, requests.exceptions.ProxyError)):
        return None
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return type(exc).__name__
    return None


def read_capture(root, day, *, now=None):
    """Use just the exchange original, independently of adjusted-price inputs."""
    folder = Path(root).parent / 'v13_public_breadth_snapshot'
    path = folder / 'receipt.json'
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if receipt['source_asof'] != day:
        return None
    proof = receipt['nse_original']
    stamp = datetime.fromisoformat(proof['first_retrieved_utc'])
    start, end = window(day)
    original = folder / 'nse_source_session.zip'
    if (proof['source_url'] != url(day) or stamp.tzinfo is None
            or not start <= stamp <= end or stamp > (now or datetime.now(timezone.utc))
            or not 0 < proof['bytes'] <= LIMIT or not original.is_file()
            or original.stat().st_size != proof['bytes']):
        raise ValueError('Invalid same-session NSE capture metadata')
    raw = original.read_bytes()
    if hashlib.sha256(raw).hexdigest() != proof['sha256']:
        raise ValueError('Same-session NSE capture hash mismatch')
    rows_in_original(raw, day)
    return raw, proof
