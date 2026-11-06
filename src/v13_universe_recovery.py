"""Fail-closed, same-day recovery from preserved original NSE master bytes.

This is transport recovery, not permission to use yesterday's universe. The
normal parser and cross-master identity checks still run in v13_collection.
"""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo


SEED_ROOT = Path(__file__).resolve().parents[1] / 'v13_evidence_seed' / 'universe_source_recovery'
FORMAT = 'V13_ORIGINAL_NSE_MASTER_CAPTURE_V1'
ALLOWED_URLS = {
    'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv',
    'https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv',
}
IST = ZoneInfo('Asia/Kolkata')


def _clock(value):
    clock = datetime.fromisoformat(value)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError('Recovery clock must include a timezone')
    return clock.astimezone(timezone.utc)


def captured_master(url, asof, folder, *, seed_root=None, now=None):
    """Return a verified capture only for today's requested completed session.

    The caller may invoke this only after exhausting transient fetch retries.
    Permission/rate-limit, identity and integrity errors never permit recovery.
    Missing or expired captures return None; malformed captures raise.
    """
    if url not in ALLOWED_URLS:
        raise ValueError('Unapproved master recovery source')
    day = date.fromisoformat(asof)
    if day.isoformat() != asof:
        raise ValueError('Recovery date must use ISO format')
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Recovery current clock must include a timezone')
    now = now.astimezone(timezone.utc)
    if now.astimezone(IST).date() != day:
        return None
    seed = Path(seed_root) if seed_root is not None else SEED_ROOT
    path = seed / asof / 'manifest.json'
    if not path.exists():
        return None
    manifest_raw = path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get('format') != FORMAT or manifest.get('source_asof') != asof:
        raise ValueError('Recovery manifest format/date mismatch')
    items = manifest.get('captures')
    if not isinstance(items, list):
        raise ValueError('Recovery manifest captures missing')
    matches = [item for item in items if item.get('source_url') == url]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('Ambiguous recovery source')
    item = matches[0]
    digest = item.get('sha256', '')
    if not re.fullmatch(r'[0-9a-f]{64}', digest) or item.get('body_file') != digest + '.csv':
        raise ValueError('Recovery body identity mismatch')
    if item.get('http_status') != 200 or item.get('final_url') != url:
        raise ValueError('Recovery original HTTP source/status mismatch')
    first = _clock(item['first_retrieved_utc'])
    checked = _clock(item['independently_reverified_utc'])
    if not first <= checked <= now or first.astimezone(IST).date() != day or checked.astimezone(IST).date() != day:
        raise ValueError('Recovery capture has stale or future observation clocks')
    expected = item.get('bytes')
    if type(expected) is not int or not 0 < expected <= 5_000_000:
        raise ValueError('Recovery original length invalid')
    body = path.parent / item['body_file']
    if body.is_symlink() or not body.is_file() or body.stat().st_size != expected:
        raise ValueError('Recovery original length/path mismatch')
    raw = body.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Recovery original hash mismatch')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    proof = {
        'requested_url': url, 'status': 200, 'sha256': digest,
        'bytes': len(raw), 'body_file': key + '.bin',
        'first_retrieved_utc': item['first_retrieved_utc'],
        'independently_reverified_utc': item['independently_reverified_utc'],
        'source_asof': asof, 'same_day_original_recovery': True,
        'recovery_used_at_utc': now.isoformat(),
        'remote_refetched_by_this_worker': False,
        'recovery_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
    }
    (folder / proof['body_file']).write_bytes(raw)
    (folder / (key + '.json')).write_text(json.dumps(proof, indent=2))
    return raw, proof
