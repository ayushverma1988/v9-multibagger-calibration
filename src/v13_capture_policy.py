"""Spend bounded source requests on recoverable gaps; retain every failure.

Identity conflicts require new evidence, not repeated downloads of the same
original. A short cooldown rotates them out without deleting the backlog.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd


def signature(master, changes):
    parser = hashlib.sha256()
    for name in ('v12_dated_training_inputs.py', 'v12_5_expanded_facts.py', 'v13_symbol_identity.py'):
        parser.update(Path(__file__).with_name(name).read_bytes())
    identities = {}
    for symbol,isin in zip(master.nse_current_symbol.astype(str),master['isin'].astype(str)):
        nodes={symbol};edges=[]
        for _ in range(21):
            add=[r for r in changes if r.get('new') in nodes and r.get('old') not in nodes]
            if not add:break
            edges.extend(add);nodes.update(r['old'] for r in add)
        identities[symbol]=hashlib.sha256(json.dumps([isin,edges],sort_keys=True).encode()).hexdigest()
    return parser.hexdigest(), identities


def failure_kind(error):
    text = str(error).lower()
    if any(x in text for x in ('403', '429', 'forbidden', 'too many requests')):
        return 'ACCESS_LIMIT'
    if any(x in text for x in ('isin conflict', 'isin unavailable', 'ambiguous original identity',
                               'rename path', 'original/master isin', 'identity changed')):
        return 'IDENTITY_REVIEW'
    if any(x in text for x in ('404', 'no usable exact-identity', 'conflicting original', 'revision review')):
        return 'DOCUMENT_REVIEW'
    return 'TRANSIENT_FETCH_OR_EXTRACTION'


def note_failure(error, clock, parser, identity):
    kind = failure_kind(error)
    days = 7 if kind in {'IDENTITY_REVIEW', 'DOCUMENT_REVIEW'} else 1 if kind == 'ACCESS_LIMIT' else .25
    now = pd.Timestamp(clock)
    if now.tzinfo is None:
        raise ValueError('Retry evidence clock requires timezone')
    return {'error': str(error)[:200], 'category': kind, 'last_attempt_utc': now.isoformat(),
            'retry_after_utc': (now + pd.Timedelta(days=days)).isoformat(),
            'parser_sha256': parser, 'identity_reference_sha256': identity}


def bootstrap(root, catalog, parser, identity):
    """Migrate the last completed pass's failures once, with its actual clock."""
    path = Path(root) / 'latest_run.json'
    if not path.exists():
        return {}
    previous = json.loads(path.read_text())
    clock = previous.get('completed_at_utc')
    if not clock:
        return {}
    by_url = {url:g[['version_key','symbol']].to_dict('records') for url,g in catalog.groupby('source_url')}
    result = {}
    for item in previous.get('steps', {}).get('financials', {}).get('errors', []):
        if not isinstance(item, dict):
            continue
        for row in by_url.get(item.get('source_url'), []):
            result[row['version_key']] = note_failure(item.get('error', ''), clock, parser, identity.get(row['symbol']))
    return result


def ready_queue(queue, journal, clock, parser, identity):
    now = pd.Timestamp(clock)
    flags = []
    deferred = {}
    for key,symbol in zip(queue.version_key,queue.symbol):
        item = journal.get(key)
        ready = not item or item.get('parser_sha256') != parser or item.get('identity_reference_sha256') != identity.get(symbol)
        if not ready:
            ready = pd.Timestamp(item['retry_after_utc']) <= now
        flags.append(ready)
        if not ready:
            category = item['category']
            deferred[category] = deferred.get(category, 0) + 1
    return queue.loc[flags].copy(), {'deferred_versions': sum(deferred.values()),
                                    'deferred_by_reason': deferred,
                                    'deferred_items_deleted': False,
                                    'identity_checks_relaxed': False}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)
