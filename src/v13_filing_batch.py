"""Validate original filings before combining any facts or marking a version done."""
import pandas as pd


KEY = ['symbol', 'security_isin', 'reporting_mode', 'period_kind', 'period_end',
       'available_at_utc', 'source_sha256']
META = ['period_start', 'source_url', 'first_retrieved_utc', 'original_document_symbol',
        'original_publication_utc', 'identity_mapping_source_sha256',
        'identity_mapping_first_seen_utc', 'identity_source']


def _conflicts(facts):
    """Test the original rows, including equal numeric values with different provenance."""
    group = facts.groupby(KEY, sort=False, dropna=False).ngroup()
    meta = [c for c in META if c in facts]
    metadata = facts[meta].groupby(group, sort=False).nunique(dropna=False).gt(1)
    values = facts.value.groupby([group, facts.metric], sort=False, dropna=False).nunique(dropna=False)
    values = values[values.gt(1)]
    bad = set(metadata.index[metadata.any(axis=1)]) | set(values.index.get_level_values(0))
    details = {}
    for gid, rows in facts.loc[group.isin(bad)].groupby(group, sort=False):
        def observed(column, subset):
            return sorted({None if pd.isna(x) else str(x) for x in subset[column]},
                          key=lambda x: (x is not None, str(x)))
        fields = {c: observed(c, rows) for c in meta if metadata.at[gid, c]}
        metrics = {metric: observed('value', rows[rows.metric.eq(metric)])
                   for metric in values.loc[gid].index} if gid in values.index.get_level_values(0) else {}
        details[gid] = {**{k: str(rows[k].iloc[0]) for k in KEY},
                        'conflicting_metadata': fields, 'conflicting_facts': metrics}
    return group, details


def _pivot(facts):
    # Only called on a validated set; duplicates now have identical evidence.
    x = facts.drop_duplicates(KEY + ['metric', 'value'])
    wide = x.pivot(index=KEY, columns='metric', values='value').reset_index()
    meta = [c for c in META if c in x]
    if meta:
        wide = wide.merge(x[KEY + meta].drop_duplicates(), on=KEY, validate='1:1')
    wide['period'] = pd.to_datetime(wide.period_end)
    wide['available'] = pd.to_datetime(wide.available_at_utc, utc=True, format='mixed')
    return wide


def financial_wide(facts):
    """Strict single-result API for callers that cannot consume quarantine receipts."""
    _, conflicts = _conflicts(facts)
    if any(item['conflicting_facts'] for item in conflicts.values()):
        raise ValueError('Conflicting original facts')
    if conflicts:
        raise ValueError('Conflicting financial source metadata')
    return _pivot(facts)


def partition(facts):
    """Quarantine every contributing document version, including cross-document collisions."""
    if facts.version_key.isna().any():
        raise ValueError('Missing original filing version key')
    groups, conflicts = _conflicts(facts)
    reviews = {}
    if conflicts:
        for gid, version in pd.DataFrame({'group': groups, 'version': facts.version_key})[
                groups.isin(conflicts)].drop_duplicates().itertuples(index=False, name=None):
            reviews.setdefault(version, []).append(conflicts[gid])
    good = facts.loc[~facts.version_key.isin(reviews)]
    return (_pivot(good) if not good.empty else pd.DataFrame(),
            set(good.version_key), reviews, len(good))
