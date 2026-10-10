"""Four-year public NSE-universe history mining; no private request cohort.

Provider adjusted close is explicitly a dividend/split-adjusted total-return
proxy in V12. It is not reused as V11's price-only 2x label. Public current
listings imply survivor bias, recorded in every output summary.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from v11_4_free_source_fallbacks import PublicEvidenceStore, sha


def bounds(asof):
    end = pd.Timestamp(asof).normalize()
    return end - pd.DateOffset(years=4), end


def parse_chart(raw, symbol, isin, asof, receipt):
    j = json.loads(raw)
    if j.get('chart', {}).get('error'):
        raise ValueError('Provider security unavailable')
    results = j.get('chart', {}).get('result')
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError('Missing/ambiguous chart')
    c = results[0]
    m = c.get('meta', {})
    if (m.get('symbol') != symbol + '.NS' or
            m.get('exchangeName') not in ('NSI', 'NSE') or
            m.get('currency') != 'INR' or m.get('instrumentType') != 'EQUITY' or
            m.get('exchangeTimezoneName') != 'Asia/Kolkata'):
        raise ValueError('Wrong symbol, venue, currency or instrument')
    low, high = bounds(asof)
    stamps = c.get('timestamp', [])
    q = c.get('indicators', {}).get('quote', [])
    adj = c.get('indicators', {}).get('adjclose', [])
    if len(q) != 1 or len(adj) != 1:
        raise ValueError('Missing original/adjusted price arrays')
    q = q[0]
    arrays = {k: q.get(k, []) for k in ('open', 'high', 'low', 'close', 'volume')}
    arrays['adj_close'] = adj[0].get('adjclose', [])
    if any(len(v) != len(stamps) for v in arrays.values()):
        raise ValueError('Misaligned price/volume arrays')
    d = pd.DataFrame(arrays)
    d['date'] = pd.to_datetime(stamps, unit='s', utc=True).tz_convert('Asia/Kolkata').tz_localize(None).normalize()
    if d.date.duplicated().any():
        raise ValueError('Duplicate provider daily date')
    d = d[d.date.between(low, high)].copy()
    for k in arrays:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    numeric = d[list(arrays)].to_numpy(float)
    valid = (np.isfinite(numeric).all(axis=1) & d[['open', 'high', 'low', 'close', 'adj_close']].gt(0).all(axis=1)
             & d.volume.ge(0) & d.low.le(d[['open', 'close']].min(axis=1))
             & d.high.ge(d[['open', 'close']].max(axis=1)))
    invalid = int((~valid).sum())
    d = d.loc[valid].sort_values('date').reset_index(drop=True)
    if len(d) < 127:
        raise ValueError('Fewer than 127 valid daily observations')
    seen = pd.Timestamp(receipt['first_retrieved_utc'])
    if seen.tzinfo is None:
        raise ValueError('Source retrieval clock lacks timezone')
    if d.date.max().tz_localize('Asia/Kolkata') + pd.Timedelta(hours=15, minutes=30) > seen:
        raise ValueError('Unfinished future session')
    d['symbol'] = symbol
    d['isin'] = isin
    d['source_sha256'] = sha(raw)
    info = {'symbol': symbol, 'isin': isin, 'provider_name': m.get('longName', ''),
            'valid_bars': len(d), 'invalid_bars_excluded': invalid,
            'source_sha256': sha(raw), 'first_retrieved_utc': receipt['first_retrieved_utc'],
            'earliest_bar': str(d.date.min().date()), 'latest_bar': str(d.date.max().date()),
            'split_events': len(c.get('events', {}).get('splits', {})),
            'dividend_events': len(c.get('events', {}).get('dividends', {})),
            'historical_ISIN_and_corporate_actions_independently_verified': False}
    return d, info


def collect(universe_path, output, asof, workers=12):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    universe = pd.read_parquet(universe_path)
    required = {'nse_current_symbol', 'nse_current_name', 'isin'}
    if not required.issubset(universe) or universe.nse_current_symbol.duplicated().any():
        raise ValueError('Public official request universe is missing or ambiguous')
    universe = universe.sort_values('nse_current_symbol').reset_index(drop=True)
    universe.to_parquet(out / 'complete_public_NSE_request_universe.parquet', index=False)
    low, high = bounds(asof)
    period1 = int(low.tz_localize('Asia/Kolkata').timestamp())
    period2 = int((high + pd.Timedelta(days=1)).tz_localize('Asia/Kolkata').timestamp())
    store = PublicEvidenceStore(out / 'raw')
    output_bars = out / 'securities'
    output_bars.mkdir(exist_ok=True)
    errors, infos = [], []

    def one(item):
        symbol, isin = item['nse_current_symbol'], item['isin']
        url = ('https://query1.finance.yahoo.com/v8/finance/chart/' + quote(symbol + '.NS', safe='')
               + f'?interval=1d&period1={period1}&period2={period2}&events=div%2Csplits')
        try:
            raw, receipt = store.get(url)
            d, info = parse_chart(raw, symbol, isin, asof, receipt)
            d.to_parquet(output_bars / (sha(symbol.encode()) + '.parquet'), index=False)
            return info, None
        except Exception as exc:
            return None, {'symbol': symbol, 'url': url, 'error': str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in universe.to_dict('records')]
        for i, future in enumerate(as_completed(futures), 1):
            info, error = future.result()
            if info:
                infos.append(info)
            if error:
                errors.append(error)
            if i % 100 == 0:
                print(json.dumps({'completed': i, 'requested': len(universe),
                                  'usable': len(infos), 'errors': len(errors)}), flush=True)
    pd.DataFrame(infos).to_parquet(out / 'history_security_audit.parquet', index=False)
    (out / 'history_errors.json').write_text(json.dumps(errors, indent=2))
    summary = {'asof': asof, 'four_year_start': str(low.date()), 'requested': len(universe),
               'usable_securities': len(infos), 'errors': len(errors),
               'valid_daily_bars': sum(i['valid_bars'] for i in infos),
               'invalid_bars_excluded': sum(i['invalid_bars_excluded'] for i in infos),
               'outbound_symbols_from_complete_official_public_lists_only': True,
               'survivor_bias_current_listing_universe': True,
               'delisted_securities_coverage_complete': False,
               'corporate_actions_independently_verified': False,
               'provider_adjusted_close_target_is_total_return_proxy_not_price_only': True,
               'historical_data_before_four_year_boundary_used': False}
    (out / 'history_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--universe', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--asof', default='2026-10-09')
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    collect(a.universe, a.output, a.asof, a.workers)
