"""Free public source adapters with explicit identity, venue and quality gates.

Provider observations are CURRENT research evidence, never reconstructed PIT
financials or proof of a future 2x return. Original frozen model inputs are not
changed. OHLC is retained in the provider's representation; adjusted close
(which includes dividend adjustments) is never substituted for the 2x label.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, SoupStrainer

from v11_4_exchange_reference_universe import isin_valid

PUBLIC_HOSTS = {"query1.finance.yahoo.com", "query2.finance.yahoo.com",
                "ticker.finology.in", "stockanalysis.com", "nsearchives.nseindia.com",
                "archives.nseindia.com"}
EQUITY_SERIES = {"EQ", "BE", "BZ", "SM", "ST"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def finite_number(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    s = str(value).strip().replace(",", "").replace("₹", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s):
        return None
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except (ValueError, OverflowError):
        return None


class PublicEvidenceStore:
    """Resumable URL cache. Receipts preserve first retrieval, including errors."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.clocks = {}
        self.sessions = threading.local()

    def get(self, url):
        host = urlparse(url).hostname
        if urlparse(url).scheme != "https" or host not in PUBLIC_HOSTS:
            raise ValueError("Unapproved public source URL")
        key = sha(url.encode())
        meta, body = self.folder/(key+'.json'), self.folder/(key+'.bin')
        if meta.exists() and body.exists():
            receipt, raw = json.loads(meta.read_text()), body.read_bytes()
            if receipt['requested_url'] != url or sha(raw) != receipt['sha256']:
                raise ValueError("Cached response or source identity changed")
            if receipt['status'] != 200:
                raise ValueError(f"Previously recorded HTTP {receipt['status']}")
            return raw, receipt
        with self.lock:
            now = time.monotonic()
            spacing = .25 if host.startswith('query') or host=='ticker.finology.in' else .5
            pause = max(0, self.clocks.get(host, 0) + spacing - now)
            if pause:
                time.sleep(pause)
            self.clocks[host] = time.monotonic()
        if not hasattr(self.sessions,'client'):
            self.sessions.client=requests.Session()
            self.sessions.client.max_redirects=3
        session=self.sessions.client
        r = session.get(url, headers={'User-Agent': 'Mozilla/5.0',
                         'Accept': 'application/json,text/html,*/*'}, timeout=(8,15))
        if urlparse(r.url).hostname not in PUBLIC_HOSTS:
            raise ValueError("Response redirected away from approved public provider")
        raw = r.content
        if not raw or len(raw) > 8_000_000:
            raise ValueError("Empty or oversized public response")
        receipt = {'requested_url': url, 'final_url': r.url, 'status': r.status_code,
                   'first_retrieved_utc': datetime.now(timezone.utc).isoformat(),
                   'bytes': len(raw), 'sha256': sha(raw),
                   'content_type': r.headers.get('Content-Type', ''),
                   'body_file': body.name}
        # Commit bytes before their receipt; readers never see partial JSON.
        body_tmp = body.with_suffix('.bin.tmp')
        meta_tmp = meta.with_suffix('.json.tmp')
        body_tmp.write_bytes(raw)
        os.replace(body_tmp, body)
        meta_tmp.write_text(json.dumps(receipt, indent=2))
        os.replace(meta_tmp, meta)
        r.raise_for_status()
        return raw, receipt


def parse_interoperability_master(raw, asof):
    if raw[:2] != b'\x1f\x8b':
        raise ValueError("Interoperability master is not a genuine gzip file")
    m = pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False)
    required = {'TckrSymb', 'SctySrs', 'ISIN', 'PrtdToTrad', 'DelFlg',
                'FinInstrmNm', 'IssdCptl', 'ParVal'}
    if not required.issubset(m):
        raise ValueError("Official interoperability master schema changed")
    x = m[m.SctySrs.isin(EQUITY_SERIES) & m.DelFlg.eq('N')].copy()
    x['ISIN'] = x.ISIN.astype(str).str.strip().str.upper()
    x = x[x.ISIN.map(isin_valid) & x.ISIN.str.startswith('INE')].copy()
    b = x[x.PrtdToTrad.eq(2)].copy()
    if not b.TckrSymb.astype(str).str.endswith('$').all():
        raise ValueError("BSE-exclusive flag disagrees with the official symbol suffix")
    priority = {'EQ': 0, 'BE': 1, 'BZ': 2, 'SM': 3, 'ST': 4}
    b['priority'] = b.SctySrs.map(priority)
    b = b.sort_values(['ISIN', 'priority']).drop_duplicates('ISIN')
    result = pd.DataFrame({'isin': b.ISIN, 'bse_interop_symbol_truncated': b.TckrSymb.str[:-1],
        'bse_interop_name_truncated': b.FinInstrmNm, 'interop_series': b.SctySrs,
        'interop_reference_date': str(pd.Timestamp(asof).date()),
        'bse_exclusive_official_interop_reference': True,
        'ordinary_NSE_trading_permitted': False,
        'BSE_active_trading_and_price_coverage_proven': False,
        'source_sha256': sha(raw)})
    return result.reset_index(drop=True), x


def normalize_name(name):
    s = re.sub(r'\bengg\b', 'engineering', str(name).lower())
    s = re.sub(r'\binds\b', 'industries', s)
    s = re.sub(r'\b(?:limited|ltd|the|company|co|corporation|corp|private|pvt)\b', '', s)
    return re.sub(r'[^a-z0-9]', '', s)


def parse_yahoo_bse(raw, requested_symbol, first_seen_utc, asof, expected_name=''):
    j = json.loads(raw)
    if j.get('chart', {}).get('error'):
        raise ValueError("Yahoo reports a missing security")
    results = j.get('chart', {}).get('result')
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("Ambiguous Yahoo chart result")
    c = results[0]
    m = c.get('meta', {})
    if m.get('symbol') != requested_symbol or not requested_symbol.endswith('.BO'):
        raise ValueError("Provider security symbol differs from requested BSE ticker")
    if m.get('exchangeName') != 'BSE' or m.get('currency') != 'INR' or m.get('instrumentType') != 'EQUITY':
        raise ValueError("Provider venue, currency or instrument differs from BSE equity")
    if m.get('exchangeTimezoneName') != 'Asia/Kolkata':
        raise ValueError("Unexpected BSE exchange timezone")
    seen = pd.Timestamp(first_seen_utc)
    if seen.tzinfo is None:
        raise ValueError("First retrieval must have an explicit timezone")
    seen = seen.tz_convert('UTC')
    quote = finite_number(m.get('regularMarketPrice'))
    t = finite_number(m.get('regularMarketTime'))
    if quote is None or quote <= 0 or t is None or t <= 0:
        raise ValueError("Missing or nonfinite latest BSE quote")
    trade_time = pd.to_datetime(t, unit='s', utc=True)
    if trade_time > seen:
        raise ValueError("Provider trade time is after first retrieval")
    day = pd.Timestamp(asof).date()
    trade_day = trade_time.tz_convert('Asia/Kolkata').date()
    names_match = bool(expected_name and normalize_name(expected_name) == normalize_name(m.get('longName', '')))
    stamps = c.get('timestamp', [])
    quotes = c.get('indicators', {}).get('quote', [])
    if len(quotes) != 1:
        raise ValueError("Missing or ambiguous OHLC arrays")
    q = quotes[0]
    if any(len(q.get(k, [])) != len(stamps) for k in ['open', 'high', 'low', 'close', 'volume']):
        raise ValueError("Misaligned Yahoo OHLC arrays")
    rows, errors = [], []
    dates_seen = set()
    for i, ts in enumerate(stamps):
        stamp = pd.to_datetime(ts, unit='s', utc=True)
        date = stamp.tz_convert('Asia/Kolkata').date()
        if stamp > seen or date > day:
            raise ValueError("Future price bar in source response")
        if date in dates_seen:
            raise ValueError("Duplicate BSE daily price bar")
        dates_seen.add(date)
        values = {k: finite_number(q[k][i]) for k in ['open', 'high', 'low', 'close', 'volume']}
        o, h, l, close, v = [values[k] for k in ['open', 'high', 'low', 'close', 'volume']]
        valid = all(z is not None for z in (o, h, l, close, v)) and min(o, h, l, close) > 0 and v >= 0
        valid = bool(valid and l <= min(o, close) <= max(o, close) <= h)
        if not valid:
            errors.append({'date': str(date), 'reason': 'missing_or_invalid_OHLCV'})
            continue
        # A daily bar seen during trading is incomplete and cannot become a
        # finalized close. Latest intraday quote remains separately observable.
        close_clock = pd.Timestamp(f'{date} 15:30:00', tz='Asia/Kolkata')
        if seen < close_clock:
            errors.append({'date': str(date), 'reason': 'session_not_closed_at_first_retrieval'})
            continue
        rows.append({'date': str(date), **values, 'venue': 'BSE', 'currency': 'INR',
                     'provider_symbol': requested_symbol, 'first_retrieved_utc': str(seen),
                     'source_sha256': sha(raw), 'provider_OHLC_preserved_no_second_split_adjustment': True})
    info = {'provider': 'Yahoo Finance public chart', 'provider_symbol': requested_symbol,
        'venue': 'BSE', 'currency': 'INR', 'latest_price_INR': quote,
        'latest_trade_utc': trade_time.isoformat(), 'latest_trade_day_IST': str(trade_day),
        'same_asof_trade_day': trade_day == day,
        'provider_long_name': m.get('longName', ''), 'official_reference_name_matches': names_match,
        'ISIN_in_quote_response': False,
        'quote_identity_level': 'BSE_TICKER_AND_NAME' if names_match else 'BSE_TICKER_ONLY_REQUIRES_IDENTITY_REVIEW',
        'guaranteed_realtime_latency_proven': False,
        'source_sha256': sha(raw), 'first_retrieved_utc': str(seen),
        'daily_bars_valid': len(rows), 'invalid_or_incomplete_bars': len(errors),
        'split_events': len(c.get('events', {}).get('splits', {})),
        'dividend_events': len(c.get('events', {}).get('dividends', {})),
        'corporate_action_adjustment_independently_verified': False,
        'historical_PIT_representation_verified': False,
        'production_eligible': False}
    return info, rows, errors


def parse_table(section, table_kind, basis):
    if section is None:
        return []
    text = section.get_text(' ', strip=True)
    if not re.search(r'(?:Figures.*(?:Cr\.|Crores)|Figures.*in\s+Cr|Figures are in Crores)', text, re.I):
        raise ValueError("Financial table lacks an explicit crore unit")
    t = section.find('table')
    if t is None:
        return []
    trs = t.select('tr')
    headings = [e.get_text(' ', strip=True) for e in trs[0].find_all(['th', 'td'])][1:]
    periods = []
    for value in headings:
        parsed = pd.to_datetime(value, format='%b %Y', errors='coerce')
        if pd.isna(parsed):
            raise ValueError("Financial table has a non-period column")
        periods.append(str((parsed + pd.offsets.MonthEnd()).date()))
    if len(set(periods)) != len(periods):
        raise ValueError("Duplicate financial periods")
    labels = {'Net Sales': 'revenue', 'Operating Profit': 'operating_profit_provider',
              'Interest': 'finance_cost', 'Depreciation': 'depreciation',
              'Profit Before Tax': 'pbt', 'Profit After Tax': 'pat',
              'Net Profit': 'pat', 'Consolidated Net Profit': 'pat_parent',
              'Operating Cash Flow': 'cfo', 'Share Capital': 'share_capital',
              'Total Reserves': 'reserves', 'Borrowings': 'borrowings_provider',
              'Net Block': 'net_block_provider', 'Capital WIP': 'capital_wip_provider',
              'Intangible WIP': 'intangible_wip_provider', 'Total Assets': 'assets_provider',
              'Exceptional Items': 'exceptional_items_provider', 'Other Income': 'other_income_provider'}
    result = []
    for tr in trs[1:]:
        cells = [e.get_text(' ', strip=True) for e in tr.find_all(['th', 'td'], recursive=False)]
        if not cells or cells[0] not in labels:
            continue
        if len(cells) != len(periods) + 1:
            raise ValueError("Financial values do not align with dates")
        metric = labels[cells[0]]
        for date, value in zip(periods, cells[1:]):
            num = finite_number(value)
            if num is not None:
                result.append({'metric': metric, 'value_INR': num*1e7, 'period_end': date,
                    'period_kind': table_kind, 'reporting_mode': basis, 'display_label': cells[0],
                    'display_value_crore': num, 'rendered_display_rounding_INR': 1e7*(.5 if '.' not in value else .5*10**-len(value.rsplit('.',1)[1])),
                    'currency': 'INR', 'source_independent_original_filing_verified': False})
    return result


def parse_finology(raw, requested_url, symbol, expected_name, first_seen_utc):
    wanted={'mainContent_pnlCompanyDetails','mainContent_ltrlCompName','mainContent_togBtn',
        'profit','mainContent_quarterly','balance','mainContent_cashflows',
        'mainContent_divSales','mainContent_divProfit','mainContent_divROE','mainContent_divROCE',
        'companyessentials','mainContent_clsprice','mainContent_hfScripCode','documents'}
    s = BeautifulSoup(raw, 'html.parser',parse_only=SoupStrainer(id=lambda v:v in wanted))
    panel = s.select_one('#mainContent_pnlCompanyDetails')
    if panel is None or s.select_one('#mainContent_ltrlCompName') is None:
        raise ValueError("Finology company financial page unavailable")
    actual_name = s.select_one('#mainContent_ltrlCompName').get_text(' ', strip=True)
    page_text = panel.get_text(' ', strip=True)
    nse_symbol_match = bool(re.search(r'\bNSE:\s*' + re.escape(symbol) + r'\b', page_text))
    name_match = normalize_name(expected_name) == normalize_name(actual_name)
    if not (nse_symbol_match or name_match):
        raise ValueError("Finology displayed issuer differs from official security identity")
    switch = s.select_one('#mainContent_togBtn')
    basis = 'consolidated' if switch is not None and switch.has_attr('checked') else 'standalone'
    # The real checkbox determines basis. CSS labels .on/.off are present in
    # both views, and requesting mode=C does not prove a basis by itself.
    sections = [('profit', 'annual'), ('mainContent_quarterly', 'quarter'),
                ('balance', 'instant'), ('mainContent_cashflows', 'annual')]
    facts = []
    for id_, kind in sections:
        facts += parse_table(s.select_one('#'+id_), kind, basis)
    if not facts:
        raise ValueError("No usable public financial table")
    seen = pd.Timestamp(first_seen_utc)
    if seen.tzinfo is None:
        raise ValueError("First retrieval timezone missing")
    if any(pd.Timestamp(r['period_end']).date() > seen.tz_convert('Asia/Kolkata').date() for r in facts):
        raise ValueError("Financial page contains a future period")
    identity = {'symbol': symbol, 'company_name': actual_name, 'reporting_mode': basis,
        'source_url': requested_url, 'source_sha256': sha(raw), 'first_retrieved_utc': str(seen),
        'issuer_symbol_matches': nse_symbol_match, 'issuer_name_matches': name_match,
        'publication_times_available': False, 'historical_PIT_training_eligible': False}
    for r in facts:
        r.update(identity)
    reported = {}
    ratio_groups = {'mainContent_divSales': ('sales_growth',),
                    'mainContent_divProfit': ('profit_growth',),
                    'mainContent_divROE': ('roe',), 'mainContent_divROCE': ('roce',)}
    for id_, (prefix,) in ratio_groups.items():
        e = s.select_one('#'+id_)
        if e:
            for years, value in re.findall(r'(\d)\s+Year\s+([+-]?[\d.,]+)%', e.get_text(' ', strip=True)):
                num = finite_number(value)
                if num is not None:
                    field = f'{prefix}_{years}y' + ('_avg' if prefix in {'roe','roce'} and years != '1' else '')
                    reported[field] = num/100
    essentials = {}
    for div in s.select('#companyessentials .compess'):
        label, val = div.find('small'), div.find('p')
        if label is not None and val is not None:
            essentials[label.get_text(' ', strip=True)] = val.get_text(' ', strip=True)
    quote_el = s.select_one('#mainContent_clsprice')
    quote_text = quote_el.get_text(' ', strip=True) if quote_el else ''
    price_match = re.match(r'\s*([\d.,]+)', quote_text)
    venue_match = re.search(r'\b(NSE|BSE):\s*(\d{1,2}\s+[A-Za-z]{3}\s+\d{1,2}:\d{2}\s*[AP]M)', quote_text, re.I)
    quote = None
    if price_match and venue_match:
        ist = seen.tz_convert('Asia/Kolkata')
        qtime = pd.to_datetime(venue_match.group(2)+f' {ist.year}', format='%d %b %I:%M %p %Y')
        if qtime.month > ist.month:
            qtime = qtime - pd.DateOffset(years=1)
        qtime = qtime.tz_localize('Asia/Kolkata')
        if qtime.tz_convert('UTC') > seen:
            raise ValueError("Finology displayed quote time is after first retrieval")
        price = finite_number(price_match.group(1))
        cap = re.search(r'([\d.,]+)\s*Cr\.', essentials.get('Market Cap',''))
        shares = re.search(r'([\d.,]+)\s*Cr\.', essentials.get('No. of Shares',''))
        mc = finite_number(cap.group(1)) if cap else None
        sc = finite_number(shares.group(1)) if shares else None
        consistent = bool(price and price > 0 and mc and mc > 0 and sc and sc > 0
                          and abs(mc-sc*price) <= .0051*price+.015)
        pe, eps = finite_number(essentials.get('P/E')), finite_number(essentials.get('EPS (TTM)'))
        pe_agrees = bool(price and pe and pe > 0 and eps and eps > 0 and abs(pe-price/eps) <= max(.011, .0051*price/(eps*eps)))
        code = s.select_one('#mainContent_hfScripCode')
        quote = {'venue': venue_match.group(1).upper(), 'latest_price_INR': price,
            'displayed_quote_utc': qtime.tz_convert('UTC').isoformat(), 'displayed_quote_day_IST': str(qtime.date()),
            'bse_scrip_code': code.get('value','') if code else '',
            'reported_market_cap_crore': mc, 'reported_shares_crore': sc,
            'cap_shares_quote_arithmetic_agrees': consistent,
            'reported_PE': pe, 'reported_EPS_TTM': eps, 'PE_quote_EPS_arithmetic_agrees': pe_agrees,
            'venue_read_from_display_not_hidden_exchange_field': True}
    reports = []
    for t in s.select('table'):
        if 'Annual Report' not in t.get_text():
            continue
        for tr in t.select('tr'):
            for a in tr.select('a[href]'):
                reports.append({'label': tr.get_text(' ',strip=True), 'url': a['href']})
    return {'identity': identity, 'facts': facts, 'reported_ratios': reported,
            'quote': quote, 'annual_report_routes': reports,
            'source_qualified_full_model': False}


def reconcile_provider_facts(provider_facts, original_facts):
    """Compare exact issuer, period, flow kind and basis, including PAT scope."""
    originals = {}
    for f in original_facts:
        if not f.get('rendered_statement_agrees'):
            continue
        key = tuple(f[k] for k in ['symbol', 'reporting_mode', 'period_end', 'period_kind', 'metric'])
        originals.setdefault(key, []).append(f)
    audit = []
    for f in provider_facts:
        key = tuple(f[k] for k in ['symbol', 'reporting_mode', 'period_end', 'period_kind', 'metric'])
        choices = originals.get(key, [])
        values = {float(z['value_INR']) for z in choices}
        record = {k:f[k] for k in ['symbol','reporting_mode','period_end','period_kind','metric','value_INR']}
        if len(values) != 1:
            record['status'] = 'NO_UNIQUE_ORIGINAL_MATCH'
        else:
            original = next(iter(values))
            tolerance = float(f.get('rendered_display_rounding_INR', 0)) + .01
            record.update({'original_value_INR': original, 'difference_INR': f['value_INR']-original,
                           'display_rounding_tolerance_INR': tolerance,
                           'original_source_sha256': choices[0]['source_sha256']})
            record['status'] = 'AGREES_WITH_ORIGINAL' if abs(f['value_INR']-original) <= tolerance else 'DISAGREES_WITH_ORIGINAL'
        audit.append(record)
    return audit


def parse_stockanalysis_financials(raw, bse_code, expected_name, first_seen_utc):
    """Keep standardized definitions separate from the user's original metrics."""
    s=BeautifulSoup(raw,'html.parser');text=s.get_text(' ',strip=True)
    seen=pd.Timestamp(first_seen_utc)
    if seen.tzinfo is None:raise ValueError('Missing source timezone')
    if not re.search(r'BOM\s*:\s*'+re.escape(str(bse_code))+r'\b',text):
        raise ValueError('StockAnalysis displayed BSE code differs from request')
    headings=[e.get_text(' ',strip=True) for e in s.select('h1')]
    suffix=r'\s+(?:Financials Overview|Income Statement|Cash Flow Statement|Ratios and Metrics)$'
    if not any(normalize_name(re.sub(suffix,'',name))==normalize_name(expected_name) for name in headings):
        raise ValueError('StockAnalysis displayed issuer name differs')
    if not re.search(r'(?:Financials|Market cap) in millions INR',text,re.I):
        raise ValueError('StockAnalysis monetary representation is not explicit million INR')
    table=s.select_one('table');trs=table.select('tr') if table else []
    if len(trs)<3:raise ValueError('StockAnalysis financial table unavailable')
    headers=[c.get_text(' ',strip=True) for c in trs[0].find_all(['th','td'],recursive=False)][1:]
    period_cells=[c.get_text(' ',strip=True) for c in trs[1].find_all(['th','td'],recursive=False)][1:]
    if len(headers)!=len(period_cells):raise ValueError('StockAnalysis date columns misaligned')
    periods=[]
    for head,cell in zip(headers,period_cells):
        match=re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}',cell)
        if not match:raise ValueError('StockAnalysis lacks an exact statement period')
        date=pd.Timestamp(match.group())
        if date.date()>seen.tz_convert('Asia/Kolkata').date():raise ValueError('Future StockAnalysis statement period')
        kind='annual' if re.fullmatch(r'FY \d{4}',head) else 'ttm' if head=='TTM' else 'current'
        periods.append((str(date.date()),kind))
    if len(set(periods))!=len(periods):raise ValueError('Duplicate standardized statement periods')
    definitions={'Revenue':'provider_standardized_revenue',
        'Operating Income':'provider_standardized_operating_income',
        'Net Income':'provider_standardized_net_income',
        'Net Income to Common':'provider_standardized_net_income_to_common',
        'Operating Cash Flow':'provider_standardized_operating_cash_flow',
        'Market Capitalization':'provider_standardized_market_cap'}
    facts=[]
    for tr in trs[2:]:
        cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'],recursive=False)]
        if not cells or cells[0] not in definitions:continue
        if len(cells)!=len(periods)+1:raise ValueError('Standardized financial values misaligned')
        for value,(date,kind) in zip(cells[1:],periods):
            amount=finite_number(value)
            if amount is None:continue
            facts.append({'metric':definitions[cells[0]],'display_label':cells[0],
                'value_INR':amount*1e6,'period_end':date,'period_kind':kind,
                'reporting_mode':'PROVIDER_STANDARDIZED_BASIS_NOT_EXPLICITLY_VERIFIED',
                'original_PAT_and_OPM_definition_equivalence_verified':False})
    if not facts:raise ValueError('No usable standardized financial facts')
    return {'bse_scrip_code':str(bse_code),'source_sha256':sha(raw),
        'first_retrieved_utc':str(seen),'source_provider':'StockAnalysis / S&P Global Market Intelligence',
        'facts':facts,'historical_PIT_training_eligible':False,
        'quote_venue_and_market_cap_basis_not_automatically_equated':True}
