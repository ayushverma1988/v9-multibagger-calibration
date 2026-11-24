"""Block prospective registration when any cross-sectional input is late or unproven."""
from datetime import datetime, timezone
import re
import pandas as pd
from v13_information_clock import cutoff


def _hash(value):
    return bool(re.fullmatch(r'[0-9a-f]{64}', str(value)))


def _clock(value):
    try:
        value=pd.Timestamp(value)
        if pd.isna(value) or value.tzinfo is None:return None
        return value.tz_convert('UTC')
    except (ValueError,TypeError,OverflowError):
        return None


def registration_time_gate(frame, observed_at=None):
    """Require timely source-bound proof for every row used in market context.

    This is an evidence gate, not a change to any model head, score or cohort.
    Incomplete proof blocks the whole new pair; it does not rewrite old pairs.
    """
    now=_clock(observed_at or datetime.now(timezone.utc))
    reasons={}; invalid=0
    def bad(name):reasons[name]=reasons.get(name,0)+1
    for row in frame.to_dict('records'):
        issues=[]
        try:
            raw=row.get('information_cutoff_utc')
            if _clock(raw) is None:raise ValueError('missing actual cutoff')
            cut=cutoff(row)
            close=(pd.Timestamp(row['date']).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
            if now is None or cut>now:raise ValueError('future cutoff')
        except (ValueError,TypeError,KeyError):
            bad('INVALID_ACTUAL_CUTOFF');invalid+=1;continue
        official=row.get('official_source_sha256');official_clock=_clock(row.get('official_first_retrieved_utc'))
        snapshot_nse=row.get('scoring_snapshot_nse_sha256');snapshot_clock=_clock(row.get('scoring_snapshot_nse_retrieved_utc'))
        direct=_hash(official) and official_clock is not None and close<=official_clock<=cut
        captured=(_hash(official) and official==snapshot_nse and snapshot_clock is not None and close<=snapshot_clock<=cut)
        if not (direct or captured):issues.append('UNPROVEN_OFFICIAL_SOURCE_TIME')
        kind=row.get('scoring_price_history_source')
        if kind=='ORIGINAL_ADJUSTED_PROVIDER':
            digest=row.get('scoring_original_provider_sha256');clock=_clock(row.get('scoring_original_provider_first_retrieved_utc'))
            if not (_hash(digest) and digest==row.get('source_sha256') and clock is not None and close<=clock<=cut):
                issues.append('UNPROVEN_ORIGINAL_PROVIDER_TIME')
        elif kind in {'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY','ALTERNATIVE_ADJUSTED_SCORING_ONLY'}:
            digest=row.get('scoring_snapshot_sha256');clock=_clock(row.get('scoring_snapshot_ready_utc'))
            if not (_hash(digest) and digest==row.get('scoring_fallback_source_sha256') and clock is not None and close<=clock<=cut):
                issues.append('UNPROVEN_SCORING_SNAPSHOT_TIME')
            if kind=='ALTERNATIVE_ADJUSTED_SCORING_ONLY':
                reference=row.get('scoring_fallback_provider_sha256');reference_clock=_clock(row.get('scoring_fallback_provider_first_retrieved_utc'))
                if not (_hash(reference) and reference_clock is not None and reference_clock<=cut):
                    issues.append('UNPROVEN_FALLBACK_REFERENCE_TIME')
        else:issues.append('UNKNOWN_SCORING_SOURCE')
        if issues:
            invalid+=1
            for issue in issues:bad(issue)
    passed=len(frame)>0 and invalid==0
    return {'passed':passed,'status':'PASS' if passed else 'BLOCKED_UNPROVEN_SOURCE_TIME',
      'rows':len(frame),'rows_with_unproven_time':invalid,'reason_counts':reasons,
      'cross_sectional_context_requires_every_row':True,'existing_forecasts_changed':False}
