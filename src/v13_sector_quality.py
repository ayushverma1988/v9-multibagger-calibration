"""Dated sector rules and size eligibility; unknown facts never become passes."""
import numpy as np,pandas as pd
from v13_financial_evidence import financial_quality

def kind(industry):
    if not isinstance(industry,str) or not industry.strip():return 'UNKNOWN'
    if industry=='Banks':return 'BANK'
    if industry=='Insurance':return 'INSURER'
    if industry in {'Finance','Financial Services','Capital Markets'}:return 'OTHER_FINANCIAL'
    return 'NONFINANCIAL'
def assess(row):
    sector=kind(row.get('industry'))
    if sector=='NONFINANCIAL':return dict(sector_kind=sector,**financial_quality(row))
    if sector=='BANK':tests={'four_profitable_quarters':lambda v:v==1,'quarter_pat_yoy':lambda v:v>0,'bank_roa':lambda v:v>0,'bank_capital_buffer_pp':lambda v:v>=0,'bank_net_npa_change_pp':lambda v:v<=0}
    elif sector=='INSURER':tests={'four_profitable_quarters':lambda v:v==1,'quarter_pat_yoy':lambda v:v>0,'insurer_solvency_buffer_pp':lambda v:v>=0,'insurer_underwriting_or_vnb_margin':lambda v:v>0}
    else:return dict(sector_kind=sector,status='UNKNOWN_SECTOR_RULES_OR_CLASSIFICATION',unknown_fields=['dated_sector_specific_rules'],failed_checks=[],complete=False,quality_pass=False)
    unknown=[k for k in tests if row.get(k) is None or not np.isfinite(row.get(k,np.nan))]
    failed=[k for k,test in tests.items() if k not in unknown and not test(row[k])]
    return dict(sector_kind=sector,status='KNOWN_FINANCIAL_FAILURE' if failed else 'UNKNOWN_SECTOR_FINANCIAL_INPUTS' if unknown else 'PASS_SECTOR_RESEARCH_CHECKS',unknown_fields=unknown,failed_checks=failed,complete=not unknown,quality_pass=not unknown and not failed)
def dated_members(frame,members):
    out=frame.copy();out['industry']=None;out['size_category']=None
    if members.empty:return out
    needed={'isin','symbol','available_at_utc','source_sha256','industry','size_category'}
    if not needed.issubset(members):raise ValueError('Incomplete membership provenance')
    m=members.copy();m['clock']=pd.to_datetime(m.available_at_utc,utc=True,errors='coerce')
    m=m[m.clock.notna()&m.source_sha256.astype(str).str.fullmatch('[0-9a-f]{64}')]
    if m.duplicated(['isin','symbol','clock']).any():raise ValueError('Ambiguous membership vintage')
    for i,r in out.iterrows():
        cut=r.get('information_cutoff_utc')
        if cut is None or pd.isna(cut):cut=pd.Timestamp(r.date).normalize().tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
        cut=pd.Timestamp(cut)
        if cut.tzinfo is None:raise ValueError('Decision clock needs timezone')
        k=m[m['isin'].eq(r['isin'])&m.symbol.eq(r.symbol)&m.clock.le(cut)]
        if len(k):
            z=k.sort_values('clock').iloc[-1];out.loc[i,['industry','size_category']]=[z.industry,z.size_category]
    return out
def annotate(frame):
    out=frame.copy();checks=[assess(r) for r in out.to_dict('records')]
    for k in ['sector_kind','status','unknown_fields','failed_checks','quality_pass']:out['sector_'+k]=[c[k] for c in checks]
    size=out.get('size_category',pd.Series(None,index=out.index,dtype=object))
    out['size_status']=np.select([size.isin(['Small Cap','Mid Cap']),size.eq('Large Cap')],['SMALL_MID_VERIFIED','LARGE_CAP_OUT_OF_SCOPE'],default='UNKNOWN_DATED_SIZE')
    return out
