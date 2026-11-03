import unittest
import numpy as np
import pandas as pd
from v12_4_profit_first import profit_evidence, choose

def facts():
    rows=[]
    for period in ['2025-09-30','2025-12-31','2026-03-31','2026-06-30']:
        rows.append(dict(security_isin='TEST',period=pd.Timestamp(period),available=pd.Timestamp(period,tz='UTC')+pd.Timedelta(days=30),
                         period_kind='quarter',reporting_mode='consolidated',metric='pat',value_INR=10.,source_sha256=period))
    for metric,val in [('pat',40.),('cfo',30.)]:
        rows.append(dict(security_isin='TEST',period=pd.Timestamp('2026-03-31'),available=pd.Timestamp('2026-05-15',tz='UTC'),
                         period_kind='annual',reporting_mode='consolidated',metric=metric,value_INR=val,source_sha256='annual'))
    return pd.DataFrame(rows)

class ProfitEvidenceTest(unittest.TestCase):
    def test_four_consecutive_positive_quarters(self):
        r=profit_evidence(facts(),'2026-10-09','TEST')
        self.assertEqual(r['profit_evidence_status'],'PASS_REPORTED_PROFIT')
        self.assertEqual(r['four_quarter_PAT_INR'],40.)
    def test_negative_quarter_rejected_despite_positive_ttm(self):
        f=facts();f.loc[0,'value_INR']=-1
        self.assertEqual(profit_evidence(f,'2026-10-09','TEST')['profit_evidence_status'],'REJECT')
    def test_future_filing_cannot_fill_quarter(self):
        f=facts();f.loc[3,'available']=pd.Timestamp('2026-10-10',tz='UTC')
        self.assertEqual(profit_evidence(f,'2026-10-09','TEST')['profit_evidence_status'],'UNKNOWN')
    def test_wrong_isin_cannot_fill(self):
        self.assertEqual(profit_evidence(facts(),'2026-10-09','OTHER')['profit_evidence_status'],'UNKNOWN')
    def test_conflicting_same_clock_blocks(self):
        f=facts();r=f.iloc[[3]].copy();r['source_sha256']='conflict'
        self.assertEqual(profit_evidence(pd.concat([f,r]),'2026-10-09','TEST')['profit_evidence_status'],'UNKNOWN')
    def test_missing_quarter_not_replaced_by_older_one(self):
        f=facts();f.loc[1,'period']=pd.Timestamp('2025-06-30')
        self.assertEqual(profit_evidence(f,'2026-10-09','TEST')['profit_evidence_status'],'UNKNOWN')
    def test_no_basis_mixing(self):
        f=facts();f.loc[0,'reporting_mode']='standalone'
        self.assertEqual(profit_evidence(f,'2026-10-09','TEST')['profit_evidence_status'],'UNKNOWN')
    def test_negative_cfo_rejected(self):
        f=facts();f.loc[f.metric.eq('cfo'),'value_INR']=-1
        self.assertEqual(profit_evidence(f,'2026-10-09','TEST')['profit_evidence_status'],'REJECT')
    def test_rsi_not_hard_filter_but_loss_veto_is_separate(self):
        d=pd.DataFrame([dict(isin='TEST',profit_evidence_status='PASS_REPORTED_PROFIT',
            quarter_sales_yoy=.2,quarter_pat_yoy=.2,quarter_pat_margin=.1,annual_CFO_PAT=.8,
            dated_debt_equity=.5,close=100,distance_ma50=.1,log_ret63=.1,rsi14=60,
            p_hit2_6=.2,p_hit2_12=.3,p_loss30_12=.8,relative_strength63=.9)])
        self.assertEqual(len(choose(d,'profit_first_top5',12)),1)
        self.assertEqual(len(choose(d,'profit_first_legacy_risk_veto',12)),0)
        d['annual_CFO_PAT']=np.nan
        self.assertEqual(len(choose(d,'profit_first_top5',12)),0)

if __name__=='__main__':unittest.main()
