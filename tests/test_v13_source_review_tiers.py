import unittest
import pandas as pd
from v13_financial_crosscheck import compare,tag_current_reviews

class CurrentSourceReview(unittest.TestCase):
    def wide(self):
        return pd.DataFrame([dict(security_isin='A',reporting_mode='consolidated',period_kind='annual',period=pd.Timestamp('2026-03-31'),available=pd.Timestamp('2026-05-01T00:00Z'),revenue=100.,cfo=20.)])
    def obs(self,metric='revenue',family='SCREENER',amount=100.,period='2026-03-31'):
        return dict(symbol='A',isin='A',reporting_mode='consolidated',period_kind='annual',period_end=period,metric=metric,value_INR=amount,rounding_half_width_INR=.5,provider_family=family,source_url='https://example.org',first_observed_utc='2026-10-20T00:00Z')
    def test_future_primary_revision_cannot_corroborate_earlier_inputs(self):
        w=self.wide();future=w.copy();future['available']=pd.Timestamp('2026-10-20T00:00Z');future['revenue']=200.
        rows,_=compare(pd.concat([w,future]),[self.obs(amount=200.)],'2026-10-19')
        self.assertEqual(rows[0]['primary_value_INR'],100.)
        self.assertEqual(rows[0]['comparison_status'],'NUMERIC_CONFLICT_REVIEW_REQUIRED')
    def test_current_annual_scope_ignores_old_conflict_and_preserves_forecast(self):
        obs=[self.obs(m,f,v) for m,v in [('revenue',100.),('cfo',20.)] for f in ['SCREENER','SP_GLOBAL']]
        obs+=[self.obs(amount=0.,period='2025-03-31')]
        rows,_=compare(self.wide(),obs,'2026-10-19');rows[-1]['comparison_status']='NUMERIC_CONFLICT_REVIEW_REQUIRED'
        original=dict(isin='A',symbol='A',p6=.1,p12=.2,loss12=.3,sector_status='PASS_AVAILABLE_SEVEN_CHECKS')
        r=tag_current_reviews([original],self.wide(),{'rows':rows},'2026-10-19',{'A':'consolidated'})[0]
        self.assertEqual(r['source_comparison_tier'],'ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE')
        self.assertEqual(r['p6'],original['p6']);self.assertNotIn('source_comparison_tier',original)
    def test_current_conflict_and_unknown_sector_remain_explicit_review_tiers(self):
        rows,_=compare(self.wide(),[self.obs(metric='cfo',amount=30.)],'2026-10-19')
        inputs=[dict(isin='A',symbol='A',p6=.1),dict(isin='A',symbol='B',p6=.2,sector_status='UNKNOWN_SECTOR_RULES_OR_CLASSIFICATION')]
        a,b=tag_current_reviews(inputs,self.wide(),{'rows':rows},'2026-10-19',{'A':'consolidated'})
        self.assertEqual(a['source_comparison_tier'],'ANNUAL_SOURCE_CONFLICT_REVIEW');self.assertEqual(a['source_conflicting_metrics'],['cfo'])
        self.assertEqual(b['source_comparison_tier'],'SECTOR_RULES_PENDING_REVIEW');self.assertEqual(b['p6'],.2)

if __name__=='__main__':unittest.main()
