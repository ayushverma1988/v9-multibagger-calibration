import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from v13_action_audit import reconstruct,verified_return_paths
from v13_catalyst_execution import project_bundles
from v13_frozen_integrity import FrozenGuard
from v13_lifecycle import register
from v13_parallel_collection import bounded_map
from v13_valuation_evidence import peer_hurdles
from v12_5_expanded_facts import parse_extra
from v13_collection import stable_event_windows
from v13_research_evaluation import paired_readiness
from v13_information_clock import cutoff,research_cutoffs

class SixAreas(unittest.TestCase):
    def test_after_close_inputs_use_actual_known_cutoff_and_not_future_2200(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2026-10-13','2026-10-14'])})
        d=research_cutoffs(frame,pd.Timestamp('2026-10-14T14:00:00Z'))
        self.assertEqual(cutoff(d.iloc[0]),pd.Timestamp('2026-10-13T16:30:00Z'))
        self.assertEqual(cutoff(d.iloc[1]),pd.Timestamp('2026-10-14T14:00:00Z'))
        self.assertEqual(cutoff(frame.iloc[1]),pd.Timestamp('2026-10-14T10:00:00Z'))
        with self.assertRaises(ValueError):cutoff({'date':'2026-10-14','information_cutoff_utc':'2026-10-14T18:00:00Z'})
        with self.assertRaises(ValueError):research_cutoffs(frame,pd.Timestamp('2026-10-14T08:00:00Z'))

    def test_forecast_cannot_claim_information_observed_after_registration(self):
        with tempfile.TemporaryDirectory() as t:
            f=pd.DataFrame([{'date':pd.Timestamp('2026-10-14'),'isin':'INE000000001','symbol':'A','p6':.1,'p12':.2,'loss12':.3,'joint12':.1,'information_cutoff_utc':'2026-10-14T15:00:00Z'}])
            with self.assertRaises(ValueError):register(f,'a'*64,t,'2026-10-14T14:00:00Z')
    def test_daily_backfill_windows_are_stable_and_never_go_outside_four_years(self):
        a=stable_event_windows('2026-10-14');b=stable_event_windows('2026-10-15')
        self.assertGreater(len(set(a)&set(b)),45)
        self.assertTrue(all(pd.Timestamp('2022-10-14')<=x<=y<=pd.Timestamp('2026-10-14') and (y-x).days<=28 for x,y in a))
        self.assertEqual(paired_readiness(pd.DataFrame()),[])
    def prices(self,close,low,adj=None):
        dates=pd.bdate_range('2026-10-12',periods=len(close));high=np.maximum(close,100)+2
        o=pd.DataFrame({'date':dates,'open':close,'close':close,'low':low,'high':high,'source_status':'VALID_POSITIVE_VOLUME_SESSION'})
        p=o[['date','open','close','low','high']].copy();p['adj_close']=close if adj is None else adj
        return o,p

    def test_intraday_loss_must_not_be_hidden_by_close(self):
        o,p=self.prices([100,210],[99,65]);x,returns=reconstruct(o,p,[]);paths=verified_return_paths(x,returns,[])
        self.assertLess(paths['total_return_low'][1],.7)
        self.assertGreater(paths['total_return_close'][1],2)

    def test_loss_after_double_is_distinct_from_frozen_first_event_target(self):
        o,p=self.prices([100,210,80],[99,205,60]);x,r=reconstruct(o,p,[]);paths=verified_return_paths(x,r,[])
        hits=np.flatnonzero(paths['total_return_close']>=2);losses=np.flatnonzero(paths['total_return_low']<=.7)
        self.assertLess(hits[0],losses[0]);self.assertTrue(len(losses))

    def test_dividend_can_not_be_mistaken_for_price_only_doubling(self):
        o,p=self.prices([100,100],[99,99],[50,100])
        actions=[{'kind':'dividend','ex_date':'2026-10-13','cash_per_old_share':100}]
        x,r=reconstruct(o,p,actions);paths=verified_return_paths(x,r,actions)
        self.assertAlmostEqual(paths['total_return_close'][1]/paths['price_only_close'][1],2)
        self.assertLess(paths['price_only_close'][1],1)

    def test_missing_lows_cannot_certify_risk(self):
        o,p=self.prices([100,110],[99,100]);p=p.drop(columns=['low']);x,r=reconstruct(o,p,[])
        with self.assertRaises(ValueError):verified_return_paths(x,r,[])

    def test_wrong_provider_intraday_factor_is_rejected(self):
        o,p=self.prices([100,110],[99,100]);p.loc[1,'low']=70;x,r=reconstruct(o,p,[])
        with self.assertRaises(ValueError):verified_return_paths(x,r,[])

    def bundle(self,project,**kw):
        return {'isin':'INE000000001','project_id':project,'source_sha256':hashlib.sha256(str(kw).encode()).hexdigest(),
          'available_at_utc':'2026-10-13T04:00:00Z',**kw}

    def test_same_project_can_join_different_original_evidence(self):
        a=self.bundle('plant-1',commissioned=True,commercial_operation_verified=True)
        b=self.bundle('plant-1',funding_sufficient=True,approval_completed=True,approval_scope='PROJECT_REQUIRED_OPERATING_APPROVALS',expected_margin_pct=10,margin_basis='EBITDA')
        q=project_bundles([a,b],a['isin'],pd.Timestamp('2026-10-14T10:00:00Z'))
        self.assertTrue(q[0]['supported']);self.assertEqual(len(q[0]['input_source_sha256']),2)
        b['project_id']='plant-2';q=project_bundles([a,b],a['isin'],pd.Timestamp('2026-10-14T10:00:00Z'))
        self.assertTrue(all(not z['supported'] for z in q))

    def test_board_approval_or_later_cancellation_does_not_complete_chain(self):
        a=self.bundle('one',firm_order=True,order_due='2027-01-01',funding_sufficient=True,approval_completed=True,expected_margin_pct=10,margin_basis='EBIT')
        cut=pd.Timestamp('2026-10-14T10:00:00Z')
        self.assertFalse(project_bundles([a],a['isin'],cut)[0]['supported'])
        a['approval_scope']='PROJECT_REQUIRED_OPERATING_APPROVALS'
        b=self.bundle('one',order_status='CANCELLED',available_at_utc='2026-10-14T09:00:00Z')
        self.assertFalse(project_bundles([a,b],a['isin'],cut)[0]['supported'])
        b['available_at_utc']='2026-10-14T11:00:00Z'
        self.assertTrue(project_bundles([a,b],a['isin'],cut)[0]['supported'])

    def test_rate_limit_stops_new_requests_without_bypassing_source(self):
        out=list(bounded_map(list(range(8)),lambda x:{'error':'HTTP 429 Too Many Requests'},workers=1))
        self.assertEqual(len(out),1)

    def test_peers_exclude_the_company_and_hurdles_do_not_claim_support(self):
        c=pd.DataFrame({'symbol':list('ABCDEF'),'isin':list('abcdef'),'close':[100]*6,'company_pe':[10,20,30,40,50,60]})
        m=pd.DataFrame({'symbol':list('ABCDEF'),'isin':list('abcdef'),'industry':['one']*6})
        q=peer_hurdles(c,m);self.assertEqual(q[0]['other_positive_PE_peers'],5)
        self.assertEqual(q[0]['covered_peer_PE_quantiles']['0.5'],40)
        self.assertFalse(q[0]['2x_scenario_supported'])
        c.loc[0,'company_pe']=-5;q=peer_hurdles(c,m);self.assertIsNone(q[0]['reported_company_PE'])

    def test_owner_pat_and_minority_are_separate_exact_currency_facts(self):
        raw=b'''<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:f="urn:test"><context id="q"><period><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></period></context><unit id="INR"><measure>iso4217:INR</measure></unit><f:Symbol>TEST</f:Symbol><f:ISIN>INE123A01012</f:ISIN><f:NatureOfReportStandaloneConsolidated>Consolidated</f:NatureOfReportStandaloneConsolidated><f:RevenueFromOperations contextRef="q" unitRef="INR">100</f:RevenueFromOperations><f:ProfitLossForPeriodAttributableToOwnersOfParent contextRef="q" unitRef="INR">7</f:ProfitLossForPeriodAttributableToOwnersOfParent><f:ProfitLossForPeriodAttributableToNonControllingInterests contextRef="q" unitRef="INR">3</f:ProfitLossForPeriodAttributableToNonControllingInterests></xbrl>'''
        r={'symbol':'TEST','catalog_isin':'INE123A01012','reporting_mode':'consolidated','period_end':'2026-06-30'}
        facts=parse_extra(raw,r,'2026-10-14');d={x['metric']:x['value'] for x in facts}
        self.assertEqual(d['pat_owners'],7);self.assertEqual(d['pat_noncontrolling'],3)

    def test_guard_restores_frozen_registry_and_head_on_mutation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);frame=pd.DataFrame([{'date':pd.Timestamp('2026-10-09'),'isin':'INE000000001','symbol':'A','p6':.1,'p12':.2,'loss12':.3,'joint12':.1}])
            p=register(frame,'a'*64,root/'registries/champion','2026-10-11T16:00:00Z');raw=p.read_bytes()
            (root/'active_heads.joblib').write_bytes(b'fixed');guard=FrozenGuard(root)
            p.write_bytes(b'changed');(root/'active_heads.joblib').write_bytes(b'bad')
            with self.assertRaises(ValueError):guard.check(pd.DataFrame())
            self.assertEqual(p.read_bytes(),raw);self.assertEqual((root/'active_heads.joblib').read_bytes(),b'fixed')

    def test_guard_rejects_new_backdated_registration(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);guard=FrozenGuard(root)
            frame=pd.DataFrame([{'date':pd.Timestamp('2026-10-09'),'isin':'INE000000001','symbol':'A','p6':.1,'p12':.2,'loss12':.3,'joint12':.1}])
            register(frame,'a'*64,root/'registries/champion','2026-10-11T16:00:00Z')
            with self.assertRaises(ValueError):guard.check(pd.DataFrame())

if __name__=='__main__':unittest.main()
