import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from v13_lifecycle import register,validate_registry,adjudicate,paired_observations,promotion_gate
from v13_paired_forecasts import register_pair,export_watchlists,selective
from v13_financial_evidence import priority_queue,earnings_scenario
from v13_catalyst_execution import reviewed
from v13_action_audit import reconstruct,expected_sessions,validate_action_notice
from v13_research_evaluation import experiments,prospective_report

class EvidencePairing(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{'date':pd.Timestamp('2026-10-09'),'isin':'INE000000001','symbol':'A','p6':.1,'p12':.2,'loss12':.4,'joint12':.15,'selected_verified':False,'selected_discovery':False},
                             {'date':pd.Timestamp('2026-10-09'),'isin':'INE000000002','symbol':'B','p6':.15,'p12':.25,'loss12':.5,'joint12':.15,'selected_verified':True,'selected_discovery':False}])

    def test_pair_same_clock_all_names_and_idempotency(self):
        f=self.frame();b=f.copy();b.p12+=.1
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=register(f,'a'*64,root/'registries/champion','2026-10-11T16:00:00Z');original=old.read_bytes()
            pair=register_pair(f,f,b,'a'*64,'b'*64,root,'2026-10-12T16:00:00Z')
            a=validate_registry(root/'registries/champion'/pair['champion_registry']);c=validate_registry(root/'registries/challenger'/pair['challenger_registry'])
            self.assertEqual(a['registered_at_utc'],c['registered_at_utc']);self.assertEqual(len(a['records']),2)
            self.assertEqual(a['paired_context'],c['paired_context']);self.assertEqual(old.read_bytes(),original)
            again=register_pair(f,f,b,'a'*64,'b'*64,root,'2026-10-13T16:00:00Z')
            self.assertTrue(again['already_registered']);self.assertEqual(again['cohort_id'],pair['cohort_id'])
            export_watchlists(root,pair);self.assertEqual(len(json.loads((root/'champion_latest_watchlist.json').read_text())['records']),1)

    def test_incomplete_or_incoherent_forecast_cannot_register(self):
        f=self.frame();f.at[0,'p12']=np.nan
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):register(f,'a'*64,t,'2026-10-11T16:00:00Z')
            f.at[0,'p12']=.05
            with self.assertRaises(ValueError):register(f,'a'*64,t,'2026-10-11T16:00:00Z')

    def test_pair_rejects_unequal_universe(self):
        f=self.frame()
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):register_pair(f,f,f.iloc[:1],'a'*64,'b'*64,t,'2026-10-12T16:00:00Z')

    def test_immature_schema_is_present_without_fabricated_outcomes(self):
        f=self.frame()
        official=pd.DataFrame([{'date':'2026-10-09','isin':'INE000000001','symbol':'A','source_status':'VALID_POSITIVE_VOLUME_SESSION','open':100,'close':200}])
        with tempfile.TemporaryDirectory() as t:
            p=register(f,'a'*64,t,'2026-10-11T16:00:00Z');o=pd.DataFrame(adjudicate(p,official,{},'2026-10-09'))
            self.assertIn('verified_hit2_12',o);self.assertTrue(o.verified_hit2_12.isna().all())
            paired=paired_observations(o,'b'*64);self.assertIn('y',paired)
            reason=promotion_gate(paired,'2026-10-11T16:00:00Z')['reasons']
            self.assertNotIn('missing_paired_prospective_schema',reason);self.assertIn('fewer_than_500_mature_paired_rows',reason)

    def test_pair_requires_same_actual_clock_and_verified_outcome(self):
        base={'cohort_id':'x','source_asof':'2025-01-01','isin':'INE000000001','registry_sha256':'r','registered_at_utc':'2025-01-01T16:00:00Z','p12':.2,'selected_verified':False,'verified_hit2_12':1,'twelve_month_mature':True,'first_entry_date':'2025-01-02','status':'TRACKING','independent_outcome_verified':True}
        base.update(entry_verified=True,entry_open_INR=100,entry_official_sha256='c'*64,entry_provider_sha256='d'*64)
        a={**base,'model_role':'champion','model_sha256':'a'*64};b={**base,'model_role':'challenger','model_sha256':'b'*64}
        self.assertEqual(len(paired_observations(pd.DataFrame([a,b]),'b'*64)),1)
        b['registered_at_utc']='2025-01-02T16:00:00Z';self.assertTrue(paired_observations(pd.DataFrame([a,b]),'b'*64).empty)
        b['registered_at_utc']=a['registered_at_utc'];b['independent_outcome_verified']=False
        self.assertTrue(paired_observations(pd.DataFrame([a,b]),'b'*64).empty)

    def test_prioritises_missing_periods_without_private_picks(self):
        q=pd.DataFrame([{'symbol':'A','period_end':'2026-06-30','last_attempt':'0000','available_at_utc':'2026-08-01T00:00:00Z'},
                        {'symbol':'A','period_end':'2026-03-31','last_attempt':'0000','available_at_utc':'2026-05-01T00:00:00Z'},
                        {'symbol':'B','period_end':'2026-06-30','last_attempt':'0000','available_at_utc':'2026-08-01T00:00:00Z'}])
        out=priority_queue(q,pd.DataFrame(),'2026-10-09',2);self.assertEqual(set(out.symbol),{'A','B'})

    def test_valuation_scenario_respects_dilution_and_missingness(self):
        z=earnings_scenario(100,1000,100,2000,200,10)
        self.assertEqual(z['scenario_multiple'],1);self.assertEqual(z['share_growth'],1)
        self.assertEqual(earnings_scenario(100,1000,100,None,100,10)['status'],'UNKNOWN_MISSING_OR_NONPOSITIVE_INPUT')

    def test_catalyst_intent_is_not_commissioning_and_source_clock_not_backdated(self):
        text='The company plans to add capacity of 9000 units per year.';sha=hashlib.sha256(text.encode()).hexdigest()
        e={'isin':'INE000000001','symbol':'A','attachment_url':'https://nsearchives.nseindia.com/a.pdf','source_sha256':'a'*64,'text_sha256':sha,'issuer_identity_read':True,'first_retrieved_utc':'2026-10-12T16:00:00Z'}
        r={k:e[k] for k in ('isin','symbol','attachment_url','source_sha256')};r['capacity_units_per_year']=9000;r['field_excerpts']={'capacity_units_per_year':'capacity of 9000 units per year'}
        z=reviewed(r,e,text);self.assertIn('commissioned',z['unknown_fields']);self.assertEqual(pd.Timestamp(z['available_at_utc']),pd.Timestamp(e['first_retrieved_utc']))
        r['commissioned']=True;r['field_excerpts']['commissioned']=text
        with self.assertRaises(ValueError):reviewed(r,e,text)
        del r['commissioned']
        r['funding_sufficient']=True
        with self.assertRaises(ValueError):reviewed(r,e,text)

    def prices(self,closes,adj):
        dates=pd.date_range('2026-10-12',periods=len(closes),freq='B')
        o=pd.DataFrame({'date':dates,'open':closes,'close':closes,'source_status':'VALID_POSITIVE_VOLUME_SESSION'})
        p=pd.DataFrame({'date':dates,'open':closes,'close':closes,'adj_close':adj})
        return o,p

    def test_split_does_not_create_false_double_or_loss(self):
        o,p=self.prices([100,50,60],[50,50,60]);_,r=reconstruct(o,p,[{'kind':'split','ex_date':'2026-10-13','share_multiplier':2}])
        np.testing.assert_allclose(r,[1,1,1.2])

    def test_cash_dividend_factor_matches_primary_notice(self):
        o,p=self.prices([100,95],[95,95]);_,r=reconstruct(o,p,[{'kind':'dividend','ex_date':'2026-10-13','cash_per_old_share':5}])
        np.testing.assert_allclose(r,[1,1])

    def test_missing_notice_and_unsupported_rights_fail_closed(self):
        o,p=self.prices([100,50],[50,50])
        with self.assertRaises(ValueError):reconstruct(o,p,[])
        with self.assertRaises(ValueError):reconstruct(o,p,[{'kind':'rights','ex_date':'2026-10-13'}])

    def test_unknown_calendar_cannot_prove_first_entry(self):
        self.assertIsNone(expected_sessions('2026-10-12','2026-10-14',None))
        c={'year':2026,'holidays':['2026-10-13']}
        self.assertEqual(len(expected_sessions('2026-10-12','2026-10-14',c)),2)
        self.assertIsNone(expected_sessions('2026-12-30','2027-01-04',c))

    def test_action_factor_needs_original_issuer_and_excerpts(self):
        raw=b'Trading Symbol: A. Ex date 13 October 2026. Shares split from 1 to 2.'
        a={'isin':'INE000000001','symbol':'A','kind':'split','ex_date':'2026-10-13','share_multiplier':2,
           'source_share_ratio':{'kind':'share_counts','old':1,'new':2},'source_sha256':hashlib.sha256(raw).hexdigest(),
           'field_excerpts':{'ex_date':'Ex date 13 October 2026','share_multiplier':'Shares split from 1 to 2'}}
        self.assertTrue(validate_action_notice(a,raw));a['symbol']='OTHER'
        with self.assertRaises(ValueError):validate_action_notice(a,raw)
        a['symbol']='A';a['field_excerpts']['share_multiplier']='made-up ratio'
        with self.assertRaises(ValueError):validate_action_notice(a,raw)

    def test_weak_evidence_abstains_without_padding(self):
        f=self.frame();f['evidence_status']='MISSING_FINANCIAL_OR_EXECUTION'
        with patch('v13_paired_forecasts.model.gates',return_value=(pd.Series(True,index=f.index),pd.Series(False,index=f.index),pd.Series(False,index=f.index))):
            self.assertTrue(selective(f,'verified').empty)
            f.at[1,'evidence_status']='FINANCIAL_AND_EXECUTION_SUPPORTED';f['scenario_base_multiple']=np.nan;f.at[1,'scenario_base_multiple']=2.1
            self.assertEqual(len(selective(f,'verified')),1)

    def test_no_mature_outcomes_means_no_accuracy_number(self):
        f=self.frame();f['model_role']='champion';f['six_month_mature']=False;f['twelve_month_mature']=False;f['independent_outcome_verified']=False
        q=prospective_report(f);self.assertTrue(all(x['brier'] is None for x in q['groups']))

    def test_worker_migration_commits_pair_and_keeps_original_registry(self):
        import v13_runner as runner
        import joblib
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);seed=root/'unused';(root/'official').mkdir();(root/'challenger').mkdir()
            f=self.frame();f['market_breadth63']=.7;f['median_turnover63']=1e7
            old=register(f,'a'*64,root/'registries/champion','2026-10-11T16:00:00Z');original=old.read_bytes()
            joblib.dump({},root/'active_heads.joblib');joblib.dump({},root/'challenger/research_heads.joblib')
            (root/'last_retrain_month.txt').write_text('2026-10')
            pd.DataFrame({'nse_current_symbol':['A','B']}).to_parquet(root/'universe.parquet')
            official=pd.DataFrame({'date':pd.to_datetime(['2026-10-09']*2),'isin':f['isin'],'symbol':f.symbol,'open':[100,100],'close':[100,100],'source_status':'VALID_POSITIVE_VOLUME_SESSION','source_sha256':'c'*64})
            official.to_parquet(root/'official/2026-10-09_official_market.parquet')
            phases=[]
            with ExitStack() as stack:
                stack.enter_context(patch.object(runner,'last_completed_session',return_value=('2026-10-09',{},[])))
                for name in ('universe','current_catalogs','collect_financials','historical_event_backfill','collect_histories'):
                    stack.enter_context(patch.object(runner.collection,name,side_effect=lambda *args,name=name,**kw:phases.append(name) or {}))
                stack.enter_context(patch.object(runner.catalyst_execution,'collect',return_value={}))
                stack.enter_context(patch.object(runner.action_audit,'backfill',return_value={}))
                stack.enter_context(patch('v13_completion_work.primary_history_backfill',return_value={}))
                stack.enter_context(patch('v13_valuation_evidence.collect',return_value={}))
                stack.enter_context(patch.object(runner,'current_frame',return_value=(f,{})))
                stack.enter_context(patch.object(runner,'scored',side_effect=lambda heads,frame,**kw:frame))
                result=runner.run(root,seed,'2026-10-12',local=True)
            self.assertEqual(old.read_bytes(),original);self.assertEqual(result['paired_registration']['rows_per_model'],2)
            self.assertLess(phases.index('collect_histories'),phases.index('collect_financials'))
            self.assertTrue((root/'challenger_latest_watchlist.json').exists())
            ledger=pd.read_parquet(root/'prospective_observations.parquet');self.assertEqual(len(ledger),6)
            self.assertEqual(ledger.cohort_id.notna().sum(),4);self.assertFalse(result['promotion']['promote'])
            self.assertNotIn('missing_paired_prospective_schema',result['promotion']['reasons'])

    def test_history_refresh_and_cold_expansion_share_budget(self):
        from v13_collection import collect_histories,sha
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'history/securities';folder.mkdir(parents=True)
            pd.DataFrame({'nse_current_symbol':['OLD','NEW'],'isin':['INE000000001','INE000000002']}).to_parquet(root/'universe.parquet')
            pd.DataFrame({'date':pd.to_datetime(['2026-10-09'])}).to_parquet(folder/(sha(b'OLD')+'.parquet'))
            with patch('v13_collection.Store.get',side_effect=ValueError('Unavailable')):
                out=collect_histories(root,'2026-10-12',2)
            self.assertEqual(out['reserved_refresh_slots'],1);self.assertEqual(out['cold_expansion_slots'],1)
            self.assertEqual({e['symbol'] for e in out['errors']},{'OLD','NEW'})

    def test_thinner_catalogue_preserves_pending_versions_inside_four_years(self):
        from v13_collection import merge_catalog
        old=pd.DataFrame([{'source_url':'https://nsearchives.nseindia.com/a.xml','available_at_utc':'2026-08-01T00:00:00Z','reporting_mode':'standalone','period_end':'2026-06-30'},
                          {'source_url':'https://nsearchives.nseindia.com/old.xml','available_at_utc':'2021-08-01T00:00:00Z','reporting_mode':'standalone','period_end':'2021-06-30'}])
        fresh=pd.DataFrame([{'source_url':'https://nsearchives.nseindia.com/b.xml','available_at_utc':'2026-08-02T00:00:00Z','reporting_mode':'standalone','period_end':'2026-06-30'}])
        merged,audit=merge_catalog(old,fresh,'2026-10-12')
        self.assertEqual(len(merged),2);self.assertEqual(audit['previous_versions_absent_from_response'],2)
        self.assertNotIn('https://nsearchives.nseindia.com/old.xml',set(merged.source_url))

if __name__=='__main__':unittest.main()
