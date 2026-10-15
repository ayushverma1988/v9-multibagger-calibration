import hashlib,json,tempfile,unittest
from pathlib import Path
import joblib,numpy as np,pandas as pd
from unittest.mock import patch
from v13_symbol_identity import parse_changes,parse_mapped,rename_path
from v13_research_selection import screen,export
from v13_prediction_replay import snapshot
from v13_upgrade_diagnostics import scenario_horizon,execution_diagnostics
from v13_statement_eps import validate,enrich
import v13_model as model

class SourceResearchUpgrade(unittest.TestCase):
    def test_financial_collector_reads_master_isin_column_at_real_entry_point(self):
        from v13_collection import collect_financials
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            pd.DataFrame({'nse_current_symbol':['A'],'isin':['INE000000001']}).to_parquet(root/'universe.parquet',index=False)
            columns=['symbol','reporting_mode','source_url','available_at_utc','period_end','filed_period']
            pd.DataFrame(columns=columns).to_parquet(root/'catalog.parquet',index=False)
            with patch('v13_symbol_identity.collect',return_value=[]):result=collect_financials(root,'2026-10-15',0)
            self.assertEqual(result['requested'],0);self.assertEqual(result['official_rename_edges'],0)

    def rename(self):
        return {'old':'OLD','new':'NEW','effective_date':'2026-01-01','source_sha256':'a'*64,'first_seen_utc':'2026-10-15T13:00:00Z'}

    def xml(self):
        return b'''<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:f="urn:test">
        <context id="q"><period><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></period></context>
        <unit id="INR"><measure>iso4217:INR</measure></unit>
        <f:Symbol>OLD</f:Symbol><f:ISIN>INE000000001</f:ISIN><f:NatureOfReportStandaloneConsolidated>Consolidated</f:NatureOfReportStandaloneConsolidated>
        <f:RevenueFromOperations contextRef="q" unitRef="INR">100</f:RevenueFromOperations></xbrl>'''

    def test_rename_requires_original_isin_and_unique_effective_path(self):
        r={'symbol':'NEW','catalog_isin':'INE000000001','reporting_mode':'consolidated','period_end':'2026-06-30','available_at_utc':'2026-08-01T00:00:00Z'}
        facts=parse_mapped(self.xml(),r,'2026-10-15','INE000000001',[self.rename()])
        self.assertEqual(facts[0]['available_at_utc'],'2026-10-15T13:00:00+00:00')
        self.assertEqual(facts[0]['original_document_symbol'],'OLD')
        for raw,isin,edges,asof in [(self.xml(),'INE000000002',[self.rename()],'2026-10-15'),
            (self.xml(),'INE000000001',[],'2026-10-15'),
            (self.xml().replace(b'<f:ISIN>INE000000001</f:ISIN>',b''),'INE000000001',[self.rename()],'2026-10-15'),
            (self.xml(),'INE000000001',[self.rename()],'2025-12-31')]:
            with self.assertRaises(ValueError):parse_mapped(raw,r,asof,isin,edges)
        alternate=[self.rename(),{**self.rename(),'new':'MID'},{**self.rename(),'old':'MID'}]
        with self.assertRaises(ValueError):rename_path('OLD','NEW',alternate,'2026-10-15')

    def test_official_reference_keeps_first_seen_and_skips_unchanged_notices(self):
        raw=b'NAME OF THE COMPANY,OLD SYMBOL,NEW SYMBOL,EFFECTIVE DATE\nA,OLD,NEW,01-JAN-2026\nB,SAME,SAME,01-JAN-2026\n'
        p={'sha256':'a'*64,'first_retrieved_utc':'2026-10-15T13:00:00Z'}
        rows=parse_changes(raw,p);self.assertEqual(len(rows),1)
        fresh=parse_changes(raw,{**p,'sha256':'b'*64,'first_retrieved_utc':'2026-10-16T13:00:00Z'},rows)
        self.assertEqual(rows,fresh)

    def test_later_identity_review_preserves_an_already_valid_earlier_fact_clock(self):
        from v13_collection import merge_financial_versions
        r={'source_sha256':'a'*64,'period_kind':'quarter','period_end':'2026-06-30','security_isin':'INE000000001',
          'reporting_mode':'consolidated','available_at_utc':'2026-08-01T00:00:00Z','pat':10.}
        old=pd.DataFrame([r]);fresh=pd.DataFrame([{**r,'available_at_utc':'2026-10-15T13:00:00Z','symbol':'NEW'}])
        d=merge_financial_versions(old,fresh);self.assertEqual(len(d),2)
        self.assertEqual(d[d.available_at_utc.lt('2026-09-01')].pat.tolist(),[10.])
        self.assertEqual(len(merge_financial_versions(old,pd.DataFrame([{**r,'pat_owners':8.}]))),1)

    def frame(self):
        r={'date':pd.Timestamp('2026-10-15'),'isin':'INE000000001','symbol':'A','close':100.,'distance_ma50':.1,
          'log_ret63':.1,'quarter_sales_yoy':.1,'quarter_pat_yoy':.1,'quarter_pat_margin':.1,
          'annual_CFO_PAT':1.,'dated_debt_equity':.2,'annual_EBIT_proxy_interest_coverage':2.,'four_profitable_quarters':1,
          'financial_quality_pass':True,'rsi14':70.0001,'p6':.1,'p12':.2,'loss12':.4,'joint12':.15,
          'evidence_status':'MISSING_FINANCIAL_OR_EXECUTION','scenario_net_multiple':np.nan}
        return pd.DataFrame([r,{**r,'isin':'INE000000002','symbol':'B','rsi14':70.},
          {**r,'isin':'INE000000003','symbol':'C','annual_CFO_PAT':.1}])

    def test_research_rsi_boundary_failures_and_zero_strict_remain_visible(self):
        f=self.frame();original=f.copy(deep=True);d,a=screen(f)
        self.assertEqual(d.loc[d.research_selected,'symbol'].tolist(),['A'])
        self.assertEqual(a['gate_counts']['strict_net_2x'],0)
        pd.testing.assert_frame_equal(f,original)
        f['rsi14']=np.nan;self.assertEqual(screen(f)[1]['gate_counts']['selected'],0)

    def test_dma_band_is_inclusive_and_optional_with_unknown_separate(self):
        r=self.frame().iloc[0].to_dict()
        f=pd.DataFrame([{**r,'isin':f'INE{i:09d}','distance_ma200':v} for i,v in enumerate([-.1,.1,.10001,-.10001,np.nan])])
        d,a=screen(f)
        self.assertEqual(d.research_selected.sum(),5)
        self.assertEqual(d.research_within_200dma_band.tolist(),[True,True,False,False,False])
        self.assertEqual(d.research_dma200_status.iloc[-1],'UNKNOWN_200DMA')
        self.assertEqual(set(d.sort_values('research_rank').head(2).index),{0,1})
        self.assertAlmostEqual(d.research_ranking_score.iloc[0]-d.research_ranking_score.iloc[2],.05)

    def test_missing_evidence_ranks_without_rewriting_probabilities_or_accepting_observed_failure(self):
        r=self.frame().iloc[0].to_dict()
        rows=[r,{**r,'isin':'INE000000004','quarter_sales_yoy':np.nan,'financial_quality_pass':False},
          {**r,'isin':'INE000000005','quarter_sales_yoy':-.01},
          {**r,'isin':'INE000000006','quarter_pat_yoy':0}]
        f=pd.DataFrame(rows);d,a=screen(f)
        self.assertEqual(d.research_selected.tolist(),[True,True,False,False])
        self.assertEqual(d.research_financial_tier.iloc[1],'INCOMPLETE_FINANCIAL_INPUTS')
        self.assertLess(d.research_ranking_score.iloc[1],d.research_ranking_score.iloc[0])
        pd.testing.assert_frame_equal(d[['p6','p12','joint12','loss12']],f[['p6','p12','joint12','loss12']])
        self.assertEqual(a['gate_counts']['strict_net_2x'],0)
        self.assertFalse(a['ranking_score_is_probability'])

    def test_dma_preference_does_not_override_material_probability_advantage(self):
        r=self.frame().iloc[0].to_dict()
        f=pd.DataFrame([r,{**r,'isin':'INE000000004','joint12':.18,'distance_ma200':.3}])
        f.at[0,'distance_ma200']=0
        d,_=screen(f)
        self.assertEqual(d.sort_values('research_rank')['isin'].iloc[0],'INE000000004')

    def test_source_scenario_bonus_requires_status_and_target_not_just_hypothetical_multiple(self):
        r=self.frame().iloc[0].to_dict()
        f=pd.DataFrame([{**r,'isin':f'INE{i:09d}','scenario_net_multiple':multiple,'earnings_scenario_status':status}
          for i,(multiple,status) in enumerate([(3,'UNKNOWN'),(1.9,'SCENARIO_NOT_PROBABILITY'),(2,'SCENARIO_NOT_PROBABILITY')])])
        d,_=screen(f)
        self.assertEqual(d.research_source_supported_net_2x.tolist(),[False,False,True])
        self.assertTrue(d.research_selected.all())

    def test_exact_real_head_replay_and_modified_predictions_reject(self):
        heads={k:{'method':'shrunk_past_event_rate','constant':.2,'calibration':{'slope':1.,'intercept':0.}} for k in model.TARGETS}
        f=self.frame();scored=f.copy()
        for k,v in model.probabilities(heads,f).items():scored[k]=v
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'heads.joblib';joblib.dump(heads,p)
            result=snapshot(f,scored,p,root,'champion');self.assertTrue(result['replay_verified'])
            scored.at[0,'p12']+=.0001
            with self.assertRaises(ValueError):snapshot(f,scored,p,root,'champion')

    def test_zero_research_selection_still_keeps_all_names_clock_and_replay_link(self):
        f=self.frame();f['rsi14']=np.nan;f['information_cutoff_utc']='2026-10-15T13:00:00Z'
        with tempfile.TemporaryDirectory() as t:
            receipt={'replay_verified':True,'rows':len(f),'input_parquet_sha256':'a'*64,'model_sha256':'b'*64}
            export(f,t,receipt);obj=json.loads((Path(t)/'research_selection/latest.json').read_text())
            self.assertEqual(len(obj['all_names_selection']),len(f));self.assertEqual(obj['records'],[])
            self.assertEqual(obj['prediction_replay'],receipt);self.assertTrue(obj['registered_at_utc'])
            self.assertEqual(obj['information_cutoffs_utc'],['2026-10-15T13:00:00Z'])

    def scenario(self):
        return {'horizon_months':12,'forecast_period_start':'2026-04-01','forecast_period_end':'2027-03-31',
          'valuation_target_date':'2027-10-15','forward_share_basis':'FULLY_DILUTED_INCLUDING_REVIEWED_ACTIONS',
          'input_source_excerpts':{'forward_PAT':'Annual owner forecast from 2026-04-01 to 2027-03-31'}}

    def test_scenario_annual_period_horizon_and_dilution_basis_required(self):
        r=self.scenario();self.assertTrue(scenario_horizon(r,'2026-10-15'))
        for k,v in [('horizon_months',6),('valuation_target_date','2028-10-15'),('forecast_period_start','2027-01-01'),
          ('forward_share_basis','ISSUED_SHARES'),('input_source_excerpts',{'forward_PAT':'Expected profit next year'})]:
            self.assertFalse(scenario_horizon({**r,k:v},'2026-10-15'))

    def test_missing_execution_is_distinct_from_adverse_update(self):
        b={'checks':{'source_funding_sufficient':False,'required_operating_approvals':False,'positive_margin_with_profit_basis':True,
          'not_cancelled_or_certificate_pending':True},'observed_values':{'funding_sufficient':False,'expected_margin_pct':10,'margin_basis':'EBITDA'}}
        d=execution_diagnostics(b)
        self.assertIn('source_funding_sufficient',d['research_failed_checks'])
        self.assertIn('required_operating_approvals',d['research_missing_checks'])
        self.assertFalse(d['margin_is_owner_net_profit']);self.assertFalse(d['order_value_is_recognised_revenue'])

    def test_statement_original_hash_rejects_and_review_never_backdates(self):
        with self.assertRaises(ValueError):validate({'statement':{'sha256':'a'*64}},b'wrong',b'',b'')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'statement_eps';p.mkdir()
            r={'isin':'INE000000001','available_at_utc':'2026-10-15T13:00:00Z','current_period_end':'2026-06-30',
              'reported_diluted_EPS_yoy_same_document':-.35,'statement_sha256':'a'*64}
            (p/'accepted.json').write_text(json.dumps([r]));f=self.frame().iloc[:1].copy()
            f['information_cutoff_utc']='2026-10-15T12:00:00Z'
            self.assertNotIn('statement_eps_source_sha256',enrich(f,t))
            f['information_cutoff_utc']='2026-10-15T14:00:00Z'
            self.assertEqual(enrich(f,t).reported_diluted_EPS_yoy_same_document.iloc[0],-.35)

    def test_unknown_legacy_cutoff_cannot_be_filled_later(self):
        from v13_frozen_integrity import FrozenGuard
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);f=self.frame().iloc[:1].copy();f['registry_sha256']='a'*64
            f['status']='TRACKING';f['information_cutoff_utc']=None
            path=root/'prospective_observations.parquet';f.to_parquet(path,index=False)
            guard=FrozenGuard(root);altered=f.copy();altered['information_cutoff_utc']='2026-10-15T13:00:00Z'
            with self.assertRaises(ValueError):guard.check(altered)
            self.assertTrue(pd.read_parquet(path).information_cutoff_utc.isna().all())

    def test_same_day_stale_volume_or_prices_retry_without_filling(self):
        from v13_collection import history_needs_refresh
        h=pd.DataFrame([{'date':pd.Timestamp('2026-10-15'),'isin':'INE000000001','symbol':'A',
          'open':100.,'high':102.,'low':99.,'close':101.,'volume':10000}])
        official=h.copy();official['source_status']='VALID_POSITIVE_VOLUME_SESSION'
        self.assertFalse(history_needs_refresh(h,official,'2026-10-15','A','INE000000001'))
        for key,val in [('volume',9999),('close',100.5),('date',pd.Timestamp('2026-10-14')),('symbol','OTHER')]:
            changed=h.copy();changed.at[0,key]=val
            self.assertTrue(history_needs_refresh(changed,official,'2026-10-15','A','INE000000001'))
        pd.testing.assert_frame_equal(h[official.columns[:-1]],official.drop(columns='source_status'))

    def test_rejected_provider_outlier_cannot_change_valid_names_context(self):
        from v13_runner import current_frame
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'history/securities';folder.mkdir(parents=True);(root/'official').mkdir()
            records=[]
            for i,(symbol,ret) in enumerate([('A',.1),('B',.2),('BAD',-100.)]):
                r={'date':pd.Timestamp('2026-10-15'),'symbol':symbol,'isin':f'INE00000000{i+1}',
                  'open':100.,'high':102.,'low':99.,'close':101.,'volume':1000,'adj_close':101.,
                  'log_ret63':ret,'log_ret20':ret,'momentum126':ret,'momentum252_skip21':ret,
                  'volume_confirmation':1.,'log_liquidity63':10.,'feature_eligible':True}
                r['source_sha256']='b'*64
                pd.DataFrame([r]).to_parquet(folder/(symbol+'.parquet'),index=False);records.append(r)
            official=pd.DataFrame(records);official.loc[official.symbol.eq('BAD'),'volume']=900
            official['series']='EQ';official['source_status']='VALID_POSITIVE_VOLUME_SESSION';official['source_sha256']='a'*64
            official.to_parquet(root/'official/2026-10-15_official_market.parquet',index=False)
            with patch('v13_runner.features.daily_features',side_effect=lambda bars,asof:bars):d,_=current_frame(root,'2026-10-15')
            self.assertEqual(set(d.symbol),{'A','B'});self.assertTrue(d.market_breadth63.eq(1).all())
            self.assertEqual(d.sort_values('symbol').relative_strength63.tolist(),[.5,1.])
            funnel=json.loads((root/'evidence/market_coverage_funnel.json').read_text());self.assertEqual(funnel['volume_mismatch'],1)

if __name__=='__main__':unittest.main()
