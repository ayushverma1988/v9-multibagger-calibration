import json,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd
from v13_financial_evidence import balancing_quarters,completion_queue
from v13_collection import financial_wide
from v12_5_expanded_facts import parse_extra
from v13_action_audit import reconstruct
from v13_research_evaluation import paired_evaluation_report
from v13_catalyst_execution import reviewed,join

class FinancialCompletion(unittest.TestCase):
    def wide(self):
        a={'symbol':'A','security_isin':'INE000000001','reporting_mode':'consolidated',
           'period_end':'2026-03-31','period':pd.Timestamp('2026-03-31'),'period_start':'2025-04-01',
           'period_kind':'annual','available':pd.Timestamp('2026-05-01',tz='UTC'),
           'available_at_utc':'2026-05-01T00:00:00+00:00','source_sha256':'a'*64,'source_url':'https://nsearchives.nseindia.com/a.xml',
           'pat':100.,'revenue':1000.,'eps_diluted_continuing':1.}
        y={**a,'period_end':'2025-12-31','period':pd.Timestamp('2025-12-31'),'period_kind':'ytd',
           'available':pd.Timestamp('2026-02-01',tz='UTC'),'available_at_utc':'2026-02-01T00:00:00+00:00',
           'source_sha256':'b'*64,'source_url':'https://nsearchives.nseindia.com/b.xml','pat':70.,'revenue':750.,'eps_diluted_continuing':.7}
        return pd.DataFrame([a,y])

    def test_balancing_quarter_uses_both_sources_later_clock_and_no_eps_subtraction(self):
        original=self.wide();out=balancing_quarters(original);q=out[out.period_kind.eq('quarter')].iloc[0]
        self.assertEqual(q.pat,30);self.assertEqual(q.revenue,250)
        self.assertTrue(pd.isna(q.eps_diluted_continuing));self.assertEqual(q.available,original.available.max())
        self.assertEqual(set(json.loads(q.input_source_sha256)),{'a'*64,'b'*64})
        self.assertTrue(out[(out.available<=pd.Timestamp('2026-04-01',tz='UTC'))&out.period_kind.eq('quarter')].empty)
        pd.testing.assert_frame_equal(out.iloc[:2][original.columns].reset_index(drop=True),original)

    def test_balancing_quarter_rejects_mixed_basis_wrong_start_or_non_nine_month(self):
        for key,value in [('reporting_mode','standalone'),('period_start','2025-07-01'),('period',pd.Timestamp('2025-09-30'))]:
            with self.subTest(key=key):
                d=self.wide();d.at[1,key]=value
                self.assertFalse(balancing_quarters(d).period_kind.eq('quarter').any())

    def test_explicit_quarter_is_not_replaced_by_balancing_math(self):
        d=self.wide();q={**d.iloc[0].to_dict(),'period_kind':'quarter','pat':99.}
        out=balancing_quarters(pd.concat([d,pd.DataFrame([q])],ignore_index=True))
        self.assertEqual(out[out.period_kind.eq('quarter')].pat.tolist(),[99.])

    def test_completion_budget_bundles_one_basis_and_rotates_issuers(self):
        rows=[]
        for s in ('A','B','C'):
            for period in ('2026-06-30','2026-03-31','2025-12-31','2025-09-30','2025-06-30','2025-03-31','2024-12-31'):
                for basis in ('standalone','consolidated'):
                    rows.append({'symbol':s,'period_end':period,'reporting_mode':basis,'available_at_utc':'2026-08-01T00:00:00Z','last_attempt':'0000','filed_period':'Quarterly'})
        q=completion_queue(pd.DataFrame(rows),pd.DataFrame(),'2026-10-12',7)
        self.assertEqual(q.symbol.nunique(),1);self.assertEqual(len(q),7)
        self.assertEqual(set(q.reporting_mode),{'consolidated'});self.assertIn('2024-12-31',set(q.period_end))

    def test_wide_retains_source_and_explicit_period_metadata(self):
        rows=[{'symbol':'A','security_isin':'INE000000001','reporting_mode':'consolidated','period_kind':'quarter',
               'period_end':'2026-06-30','period_start':'2026-04-01','available_at_utc':'2026-08-01T00:00:00Z',
               'source_sha256':'a'*64,'source_url':'https://nsearchives.nseindia.com/a.xml','metric':m,'value':v}
              for m,v in [('revenue',100),('pat',10)]]
        w=financial_wide(pd.DataFrame(rows));self.assertEqual(w.source_url.iloc[0],rows[0]['source_url'])
        self.assertEqual(w.period_start.iloc[0],'2026-04-01')

    def test_ytd_is_explicit_opt_in_and_wrong_identity_is_rejected(self):
        raw=b'''<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:f="urn:test">
        <context id="q"><period><startDate>2025-04-01</startDate><endDate>2025-12-31</endDate></period></context>
        <unit id="INR"><measure>iso4217:INR</measure></unit>
        <f:Symbol>A</f:Symbol><f:ISIN>INE000000001</f:ISIN><f:NatureOfReportStandaloneConsolidated>Consolidated</f:NatureOfReportStandaloneConsolidated>
        <f:RevenueFromOperations contextRef="q" unitRef="INR">100</f:RevenueFromOperations></xbrl>'''
        r={'symbol':'A','catalog_isin':'INE000000001','reporting_mode':'consolidated','period_end':'2025-12-31'}
        with self.assertRaises(ValueError):parse_extra(raw,r,'2026-10-12')
        facts=parse_extra(raw,r,'2026-10-12',include_ytd=True);self.assertEqual(facts[0]['period_kind'],'ytd')
        with self.assertRaises(ValueError):parse_extra(raw,{**r,'symbol':'OTHER'},'2026-10-12',include_ytd=True)

    def test_provider_split_scaled_entry_needs_reviewed_explanatory_factors(self):
        dates=pd.date_range('2026-10-12',periods=3,freq='B')
        o=pd.DataFrame({'date':dates,'open':[100,50,60],'close':[100,50,60],'source_status':'VALID_POSITIVE_VOLUME_SESSION'})
        p=pd.DataFrame({'date':dates,'open':[50,50,60],'close':[50,50,60],'adj_close':[50,50,60]})
        _,ratios=reconstruct(o,p,[{'kind':'split','ex_date':'2026-10-13','share_multiplier':2}])
        np.testing.assert_allclose(ratios,[1,1,1.2])
        with self.assertRaises(ValueError):reconstruct(o,p,[])
        p['open']=[50,25,30];p['close']=[50,25,30]
        with self.assertRaises(ValueError):reconstruct(o,p,[{'kind':'split','ex_date':'2026-10-13','share_multiplier':2}])

    def test_independent_split_proof_can_verify_early_entry_without_mature_claim(self):
        from v13_lifecycle import register,adjudicate
        frame=pd.DataFrame([{'date':pd.Timestamp('2026-10-09'),'isin':'INE000000001','symbol':'A','p6':.1,'p12':.2,'loss12':.3,'joint12':.1}])
        dates=pd.to_datetime(['2026-10-09','2026-10-12','2026-10-13'])
        provider=pd.DataFrame({'date':dates,'symbol':'A','isin':'INE000000001','open':[50,50,60],'close':[50,50,60],'adj_close':[50,50,60]})
        official=pd.DataFrame({'date':dates[1:],'symbol':'A','isin':'INE000000001','open':[100,60],'close':[100,60],'source_status':'VALID_POSITIVE_VOLUME_SESSION'})
        with tempfile.TemporaryDirectory() as t:
            path=register(frame,'a'*64,t,'2026-10-11T16:00:00Z')
            proof={(path.stem,'INE000000001'):{'entry_verified':True,'independent_outcome_verified':False,
              'verification_status':'INDEPENDENT_PRIMARY_PRICES_ACTIONS_AND_PROVIDER_FACTORS_MATCHED'}}
            row=adjudicate(path,official,{'INE000000001':provider},'2026-10-13',proof)[0]
            self.assertEqual(row['status'],'INDEPENDENT_PRIMARY_ENTRY_PROVIDER_ACTION_RECONCILED')
            self.assertFalse(row['six_month_mature']);self.assertFalse(row['independent_outcome_verified'])

    def test_paired_metrics_keep_unselected_unknown_names_and_exclude_old_unpaired(self):
        rows=[]
        for role,p in [('champion',.2),('challenger',.7)]:
            for isin,known in [('INE000000001',True),('INE000000002',False)]:
                rows.append({'cohort_id':'pair','isin':isin,'symbol':'A','model_role':role,'source_asof':'2025-01-01',
                  'registered_at_utc':'2025-01-01T16:00:00Z','first_entry_date':'2025-01-02','p6':p,'p12':p,
                  'selected_verified':False,'selected_discovery':False,'financial_tier':'discovery','market_regime':'MIXED','liquidity_tier':'LOW',
                  'six_month_mature':True,'twelve_month_mature':True,'independent_outcome_verified':known,
                  'verified_hit2_6':1 if known else None,'verified_hit2_12':1 if known else None})
        rows.append({**rows[0],'cohort_id':None,'isin':'INE000000003'})
        r=paired_evaluation_report(pd.DataFrame(rows));self.assertEqual(r['cohorts'][0]['registered_stock_pairs'],2)
        for z in r['comparisons']:
            self.assertEqual(z['verified_pairs'],1);self.assertEqual(z['unknown_or_immature_pairs'],1)
            self.assertGreater(z['paired_brier_gain'],0);self.assertIsNone(z['challenger']['selected_precision'])
        rows[2]['registered_at_utc']='2025-01-02T16:00:00Z'
        self.assertFalse(paired_evaluation_report(pd.DataFrame(rows))['cohorts'][0]['same_clock_same_names'])

    def test_policy_and_input_contracts_cannot_be_pooled_for_promotion(self):
        from v13_lifecycle import paired_observations
        base={'cohort_id':'pair','source_asof':'2025-01-01','isin':'INE000000001','registry_sha256':'r',
          'registered_at_utc':'2025-01-01T16:00:00Z','p12':.2,'selected_verified':False,'verified_hit2_12':1,
          'twelve_month_mature':True,'first_entry_date':'2025-01-02','status':'TRACKING','independent_outcome_verified':True,
          'comparison_contract':'OLD_POLICY'}
        a={**base,'model_role':'champion','model_sha256':'a'*64};b={**base,'model_role':'challenger','model_sha256':'b'*64}
        d=pd.DataFrame([a,b]);self.assertEqual(len(paired_observations(d,'b'*64,comparison_contract='OLD_POLICY')),1)
        self.assertTrue(paired_observations(d,'b'*64,comparison_contract='NEW_POLICY').empty)

    def test_review_clock_is_not_backdated_to_publication_and_capacity_needs_units(self):
        text='Symbol: A. has commissioned 5 MW. formal certificate is expected shortly.'
        import hashlib
        e={'isin':'INE000000001','symbol':'A','source_sha256':'a'*64,'attachment_url':'https://nsearchives.nseindia.com/a.pdf',
           'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'issuer_identity_read':True,
           'first_retrieved_utc':'2026-10-12T16:00:00Z','first_reviewed_utc':'2026-10-13T01:00:00Z','available_at_utc':'2026-10-12T15:00:00+00:00'}
        r={k:e[k] for k in ('isin','symbol','source_sha256','attachment_url')}
        r.update(published_at_utc='2026-10-12T15:00:00Z',capacity_quantity=5,field_excerpts={'capacity_quantity':'has commissioned 5 MW'})
        with self.assertRaises(ValueError):reviewed(r,e,text)
        r.update(capacity_unit='MW');r['field_excerpts']['capacity_unit']='has commissioned 5 MW'
        self.assertEqual(pd.Timestamp(reviewed(r,e,text)['available_at_utc']),pd.Timestamp(e['first_reviewed_utc']))

    def test_planned_funding_cannot_pass_with_an_exact_source_quote(self):
        import hashlib
        text='A plans to arrange funding after regulatory approval.'
        e={'isin':'INE000000001','symbol':'A','source_sha256':'a'*64,'attachment_url':'https://nsearchives.nseindia.com/a.pdf',
           'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'issuer_identity_read':True,'first_retrieved_utc':'2026-10-13T01:00:00Z'}
        r={k:e[k] for k in ('isin','symbol','source_sha256','attachment_url')}
        r.update(funding_sufficient=True,field_excerpts={'funding_sufficient':text})
        with self.assertRaises(ValueError):reviewed(r,e,text)

    def test_scenario_source_hash_alone_cannot_establish_forward_earnings(self):
        from v13_financial_evidence import apply_scenarios
        import hashlib
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'evidence').mkdir();raw=b'Company A: historical reported numbers only.'
            (root/'source.bin').write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
            rec={'isin':'INE000000001','available_at_utc':'2026-10-12T00:00:00Z','first_reviewed_utc':'2026-10-13T01:00:00Z',
              'input_source_sha256':{k:digest for k in ('forward_PAT','forward_diluted_shares','sector_PE')},
              'forward_PAT_INR':200,'forward_diluted_shares':10,'sector_PE':20}
            (root/'evidence/reviewed_earnings_scenarios.json').write_text(json.dumps([rec]))
            f=pd.DataFrame([{'isin':'INE000000001','date':pd.Timestamp('2026-10-13'),'close':100,'ttm_reported_PAT_INR':100,'latest_diluted_shares':10,'earnings_scenario_status':'UNKNOWN'}])
            out=apply_scenarios(f,root);self.assertEqual(out.earnings_scenario_status.iloc[0],'UNKNOWN')
            self.assertEqual(json.loads((root/'evidence/earnings_scenario_audit.json').read_text())[0]['status'],'UNKNOWN_UNBOUND_SCENARIO_INPUTS')

if __name__=='__main__':unittest.main()
