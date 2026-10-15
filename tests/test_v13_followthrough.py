import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from v13_financial_evidence import owner_completion_queue
from v13_action_sources import notice_queue
from v13_pair_proof import entry_match,baseline_match,same_metadata
from v13_completion_work import completion_plan,primary_history_backfill

class Followthrough(unittest.TestCase):
    def test_commercial_production_commencing_today_is_distinct_from_future_intent(self):
        from v13_catalyst_execution import reviewed
        import hashlib
        for text,accept in [('commercial production and dispatch of the product commencing from today.',True),
                            ('commercial production is expected, commencing from today.',False),
                            ('commercial production commencing from next month.',False)]:
            e={'isin':'INE000000001','symbol':'A','source_sha256':'a'*64,'attachment_url':'https://nsearchives.nseindia.com/a.pdf',
              'issuer_identity_read':True,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'first_retrieved_utc':'2026-10-15T06:30:00Z'}
            r={**{k:e[k] for k in ('isin','symbol','source_sha256','attachment_url')},'commercial_operation_verified':True,'field_excerpts':{'commercial_operation_verified':text}}
            if accept:self.assertTrue(reviewed(r,e,text)['commercial_operation_verified'])
            else:
                with self.assertRaises(ValueError):reviewed(r,e,text)

    def owner_inputs(self):
        rows=[]
        for period in ('2026-06-30','2026-03-31','2025-12-31','2025-09-30','2025-06-30'):
            rows.append({'symbol':'A','reporting_mode':'consolidated','period_end':period,'period':pd.Timestamp(period),
              'available':pd.Timestamp('2026-08-01',tz='UTC'),'available_at_utc':'2026-08-01T00:00:00Z',
              'period_kind':'quarter','revenue':100,'pat':10,'pat_owners':np.nan})
        wide=pd.DataFrame(rows)
        q=wide[['symbol','reporting_mode','period_end','available_at_utc']].copy();q['last_attempt']='0000'
        return q,wide

    def test_owner_queue_does_not_stop_at_basic_revenue_and_total_pat(self):
        q,w=self.owner_inputs();out=owner_completion_queue(q,w,'2026-10-14',2)
        self.assertEqual(out.period_end.tolist(),['2026-06-30','2025-06-30'])
        self.assertTrue(w.pat_owners.isna().all())

    def test_observed_zero_owner_is_not_missing_and_standalone_not_substituted(self):
        q,w=self.owner_inputs();w['pat_owners']=0
        self.assertTrue(owner_completion_queue(q,w,'2026-10-14',5).empty)
        q['reporting_mode']='standalone';w['reporting_mode']='standalone';w['pat_owners']=np.nan
        self.assertTrue(owner_completion_queue(q,w,'2026-10-14',5).empty)

    def test_owner_queue_excludes_future_publications_and_four_year_boundary(self):
        q,w=self.owner_inputs();q['available_at_utc']='2026-10-15T10:00:00Z'
        self.assertTrue(owner_completion_queue(q,w,'2026-10-14',5).empty)
        self.assertTrue(owner_completion_queue(q,w,'2026-10-14',0).empty)

    def test_failed_action_notice_rotates_behind_unattempted_original(self):
        e=pd.DataFrame([{'symbol':'A','attachment_url':'a','available_at_utc':'2026-10-14T16:00:00Z'},
                        {'symbol':'B','attachment_url':'b','available_at_utc':'2026-10-13T16:00:00Z'}])
        self.assertEqual(notice_queue(e,{},1).attachment_url.tolist(),['a'])
        self.assertEqual(notice_queue(e,{'a':'2026-10-14'},1).attachment_url.tolist(),['b'])

    def pair(self):
        a=pd.DataFrame([{'entry_verified':True,'first_entry_date':'2026-10-12','entry_open_INR':100,
          'entry_official_sha256':'a'*64,'entry_provider_sha256':'b'*64,'baseline_p12':.1,
          'comparison_contract':'frozen','information_cutoff_utc':'2026-10-09T16:30:00Z'}])
        return a,a.copy()

    def test_paired_entry_needs_same_price_and_both_primary_provider_hashes(self):
        a,b=self.pair();self.assertTrue(entry_match(a,b).iloc[0])
        for key,value in [('entry_open_INR',110),('entry_provider_sha256','c'*64),('entry_official_sha256',None),('entry_verified',False),('first_entry_date',None)]:
            q=b.copy();q.at[0,key]=value
            with self.subTest(key=key):self.assertFalse(entry_match(a,q).iloc[0])
        self.assertFalse(entry_match(a,b.drop(columns='entry_open_INR')).iloc[0])

    def test_paired_baselines_and_information_contracts_cannot_be_spliced(self):
        a,b=self.pair();self.assertTrue(baseline_match(a,b,12).iloc[0]);self.assertTrue(same_metadata(a,b).iloc[0])
        b.at[0,'baseline_p12']=.2;self.assertFalse(baseline_match(a,b,12).iloc[0])
        b.at[0,'information_cutoff_utc']='2026-10-09T10:00:00Z';self.assertFalse(same_metadata(a,b).iloc[0])

    def test_completion_plan_separates_queue_pass_from_evidence_and_market_time(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'evidence').mkdir()
            (p/'evidence/financial_completeness.json').write_text(json.dumps({'complete_rows':1,'issuer_backlog':[]}))
            obs=pd.DataFrame([{'entry_verified':True,'first_entry_date':'2026-10-12'}])
            result=completion_plan(p,{'asof':'2026-10-14','current_verified_market_rows':2,
              'steps':{'financials':{'remaining':1601}}},obs)
            self.assertEqual(result['financial']['optimistic_minimum_runs_for_one_catalogue_pass'],3)
            self.assertFalse(result['financial']['is_evidence_completion_or_calendar_deadline'])
            self.assertEqual(result['future_evaluation']['earliest_possible_full_horizon_labels'],{'6m':'2027-04-12','12m':'2027-10-12'})
            self.assertFalse(result['promotion_authorised']);self.assertFalse(result['frozen_records_modified'])

    def test_primary_history_collects_whole_public_sessions_and_resumes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'action_audit').mkdir()
            (root/'action_audit/calendar.json').write_text(json.dumps({'year':2026,'holidays':[]}))
            calls=[]
            def collect(date,folder):
                calls.append(date);(folder/(date+'_official_market.parquet')).write_bytes(b'fixture')
                return {'date':date,'source_sha256':'a'*64}
            with patch('v12_3_official_monitor.collect',side_effect=collect):
                a=primary_history_backfill(root,'2026-10-14',budget=2,goal_sessions=3)
                b=primary_history_backfill(root,'2026-10-14',budget=2,goal_sessions=3)
            self.assertEqual(a['new_session_files'],2);self.assertEqual(b['new_session_files'],1)
            self.assertEqual(len(calls),3);self.assertFalse(b['scoring_coverage_or_outcome_certification_added'])
            self.assertGreaterEqual(min(calls),'2022-10-14')

    def test_official_rate_limit_stops_backfill_without_certifying_missing_dates(self):
        with tempfile.TemporaryDirectory() as t:
            with patch('v12_3_official_monitor.collect',side_effect=ValueError('429 Too many requests')) as c:
                r=primary_history_backfill(Path(t),'2026-10-14',budget=8,goal_sessions=8)
            self.assertEqual(c.call_count,1);self.assertEqual(r['available_session_files'],0)
            self.assertEqual(r['remaining_session_files'],8)

if __name__=='__main__':unittest.main()
