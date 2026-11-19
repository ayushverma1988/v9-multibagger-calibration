import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from v13_market_sessions import SEED,reviewed_sessions,session_open,session_close,retain_originals,open_clocks
from v13_action_audit import expected_sessions,audit
from v13_lifecycle import register,adjudicate
from v13_history_integrity import validated_cached_entry,cache_digest

class SpecialSessions(unittest.TestCase):
    def test_empty_price_series_retains_timezone(self):
        clocks=open_clocks(pd.Series([],dtype='datetime64[ns]'))
        self.assertEqual(str(clocks.dtype),'datetime64[ns, Asia/Kolkata]')
        self.assertTrue(clocks.gt(pd.Timestamp('2026-11-06T00:00:00Z')).empty)

    def test_reviewed_original_includes_sunday_and_overrides_holiday_label(self):
        calendar={'year':2026,'holidays':['2026-11-08','2026-11-10']}
        self.assertEqual([str(x.date()) for x in expected_sessions('2026-11-06','2026-11-11',calendar)],['2026-11-06','2026-11-08','2026-11-09','2026-11-11'])
        self.assertEqual(session_open('2026-11-08'),pd.Timestamp('2026-11-08T08:50:00Z'))
        self.assertEqual(session_close('2026-11-08'),pd.Timestamp('2026-11-08T10:45:00Z'))

    def test_regular_days_and_pre_special_interval_unchanged(self):
        self.assertEqual(session_open('2026-11-06'),pd.Timestamp('2026-11-06T03:45:00Z'))
        self.assertEqual([str(d.date()) for d in expected_sessions('2026-11-02','2026-11-06',{'year':2026,'holidays':[]})],['2026-11-02','2026-11-03','2026-11-04','2026-11-05','2026-11-06'])
        self.assertIsNone(expected_sessions('2026-11-06','2027-01-04',{'year':2026,'holidays':[]}))

    def altered_seed(self,change):
        items=json.loads(SEED.read_text());change(items)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'reviews.json';p.write_text(json.dumps(items))
            with self.assertRaises(ValueError):reviewed_sessions(p)

    def test_corrupt_or_wrong_original_is_rejected(self):
        for field,value in [('source_sha256','0'*64),('bytes',1),('source_url','https://example.com/session.pdf')]:
            self.altered_seed(lambda xs:xs[0].update({field:value}))

    def test_times_and_date_must_match_exact_original_excerpts(self):
        for field,value in [('date','2026-11-09'),('open','09:15'),('close','15:30')]:
            self.altered_seed(lambda xs:xs[0].update({field:value}))

    def test_duplicate_or_inverted_or_future_review_clock_rejected(self):
        self.altered_seed(lambda xs:xs.append(xs[0]))
        self.altered_seed(lambda xs:xs[0].update({'reviewed_at_utc':'2099-01-01T00:00:00Z'}))
        self.altered_seed(lambda xs:xs[0].update({'retrieved_at_utc':'2026-11-06T23:00:00Z','reviewed_at_utc':'2026-11-06T15:00:00Z'}))

    def test_original_is_retained_without_certifying_prices(self):
        with tempfile.TemporaryDirectory() as t:
            items=retain_originals(t);self.assertEqual(len(items),1)
            raw=(Path(t)/(items[0]['source_sha256']+'.bin')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),items[0]['source_sha256'])
            self.assertTrue(items[0]['primary_price_snapshot_still_required'])
            self.assertIsNone(items[0]['intraday_publication_time'])

    def rows(self,dates):
        return pd.DataFrame({'date':pd.to_datetime(dates),'isin':'INE000000001','symbol':'A','open':100.,'close':100.,'source_status':'VALID_POSITIVE_VOLUME_SESSION','source_sha256':'a'*64})

    def reg(self,root,clock):
        frame=pd.DataFrame([{'date':pd.Timestamp('2026-11-06'),'isin':'INE000000001','symbol':'A','p6':.1,'p12':.2,'loss12':.3,'joint12':.1}])
        return register(frame,'f'*64,root/'registries/champion',clock)

    def test_missing_sunday_does_not_certify_monday_as_first_entry(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'action_audit').mkdir()
            (root/'action_audit/calendar.json').write_text(json.dumps({'year':2026,'holidays':['2026-11-08']}))
            reg=self.reg(root,'2026-11-06T16:00:00Z');o=self.rows(['2026-11-09'])
            proofs,_=audit(root,o,{'INE000000001':o},'2026-11-09')
            self.assertFalse(proofs[(reg.stem,'INE000000001')]['entry_verified'])

    def test_sunday_prices_verify_entry_only_and_do_not_change_frozen_registry(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'action_audit').mkdir()
            (root/'action_audit/calendar.json').write_text(json.dumps({'year':2026,'holidays':['2026-11-08']}))
            reg=self.reg(root,'2026-11-06T16:00:00Z');raw=reg.read_bytes()
            o=self.rows(['2026-11-08','2026-11-09'])
            proofs,_=audit(root,o,{'INE000000001':o},'2026-11-09');p=proofs[(reg.stem,'INE000000001')]
            self.assertTrue(p['entry_verified']);self.assertEqual(p['entry_date'],'2026-11-08')
            self.assertFalse(p['independent_outcome_verified']);self.assertFalse(p['confirmed_risk_event'])
            self.assertEqual(reg.read_bytes(),raw)

    def test_registration_after_special_open_uses_later_session(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);o=self.rows(['2026-11-08','2026-11-09'])
            reg=self.reg(root,'2026-11-08T09:00:00Z')
            self.assertEqual(adjudicate(reg,o,{},'2026-11-09')[0]['first_entry_date'],'2026-11-09')

    def test_registration_before_special_open_can_use_sunday(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);o=self.rows(['2026-11-08','2026-11-09'])
            reg=self.reg(root,'2026-11-08T08:30:00Z')
            self.assertEqual(adjudicate(reg,o,{},'2026-11-09')[0]['first_entry_date'],'2026-11-08')

    def test_cached_special_entry_cannot_predate_actual_open(self):
        value={'isin':'INE000000001','symbol':'A','entry_date':'2026-11-08','entry_open_INR':100.,'entry_official_sha256':'a'*64,'entry_provider_sha256':'b'*64,'entry_evidence_recorded_at_utc':'2026-11-08T04:00:00Z'}
        c={'evidence':value,'evidence_sha256':cache_digest(value)};o=self.rows(['2026-11-08'])
        self.assertIsNone(validated_cached_entry(c,o,'2026-11-08','A',value['isin']))
        value['entry_evidence_recorded_at_utc']='2026-11-08T11:00:00Z';c['evidence_sha256']=cache_digest(value)
        self.assertIsNotNone(validated_cached_entry(c,o,'2026-11-08','A',value['isin']))

    def test_last_completed_session_attempts_verified_weekend(self):
        import v13_runner as runner
        calls=[]
        def collect(date,out):
            calls.append(date)
            if date=='2026-11-08':return {'date':date}
            raise ValueError('not yet available')
        with patch.object(runner,'collect_official',side_effect=collect),patch.object(pd.Timestamp,'now',return_value=pd.Timestamp('2026-11-09T00:00:00Z')):
            asof,_,_=runner.last_completed_session(Path('/unused'),'2026-11-09')
        self.assertEqual(asof,'2026-11-08');self.assertEqual(calls,['2026-11-08'])

    def test_unfinished_special_session_not_requested(self):
        import v13_runner as runner
        calls=[]
        def collect(date,out):calls.append(date);return {'date':date}
        with patch.object(runner,'collect_official',side_effect=collect),patch.object(pd.Timestamp,'now',return_value=pd.Timestamp('2026-11-08T10:00:00Z')):
            asof,_,_=runner.last_completed_session(Path('/unused'),'2026-11-08')
        self.assertEqual(asof,'2026-11-06');self.assertEqual(calls,['2026-11-06'])

if __name__=='__main__':unittest.main()
