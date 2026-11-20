import hashlib,io,json,tempfile,unittest,zipfile
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from v13_breadth_fallback import extend,load,validate_receipt,single_source_view


class BreadthFallbackTests(unittest.TestCase):
    @staticmethod
    def primary_bytes(year=2026):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:z.writestr('official.csv','date,symbol\n')
        calendar=json.dumps({'CM':[{'tradingDate':'26-Jan-'+str(year)}]}).encode()
        return buf.getvalue(),calendar

    def add_primary(self,folder,receipt,asof,at):
        nse,calendar=self.primary_bytes(int(asof[:4]));folder=Path(folder)
        (folder/'nse_source_session.zip').write_bytes(nse)
        (folder/'nse_holiday_calendar.json').write_bytes(calendar)
        receipt['nse_original']={'source_url':'https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_'+asof.replace('-','')+'_F_0000.csv.zip',
          'first_retrieved_utc':at,'sha256':hashlib.sha256(nse).hexdigest(),'bytes':len(nse)}
        receipt['nse_calendar']={'source_url':'https://www.nseindia.com/api/holiday-master?type=trading',
          'first_retrieved_utc':at,'sha256':hashlib.sha256(calendar).hexdigest(),'bytes':len(calendar)}
        actions=b'[]';(folder/'nse_corporate_actions.json').write_bytes(actions)
        day=pd.Timestamp(asof).strftime('%d-%m-%Y')
        receipt['nse_corporate_actions']={'source_url':'https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date='+day+'&to_date='+day,
          'first_retrieved_utc':at,'sha256':hashlib.sha256(actions).hexdigest(),'bytes':len(actions),
          'rows':0,'no_action_interval_certified':False}
        receipt.setdefault('adjusted_through_asof',asof)
        receipt.setdefault('source_day_in_public_file',True)
        return receipt

    def setUp(self):
        self.dates=pd.bdate_range(end='2026-11-17',periods=160)
        prices=np.linspace(100,180,len(self.dates))
        self.old=pd.DataFrame({'date':self.dates,'symbol':'ABC','isin':'INE000000001',
          'open':prices,'high':prices+2,'low':prices-2,'close':prices,'volume':1000,
          'adj_close':prices,'source_sha256':'a'*64})
        self.extra=pd.DataFrame([{'date':pd.Timestamp('2026-11-18'),'symbol':'ABC','isin':'INE000000001',
          'series':'EQ','open':181.,'high':183.,'low':179.,'close':181.,'volume':1200,
          'adj_close':181.,'adj_factor_cumulative':1.}])
        self.alt=pd.concat([self.old.assign(series='EQ',adj_factor_cumulative=1.),self.extra],ignore_index=True)
        self.official=self.extra.drop(columns=['adj_close','adj_factor_cumulative']).assign(
          source_status='VALID_POSITIVE_VOLUME_SESSION',source_sha256='b'*64)
        self.sessions=[pd.Timestamp('2026-11-18')]

    def check(self,old=None,alt=None,official=None,sessions=None):
        return extend(self.old if old is None else old,self.alt if alt is None else alt,
          self.official if official is None else official,self.sessions if sessions is None else sessions,
          '2026-11-18','ABC','INE000000001','c'*64)

    def test_score_view_preserves_original_and_does_not_certify_outcomes(self):
        copy=self.old.copy(deep=True);view,proof=self.check()
        pd.testing.assert_frame_equal(self.old,copy)
        self.assertEqual(len(view),len(copy)+1)
        self.assertEqual(proof['new_sessions'],1)
        self.assertFalse(proof['action_interval_certified'])
        self.assertFalse(proof['entry_or_outcome_certified'])
        self.assertEqual(view.iloc[-1].source_sha256,'c'*64)

    def test_single_source_view_allows_short_or_absent_reference_but_labels_it(self):
        original=self.alt.copy(deep=True)
        view,proof=single_source_view(self.alt,self.official,self.sessions,'2026-11-18',
          'ABC','INE000000001','c'*64,reference=self.old.tail(10))
        pd.testing.assert_frame_equal(self.alt,original)
        self.assertEqual(proof['overlap_sessions'],10)
        self.assertEqual(proof['reference_status'],'OVERLAP_MATCHED')
        self.assertFalse(proof['action_interval_certified'])
        self.assertFalse(proof['entry_or_outcome_certified'])
        self.assertEqual(len(view),len(self.alt))
        _,proof=single_source_view(self.alt,self.official,self.sessions,'2026-11-18',
          'ABC','INE000000001','c'*64)
        self.assertEqual(proof['reference_status'],'NO_TIMELY_COMPARABLE_PROVIDER')

    def test_single_source_view_never_waives_known_conflicts(self):
        bad=self.old.tail(10).copy();bad.loc[bad.index[-1],'adj_close']*=.9
        with self.assertRaisesRegex(ValueError,'conflict'):
            single_source_view(self.alt,self.official,self.sessions,'2026-11-18',
              'ABC','INE000000001','c'*64,reference=bad)
        with self.assertRaisesRegex(ValueError,'corporate action'):
            single_source_view(self.alt,self.official,self.sessions,'2026-11-18',
              'ABC','INE000000001','c'*64,('ABC',))
        bad=self.alt.copy();bad.loc[bad.index[-1],'isin']='INE999999999'
        with self.assertRaisesRegex(ValueError,'identity'):
            single_source_view(bad,self.official,self.sessions,'2026-11-18','ABC','INE000000001','c'*64)
        bad=self.official.copy();bad.loc[bad.index[-1],'volume']+=1
        with self.assertRaisesRegex(ValueError,'volume conflict'):
            single_source_view(self.alt,bad,self.sessions,'2026-11-18','ABC','INE000000001','c'*64)
        bad=self.alt.copy();bad.loc[bad.index[0],'adj_close']*=.9
        with self.assertRaisesRegex(ValueError,'factors disagree'):
            single_source_view(bad,self.official,self.sessions,'2026-11-18','ABC','INE000000001','c'*64)

    def test_single_source_lag_is_only_one_session(self):
        view,proof=single_source_view(self.alt.iloc[:-1],self.official,self.sessions,'2026-11-18',
          'ABC','INE000000001','c'*64)
        self.assertTrue(proof['lagged_adjusted_source_extended_with_primary_raw'])
        self.assertEqual(view.iloc[-1].adj_close,181.)
        previous=self.official.copy();previous['date']=pd.Timestamp('2026-11-17')
        with self.assertRaisesRegex(ValueError,'one exchange session'):
            single_source_view(self.alt.iloc[:-2],pd.concat([previous,self.official]),
              [pd.Timestamp('2026-11-17'),pd.Timestamp('2026-11-18')],
              '2026-11-18','ABC','INE000000001','c'*64)

    def test_single_source_retains_same_isin_series_change_but_flags_it(self):
        alt=self.alt.copy();alt.loc[alt.index[:40],'series']='BE'
        _,proof=single_source_view(alt,self.official,self.sessions,'2026-11-18',
          'ABC','INE000000001','c'*64)
        self.assertTrue(proof['historical_series_transition_unverified'])
        alt.loc[alt.index[-1],'series']='BE'
        with self.assertRaisesRegex(ValueError,'current exchange series'):
            single_source_view(alt,self.official,self.sessions,'2026-11-18','ABC','INE000000001','c'*64)

    def test_low_turnover_expands_scoring_but_keeps_history_quality_gates(self):
        from v13_features import daily_features,relax_scoring_liquidity
        standard=daily_features(self.alt,'2026-11-18')
        self.assertFalse(bool(standard.iloc[-1].feature_eligible))
        expanded=relax_scoring_liquidity(standard)
        self.assertTrue(bool(expanded.iloc[-1].feature_eligible))
        self.assertFalse(bool(expanded.iloc[-1].feature_eligible_standard))
        self.assertEqual(expanded.iloc[-1].scoring_liquidity_status,'BELOW_25_LAKH_OUTSIDE_TRAINING_BAND')
        broken=standard.copy();broken.loc[broken.index[-1],'gap_max63']=11
        self.assertFalse(bool(relax_scoring_liquidity(broken).iloc[-1].feature_eligible))

    def test_single_source_worker_adds_identity_without_mutating_history_or_outcomes(self):
        from v13_runner import current_frame
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'v13_state';(root/'history'/'securities').mkdir(parents=True)
            official=root/'official';official.mkdir()
            pd.DataFrame([{'isin':'INE000000001','nse_current_symbol':'ABC'}]).to_parquet(root/'universe.parquet',index=False)
            snap=base/'v13_public_breadth_snapshot';snap.mkdir()
            data=snap/'nse_adjusted.parquet';self.alt.to_parquet(data,index=False)
            receipt={'source_asof':'2026-11-18','publisher_revision':'1'*40,
              'publisher_commit_utc':'2026-11-18T14:00:00Z','first_retrieved_utc':'2026-11-18T14:20:00Z',
              'source_url':'https://huggingface.co/datasets/tejhq/indian-markets/resolve/'+('1'*40)+'/prices_adjusted/nse_2026.parquet?download=true',
              'bytes':data.stat().st_size,'sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
            self.add_primary(snap,receipt,'2026-11-18','2026-11-18T14:20:00Z')
            self.official.assign(source_sha256=receipt['nse_original']['sha256']).to_parquet(official/'2026-11-18_official_market.parquet',index=False)
            (snap/'receipt.json').write_text(json.dumps(receipt))
            frame,history=current_frame(root,'2026-11-18')
            self.assertEqual(len(frame),1)
            self.assertEqual(frame.iloc[0].scoring_price_history_source,'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY')
            self.assertEqual(frame.iloc[0].scoring_fallback_reference_status,'NO_TIMELY_COMPARABLE_PROVIDER')
            self.assertEqual(frame.iloc[0].scoring_liquidity_status,'BELOW_25_LAKH_OUTSIDE_TRAINING_BAND')
            self.assertEqual(history,{})
            self.assertEqual(list((root/'history'/'securities').glob('*.parquet')),[])
            funnel=json.loads((root/'evidence'/'market_coverage_funnel.json').read_text())
            self.assertEqual(funnel['single_source_scoring_only']['new_identities_without_provider_history'],1)
            self.assertFalse(funnel['single_source_scoring_only']['entry_or_outcome_certified'])

    def test_original_low_liquidity_requires_a_real_pre_cutoff_receipt(self):
        from v13_runner import current_frame
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'v13_state';hist=root/'history'/'securities';hist.mkdir(parents=True)
            official=root/'official';official.mkdir()
            source=root/'sources'/'2026-11-18';source.mkdir(parents=True)
            history=self.alt.assign(source_sha256='a'*64)
            history.to_parquet(hist/'ABC.parquet',index=False)
            self.official.to_parquet(official/'2026-11-18_official_market.parquet',index=False)
            receipt=source/'history_refresh_test.json'
            def save_clock(clock):
                receipt.write_text(json.dumps([{'symbol':'ABC','info':{'symbol':'ABC','isin':'INE000000001',
                  'source_sha256':'a'*64,'first_retrieved_utc':clock}}]))
            save_clock('2026-11-18T16:31:00Z')
            late,_=current_frame(root,'2026-11-18')
            self.assertTrue(late.empty)
            funnel=json.loads((root/'evidence'/'market_coverage_funnel.json').read_text())
            self.assertEqual(funnel['low_liquidity_original_timing_blocked'],1)
            save_clock('2026-11-18T14:30:00Z')
            timely,_=current_frame(root,'2026-11-18')
            self.assertEqual(len(timely),1)
            self.assertEqual(timely.iloc[0].scoring_original_provider_sha256,'a'*64)
            self.assertEqual(timely.iloc[0].scoring_original_provider_first_retrieved_utc,'2026-11-18T14:30:00+00:00')
            from v13_lifecycle import register,validate_registry
            row=timely.assign(p6=.1,p12=.2,loss12=.3,joint12=.1,information_cutoff_utc='2026-11-18T14:30:00Z')
            p=register(row,'d'*64,root/'registries'/'champion',registered_at='2026-11-18T14:40:00Z')
            frozen=validate_registry(p)['records'][0]
            self.assertEqual(frozen['scoring_original_provider_sha256'],'a'*64)
            self.assertEqual(frozen['scoring_original_provider_first_retrieved_utc'],'2026-11-18T14:30:00+00:00')

    def test_wrong_isin_or_series_blocks(self):
        for col,value in [('isin','INE999999999'),('series','BE')]:
            alt=self.alt.copy();alt.loc[alt.index[-1],col]=value
            with self.subTest(col=col),self.assertRaises(ValueError):self.check(alt=alt)

    def test_any_official_ohlcv_disagreement_blocks(self):
        for col in ['open','high','low','close','volume']:
            alt=self.alt.copy();alt.loc[alt.index[-1],col]+=1
            with self.subTest(col=col),self.assertRaises(ValueError):self.check(alt=alt)

    def test_missing_or_changed_factors_block(self):
        alt=self.alt.copy();alt.loc[alt.index[7],'adj_close']*=.9
        with self.assertRaisesRegex(ValueError,'factors disagree'):self.check(alt=alt)
        alt=self.alt.copy();alt.loc[alt.index[-1],'adj_factor_cumulative']=.5
        with self.assertRaisesRegex(ValueError,'factor'):self.check(alt=alt)
        with self.assertRaisesRegex(ValueError,'Insufficient'):self.check(old=self.old.tail(125))

    def test_missing_or_duplicate_intervening_session_blocks(self):
        with self.assertRaises(ValueError):self.check(official=self.official.iloc[0:0])
        with self.assertRaises(ValueError):self.check(alt=pd.concat([self.alt,self.alt.tail(1)]))
        with self.assertRaisesRegex(ValueError,'omitted a recent'):self.check(alt=self.alt.drop(self.alt.index[-2]))

    def test_lagged_public_adjusted_file_extends_scoring_only_from_primary_raw(self):
        view,proof=extend(self.old,self.alt.iloc[:-1],self.official,self.sessions,
          '2026-11-18','ABC','INE000000001','c'*64,())
        self.assertEqual(view.iloc[-1].date,pd.Timestamp('2026-11-18'))
        self.assertEqual(view.iloc[-1].adj_close,181.)
        self.assertTrue(proof['lagged_adjusted_source_extended_with_primary_raw'])
        self.assertFalse(proof['action_interval_certified'])
        self.assertFalse(proof['entry_or_outcome_certified'])
        with self.assertRaisesRegex(ValueError,'corporate action'):
            extend(self.old,self.alt.iloc[:-1],self.official,self.sessions,
              '2026-11-18','ABC','INE000000001','c'*64,('ABC',))

    def test_actual_capture_and_publisher_clocks_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'original')
            r={'source_asof':'2026-11-18','publisher_revision':'1'*40,
               'publisher_commit_utc':'2026-11-18T14:00:00Z',
               'first_retrieved_utc':'2026-11-18T14:30:00Z',
               'source_url':'https://huggingface.co/datasets/tejhq/indian-markets/resolve/'+('1'*40)+'/prices_adjusted/nse_2026.parquet?download=true','bytes':len(b'original'),
               'sha256':hashlib.sha256(b'original').hexdigest()}
            self.add_primary(d,r,'2026-11-18','2026-11-18T14:20:00Z')
            validate_receipt(r,p,'2026-11-18')
            for field,value in [('publisher_commit_utc','2026-11-18T16:31:00Z'),
                                ('first_retrieved_utc','2026-11-18T16:31:00Z'),
                                ('first_retrieved_utc','2026-11-18T14:30:00'),
                                ('publisher_revision','a'*39),('sha256','0'*64)]:
                q={**r,field:value}
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate_receipt(q,p,'2026-11-18')
            for key in ('nse_original','nse_calendar'):
                q={**r,key:{**r[key],'first_retrieved_utc':'2026-11-18T16:31:00Z'}}
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Primary NSE source'):validate_receipt(q,p,'2026-11-18')
            with self.assertRaises(ValueError):validate_receipt(r,p,'2026-11-19')

    def test_absent_or_late_snapshot_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'v13_state';root.mkdir()
            data,status=load(root,'2026-11-18')
            self.assertIsNone(data);self.assertEqual(status['status'],'NO_TIMELY_PUBLIC_FALLBACK_SNAPSHOT')
            folder=Path(d)/'v13_public_breadth_snapshot';folder.mkdir()
            (folder/'nse_adjusted.parquet').write_bytes(b'raw')
            (folder/'receipt.json').write_text(json.dumps({'source_asof':'2026-11-18',
              'publisher_revision':'1'*40,'publisher_commit_utc':'2026-11-18T21:41:00Z',
              'first_retrieved_utc':'2026-11-19T04:00:00Z','sha256':hashlib.sha256(b'raw').hexdigest()}))
            data,status=load(root,'2026-11-18')
            self.assertIsNone(data);self.assertEqual(status['status'],'BLOCKED_SNAPSHOT')

    def test_worker_scores_fallback_without_changing_outcome_history(self):
        from v13_runner import current_frame
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'v13_state';hist=root/'history'/'securities';hist.mkdir(parents=True)
            official=root/'official';official.mkdir();audit=root/'action_audit';audit.mkdir()
            p=hist/'ABC.parquet';self.old.to_parquet(p,index=False);before=p.read_bytes()
            (audit/'calendar_2026.json').write_text(json.dumps({'year':2026,'holidays':[],'source_sha256':'a'*64}))
            snap=base/'v13_public_breadth_snapshot';snap.mkdir()
            data=snap/'nse_adjusted.parquet';self.alt.to_parquet(data,index=False)
            receipt={'source_asof':'2026-11-18','publisher_revision':'1'*40,
              'publisher_commit_utc':'2026-11-18T14:00:00Z','first_retrieved_utc':'2026-11-18T14:20:00Z',
              'source_url':'https://huggingface.co/datasets/tejhq/indian-markets/resolve/'+('1'*40)+'/prices_adjusted/nse_2026.parquet?download=true','bytes':data.stat().st_size,
              'sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
            self.add_primary(snap,receipt,'2026-11-18','2026-11-18T14:20:00Z')
            self.official.assign(source_sha256=receipt['nse_original']['sha256']).to_parquet(official/'2026-11-18_official_market.parquet',index=False)
            (snap/'receipt.json').write_text(json.dumps(receipt))
            source=root/'sources'/'2026-11-17';source.mkdir(parents=True)
            (source/'history_refresh_test.json').write_text(json.dumps([{'symbol':'ABC','info':{'symbol':'ABC','isin':'INE000000001',
              'source_sha256':'a'*64,'first_retrieved_utc':'2026-11-18T12:00:00Z'}}]))
            def features(bars,asof):
                result=bars.copy()
                for col in ('log_ret63','log_ret20','momentum126','momentum252_skip21','volume_confirmation','log_liquidity63'):
                    result[col]=1.
                result['feature_eligible']=True
                return result
            with patch('v13_runner.features.daily_features',side_effect=features):frame,history=current_frame(root,'2026-11-18')
            self.assertEqual(len(frame),1,(root/'evidence'/'market_coverage_funnel.json').read_text())
            self.assertEqual(frame.iloc[0].scoring_price_history_source,'ALTERNATIVE_ADJUSTED_SCORING_ONLY')
            self.assertEqual(history['INE000000001'].date.max(),pd.Timestamp('2026-11-17'))
            self.assertEqual(p.read_bytes(),before)
            funnel=json.loads((root/'evidence'/'market_coverage_funnel.json').read_text())
            self.assertEqual(funnel['scoring_only_fallback']['accepted_before_feature_gate'],1)
            self.assertFalse(funnel['scoring_only_fallback']['entry_or_outcome_certified'])
            from v13_breadth_fallback import snapshot_prior_histories
            prior=base/'temporary_prior';summary=snapshot_prior_histories(root,'2026-11-18',prior)
            self.assertEqual(summary['histories'],1)
            # A post-cutoff refresh has a different raw hash. Scoring can use
            # the earlier timed copy while outcome history remains the refresh.
            self.old.assign(source_sha256='e'*64).to_parquet(p,index=False);refreshed=p.read_bytes()
            with patch('v13_runner.features.daily_features',side_effect=features):restored,actual_history=current_frame(root,'2026-11-18',fallback_prior_dir=prior)
            self.assertEqual(len(restored),1)
            self.assertEqual(restored.iloc[0].scoring_fallback_provider_sha256,'a'*64)
            self.assertEqual(set(actual_history['INE000000001'].source_sha256),{'e'*64})
            self.assertEqual(p.read_bytes(),refreshed)
            saved=json.loads((prior/'index.json').read_text())['INE000000001|ABC']['file']
            (prior/saved).write_bytes(b'corrupted')
            with patch('v13_runner.features.daily_features',side_effect=features):blocked,_=current_frame(root,'2026-11-18',fallback_prior_dir=prior)
            self.assertTrue(blocked.empty)
            p.write_bytes(before)
            self.official.to_parquet(official/'2026-11-18_official_market.parquet',index=False)
            with patch('v13_runner.features.daily_features',side_effect=features):blocked,_=current_frame(root,'2026-11-18')
            self.assertTrue(blocked.empty)
            funnel=json.loads((root/'evidence'/'market_coverage_funnel.json').read_text())
            self.assertIn('Post-capture exchange bytes changed',funnel['scoring_only_fallback']['blocked_reasons'])
            self.official.assign(source_sha256=receipt['nse_original']['sha256']).to_parquet(official/'2026-11-18_official_market.parquet',index=False)
            (source/'history_refresh_test.json').write_text(json.dumps([{'symbol':'ABC','info':{'symbol':'ABC','isin':'INE000000001',
              'source_sha256':'a'*64,'first_retrieved_utc':'2026-11-18T16:31:00Z'}}]))
            with patch('v13_runner.features.daily_features',side_effect=features):single,history2=current_frame(root,'2026-11-18')
            # The independently timed alternate remains usable as a labelled
            # single source; the later provider is not a comparison input.
            self.assertEqual(len(single),1)
            self.assertEqual(single.iloc[0].scoring_price_history_source,'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY')
            self.assertEqual(single.iloc[0].scoring_fallback_reference_status,'NO_TIMELY_COMPARABLE_PROVIDER')
            self.assertIsNone(single.iloc[0].scoring_fallback_provider_sha256)
            self.assertEqual(history2['INE000000001'].date.max(),pd.Timestamp('2026-11-17'))
            self.assertEqual(p.read_bytes(),before)
            funnel=json.loads((root/'evidence'/'market_coverage_funnel.json').read_text())
            self.assertIn('No pre-cutoff original provider adjustment receipt',funnel['scoring_only_fallback']['blocked_reasons'])

    def test_new_registry_freezes_fallback_provenance(self):
        from v13_lifecycle import register,validate_registry
        row={'date':pd.Timestamp('2026-11-18'),'isin':'INE000000001','symbol':'ABC',
          'p6':.1,'p12':.2,'loss12':.3,'joint12':.1,
          'scoring_price_history_source':'ALTERNATIVE_ADJUSTED_SCORING_ONLY',
          'scoring_fallback_source_sha256':'c'*64,'scoring_fallback_action_interval_certified':False,
          'scoring_fallback_provider_sha256':'a'*64,'scoring_fallback_provider_first_retrieved_utc':'2026-11-18T12:00:00Z'}
        with tempfile.TemporaryDirectory() as d:
            path=register(pd.DataFrame([row]),'d'*64,d,registered_at='2026-11-18T14:30:00Z')
            rec=validate_registry(path)['records'][0]
            self.assertEqual(rec['scoring_fallback_source_sha256'],'c'*64)
            self.assertFalse(rec['scoring_fallback_action_interval_certified'])
            row.update(scoring_price_history_source='SINGLE_SOURCE_ADJUSTED_SCORING_ONLY',
              scoring_fallback_provider_sha256=None,scoring_fallback_provider_first_retrieved_utc=None,
              scoring_fallback_reference_status='NO_TIMELY_COMPARABLE_PROVIDER',scoring_fallback_overlap_sessions=0,
              scoring_history_series_transition_unverified=True)
            path=register(pd.DataFrame([row]),'d'*64,d,registered_at='2026-11-18T14:30:00Z')
            rec=validate_registry(path)['records'][0]
            self.assertEqual(rec['scoring_price_history_source'],'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY')
            self.assertEqual(rec['scoring_fallback_reference_status'],'NO_TIMELY_COMPARABLE_PROVIDER')
            self.assertIsNone(rec['scoring_fallback_provider_sha256'])
            self.assertTrue(rec['scoring_history_series_transition_unverified'])

    def test_capture_requires_an_immutable_current_publisher_revision(self):
        from v13_breadth_capture import capture
        rev='1'*40
        tree=json.dumps([{'path':'prices_adjusted/nse_2026.parquet',
          'lastCommit':{'id':rev,'date':'2026-11-18T14:00:00Z'}}]).encode()
        raw=self.alt.to_parquet(index=False)
        nse,calendar=self.primary_bytes()
        with tempfile.TemporaryDirectory() as d,patch('v13_breadth_capture.get',side_effect=[tree,raw,nse,calendar,b'[]']) as fetch,patch('v13_breadth_capture.datetime') as clock:
            clock.now.return_value=datetime(2026,11,18,14,20,tzinfo=timezone.utc)
            receipt=capture('2026-11-18',d)
            self.assertIn('/'+rev+'/',fetch.call_args_list[1].args[0])
            self.assertEqual(receipt['sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(receipt['first_retrieved_utc'],'2026-11-18T14:20:00+00:00')
            self.assertTrue((Path(d)/'receipt.json').exists())
        late=json.dumps([{'path':'prices_adjusted/nse_2026.parquet',
          'lastCommit':{'id':rev,'date':'2026-11-18T17:00:00Z'}}]).encode()
        with tempfile.TemporaryDirectory() as d,patch('v13_breadth_capture.get',return_value=late):
            with self.assertRaisesRegex(ValueError,'cutoff'):capture('2026-11-18',d)

    def test_early_year_capture_preserves_previous_year_original_and_hash(self):
        from v13_breadth_capture import capture
        rev='1'*40;old_rev='2'*40
        tree=json.dumps([
          {'path':'prices_adjusted/nse_2026.parquet','lastCommit':{'id':rev,'date':'2026-01-07T12:00:00Z'}},
          {'path':'prices_adjusted/nse_2025.parquet','lastCommit':{'id':old_rev,'date':'2025-12-31T12:00:00Z'}}]).encode()
        older=self.alt.head(130).copy();older['date']=pd.bdate_range(end='2025-12-31',periods=130)
        current=self.alt.tail(5).copy();current['date']=pd.bdate_range(end='2026-01-07',periods=5)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'v13_public_breadth_snapshot';root=Path(d)/'v13_state';root.mkdir()
            nse,calendar=self.primary_bytes()
            with patch('v13_breadth_capture.get',side_effect=[tree,current.to_parquet(index=False),older.to_parquet(index=False),nse,calendar,b'[]']),patch('v13_breadth_capture.datetime') as clock:
                clock.now.return_value=datetime(2026,1,7,12,20,tzinfo=timezone.utc)
                receipt=capture('2026-01-07',folder)
            self.assertEqual(receipt['previous_year']['year'],2025)
            data,status=load(root,'2026-01-07')
            self.assertEqual(status['status'],'TIMELY_SNAPSHOT_VERIFIED');self.assertEqual(len(data[0]),135)
            session=(folder/'nse_source_session.zip').read_bytes()
            (folder/'nse_source_session.zip').write_bytes(b'tampered')
            data,status=load(root,'2026-01-07')
            self.assertIsNone(data);self.assertEqual(status['status'],'BLOCKED_SNAPSHOT')
            (folder/'nse_source_session.zip').write_bytes(session)
            (folder/'nse_adjusted_previous.parquet').write_bytes(b'tampered')
            data,status=load(root,'2026-01-07')
            self.assertIsNone(data);self.assertEqual(status['status'],'BLOCKED_SNAPSHOT')


if __name__=='__main__':unittest.main()
