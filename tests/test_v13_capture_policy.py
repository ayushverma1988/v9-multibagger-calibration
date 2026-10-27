import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from v13_capture_policy import failure_kind,note_failure,ready_queue,bootstrap,signature
from v13_catalyst_field_capture import capture,capture_priority,export
from v12_dated_source_collection import sha


class CapturePolicy(unittest.TestCase):
    def test_identity_failure_rotates_and_recovers_after_cooldown(self):
        q=pd.DataFrame({'symbol':['A','B'],'version_key':['bad','new']})
        journal={'bad':note_failure('Document/current master ISIN conflict','2026-10-26T18:00Z','p','a')}
        ready,audit=ready_queue(q,journal,'2026-10-27T18:00Z','p',{'A':'a','B':'b'})
        self.assertEqual(ready.symbol.tolist(),['B'])
        self.assertEqual(audit['deferred_by_reason'],{'IDENTITY_REVIEW':1})
        ready,_=ready_queue(q,journal,'2026-11-03T18:00Z','p',{'A':'a','B':'b'})
        self.assertEqual(len(ready),2)
        for parser,identities in [('new',{'A':'a'}),('p',{'A':'changed'})]:
            ready,_=ready_queue(q,journal,'2026-10-27T18:00Z',parser,identities)
            self.assertIn('A',set(ready.symbol))

    def test_adding_unrelated_issuer_does_not_reset_failed_issuer(self):
        a=pd.DataFrame({'nse_current_symbol':['A'],'isin':['INE000000001']})
        _,prior=signature(a,[])
        _,new=signature(pd.concat([a,pd.DataFrame({'nse_current_symbol':['B'],'isin':['INE000000002']})]),[])
        self.assertEqual(prior['A'],new['A'])

    def test_bootstrap_preserves_actual_attempt_clock_and_all_public_versions(self):
        q=pd.DataFrame({'symbol':['A','A'],'source_url':['https://x/a','https://x/a'],'version_key':['v1','v2']})
        with tempfile.TemporaryDirectory() as t:
            Path(t,'latest_run.json').write_text(json.dumps({'completed_at_utc':'2026-10-26T18:00Z',
                'steps':{'financials':{'errors':[{'source_url':'https://x/a','error':'Missing or ambiguous original identity'}]}}}))
            state=bootstrap(t,q,'p',{'A':'a'})
        self.assertEqual(set(state),{'v1','v2'})
        self.assertEqual(pd.Timestamp(state['v1']['last_attempt_utc']),pd.Timestamp('2026-10-26T18:00Z'))
        self.assertEqual(failure_kind('429 Too Many Requests'),'ACCESS_LIMIT')
        self.assertEqual(failure_kind('Read timed out'),'TRANSIENT_FETCH_OR_EXTRACTION')

    def document(self,text):
        return {'isin':'INE000000001','symbol':'A','issuer_identity_read':True,
            'source_sha256':'b'*64,'text_sha256':sha(text.encode()),'attachment_url':'https://nsearchives.nseindia.com/a.pdf',
            'available_at_utc':'2026-10-26T12:00Z','first_retrieved_utc':'2026-10-26T13:00Z'}

    def test_candidates_preserve_units_excerpt_and_actual_observation(self):
        text='A has commenced commercial production. Capacity of 15 MW. Utilisation at 89%. EBITDA margin of 18%.'
        row=capture(text,self.document(text),'2026-10-28T00:00Z')
        fields={x['field']:x for x in row['fields']}
        self.assertEqual(fields['capacity_quantity']['value'],15)
        self.assertEqual(fields['capacity_quantity']['unit'],'MW')
        self.assertEqual(fields['utilisation_pct']['value'],89)
        self.assertEqual(fields['reported_margin_pct']['unit'],'EBITDA_PERCENT')
        self.assertFalse(row['full_execution_chain_verified'])
        self.assertEqual(row['available_for_new_analysis_at_utc'],'2026-10-28T00:00:00+00:00')
        self.assertTrue(all(x['source_excerpt'] in text for x in row['fields']))

    def test_intent_loi_and_legal_order_are_not_certified(self):
        for text in ['A plans a capacity of 15 MW.',
                     'A has received a letter of intent for a contract.',
                     'A has received a tax order from a court.']:
            row=capture(text,self.document(text),'2026-10-28T00:00Z')
            self.assertFalse(row['full_execution_chain_verified'])
            self.assertTrue(all(x['status']=='CONDITIONAL_OR_CONTEXT_REVIEW' for x in row['fields']))

    def test_wrong_hash_identity_and_future_clock_rejected(self):
        text='A has commenced commercial production.'
        for change in [{'text_sha256':'a'*64},{'issuer_identity_read':False},
                       {'first_retrieved_utc':'2026-10-29T00:00Z'}]:
            with self.assertRaises(ValueError):
                capture(text,{**self.document(text),**change},'2026-10-28T00:00Z')

    def test_export_deduplicates_original_and_prioritises_business_disclosure(self):
        text='A has commenced commercial production. Capacity of 15 MW.'
        doc=self.document(text)
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t,'catalyst_evidence');folder.mkdir()
            (folder/'document_evidence.json').write_text(json.dumps([doc,doc]))
            (folder/(doc['source_sha256']+'.txt')).write_text(text)
            result=export(t,'2026-10-26')
            self.assertEqual(result['documents_with_field_candidates'],1)
            self.assertEqual(result['reviewed_execution_chains_added'],0)
            self.assertEqual(result['original_downloads_added'],0)
        self.assertLess(capture_priority({'headline':'Bagging/Receiving of orders/contracts'}),
                        capture_priority({'headline':'Newspaper publication'}))


if __name__=='__main__':unittest.main()
