import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
from v13_review_restore import bound_doc,digest,floor_clock,key_id,present_hashes,receipts,restore,source_index

class SourceRecovery(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)
        self.body=b'%PDF-archived-source';self.text='a retained source excerpt';self.h=digest(self.body)
        self.seed=dict(source_sha256=self.h,symbol='EX',isin='INE000000000',attachment_url='https://nsearchives.nseindia.com/corporate/a.pdf',order_id='A')
        self.doc={**self.seed,'issuer_identity_read':True,'text_sha256':digest(self.text.encode()),'available_at_utc':'2026-10-01T12:00:00Z'}
        self.fetch=Mock(return_value=(self.body,self.seed['attachment_url']));self.verify=Mock()
        self.kw=dict(verify=self.verify,extract=lambda b:self.text,fetch=self.fetch,clock=lambda:'2026-11-18T09:33:11+00:00')
    def run_case(self,seeds=None,stored=None,docs=None,**kw):
        return restore(self.p,[self.seed] if seeds is None else seeds,[self.seed] if stored is None else stored,[self.doc] if docs is None else docs,**{**self.kw,**kw})
    def test_bound_exact_origin_across_mirrors(self):
        d2={**self.doc,'attachment_url':'https://www.aeroflexindia.com/uploads/topics/m.pdf','available_at_utc':'2026-07-01T12:00:00Z'}
        for docs in ([d2,self.doc],[self.doc,d2]):
            self.assertEqual(bound_doc(source_index(docs),self.seed),self.doc)
        with self.assertRaises(ValueError):bound_doc(source_index([d2]),self.seed)
    def test_shared_original_revalidates_both_reviews_without_duplication(self):
        b={**self.seed,'order_id':'B'};r=self.run_case([self.seed,b],[self.seed,b]);self.assertEqual((r['restored_originals'],r['revalidated_review_rows']),(1,2))
        self.fetch.assert_called_once();self.assertEqual(self.verify.call_count,2)
        self.assertEqual(present_hashes(self.p/'raw',{self.h}),{self.h})
        self.assertEqual(self.run_case()['restored_originals'],0)
    def test_retirement_and_repeat_keep_first_recovery_clock(self):
        self.run_case();(self.p/'raw'/(self.h+'.bin')).unlink()
        self.run_case(clock=lambda:'2026-11-19T06:20:00+00:00')
        j=receipts(self.p);z=floor_clock(dict(available_at_utc='2026-10-01T00:00:00Z'),self.seed,j)
        self.assertEqual(pd.Timestamp(z['available_at_utc']),pd.Timestamp('2026-11-18T09:33:11Z'))
        self.assertEqual(j[key_id(self.seed)]['last_revalidated_utc'],'2026-11-19T06:20:00+00:00')
    def test_changed_pdf_extraction_or_rejected_review_never_installed(self):
        for kw in [dict(fetch=lambda url:(b'%PDF-WRONG',url)),dict(extract=lambda b:'different text'),dict(verify=Mock(side_effect=ValueError('unbound quantity')))]:
            self.assertTrue(self.run_case(**kw)['errors']);self.assertFalse(list((self.p/'raw').glob('*.bin')));self.assertFalse(receipts(self.p))
    def test_corrupted_existing_text_or_content_addressed_bytes_kept_for_review(self):
        tp=self.p/(self.h+'.txt');tp.write_text('quarantined');self.assertTrue(self.run_case()['errors']);self.assertEqual(tp.read_text(),'quarantined')
        tp.unlink();raw=self.p/'raw'/(self.h+'.bin');raw.write_bytes(b'conflict');self.assertTrue(self.run_case()['errors'])
        self.assertEqual(raw.read_bytes(),b'conflict');self.fetch.assert_not_called()
    def test_ambiguous_provenance_and_untrusted_seed_never_fetched(self):
        self.assertTrue(self.run_case(docs=[self.doc,{**self.doc,'first_reviewed_utc':'2026-10-22T10:00:00Z'}])['errors'])
        self.assertTrue(self.run_case(docs=[{**self.doc,'isin':'different'}])['errors'])
        self.assertEqual(self.run_case(stored=[{**self.seed,'attachment_url':'https://invalid.example/x.pdf'}])['restored_originals'],0);self.fetch.assert_not_called()
    def test_recovery_receipt_cannot_replace_source_identity_or_newer_review(self):
        self.run_case();j=receipts(self.p);z=floor_clock({'available_at_utc':'2026-11-20T12:00:00Z'},self.seed,j)
        self.assertEqual(pd.Timestamp(z['available_at_utc']),pd.Timestamp('2026-11-20T12:00:00Z'))
        j[key_id(self.seed)]['isin']='wrong'
        with self.assertRaises(ValueError):floor_clock({'available_at_utc':'2026-11-20T12:00:00Z'},self.seed,j)
