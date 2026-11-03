import base64,gzip,json,tempfile,unittest
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd
from v13_financial_recovery_seed import apply

class VerifiedFinancialRecovery(unittest.TestCase):
    def doc(self):
        path=Path(__file__).resolve().parents[1]/'v13_evidence_seed/financial_primary_recovery_20261020.json'
        return next(d for d in json.loads(path.read_text())['documents'] if d['record']['symbol']=='MASTEK')
    def setup_root(self,folder,doc,wrong=False):
        root=Path(folder)/'state';root.mkdir();raw=gzip.decompress(base64.b64decode(doc['raw_gzip_base64']))
        values=[e.text for e in ET.fromstring(raw).iter() if e.tag.split('}')[-1]=='ISIN']
        isin=values[0] if values else doc['record']['catalog_isin']
        pd.DataFrame([dict(nse_current_symbol='MASTEK',isin='INE000000001' if wrong else isin)]).to_parquet(root/'universe.parquet')
        pack=Path(folder)/'pack.json';pack.write_text(json.dumps({'documents':[doc]}));return root,pack
    def test_original_clock_and_identity_preserved_and_existing_forecast_untouched(self):
        d=self.doc()
        with tempfile.TemporaryDirectory() as t:
            root,pack=self.setup_root(t,d);(root/'frozen_forecast.json').write_bytes(b'original probabilities')
            r=apply(root,pack,'2026-10-19');self.assertEqual(r['errors'],[]);self.assertEqual(r['original_documents_replayed'],1)
            w=pd.read_parquet(root/'financial_wide.parquet');self.assertEqual(set(w.available_at_utc),{d['record']['available_at_utc']})
            self.assertEqual((root/'frozen_forecast.json').read_bytes(),b'original probabilities')
    def test_tampered_bytes_and_wrong_current_isin_never_enter_financials(self):
        for wrong,tamper in [(True,False),(False,True)]:
            d=self.doc()
            if tamper:d['receipt']['sha256']='0'*64
            with tempfile.TemporaryDirectory() as t:
                root,pack=self.setup_root(t,d,wrong);r=apply(root,pack,'2026-10-19')
                self.assertEqual(len(r['errors']),1);self.assertFalse((root/'financial_wide.parquet').exists())

if __name__=='__main__':unittest.main()
