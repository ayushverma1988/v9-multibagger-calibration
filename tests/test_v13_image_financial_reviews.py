import base64
import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from v12_5_expanded_facts import parse_extra
from v13_image_financial_reviews import validate_table, collect


class ImageFinancialReview(unittest.TestCase):
    def record(self):
        path=Path(__file__).resolve().parents[1]/'v13_evidence_seed/image_financial_reviews_20261112.json'
        r=json.loads(path.read_text())['documents'][0]
        r['source_sha256']=hashlib.sha256(b'original PDF').hexdigest()
        r['page_image_sha256']=hashlib.sha256(b'original scanned page').hexdigest()
        r['first_reviewed_utc']='2026-11-12T00:00:00+00:00'
        return r

    def pdf(self, r):
        front=MagicMock();front.extract_text.return_value=' '.join(r[k] for k in ('symbol','isin','cin'))
        image=MagicMock();image.get_rawdata.return_value=b'original scanned page'
        page=MagicMock();page.images=[{'stream':image}]
        pdf=MagicMock();pdf.__enter__.return_value.pages=[front]+[page]*6
        return pdf

    def validate(self,r):
        with patch('v13_image_financial_reviews.pdfplumber.open',return_value=self.pdf(r)):
            return validate_table(r,b'original PDF','2026-11-12',now='2026-11-12T12:00:00Z')

    def test_original_columns_preserve_money_owner_profit_and_unscaled_diluted_EPS(self):
        values=self.validate(self.record())
        self.assertEqual(values['revenue'][2],2252645000.)
        self.assertEqual(values['pat'][2],300271000.)
        self.assertEqual(values['pat_owners'][2],311114000.)
        self.assertEqual(values['pat_noncontrolling'][2],-10843000.)
        self.assertEqual(values['eps_diluted_continuing'][2],.91)
        self.assertAlmostEqual(values['eps_diluted_continuing'][0]/values['eps_diluted_continuing'][2]-1,.4505494505494505)

    def test_changed_original_image_basis_columns_units_and_future_review_are_rejected(self):
        changes=[{'source_sha256':'a'*64},{'page_image_sha256':'b'*64},
            {'reporting_mode':'standalone'},{'comparative_column':5},
            {'scale_INR':1000000},{'page_number':0},
            {'first_reviewed_utc':'2026-11-13T00:00:00Z'},
            {'review_status':'AUTOMATIC_OCR_CANDIDATE'}]
        for changeset in changes:
            with self.subTest(changes=changeset),self.assertRaises(ValueError):self.validate({**self.record(),**changeset})

    def test_missing_cells_basic_EPS_and_inferred_owner_profit_are_not_accepted(self):
        for mode in ('dash','basic','units','derived_owner','owner_conflict'):
            r=self.record()
            if mode=='dash':r['fields'][0]['reported_cells'][2]='-'
            if mode=='basic':r['fields'][-1]['label']='Basic'
            if mode=='units':r['fields'][-1]['unit']='INR'
            if mode=='derived_owner':r['fields'][2]['metric']='pat_owners_by_subtraction'
            if mode=='owner_conflict':r['fields'][2]['reported_cells'][2]='2,111.14'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.validate(r)

    def fixture(self,root,existing=False):
        r=self.record();source=MagicMock()
        catalog=[{'symbol':r['symbol'],'sm_isin':'INE713T01010','attchmntFile':r['source_url'],'exchdisstime':'24-Oct-2026 15:54:02'}]
        proof={'sha256':r['source_sha256'],'first_retrieved_utc':'2026-11-12T00:01:00Z'}
        source.get.side_effect=lambda url:(json.dumps(catalog).encode(),{'sha256':'c'*64}) if url==r['catalogue_url'] else (b'original PDF',proof)
        seed=root/'seed.json';seed.write_text(json.dumps({'documents':[r]}))
        pd.DataFrame({'nse_current_symbol':[r['symbol']],'isin':[r['isin']]}).to_parquet(root/'universe.parquet',index=False)
        values=self.validate(r)
        anchor={'security_isin':r['isin'],'symbol':r['symbol'],'reporting_mode':'consolidated','period_kind':'quarter',
            'period_end':'2026-09-30','period':pd.Timestamp('2026-09-30'),'available':pd.Timestamp('2026-10-24T11:41:30Z'),
            'available_at_utc':'2026-10-24T11:41:30Z','source_sha256':'d'*64,
            **{k:v[0] for k,v in values.items()}}
        rows=[anchor]
        if existing:rows.append({**anchor,'period_end':'2025-09-30','period':pd.Timestamp('2025-09-30'),**{k:v[2] for k,v in values.items()}})
        pd.DataFrame(rows).to_parquet(root/'financial_wide.parquet',index=False)
        (root/'evidence').mkdir();(root/'frozen_forecasts.json').write_bytes(b'unchanged original forecast')
        return seed,source,r

    def test_collection_preserves_real_clocks_originals_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);seed,source,r=self.fixture(root)
            with patch('v13_image_financial_reviews.Store',return_value=source),patch('v13_image_financial_reviews.pdfplumber.open',return_value=self.pdf(r)):
                result=collect(root,seed,'2026-11-12');again=collect(root,seed,'2026-11-12')
            self.assertTrue(result['complete']);self.assertEqual(result['missing_comparative_fact_observations_added'],9)
            self.assertEqual(result,again);self.assertEqual(source.get.call_count,2)
            data=pd.read_parquet(root/'financial_wide.parquet');new=data[data.period_end.eq('2025-09-30')]
            self.assertEqual(new.available.iloc[0],pd.Timestamp('2026-11-12T00:01:00Z'))
            self.assertEqual((root/'frozen_forecasts.json').read_bytes(),b'unchanged original forecast')
            eps=json.loads((root/'statement_eps/accepted.json').read_text())[0]
            self.assertAlmostEqual(eps['reported_diluted_EPS_yoy_same_document'],1.32/.91-1)
            self.assertFalse(eps['recurring_EPS_verified'])

    def test_existing_correct_comparative_is_preserved_and_conflict_is_flagged(self):
        for conflict in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);seed,source,r=self.fixture(root,existing=True)
                old=pd.read_parquet(root/'financial_wide.parquet')
                if conflict:old.loc[old.period_end.eq('2025-09-30'),'revenue']=999.
                old.to_parquet(root/'financial_wide.parquet',index=False);before=(root/'financial_wide.parquet').read_bytes()
                with patch('v13_image_financial_reviews.Store',return_value=source),patch('v13_image_financial_reviews.pdfplumber.open',return_value=self.pdf(r)):
                    result=collect(root,seed,'2026-11-12')
                self.assertEqual(result['complete'],not conflict)
                self.assertEqual(result['missing_comparative_fact_observations_added'],0)
                self.assertEqual((root/'financial_wide.parquet').read_bytes(),before)

    def test_half_year_cash_flow_is_captured_separately_from_annual_cash_flow(self):
        seed=Path(__file__).resolve().parents[1]/'v13_evidence_seed/financial_primary_recovery_20261112.json'
        doc=json.loads(seed.read_text())['documents'][0]
        raw=gzip.decompress(base64.b64decode(doc['raw_gzip_base64']))
        old=parse_extra(raw,doc['record'],'2026-11-12',include_ytd=True)
        new=parse_extra(raw,doc['record'],'2026-11-12',include_ytd=True,include_ytd_cashflow=True)
        self.assertFalse(any(r['metric']=='cfo' and r['period_kind']=='ytd' for r in old))
        cash=[r for r in new if r['metric']=='cfo']
        self.assertEqual([(r['period_kind'],r['period_end'],r['value']) for r in cash],[('ytd','2026-09-30',224373000.)])


if __name__=='__main__':unittest.main()
