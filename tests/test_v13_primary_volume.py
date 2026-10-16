import unittest
import pandas as pd
from v13_primary_volume import reconcile

class PrimaryVolume(unittest.TestCase):
    def inputs(self):
        h=pd.DataFrame([dict(date=pd.Timestamp('2026-10-15'),symbol='A',isin='INE000000001',open=100.,high=103.,low=99.,close=102.,adj_close=101.,volume=1000,source_sha256='a'*64)])
        o=h.copy();o['volume']=1100;o['source_sha256']='b'*64;o['source_status']='VALID_POSITIVE_VOLUME_SESSION';o['first_retrieved_utc']='2026-10-15T16:00:00Z'
        return h,o
    def test_lineage_and_original_prices_preserved(self):
        h,o=self.inputs();before=h.copy();d,changed=reconcile(h,o,'2026-10-15')
        self.assertTrue(changed);self.assertEqual(d.volume.iloc[0],1100)
        self.assertEqual(d.provider_original_volume.iloc[0],1000)
        self.assertEqual(d.source_sha256.iloc[0],'a'*64)
        self.assertEqual(d.volume_primary_source_sha256.iloc[0],'b'*64)
        pd.testing.assert_frame_equal(h,before)
        pd.testing.assert_frame_equal(d[['open','high','low','close','adj_close']],h[['open','high','low','close','adj_close']])
    def test_no_missing_or_conflicting_prices_dates_identity_or_clocks_filled(self):
        for field,value in [('open',101.),('close',float('nan')),('isin','INE000000002'),('date',pd.Timestamp('2026-10-14')),('source_sha256','bad'),('first_retrieved_utc','2026-10-15T02:00:00Z')]:
            h,o=self.inputs();o[field]=value;d,changed=reconcile(h,o,'2026-10-15')
            self.assertFalse(changed,field);pd.testing.assert_frame_equal(d,h)
