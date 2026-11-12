import unittest
import pandas as pd
from primary_history_extension import extend

class ExtensionTests(unittest.TestCase):
    def example(self):
        b=pd.DataFrame({'date':pd.to_datetime(['2026-11-06','2026-11-09']),
            'isin':['INE000000001']*2,'symbol':['TEST']*2,'open':[99.,100.],
            'high':[102.,104.],'low':[98.,99.],'close':[100.,102.],
            'adj_close':[98.,99.96],'volume':[1000,1200],'source_sha256':['a'*64]*2})
        anchor={**b.iloc[-1].to_dict(),'series':'BE','source_status':'VALID_POSITIVE_VOLUME_SESSION'}
        current={**anchor,'date':pd.Timestamp('2026-11-11'),'open':103.,'high':107.,'low':101.,'close':106.,'volume':2300,'source_sha256':'b'*64}
        return b,anchor,current
    def test_real_new_prices_and_volume_without_changing_old_values(self):
        b,a,c=self.example();d,r=extend(b,a,c,set(),True,True)
        pd.testing.assert_frame_equal(b,d.iloc[:2].reset_index(drop=True),check_dtype=False)
        self.assertEqual(d.iloc[-1].volume,2300)
        self.assertEqual(d.iloc[-1].close,106.)
        self.assertAlmostEqual(d.iloc[-1].adj_close,103.88)
        self.assertFalse(r['latest_session_independently_vendor_verified'])
    def test_rejects_unverified_action_or_calendar_scope(self):
        b,a,c=self.example()
        for feed,cal in [(False,True),(True,False)]:
            with self.assertRaises(ValueError):extend(b,a,c,set(),feed,cal)
    def test_rejects_any_disclosed_action_even_cash_dividend(self):
        b,a,c=self.example()
        with self.assertRaises(ValueError):extend(b,a,c,{'TEST'},True,True)
    def test_actual_primary_string_dates_remain_datetime_and_serialize(self):
        b,a,c=self.example();a['date']='2026-11-09';c['date']='2026-11-11'
        d,_=extend(b,a,c,set(),True,True)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(d.date))
        self.assertEqual(d.date.iloc[-1],pd.Timestamp('2026-11-11'))
        import importlib.util,io,os
        if importlib.util.find_spec('pyarrow'):
            d.to_parquet(io.BytesIO(),index=False)
        elif os.environ.get('CI'):self.fail('Runner must exercise actual Parquet serialization')
    def test_rejects_stale_history_and_missing_trading_session(self):
        b,a,c=self.example();b=b.iloc[:1]
        with self.assertRaises(ValueError):extend(b,a,c,set(),True,True)
    def test_rejects_price_volume_identity_and_SME_mismatches(self):
        for changes in [{'volume':1201},{'close':102.02},{'isin':'INE000000002'},{'series':'SM'}]:
            b,a,c=self.example();a.update(changes)
            with self.assertRaises(ValueError):extend(b,a,c,set(),True,True)

if __name__=='__main__':unittest.main(verbosity=2)
