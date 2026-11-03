import io,unittest
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from v13_dated_export import compact_parquet

class DatedTableExport(unittest.TestCase):
    def test_precision_missing_values_issuer_and_dates_survive(self):
        data=pd.DataFrame({'date':pd.to_datetime(['2025-04-01','2026-11-02',None]),
          'isin':['INE000000001','INE000000002',None],'value':[0.12345678901234566,np.nan,-1.2345678901234567]})
        source=io.BytesIO();data.to_parquet(source,index=False)
        encoded=compact_parquet(source.getvalue())
        pd.testing.assert_frame_equal(data,pq.read_table(io.BytesIO(encoded)).to_pandas(),check_exact=True)
