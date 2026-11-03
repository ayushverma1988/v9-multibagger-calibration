"""Guard source-only fiscal runs against year, mode and as-of leakage."""
import unittest
import pandas as pd
from v11_4_annual_index_pit_v2 import fold_close,find_run,best_mode_run,canonical_mode
def fy(n):return pd.Timestamp(f"{n}-03-31T00:00:00Z")

class AnnualIndexTests(unittest.TestCase):
    def setUp(self):self.cut=fold_close("2025-12-31")
    def test_both_modes_not_mixed(self):
        # Four consolidated years + two standalone years cannot be 5-year CAGR.
        mode,run=best_mode_run({"consolidated":[fy(y) for y in range(2021,2025)],
                      "standalone":[fy(2020),fy(2025)]},self.cut,5)
        self.assertFalse(run)
    def test_valid_contiguous_5y(self):
        mode,run=best_mode_run({"consolidated":[fy(y) for y in range(2020,2026)]},self.cut,5)
        self.assertEqual(mode,"consolidated")
        self.assertEqual(len(run),6)
    def test_standalone_choice_not_shorter_consolidated(self):
        mode,run=best_mode_run({"standalone":[fy(y) for y in range(2019,2026)],
                              "consolidated":[fy(2024),fy(2025)]},self.cut,5)
        self.assertEqual(mode,"standalone")
        self.assertEqual(len(run),6)
    def test_fiscal_gap_rejected(self):
        self.assertEqual(find_run([fy(y) for y in [2019,2020,2021,2023,2024,2025]],self.cut,5),[])
    def test_stale_run_rejected(self):
        self.assertEqual(find_run([fy(y) for y in range(2017,2023)],self.cut,5),[])
    def test_mode_canonicalization(self):
        self.assertEqual(canonical_mode("Non-Consolidated"),"standalone")
        self.assertEqual(canonical_mode("Standalone"),"standalone")
        self.assertEqual(canonical_mode("Consolidated"),"consolidated")
    def test_old_and_new_source_date_formats(self):
        a=pd.Series(["2020-05-27T08:50:00+00:00","2025-05-29 08:50:00+00:00"])
        parsed=pd.to_datetime(a,utc=True,format="mixed",errors="coerce")
        self.assertEqual(int(parsed.notna().sum()),2)

    def test_close(self):
        self.assertEqual(self.cut.isoformat(),"2025-12-31T10:00:00+00:00")
if __name__=="__main__":unittest.main(verbosity=2)
