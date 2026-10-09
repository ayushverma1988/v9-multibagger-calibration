"""Official NSE old annual report metadata source candidates never count as 3FY numeric."""
import unittest
import pandas as pd
import v11_4_2021Dec_original_NSE_annual_reports_metadata_candidate_probe as m

class Response:
 def __init__(self,data):self.data=data
 def json(self):return self.data
 def raise_for_status(self):pass
class Session:
 def __init__(self,data):self.data=data;self.calls=0
 def get(self,url,params=None,headers=None,timeout=None):
  if "api/annual-reports" not in url:return Response([])
  self.calls+=1
  return Response(self.data)
def docs():
 return [
  {"fromYr":"2020","toYr":"2021",
   "broadcast_dttm":"25-Jun-2021 17:20:00",
   "disseminationDateTime":"25-Jun-2021 17:21:00",
   "fileName":"https://nsearchives.nseindia.com/annual_reports/AR_A_2020_2021.pdf"},
  {"fromYr":"2020","toYr":"2021",
   "broadcast_dttm":"03-Jan-2022 11:00:00",
   "disseminationDateTime":"03-Jan-2022 11:00:01",
   "fileName":"https://nsearchives.nseindia.com/annual_reports/AR_A_LATE.pdf"},
  {"fromYr":"2020","toYr":"2021",
   "broadcast_dttm":"25-Jun-2021 17:20:00",
   "disseminationDateTime":"25-Jun-2021 17:21:00",
   "fileName":"https://malicious.example/2021.xml"}]
def missing():
 return pd.DataFrame([{"date":m.DATE,"symbol":f"SYM{i:03d}",
  "missing_original_FY_under_same_mode":[2021],
  "asof_best_same_mode_years":2,
  "failure_reason":"ONLY_TWO_EXACT_FYS_IN_SAME_MODE_ASOF_CLOSE"} for i in range(25)])
class OldOriginalAnnualMetadata(unittest.TestCase):
 def test_only_official_original_2021_preclose_report_is_accepted(self):
  accepted,audit=m.strict_original_reports(docs(),"SAMPLE",{2021})
  self.assertEqual(len(accepted),1)
  self.assertEqual(audit["target_FY_late_publication_rejected"],1)
  self.assertEqual(audit["invalid_PIT_metadata_or_host_rejected"],1)
  self.assertFalse(accepted[0]["original_NSE_annual_report_revenue_PAT_INR_numeric_verified"])
 def test_source_incorrect_or_future_asof_rejected(self):
  item=docs()[0].copy()
  item["broadcast_dttm"]="01-Jan-2022 11:00:00"
  accepted,_=m.strict_original_reports([item],"SAMPLE",{2021})
  self.assertEqual(len(accepted),0)
 def test_annual_report_metadata_not_three_year_verified_company(self):
  r,events,found,errors=m.inspect(missing(),limit=8,session=Session(docs()))
  self.assertEqual(r["annual_report_metadata_issuer_FY_candidates_available_before_historical_asof"],8)
  self.assertEqual(r["new_original_3FY_numeric_company_years_verified"],0)
  self.assertEqual(r["max_sample_48_deterministic_missing_annual_NSE_XBRL_issuers_not_selected_by_past_returns"],8)
  self.assertFalse(r["new_original_3FY_scientifically_validated_2021_Dec_fold"])
  self.assertEqual(len(found),8)
 def test_future_market_winner_label_in_missing_source_rejected(self):
  f=missing();f["y6"]=1
  with self.assertRaisesRegex(ValueError,"winners"):
   m.inspect(f,limit=8,session=Session(docs()))
 def test_no_original_filing_needed_missing_year_drift_fails(self):
  f=missing();f.loc[0,"missing_original_FY_under_same_mode"]=[2025]
  with self.assertRaisesRegex(ValueError,"missing historical"):
   m.inspect(f,limit=25,session=Session(docs()))
if __name__=="__main__":unittest.main()
