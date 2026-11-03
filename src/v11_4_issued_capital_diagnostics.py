"""Original NSE security-master IssdCptl unit investigation ONLY.

Do not infer fully listed outstanding shares or market cap until master-field
unit and same-date security eligibility are corroborated. Show fixed public
large-cap examples and magnitude without publishing guessed valuation fields.
"""
import argparse,gzip,hashlib,io,json
from pathlib import Path
import pandas as pd,requests
from v11_4_live_nse_market_catalyst import exchange_primary_for_day
from v11_4_nse_archive_valuation_probe import HEADERS
EXAMPLES=("RELIANCE","TCS","INFY","HDFCBANK","SBIN","ICICIBANK","ITC","TATAMOTORS")
def check(asof,out):
 date=pd.Timestamp(asof).normalize()
 url=f"https://archives.nseindia.com/content/cm/NSE_CM_security_{date.strftime('%d%m%Y')}.csv.gz"
 r=requests.get(url,headers=HEADERS,timeout=55);r.raise_for_status()
 assert r.content[:2]==b'\x1f\x8b', "Not a true NSE gzip master file"
 frame=pd.read_csv(io.BytesIO(gzip.decompress(r.content)),low_memory=False)
 mandatory={"TckrSymb","SctySrs","ISIN","IssdCptl","ParVal"}
 if not mandatory.issubset(frame):raise ValueError("Security master missing issuance/capital metadata")
 df=frame[frame["TckrSymb"].astype(str).isin(EXAMPLES)&frame["SctySrs"].eq("EQ")].copy()
 official,_=exchange_primary_for_day(date)
 official=official[official["series"].eq("EQ")]
 df=df.merge(official[["symbol","isin","close"]],left_on=["TckrSymb","ISIN"],right_on=["symbol","isin"],how="inner")
 df["issued_capital_raw"]=pd.to_numeric(df["IssdCptl"],errors="coerce")
 df["par_value_raw"]=pd.to_numeric(df["ParVal"],errors="coerce")
 df["candidate_if_issued_capital_is_shares_cr"]=df["issued_capital_raw"]*df["close"]/1e7
 df["candidate_if_issued_capital_is_nominal_rupees_cr"]=df["issued_capital_raw"]*df["close"]/df["par_value_raw"]/1e7
 fields=["symbol","close","issued_capital_raw","par_value_raw",
 "candidate_if_issued_capital_is_shares_cr","candidate_if_issued_capital_is_nominal_rupees_cr"]
 data=df[fields].sort_values("symbol").fillna(-1).to_dict("records")
 summary={"scope":"UNIT_SENSE_CHECK_NSE_EXCHANGE_ISSUED_CAPITAL_NOT_VERIFIED_MARKET_CAP",
          "date":str(date.date()),"official_master_url":url,
          "sha256":hashlib.sha256(r.content).hexdigest(),"matched_trusted_examples":len(data),
          "raw_and_TWO_possible_units_for_recognizable_public_company_market_caps":data,
          "market_cap_inferred_as_verified":False,"frozen_V11_4_unchanged":True}
 p=Path(out);p.mkdir(exist_ok=True,parents=True)
 (p/"issued_capital_units_competing_interpretations.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 return summary
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--asof",required=True);p.add_argument("--output",required=True)
 a=p.parse_args();check(a.asof,a.output)
if __name__=="__main__":main()
