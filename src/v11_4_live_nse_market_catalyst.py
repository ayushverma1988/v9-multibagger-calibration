"""V11.4 standalone 2026+ fresh NSE cash-market and announcement PIT collector.

Research-only inputs. No older model, no cached stale 'today', no news
after the cash-market close, no silent third-party-only price fallback.
The independently verified two-file NSE UDiFF + full-security bhavcopy
are mandatory. Hugging Face nse/year parquet is a historical price
source only; the target close is ALWAYS overwritten by validated NSE.
"""
from __future__ import annotations
import argparse,hashlib,io,json,re,zipfile
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from v11_4_standalone_train_walkforward import PRICE_COLUMNS,CATALYST_PREFIXES,fold_close

NSE_U="https://archives.nseindia.com/content/cm"
NSE_F="https://archives.nseindia.com/products/content"
NSE_PAGE="https://www.nseindia.com/companies-listing/corporate-filings-announcements"
NSE_API="https://www.nseindia.com/api/corporate-announcements"
HF_MARKET="tejhq/indian-markets"
HEADERS={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/129 Safari/537.36",
 "Accept":"application/json,text/plain,*/*","Accept-Language":"en-US,en;q=0.9",
 "Referer":NSE_PAGE}
SERIES={"EQ","BE","BZ"}
MIN_NSE_ROWS=500
MIN_SECONDARY_OVERLAP=.90
MIN_UNIQUE_MARKET_CANDIDATES=500
RECENT_DAYS=185
TYPES={"order_win","capacity_expansion","regulatory","promoter_activity",
       "corporate_action","dilution","earnings","buyback"}

def sha(raw):
 return hashlib.sha256(raw).hexdigest()

def col(x,*aliases,optional=False):
 cmap={str(k).upper().strip():k for k in x.columns}
 for k in aliases:
  if k.upper() in cmap:return cmap[k.upper()]
 if optional:return None
 raise ValueError(f"Missing source fields {aliases}; available {list(x)}")

def get_bytes(url,session=None):
 s=session or requests
 r=s.get(url,headers=HEADERS,timeout=70)
 r.raise_for_status()
 if not r.content:raise ValueError("Empty exchange response: "+url)
 return r.content

def market_udiff(data,target):
 x=pd.read_csv(io.BytesIO(data),low_memory=False)
 date=col(x,"TradDt","TRADE_DATE")
 sy=col(x,"TckrSymb","SYMBOL")
 ser=col(x,"SctySrs","SERIES")
 isin=col(x,"ISIN",optional=True)
 fields={"open":("OpnPric","OPEN_PRICE"),
         "high":("HghPric","HIGH_PRICE"),
         "low":("LwPric","LOW_PRICE"),
         "close":("ClsPric","CLOSE_PRICE"),
         "volume":("TtlTradgVol","TOTTRDQTY"),
         "turnover":("TtlTrfVal","TOTTRDVAL")}
 out=pd.DataFrame({"date":pd.to_datetime(x[date],dayfirst=True,errors="coerce").dt.normalize(),
       "symbol":x[sy].astype(str).str.upper().str.strip(),
       "series":x[ser].astype(str).str.upper().str.strip(),
       "isin":x[isin].astype(str).str.upper().str.strip() if isin else ""})
 for name,alternates in fields.items():
  out[name]=pd.to_numeric(x[col(x,*alternates)],errors="coerce")
 out=out[out["date"].eq(pd.Timestamp(target))&out["series"].isin(SERIES)]
 out=out.dropna(subset=["close","turnover","volume"])
 if out.duplicated(["symbol","series"]).any():raise ValueError("NSE UDiFF duplicated listed security")
 if len(out)<MIN_NSE_ROWS:raise ValueError("NSE primary day incomplete")
 return out

def market_full(data):
 x=pd.read_csv(io.BytesIO(data),skipinitialspace=True,low_memory=False)
 out=pd.DataFrame({"symbol":x[col(x,"SYMBOL")].astype(str).str.upper().str.strip(),
                   "series":x[col(x,"SERIES")].astype(str).str.upper().str.strip(),
                   "check_close":pd.to_numeric(x[col(x,"CLOSE_PRICE","LAST_PRICE")],errors="coerce")})
 return out[out["series"].isin(SERIES)].dropna(subset=["check_close"]).drop_duplicates(
     ["symbol","series"],keep="last")

def validate_market(primary,secondary):
 y=primary.merge(secondary,on=["symbol","series"],how="inner",validate="1:1")
 overlap=len(y)/len(primary) if len(primary) else 0
 if len(primary)<MIN_NSE_ROWS or len(y)<MIN_NSE_ROWS or overlap<MIN_SECONDARY_OVERLAP:
  raise ValueError(f"Two NSE source overlap failed: {len(y)}/{len(primary)}")
 absdiff=(y["close"]-y["check_close"]).abs()
 med=float(absdiff.median());p99=float(absdiff.quantile(.99))
 if med>.01 or p99>.10 or float(absdiff.max())>.10:
  raise ValueError(f"Official NSE two-source prices disagree: median={med}, p99={p99}")
 return {"official_udiff_rows":len(primary),"official_overlap_rows":len(y),
         "official_overlap_fraction":overlap,"close_median_abs_diff":med,
         "close_p99_abs_diff":p99,"close_max_abs_diff":float(absdiff.max())}

def exchange_primary_for_day(target):
 target=pd.Timestamp(target).normalize()
 ymd=target.strftime("%Y%m%d")
 dmy=target.strftime("%d%m%Y")
 u=f"{NSE_U}/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
 f=f"{NSE_F}/sec_bhavdata_full_{dmy}.csv"
 zb=get_bytes(u)
 with zipfile.ZipFile(io.BytesIO(zb)) as zipdata:
  names=[n for n in zipdata.namelist() if n.lower().endswith(".csv")]
  if len(names)!=1:raise ValueError("Unexpected official NSE UDiFF archive files")
  csv=zipdata.read(names[0])
 fb=get_bytes(f)
 p=market_udiff(csv,target)
 v=validate_market(p,market_full(fb))
 v.update({"official_primary_url":u,"official_secondary_url":f,
           "official_udiff_zip_sha256":sha(zb),"official_full_csv_sha256":sha(fb),
           "official_primary_used":True})
 return p,v

def hf_source(repo_path):
 path=Path(hf_hub_download(HF_MARKET,repo_path,repo_type="dataset"))
 return path,sha(path.read_bytes())

def adjustment_factors(history,actions,target):
 # All split/bonus factors with ex_date AFTER a historical day, but
 # ON/BEFORE the current signal date; no future/unannounced actions.
 history=history.sort_values(["symbol","date"]).copy()
 history["adj_close"]=history["close"].astype(float)
 if len(actions)==0:return history
 for symbol,g in actions.groupby("symbol"):
  ids=history.index[history["symbol"].eq(symbol)]
  if not len(ids):continue
  dates=history.loc[ids,"date"]
  mult=np.ones(len(ids),dtype=float)
  for r in g.itertuples(index=False):
   if r.ex_date>target:continue
   mult[dates<r.ex_date]*=r.factor
  history.loc[ids,"adj_close"]=history.loc[ids,"close"].astype(float).to_numpy()*mult
 return history

def verified_corporate_actions(paths,year0,target):
 tables=[];audit=[]
 for year in range(year0,target.year+1):
  path,digest=hf_source(f"actions/nse_{year}.parquet")
  x=pd.read_parquet(path)
  req={"symbol","ex_date","type"}
  if not req.issubset(x):raise ValueError(f"Corporate-action history missing required columns year={year}")
  x["ex_date"]=pd.to_datetime(x["ex_date"],errors="coerce").dt.normalize()
  x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
  x["type"]=x["type"].astype(str).str.lower()
  x=x[x["type"].isin(["split","bonus"]) & x["ex_date"].notna() &
       x["ex_date"].le(pd.Timestamp(target))].copy()
  factors=[]
  for row in x.to_dict("records"):
   fac=np.nan
   if row["type"]=="split":
    try:fac=float(row.get("face_value_to"))/float(row.get("face_value_from"))
    except (TypeError,ValueError,ZeroDivisionError):pass
   else:
    try:
     n=float(row.get("ratio_num"));d=float(row.get("ratio_den"))
     fac=d/(d+n) if (n+d)>0 else np.nan
    except (TypeError,ValueError):pass
   factors.append(fac)
  x["factor"]=factors
  incomplete=x["factor"].isna().sum()
  if incomplete:raise ValueError(f"Corporate actions have {incomplete} unverified split or bonus factors year={year}")
  x=x[x["factor"].between(.001,1.0)].copy()
  if len(x):tables.append(x[["symbol","ex_date","factor"]])
  audit.append({"year":year,"verified_action_source_sha256":digest,
                "split_bonus_actions":len(x)})
 actions=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame(
   columns=["symbol","ex_date","factor"])
 if actions.duplicated(["symbol","ex_date","factor"]).any():
  actions=actions.drop_duplicates(["symbol","ex_date","factor"])
 return actions,audit

def market_history(official,target):
 # At least 252 sessions of history, with official current-day close. Use
 # only original market OHLCV and documented ex-date split/bonus adjustments.
 target=pd.Timestamp(target).normalize()
 source_hashes={}
 frames=[]
 for year in range(max(2010,target.year-2),target.year+1):
  path,digest=hf_source(f"nse/year={year}/nse_{year}.parquet")
  x=pd.read_parquet(path,columns=["date","symbol","series","isin","close","volume","turnover"])
  x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
  x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
  x["series"]=x["series"].astype(str).str.upper().str.strip()
  frames.append(x)
  source_hashes[str(year)]=digest
 history=pd.concat(frames,ignore_index=True)
 history=history[history["series"].isin(SERIES)&history["date"].le(target)].copy()
 official=official[["date","symbol","series","isin","close","volume","turnover"]]
 history=history[~history["date"].eq(target)]
 history=pd.concat([history,official],ignore_index=True)
 history=history.sort_values(["symbol","date","series"]).drop_duplicates(
     ["date","symbol"],keep="first")
 history=history[history["isin"].isna()|history["isin"].astype(str).str.startswith("INE")].copy()
 if len(history.loc[history["date"].eq(target)])<MIN_UNIQUE_MARKET_CANDIDATES:
  raise ValueError("Too few corporate equities at official NSE target date")
 if history.duplicated(["date","symbol"]).any():raise ValueError("Ambiguous company/date quote after series resolution")
 # HF archive may be late; do not make current momentum using a 2-week gap.
 recent=history[(history["date"]>=target-pd.Timedelta(days=25)) &
                (history["date"]<target)]
 daily=recent.groupby("date")["symbol"].count()
 if (daily>=MIN_NSE_ROWS).sum()<10:
  raise ValueError("HF cash-market history too stale/sparse for 20d/60d features")
 action_year0=target.year-2
 acts,action_audit=verified_corporate_actions(None,action_year0,target)
 history=adjustment_factors(history,acts,target)
 return history,{"HF_historical_market_file_SHA256":source_hashes,
                 "corporate_action_source_audit":action_audit,
                 "historical_rows":len(history),
                 "sufficient_recent_historical_market_days":int((daily>=MIN_NSE_ROWS).sum())}

def rsi_wilder_last(prices,period=14):
 # Standard RSI(14): 14 changes SMA seed; Wilder recursive smoothing
 # thereafter. No future bars, and flat prices score neutral 50.
 x=pd.to_numeric(pd.Series(prices),errors="coerce")
 if len(x)<period+1 or x.isna().any() or (x<=0).any():return float("nan")
 d=x.diff().iloc[1:]
 gains=d.clip(lower=0).to_numpy(float)
 losses=(-d.clip(upper=0)).to_numpy(float)
 g=float(gains[:period].mean());l=float(losses[:period].mean())
 for i in range(period,len(gains)):
  g=(g*(period-1)+gains[i])/period
  l=(l*(period-1)+losses[i])/period
 if l<=1e-14:
  return 100. if g>1e-14 else 50.
 rs=g/l
 return float(100-100/(1+rs))

def market_features(history,target,min_company_rows=MIN_UNIQUE_MARKET_CANDIDATES):
 out=[]
 for symbol,g in history.groupby("symbol",sort=False):
  g=g.sort_values("date")
  if pd.Timestamp(g["date"].iloc[-1])!=pd.Timestamp(target):continue
  if len(g)<252:continue
  x=g["adj_close"].astype(float)
  v=g["volume"].astype(float)
  t=g["turnover"].astype(float)
  ret=x.pct_change(fill_method=None)
  r20=x/x.shift(20)-1
  r60=x/x.shift(60)-1
  r120=x/x.shift(120)-1
  r252=x/x.shift(252)-1
  hi252=x.rolling(252,min_periods=126).max()
  lo252=x.rolling(252,min_periods=126).min()
  vz=v.rolling(20,min_periods=15).mean()/v.rolling(60,min_periods=40).mean()-1
  tz=t.rolling(20,min_periods=15).mean()/t.rolling(60,min_periods=40).mean()-1
  d={
    "date":pd.Timestamp(target),"symbol":symbol,
    "close":float(g["close"].iloc[-1]),
    "avg_turnover_63":float(t.rolling(63,min_periods=40).mean().iloc[-1]),
    "ret_20":float(r20.iloc[-1]),"ret_60":float(r60.iloc[-1]),
    "ret_120":float(r120.iloc[-1]),"ret_252":float(r252.iloc[-1]),
    "mom_accel":float((r20-r60/3).iloc[-1]),
    "vol_accel":float(vz.iloc[-1]),"turnover_accel":float(tz.iloc[-1]),
    "off_high_252":float((x/hi252-1).iloc[-1]),
    "above_low_252":float((x/lo252-1).iloc[-1]),
    "dma50_prev":float(x.shift(1).rolling(50,min_periods=40).mean().iloc[-1]),
    "dma200_prev":float(x.shift(1).rolling(200,min_periods=160).mean().iloc[-1]),
    "rsi14_wilder":rsi_wilder_last(x),
    "trend_consistency_60":float((ret>0).rolling(60,min_periods=40).mean().iloc[-1]),
    "volatility_60":float((ret.rolling(60,min_periods=40).std()*np.sqrt(252)).iloc[-1])
  }
  d["price_gt_dma50_prev"]=bool(d["adj_close"]>d["dma50_prev"]) if "adj_close" in d else bool(x.iloc[-1]>d["dma50_prev"])
  d["price_lt_dma200_prev"]=bool(x.iloc[-1]<d["dma200_prev"])
  d["up_from_52w_low"]=float((x/lo252-1).iloc[-1])
  d["down_from_52w_high"]=float((1-x/hi252).iloc[-1])
  d["rsi14_gt80"]=bool(d["rsi14_wilder"]>80)
  d["integrity_feature_clean"]=bool(np.isfinite([d[k] for k in PRICE_COLUMNS]).all() and
                    20<=d["close"]<=2000 and np.isfinite(d["avg_turnover_63"]) and
                    d["avg_turnover_63"]>0)
  out.append(d)
 result=pd.DataFrame(out)
 if len(result)<min_company_rows:
  raise ValueError(f"Fewer than 500 equities with at least 252 market sessions: {len(result)}")
 return result

def parse_nse_ts(raw):
 if raw is None or str(raw).strip() in {"","nan","None"}:return pd.NaT
 val=pd.to_datetime(raw,dayfirst=True,errors="coerce")
 if pd.isna(val):return pd.NaT
 if val.tzinfo is None:val=val.tz_localize("Asia/Kolkata")
 return val.tz_convert("UTC")

def first(record,*keys):
 for k in keys:
  v=record.get(k)
  if v is not None and str(v).strip() not in {"","None","nan"}:return v
 return None

def classify_event(subject,details):
 # High precision deterministic NSE headline metadata taxonomy.
 # Earnings positive or promoter purchases are NEVER inferred from a
 # generic announcement. Categories can overlap; preserve one event per
 # source ID AND per category, not arbitrary textual sentiment.
 text=(" "+str(subject)+" "+str(details)+" ").lower()
 text=re.sub(r"\s+"," ",text)
 labels=[]
 def present(*words):return any(w in text for w in words)
 if present("receipt of order","received order","order received","orders received",
            "award of contract","letter of award","work order received",
            "purchase order received","new order win","contract awarded"):
  labels.append(("order_win",1))
 if present("capacity expansion","commissioning of","commercial production",
            "production commenced","new manufacturing facility","new manufacturing plant",
            "commencement of operations","new production line","plant commissioning"):
  labels.append(("capacity_expansion",1))
 if present("regulatory approval","approval received","approval granted",
            "usfda approval","fda approval","drug approval","license granted"):
  labels.append(("regulatory",1))
 if present("warning letter","regulatory action","show cause notice","approval rejected"):
  labels.append(("regulatory",-1))
 if present("promoter acquisition","acquisition by promoter","promoter bought",
            "promoters purchased","purchase of shares by promoters"):
  labels.append(("promoter_activity",1))
 if present("merger","demerger","bonus issue","stock split","change in name",
            "name change","scheme of arrangement","consolidation of shares"):
  labels.append(("corporate_action",0))
 if present("qualified institutions placement","qip","preferential allotment",
            "rights issue","issue of equity shares","fund raising through equity"):
  labels.append(("dilution",-1))
 if present("share buyback","buy-back","buyback of shares","buy back of shares"):
  labels.append(("buyback",1))
 # Earnings releases without verified numbers do not constitute positive
 # earnings inflections: conservatively the feature stays zero.
 return list(dict.fromkeys(labels))

def collect_nse_events(target,sess=None):
 cutoff=fold_close(target)
 start=cutoff-pd.Timedelta(days=RECENT_DAYS)
 s=sess or requests.Session()
 r=s.get(NSE_PAGE,headers=HEADERS,timeout=45)
 r.raise_for_status()  # NSE anti-bot and rate limits fail closed.
 parts=[];ranges=[];d=pd.Timestamp(start.tz_convert("Asia/Kolkata").date())
 last=pd.Timestamp(target)
 while d<=last:
  end=min(d+pd.Timedelta(days=6),last)
  urlparams={"index":"equities","from_date":d.strftime("%d-%m-%Y"),
              "to_date":end.strftime("%d-%m-%Y")}
  r=s.get(NSE_API,params=urlparams,headers=HEADERS,timeout=90)
  r.raise_for_status()
  data=r.json()
  if isinstance(data,dict):data=data.get("data")
  if not isinstance(data,list):raise ValueError("NSE announcements API unexpected schema")
  ranges.append({"start":str(d.date()),"end":str(end.date()),"count":len(data)})
  parts.extend(data)
  d=end+pd.Timedelta(days=1)
 if len(parts)<200:raise ValueError("NSE 185-day corporate disclosures source appears incomplete")
 rows=[];ids=set();dates=[]
 for rec in parts:
  sym=str(first(rec,"symbol","sm_symbol") or "").upper().strip()
  if not sym:continue
  published=parse_nse_ts(first(rec,"an_dt","broadcastDateTime","sort_date","dt"))
  received=parse_nse_ts(first(rec,"broadcastDateTime","an_dt"))
  if pd.isna(published):continue
  avail=max(published,received) if pd.notna(received) else published
  dates.append(avail)
  if avail>cutoff or avail<start:continue
  eid=str(first(rec,"seq_id","csvName","orgid") or "")
  subject=str(first(rec,"desc","subject","SUBJECT") or "")
  detail=str(first(rec,"attchmntText","details","DETAILS") or "")
  # Do not duplicate repeated feeds with same source id.
  if not eid:eid=sha((sym+str(published)+subject).encode())
  if eid in ids:continue
  ids.add(eid)
  attach=first(rec,"attchmntFile","attachment","ATTACHMENT")
  url=urljoin("https://www.nseindia.com/",str(attach)) if attach else ""
  for typ,direction in classify_event(subject,detail):
   rows.append({"symbol":sym,"type":typ,"direction":direction,
                "available_utc":avail,"source_record_id":eid,
                "headline":subject[:240],"original_exchange_document_url":url})
 if not dates or min(dates)>cutoff-pd.Timedelta(days=155):
  raise ValueError("Original NSE announcements feed lacks 180-day span")
 return pd.DataFrame(rows,columns=[
  "symbol","type","direction","available_utc","source_record_id",
  "headline","original_exchange_document_url"]),{
  "raw_records_fetched":len(parts),"source_query_windows":ranges,
  "source_record_timestamps_min":str(min(dates)),
  "source_record_timestamps_max":str(max(dates)),
  "classified_catalysts":len(rows),
  "exchange_history_window_complete":True,
  "event_source_url":NSE_API}

def with_catalyst_features(market,events,target):
 cutoff=fold_close(target)
 res=market.copy()
 if len(events):
  ev=events.copy()
  ev["available_utc"]=pd.to_datetime(ev["available_utc"],utc=True,errors="coerce")
  if ev["available_utc"].isna().any() or (ev["available_utc"]>cutoff).any():
   raise ValueError("NSE post-close events in snapshot")
 else:ev=events
 for key in CATALYST_PREFIXES:
  match=re.fullmatch(r"nse_(.*)_(90|180)d(_positive)?",key)
  if match is None:raise ValueError("Unsupported frozen V11.4 catalyst key: "+key)
  kind,days,positive=match.groups()
  start=cutoff-pd.Timedelta(days=int(days))
  sub=ev[ev["type"].eq(kind)&(ev["available_utc"]>=start)&
         (ev["available_utc"]<=cutoff)] if len(ev) else ev
  if positive:sub=sub[sub["direction"]>0]
  count=sub.groupby("symbol").size() if len(sub) else pd.Series(dtype=int)
  res[key]=res["symbol"].map(count).fillna(0).astype("int32")
 res["historical_asof_utc"]=cutoff.isoformat()
 return res

def run(target,out):
 target=pd.Timestamp(target).normalize()
 now=pd.Timestamp.now(tz="Asia/Kolkata")
 if target.date()!=now.date():
  raise ValueError("Only CURRENT Indian market day can be sourced in live forward workflow")
 if now<target.tz_localize("Asia/Kolkata")+pd.Timedelta(hours=17):
  raise ValueError("NSE close not independently released yet; only run after 17:00 IST")
 p=Path(out);p.mkdir(parents=True,exist_ok=True)
 primary,marketproof=exchange_primary_for_day(target)
 history,historyproof=market_history(primary,target)
 m=market_features(history,target)
 events,eventproof=collect_nse_events(target)
 x=with_catalyst_features(m,events,target)
 if len(x)<500 or len(x.loc[x["integrity_feature_clean"]])<300:
  raise ValueError("Insufficient integrity-clean market universe; no shortlist")
 par=p/"v11_4_live_NSE_verified_stock_features.parquet"
 x.to_parquet(par,index=False)
 ep=p/"v11_4_NSE_filing_evidence_sample.csv"
 events.sort_values("available_utc",ascending=False).head(300).to_csv(ep,index=False)
 cutoff=fold_close(target)
 meta={
  "snapshot_date":str(target.date()),"time_created_utc":datetime.now(timezone.utc).isoformat(),
  "original_NSE_event_catalog_verified":True,"market_close_source_verified":True,
  "NSE_event_coverage_through_utc":cutoff.isoformat(),
  "market_data_coverage_through_utc":cutoff.isoformat(),
  "market_source_verified":marketproof,"history_verified":historyproof,
  "filing_source_verified":eventproof,"company_rows":len(x),
  "integrity_clean_company_rows":int(x["integrity_feature_clean"].sum()),
  "live_features_SHA256":sha(par.read_bytes()),
  "research_only_not_trade_at_recorded_close":True}
 (p/"v11_4_live_source_validation.json").write_text(json.dumps(meta,indent=2))
 print(json.dumps({k:meta[k] for k in ("snapshot_date","company_rows",
                 "integrity_clean_company_rows","live_features_SHA256")},
                  indent=2),flush=True)
 return x,meta

def main():
 a=argparse.ArgumentParser()
 a.add_argument("--date",default=None)
 a.add_argument("--output",required=True)
 o=a.parse_args()
 target=o.date or str(pd.Timestamp.now(tz="Asia/Kolkata").date())
 run(target,o.output)

if __name__=="__main__":main()
