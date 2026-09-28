import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

IN=Path("results")
OUT=Path("diagnostics_out"); OUT.mkdir(exist_ok=True)
snap=pd.read_parquet(IN/"snapshot_dataset.parquet")
oos=pd.read_parquet(IN/"oos_predictions.parquet")
cur=pd.read_csv(IN/"current_predictions.csv")
top=pd.read_csv(IN/"topk_by_fold.csv")

summary={}
rates=[]
for y in ["y6","y12","y24","dd30_6m"]:
    d=snap[y].dropna()
    rec={"label":y,"n":int(len(d)),"positives":int(d.sum()),"rate":float(d.mean())}
    rates.append(rec); summary[f"{y}_all_mature_rate"]=rec["rate"]
pd.DataFrame(rates).to_csv(OUT/"horizon_base_rates.csv",index=False)

m=snap.dropna(subset=["y24"]).copy()
w=m[m.y24==1]
summary["mature_24m_n"]=int(len(m)); summary["winners_24m_n"]=int(len(w))
summary["winners_fast_6m_share"]=float((w.y6==1).mean()) if len(w) else None
summary["winners_6to12m_share"]=float(((w.y6==0)&(w.y12==1)).mean()) if len(w) else None
summary["winners_12to24m_share"]=float(((w.y12==0)&(w.y24==1)).mean()) if len(w) else None
summary["median_days_to_2x_24m_winners"]=float(w.days_to_2x.dropna().median()) if len(w.days_to_2x.dropna()) else None

comp=[]
evalmask=oos.y6.notna()
for c in ["p_struct","p_elastic","p_gbm","p_horizon","p_raw","p_cal"]:
    d=oos.loc[evalmask & oos[c].notna(),["y6",c]]
    if len(d) and d.y6.nunique()==2:
        comp.append({"component":c,"n":int(len(d)),"positive_rate":float(d.y6.mean()),
                     "pr_auc":float(average_precision_score(d.y6,d[c])),
                     "roc_auc":float(roc_auc_score(d.y6,d[c])),
                     "brier":float(brier_score_loss(d.y6,np.clip(d[c],1e-6,1-1e-6)))})
pd.DataFrame(comp).to_csv(OUT/"component_metrics.csv",index=False)

d=oos.dropna(subset=["y6","p_cal"]).copy()
d["prob_decile"]=pd.qcut(d.p_cal,10,labels=False,duplicates="drop")
rel=d.groupby("prob_decile").agg(n=("y6","size"),predicted=("p_cal","mean"),actual=("y6","mean"),
                                  min_p=("p_cal","min"),max_p=("p_cal","max")).reset_index()
rel.to_csv(OUT/"reliability_deciles.csv",index=False)

tails=[]
for dt,g in d.groupby("date"):
    base=float(g.y6.mean())
    g=g.sort_values("p_cal",ascending=False)
    for pct in [0.01,0.05,0.10]:
        n=max(1,int(np.ceil(len(g)*pct))); q=g.head(n)
        tails.append({"date":dt,"pct":pct,"n":n,"base_rate":base,"precision":float(q.y6.mean()),
                      "lift":float(q.y6.mean()/base) if base>0 else np.nan})
taildf=pd.DataFrame(tails); taildf.to_csv(OUT/"top_tail_by_fold.csv",index=False)
for pct in [0.01,0.05,0.10]:
    q=taildf[taildf.pct==pct]
    summary[f"top_{int(pct*100)}pct_precision_median"]=float(q.precision.median())
    summary[f"top_{int(pct*100)}pct_lift_median"]=float(q.lift.median())

for k in [5,10,20]:
    q=top[top.k==k].sort_values("date")
    recent=q.tail(10)
    summary[f"recent10_precision_at_{k}_mean"]=float(recent.precision.mean())
    summary[f"recent10_lift_at_{k}_median"]=float(recent.lift.median())

cur["isin_prefix"]=cur.isin.astype(str).str[:3]
non_eq=cur[cur.isin_prefix!="INE"].copy()
non_eq.to_csv(OUT/"current_non_company_securities.csv",index=False)
summary["current_eligible_rows_before_security_filter"]=int(len(cur))
summary["current_non_INE_count"]=int(len(non_eq))
summary["current_INE_count"]=int((cur.isin_prefix=="INE").sum())

known=["BONDADA","SHANTIGOLD","LOHIA","SRM","RAJOOENG","ELLEN","DDEVPLSTIK","TRANSRAILL","ARROWGREEN","GKENERGY","KPIGREEN","GAYAPROJ"]
cur[cur.symbol.isin(known)].to_csv(OUT/"known_candidates.csv",index=False)

json.dump(summary,open(OUT/"diagnostics.json","w"),indent=2)
print(json.dumps(summary,indent=2))
