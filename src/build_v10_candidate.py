from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path):
    p=Path(path)
    return json.load(open(p)) if p.exists() else {}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-summary",required=True)
    ap.add_argument("--baseline-current",required=True)
    ap.add_argument("--v95-summary",required=True)
    ap.add_argument("--v96b-summary",required=True)
    ap.add_argument("--v97-summary",required=True)
    ap.add_argument("--v98-path-summary",required=True)
    ap.add_argument("--v98-sector-summary",required=True)
    ap.add_argument("--v99-summary",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    outdir=Path(args.output)
    outdir.mkdir(parents=True,exist_ok=True)

    baseline=read_json(args.baseline_summary)
    v95=read_json(args.v95_summary)
    v96b=read_json(args.v96b_summary)
    v97=read_json(args.v97_summary)
    v98p=read_json(args.v98_path_summary)
    v98s=read_json(args.v98_sector_summary)
    v99=read_json(args.v99_summary)

    current=pd.read_csv(args.baseline_current)
    # Use the validated production selection column if present.
    sel_col=None
    for c in ["selected_v941","selected_v931","selected"]:
        if c in current.columns:
            sel_col=c
            break
    if sel_col is None:
        raise SystemExit("No validated selection column in baseline current_selection.csv")

    # Every non-baseline component must have an explicit promotion gate.
    fund_gate=bool(v95.get("production_fundamental_gate",{}).get("passed",False))
    fund_weight=float(v95.get("production_applied_fundamental_weight",0.0) or 0.0)
    regime_gate=bool(v96b.get("production_gate",False))
    event_gate=bool(v97.get("production_gate",False))
    path_gate=bool(v98p.get("production_gate",False))
    sector_gate=bool(v98s.get("production_gate",False))

    # V10.0 does not invent any new weights. It only admits already-promoted
    # components. Current component results all resolve to zero contribution.
    applied={
        "technical_v9_4_1":1.0,
        "fundamental_v9_5":fund_weight if fund_gate else 0.0,
        "regime_v9_6b":0.0 if not regime_gate else None,
        "events_v9_7":0.0 if not event_gate else None,
        "path_v9_8":0.0 if not path_gate else None,
        "sector_v9_8":0.0 if not sector_gate else None,
    }

    unsupported=[k for k,v in applied.items() if v is None]
    if unsupported:
        raise SystemExit(
            "A challenger passed its promotion gate but V10.0 has no current-signal "
            f"integration implementation for: {unsupported}. Refuse silent fallback."
        )

    current["selected_v10"]=current[sel_col].astype(bool)
    current["v10_rank"]=np.nan
    selected=current[current["selected_v10"]].copy()

    rank_col=None
    for c in ["selection_rank_v941","selection_rank_v931","selection_rank","rank"]:
        if c in selected.columns:
            rank_col=c
            break
    if rank_col is not None:
        current.loc[current["selected_v10"],"v10_rank"]=pd.to_numeric(
            current.loc[current["selected_v10"],rank_col],errors="coerce"
        )
    else:
        score_col=None
        for c in ["selection_score_v941","selection_score_v931","selection_score","p_cal"]:
            if c in selected.columns:
                score_col=c
                break
        if score_col is None:
            raise SystemExit("No ranking/score column for current baseline selection")
        ranks=selected[score_col].rank(method="first",ascending=False)
        current.loc[selected.index,"v10_rank"]=ranks

    current["v10_status"]="validated_fallback_integrated_candidate"
    current["v10_fundamental_weight"]=applied["fundamental_v9_5"]
    current["v10_regime_weight"]=applied["regime_v9_6b"]
    current["v10_event_weight"]=applied["events_v9_7"]
    current["v10_path_weight"]=applied["path_v9_8"]
    current["v10_sector_weight"]=applied["sector_v9_8"]

    top=current[current["selected_v10"]].sort_values("v10_rank")
    current.to_csv(outdir/"current_selection_v10.csv",index=False)
    top.to_csv(outdir/"current_top10_v10.csv",index=False)

    manifest={
        "model":"V10.0 integrated candidate",
        "policy":"admit only already-promoted forward-validated components; exact V9.4.1 fallback otherwise",
        "component_status":{
            "V9.4.1":{"promoted":True,"weight":1.0},
            "V9.5_fundamental":{
                "standalone_gate_passed":fund_gate,
                "production_weight":applied["fundamental_v9_5"],
                "note":"fixed linear blend not promoted; adaptive use reserved for V10.1",
            },
            "V9.6B_regime":{
                "production_gate":regime_gate,
                "weight":applied["regime_v9_6b"],
            },
            "V9.7_events":{
                "production_gate":event_gate,
                "weight":applied["events_v9_7"],
            },
            "V9.8_path":{
                "production_gate":path_gate,
                "weight":applied["path_v9_8"],
            },
            "V9.8_sector":{
                "production_gate":sector_gate,
                "weight":applied["sector_v9_8"],
            },
        },
        "robustness_v9_9":v99,
        "test_run_ready":True,
        "test_run_scope":"paper/simulation test only; no claim of guaranteed future returns",
        "selected_count":int(current["selected_v10"].sum()),
    }
    json.dump(manifest,open(outdir/"manifest.json","w"),indent=2,default=str)
    print(json.dumps(manifest,indent=2,default=str))
    cols=[c for c in ["v10_rank","symbol","close","p_cal","p_dd30","model_dispersion"] if c in top.columns]
    print(top[cols].head(20).to_string(index=False))


if __name__=="__main__":
    main()
