from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_5 as v95
import calibrate_v9_3_1 as v931


def spec_from_row(row):
    return {
        "name": row.config,
        "pool_n": int(row.pool_n),
        "risk_drop": float(row.risk_drop),
        "w_safety": float(row.w_safety),
        "w_consensus": float(row.w_consensus),
        "baseline": bool(row.fell_back_to_v92),
    }


def technical_survivors(g: pd.DataFrame, spec: dict, k: int) -> pd.DataFrame:
    q = g.dropna(subset=["p_cal", "p_dd30_cal", "model_dispersion"]).copy()
    if len(q) < k:
        return pd.DataFrame()

    if spec.get("baseline", False):
        q["selection_score"] = q["p_cal"]
        return q.sort_values(
            ["selection_score", "p_cal", "model_dispersion"],
            ascending=[False, False, True],
        )

    pool_n = min(int(spec["pool_n"]), len(q))
    pool = q.sort_values(
        ["p_cal", "model_dispersion"], ascending=[False, True]
    ).head(pool_n).copy()
    if len(pool) < k:
        return pd.DataFrame()

    rd = float(spec["risk_drop"])
    if rd > 0:
        cutoff = float(pool["p_dd30_cal"].quantile(1.0 - rd))
        surv = pool[pool["p_dd30_cal"] <= cutoff].copy()
    else:
        surv = pool.copy()
    if len(surv) < k:
        return pd.DataFrame()

    surv["comp_alpha"] = surv["p_cal"].rank(pct=True, method="average")
    surv["comp_safety"] = surv["p_dd30_cal"].rank(
        pct=True, method="average", ascending=False
    )
    surv["comp_consensus"] = surv["model_dispersion"].rank(
        pct=True, method="average", ascending=False
    )
    ws = float(spec["w_safety"])
    wc = float(spec["w_consensus"])
    wa = max(0.0, 1.0 - ws - wc)
    surv["selection_score"] = (
        wa * surv["comp_alpha"]
        + ws * surv["comp_safety"]
        + wc * surv["comp_consensus"]
    )
    return surv.sort_values(
        ["selection_score", "p_cal", "model_dispersion"],
        ascending=[False, False, True],
    )


def policy_grid():
    yield {
        "name": "baseline",
        "veto_q": 0.0,
        "confirm_q": 1.0,
        "confirm_bonus": 0.0,
    }
    for veto in [0.10, 0.20, 0.30]:
        yield {
            "name": f"veto{int(veto*100)}",
            "veto_q": veto,
            "confirm_q": 1.0,
            "confirm_bonus": 0.0,
        }
    for q in [0.60, 0.70, 0.80]:
        for bonus in [0.02, 0.05, 0.10]:
            yield {
                "name": f"confirm{int(q*100)}_b{int(bonus*100)}",
                "veto_q": 0.0,
                "confirm_q": q,
                "confirm_bonus": bonus,
            }
    for veto in [0.10, 0.20]:
        for q in [0.70, 0.80]:
            for bonus in [0.02, 0.05]:
                yield {
                    "name": (
                        f"veto{int(veto*100)}_"
                        f"confirm{int(q*100)}_b{int(bonus*100)}"
                    ),
                    "veto_q": veto,
                    "confirm_q": q,
                    "confirm_bonus": bonus,
                }


def coverage(surv: pd.DataFrame) -> dict:
    if surv.empty:
        return {"survivors": 0, "covered": 0, "coverage": 0.0}
    covered = surv["p_fund_cal"].notna()
    return {
        "survivors": int(len(surv)),
        "covered": int(covered.sum()),
        "coverage": float(covered.mean()),
    }


def select_policy(g: pd.DataFrame, spec: dict, policy: dict, cfg: dict):
    k = int(cfg.get("selection_k", 10))
    surv = technical_survivors(g, spec, k)
    if len(surv) < k:
        return pd.DataFrame(), False, coverage(surv)

    cov = coverage(surv)
    min_cov = float(cfg.get("fund_min_candidate_coverage", 0.70))
    if (
        policy["name"] == "baseline"
        or cov["coverage"] < min_cov
        or cov["covered"] < k
    ):
        return surv.head(k).copy(), False, cov

    pool = surv.copy()
    covered = pool["p_fund_cal"].notna()
    pool["fund_rank"] = np.nan
    if covered.any():
        pool.loc[covered, "fund_rank"] = pool.loc[
            covered, "p_fund_cal"
        ].rank(pct=True, method="average")
    pool["fund_rank_neutral"] = pool["fund_rank"].fillna(0.5)

    veto_q = float(policy.get("veto_q", 0.0))
    if veto_q > 0:
        keep = (~covered) | (pool["fund_rank"] > veto_q)
        candidate = pool[keep].copy()
        if len(candidate) < k:
            return surv.head(k).copy(), False, cov
        pool = candidate

    pool["overlay_score"] = pool["selection_score"]
    confirm_q = float(policy.get("confirm_q", 1.0))
    bonus = float(policy.get("confirm_bonus", 0.0))
    if bonus > 0 and confirm_q < 1.0:
        strength = np.clip(
            (pool["fund_rank_neutral"] - confirm_q)
            / max(1e-9, 1.0 - confirm_q),
            0.0,
            1.0,
        )
        pool["overlay_score"] = pool["selection_score"] + bonus * strength

    sel = pool.sort_values(
        ["overlay_score", "selection_score", "p_cal", "model_dispersion"],
        ascending=[False, False, False, True],
    ).head(k).copy()
    return sel, True, cov


def fold_metric(g, spec, policy, cfg):
    q = g.dropna(
        subset=["y6", "dd30_6m", "p_cal", "p_dd30_cal", "model_dispersion"]
    ).copy()
    k = int(cfg.get("selection_k", 10))
    if len(q) < max(k, 20):
        return None
    sel, applied, cov = select_policy(q, spec, policy, cfg)
    if len(sel) != k:
        return None
    base_rate = float(q["y6"].mean())
    precision = float(sel["y6"].mean())
    return {
        "precision_2x": precision,
        "lift_2x": precision / base_rate if base_rate > 0 else np.nan,
        "hit": float(precision > 0),
        "dd30_rate": float(sel["dd30_6m"].mean()),
        "coverage": cov["coverage"],
        "applied": float(applied),
    }


def eligible_fold(g, spec, cfg):
    k = int(cfg.get("selection_k", 10))
    surv = technical_survivors(g, spec, k)
    cov = coverage(surv)
    return (
        len(surv) >= k
        and cov["coverage"] >= float(cfg.get("fund_min_candidate_coverage", 0.70))
        and cov["covered"] >= k
    )


def eval_policy(hist, chosen_map, policy, cfg, eligible_only=True):
    rows = []
    for td in sorted(chosen_map):
        g = hist[hist["date"] == td].copy()
        if g.empty:
            continue
        spec = spec_from_row(chosen_map[td])
        if eligible_only and not eligible_fold(g, spec, cfg):
            continue
        m = fold_metric(g, spec, policy, cfg)
        if m is not None:
            rows.append({"date": td, **m})
    return pd.DataFrame(rows)


def aggregate(t: pd.DataFrame):
    if t.empty:
        return None
    lift = np.clip(
        t["lift_2x"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float),
        0, 10,
    )
    return {
        "folds": int(t["date"].nunique()),
        "mean_precision_2x": float(t["precision_2x"].mean()),
        "median_precision_2x": float(t["precision_2x"].median()),
        "mean_capped_lift_2x": float(np.mean(lift)) if len(lift) else np.nan,
        "hit_fold_rate": float(t["hit"].mean()),
        "hit_folds": int(t["hit"].sum()),
        "mean_dd30_rate": float(t["dd30_rate"].mean()),
        "median_dd30_rate": float(t["dd30_rate"].median()),
        "mean_coverage": float(t["coverage"].mean()),
        "applied_folds": int(t["applied"].sum()),
    }


def retention(candidate, baseline, cfg):
    r = float(cfg.get("fund_alpha_retention", 0.95))
    fields = [
        "mean_precision_2x",
        "mean_capped_lift_2x",
        "hit_fold_rate",
        "median_precision_2x",
    ]
    gates = {}
    ok = True
    for f in fields:
        actual = candidate.get(f, np.nan)
        required = r * baseline.get(f, np.nan)
        gates[f] = {"actual": actual, "required": required}
        if not (
            np.isfinite(actual)
            and np.isfinite(required)
            and actual + 1e-12 >= required
        ):
            ok = False
    return ok, gates


def utility(a, b):
    if a is None or b is None:
        return -np.inf
    ratios = []
    for f in ["mean_precision_2x", "mean_capped_lift_2x", "hit_fold_rate"]:
        av, bv = a.get(f, np.nan), b.get(f, np.nan)
        ratios.append(
            av / bv if np.isfinite(av) and np.isfinite(bv) and bv > 0 else 0.0
        )
    dd_a, dd_b = a.get("mean_dd30_rate", np.nan), b.get("mean_dd30_rate", np.nan)
    dd_gain = (
        (dd_b - dd_a) / dd_b
        if np.isfinite(dd_a) and np.isfinite(dd_b) and dd_b > 0
        else 0.0
    )
    return float(
        0.45 * ratios[0]
        + 0.35 * ratios[1]
        + 0.20 * ratios[2]
        + 0.10 * dd_gain
    )


def optimize_overlay(prior, chosen_map, cfg):
    gate = v95.fundamental_gate(prior, cfg)
    baseline_policy = next(policy_grid())
    base_t = eval_policy(prior, chosen_map, baseline_policy, cfg, eligible_only=True)
    base = aggregate(base_t)
    min_folds = int(cfg.get("fund_overlay_min_folds", 8))
    if (
        not gate["passed"]
        or base is None
        or base["folds"] < min_folds
    ):
        return baseline_policy, pd.DataFrame(), base, gate

    base_u = utility(base, base)
    min_gain = float(cfg.get("fund_overlay_min_utility_gain", 0.02))
    rows = []
    best = baseline_policy
    best_u = base_u

    for p in policy_grid():
        t = eval_policy(prior, chosen_map, p, cfg, eligible_only=True)
        a = aggregate(t)
        if a is None or a["folds"] < min_folds:
            continue
        feasible, gates = retention(a, base, cfg)
        u = utility(a, base)
        if p["name"] == "baseline":
            feasible = True
        row = {
            **p,
            **a,
            "feasible": bool(feasible),
            "utility": u,
            "utility_gain": float(u - base_u),
            "required_precision": gates.get("mean_precision_2x", {}).get("required", np.nan),
            "required_lift": gates.get("mean_capped_lift_2x", {}).get("required", np.nan),
            "required_hit": gates.get("hit_fold_rate", {}).get("required", np.nan),
            "required_median_precision": gates.get("median_precision_2x", {}).get("required", np.nan),
        }
        rows.append(row)
        if (
            p["name"] != "baseline"
            and feasible
            and u >= base_u + min_gain
            and u > best_u
        ):
            best = p
            best_u = u

    tab = pd.DataFrame(rows)
    if len(tab):
        tab = tab.sort_values(
            ["feasible", "utility", "mean_precision_2x"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return best, tab, base, gate


def forward_overlay(oos, chosen, cfg):
    out = oos.copy()
    out["selected_v95_overlay"] = False
    out["selection_rank_v95_overlay"] = np.nan
    out["overlay_policy"] = "baseline"
    out["overlay_applied"] = False
    records = []

    chosen["date"] = pd.to_datetime(chosen["date"])
    chosen_map = {
        pd.Timestamp(r.date): r
        for r in chosen.itertuples(index=False)
    }

    for td in sorted(chosen_map):
        g = out[out["date"] == td].copy()
        prior = out[out["date"] < td].copy()
        prior_map = {d:r for d,r in chosen_map.items() if d < td}
        policy, search, base, gate = optimize_overlay(
            prior, prior_map, cfg
        )
        spec = spec_from_row(chosen_map[td])
        sel, applied, cov = select_policy(g, spec, policy, cfg)
        if len(sel) != int(cfg.get("selection_k", 10)):
            policy = next(policy_grid())
            sel, applied, cov = select_policy(g, spec, policy, cfg)
        if len(sel) != int(cfg.get("selection_k", 10)):
            continue

        for rank, idx in enumerate(sel.index.tolist(), start=1):
            out.loc[idx, "selected_v95_overlay"] = True
            out.loc[idx, "selection_rank_v95_overlay"] = rank
        idxd = out.index[out["date"] == td]
        out.loc[idxd, "overlay_policy"] = policy["name"]
        out.loc[idxd, "overlay_applied"] = bool(applied and policy["name"] != "baseline")

        rec = {
            "date": td,
            "policy": policy["name"],
            "overlay_applied": bool(applied and policy["name"] != "baseline"),
            "prior_fund_gate": bool(gate["passed"]),
            "prior_fund_folds": int(prior.dropna(subset=["p_fund_cal"])["date"].nunique()),
            "current_coverage": cov["coverage"],
            "eligible_prior_folds": int(base["folds"]) if base else 0,
        }
        if len(search):
            z=search[search["name"]==policy["name"]]
            if len(z):
                rec["train_utility_gain"]=float(z.iloc[0]["utility_gain"])
                rec["train_precision"]=float(z.iloc[0]["mean_precision_2x"])
        records.append(rec)

    return out, pd.DataFrame(records)


def compare_forward(out, cfg):
    rows=[]
    k=int(cfg.get("selection_k",10))
    dates=sorted(pd.to_datetime(out.loc[out["selected_v95_overlay"],"date"].unique()))
    for td in dates:
        g=out[out["date"]==td].copy()
        q=g.dropna(subset=["y6","dd30_6m","p_cal","p_dd30_cal","model_dispersion"])
        if len(q)<max(k,20):
            continue
        br=float(q["y6"].mean())
        for name,mask in [
            ("overlay",g["selected_v95_overlay"]),
            ("baseline",g["selected_v931"]),
        ]:
            s=g[mask].copy()
            if len(s)!=k:
                continue
            p=float(s["y6"].mean())
            rows.append({
                "date":td,"strategy":name,
                "precision_2x":p,
                "lift_2x":p/br if br>0 else np.nan,
                "hit":float(p>0),
                "dd30_rate":float(s["dd30_6m"].mean()),
                "overlay_policy":str(g["overlay_policy"].iloc[0]) if name=="overlay" else "baseline",
                "overlay_applied":bool(g["overlay_applied"].iloc[0]) if name=="overlay" else False,
            })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--current",required=True)
    ap.add_argument("--summary",required=True)
    ap.add_argument("--config",default="config_v9_5.json")
    ap.add_argument("--output",default="outputs_v9_5_overlay")
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    outdir=Path(args.output)
    outdir.mkdir(parents=True,exist_ok=True)

    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931)
    chosen["date"]=pd.to_datetime(chosen["date"])

    # Diagnostic policy table across all currently eligible calibrated folds.
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}
    diag_rows=[]
    base_t=eval_policy(oos,cmap,next(policy_grid()),cfg,eligible_only=True)
    base=aggregate(base_t)
    for p in policy_grid():
        t=eval_policy(oos,cmap,p,cfg,eligible_only=True)
        a=aggregate(t)
        if a is None or base is None:
            continue
        feasible,gates=retention(a,base,cfg)
        u=utility(a,base)
        diag_rows.append({
            **p,**a,
            "feasible":bool(feasible),
            "utility":u,
            "utility_gain":float(u-utility(base,base)),
        })
    diag=pd.DataFrame(diag_rows).sort_values(
        ["feasible","utility","mean_precision_2x"],
        ascending=[False,False,False],
    )
    diag.to_csv(outdir/"overlay_policy_diagnostic.csv",index=False)

    # True forward-only historical chooser.
    forward, decisions=forward_overlay(oos,chosen,cfg)
    comp=compare_forward(forward,cfg)
    forward.to_parquet(outdir/"oos_overlay_predictions.parquet",index=False)
    decisions.to_csv(outdir/"overlay_decisions_by_fold.csv",index=False)
    comp.to_csv(outdir/"overlay_forward_metrics.csv",index=False)

    # Current decision uses all mature prior folds and the same prior-only optimizer.
    current=pd.read_csv(args.current)
    current["date"]=pd.to_datetime(current["date"])
    old_summary=json.load(open(args.summary))
    current_spec=old_summary["production_risk_config"]
    policy,search,base_prior,gate=optimize_overlay(oos,cmap,cfg)
    current_sel,applied,cov=select_policy(current,current_spec,policy,cfg)
    if len(current_sel)!=int(cfg.get("selection_k",10)):
        policy=next(policy_grid())
        current_sel,applied,cov=select_policy(current,current_spec,policy,cfg)
    search.to_csv(outdir/"current_overlay_search.csv",index=False)

    current["selected_v95_overlay"]=False
    current["selection_rank_v95_overlay"]=np.nan
    current["overlay_policy"]=policy["name"]
    current["overlay_applied"]=bool(applied and policy["name"]!="baseline")
    for rank,idx in enumerate(current_sel.index.tolist(),start=1):
        current.loc[idx,"selected_v95_overlay"]=True
        current.loc[idx,"selection_rank_v95_overlay"]=rank
    current[current["selected_v95_overlay"]].sort_values(
        "selection_rank_v95_overlay"
    ).to_csv(outdir/"current_top50.csv",index=False)

    active=decisions[decisions["overlay_applied"]] if len(decisions) else pd.DataFrame()
    summary={
        "experiment":"V9.5 conditional fundamental overlay",
        "production_calibration":"rank_platt_anchor_r8",
        "policy_types":[
            "bottom-fundamental veto inside risk survivors",
            "high-fundamental confirmation bonus inside risk survivors",
            "veto+confirmation hybrid",
        ],
        "leakage_policy":"each historical policy is selected only from strictly prior folds; current fold outcomes are never used; low coverage or failed prior fundamental gate falls back exactly to V9.4.1",
        "diagnostic_best":diag.iloc[0].to_dict() if len(diag) else None,
        "forward_active_folds":int(len(active)),
        "forward_decisions":decisions.to_dict(orient="records"),
        "current_policy":policy,
        "current_overlay_applied":bool(applied and policy["name"]!="baseline"),
        "current_coverage":cov,
        "prior_fundamental_gate":gate,
        "remaining_requirement":"promote overlay only if prior-fold search finds a non-baseline policy with >=95% alpha retention and >= configured utility gain",
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))
    if len(current_sel):
        cols=["symbol","close","p_cal","p_fund_cal","p_dd30","selection_score"]
        cols=[c for c in cols if c in current_sel.columns]
        print(current_sel[cols].head(20).to_string(index=False))


if __name__=="__main__":
    main()
