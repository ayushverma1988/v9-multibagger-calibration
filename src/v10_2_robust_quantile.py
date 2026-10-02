from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v10_2_compression_switch import SwitchPolicy, spec_from_row, select_switch, perturb_all
from v10_2_raw_rank_stability import outcome, agg, alpha_pass, select_policy, Policy as RawPolicy

BASE = SwitchPolicy("compress_030_raw50", .030, .50)
INC = RawPolicy("incumbent", "p_cal", "p_dd30_cal")


@dataclass(frozen=True)
class RobustPolicy:
    name: str
    rank_quantile: float
    internal_sims: int = 25


POLICIES = [
    RobustPolicy("rq60_i25", .60, 25),
    RobustPolicy("rq65_i25", .65, 25),
    RobustPolicy("rq70_i25", .70, 25),
    RobustPolicy("rq75_i25", .75, 25),
]


def seed_for(date, name, salt):
    raw = f"{pd.Timestamp(date).date()}|{name}|{salt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def base_select(g, spec, k=10):
    return select_switch(g, spec, BASE, k)


def robust_rank_table(g, spec, rp, k=10):
    base = base_select(g, spec, k)
    if len(base) != k:
        return base, pd.DataFrame()

    # CRN: every evaluation of a given fold/policy uses the same internal
    # perturbation stream. This removes Monte-Carlo noise when external
    # stability perturbations are compared with the unperturbed fold.
    rng = np.random.default_rng(seed_for(g["date"].iloc[0], rp.name, "internal_crn"))
    sim_lists = []
    universe = set(base["symbol"].astype(str))

    for _ in range(int(rp.internal_sims)):
        z = perturb_all(g, .01, rng)
        s = base_select(z, spec, k)
        if len(s) != k:
            continue
        syms = s["symbol"].astype(str).tolist()
        sim_lists.append(syms)
        universe.update(syms)

    if not sim_lists:
        return base, pd.DataFrame()

    base_order = {s: i for i, s in enumerate(base["symbol"].astype(str), 1)}
    gsym = g.copy()
    gsym["_sym"] = gsym["symbol"].astype(str)
    meta = gsym.drop_duplicates("_sym").set_index("_sym")

    rec = []
    for sym in universe:
        ranks = []
        selected = 0
        for syms in sim_lists:
            try:
                r = syms.index(sym) + 1
                selected += 1
            except ValueError:
                r = k + 1
            ranks.append(float(r))

        qrank = float(np.quantile(np.asarray(ranks), rp.rank_quantile, method="higher"))
        freq = float(selected / len(sim_lists))
        selected_ranks = [r for r in ranks if r <= k]
        mean_rank = float(np.mean(selected_ranks)) if selected_ranks else float(k + 1)
        row = meta.loc[sym] if sym in meta.index else None
        p_raw = float(row["p_raw"]) if row is not None and pd.notna(row.get("p_raw", np.nan)) else np.nan
        p_cal = float(row["p_cal"]) if row is not None and pd.notna(row.get("p_cal", np.nan)) else np.nan
        rec.append({
            "symbol": sym,
            "qrank": qrank,
            "freq": freq,
            "mean_rank": mean_rank,
            "base_rank": float(base_order.get(sym, k + 20)),
            "p_raw": p_raw,
            "p_cal": p_cal,
        })
    return base, pd.DataFrame(rec)


def robust_select(g, spec, rp, k=10):
    base, tab = robust_rank_table(g, spec, rp, k)
    if len(base) != k or tab.empty:
        return base

    # A certified member remains inside Top-K through at least the requested
    # upper rank quantile. This creates a label-free dead-band around the
    # membership boundary instead of forcing an unstable fixed-size core.
    core = tab[tab["qrank"] <= float(k)].copy()
    core = core.sort_values(
        ["qrank", "freq", "mean_rank", "base_rank", "p_raw", "symbol"],
        ascending=[True, False, True, True, False, True],
    )

    selected_syms = core["symbol"].astype(str).head(k).tolist()
    for sym in base["symbol"].astype(str):
        if sym not in selected_syms:
            selected_syms.append(sym)
        if len(selected_syms) >= k:
            break

    if len(selected_syms) < k:
        rest = tab[~tab["symbol"].astype(str).isin(selected_syms)].sort_values(
            ["qrank", "freq", "mean_rank", "p_raw", "symbol"],
            ascending=[True, False, True, False, True],
        )
        selected_syms.extend(rest["symbol"].astype(str).head(k - len(selected_syms)).tolist())

    order = {s: i for i, s in enumerate(selected_syms[:k])}
    out = g[g["symbol"].astype(str).isin(order)].copy()
    out["_robust_order"] = out["symbol"].astype(str).map(order)
    out = out.sort_values("_robust_order").head(k).copy()
    if len(out) != k:
        return base
    out["_robust_core_n"] = float(len(core))
    return out


def external_stability(g, spec, rp, k, sims):
    base = robust_select(g, spec, rp, k)
    if len(base) != k:
        return None
    bs = set(base["symbol"].astype(str))
    top = str(base.iloc[0]["symbol"])

    rng = np.random.default_rng(seed_for(g["date"].iloc[0], rp.name, "external"))
    js, tops, repl, cores = [], [], [], []
    for _ in range(int(sims)):
        z = perturb_all(g, .01, rng)
        s = robust_select(z, spec, rp, k)
        if len(s) != k:
            continue
        ss = set(s["symbol"].astype(str))
        js.append(len(bs & ss) / len(bs | ss))
        tops.append(float(str(s.iloc[0]["symbol"]) == top))
        repl.append(len(bs - ss) / k)
        cores.append(float(s["_robust_core_n"].iloc[0]) if "_robust_core_n" in s else np.nan)

    if not js:
        return None
    return {
        "mean_jaccard": float(np.mean(js)),
        "p05_jaccard": float(np.quantile(js, .05)),
        "top1_stability": float(np.mean(tops)),
        "names_replaced_fraction": float(np.mean(repl)),
        "mean_robust_core_n": float(np.nanmean(cores)) if len(cores) else np.nan,
    }


def incumbent_metrics(oos, cmap, cfg):
    k = int(cfg.get("selection_k", 10))
    rows = []
    for td, r in cmap.items():
        g = oos[oos["date"] == td].copy()
        if g.empty:
            continue
        s = select_policy(g, spec_from_row(r), INC, k)
        m = outcome(g, s, k)
        if m:
            rows.append({"date": td, **m})
    return agg(pd.DataFrame(rows))


def evaluate(oos, cmap, cfg, rp, sims):
    k = int(cfg.get("selection_k", 10))
    mets, sts = [], []
    for td, r in cmap.items():
        g = oos[oos["date"] == td].copy()
        if g.empty:
            continue
        spec = spec_from_row(r)
        s = robust_select(g, spec, rp, k)
        m = outcome(g, s, k)
        if m:
            mets.append({"date": td, **m})
        st = external_stability(g, spec, rp, k, sims)
        if st:
            sts.append({"date": td, **st})
    return pd.DataFrame(mets), pd.DataFrame(sts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True)
    ap.add_argument("--chosen931", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--grid-sims", type=int, default=100)
    ap.add_argument("--confirm-sims", type=int, default=500)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    oos = pd.read_parquet(args.oos)
    oos["date"] = pd.to_datetime(oos["date"])
    chosen = pd.read_csv(args.chosen931)
    chosen["date"] = pd.to_datetime(chosen["date"])
    cmap = {pd.Timestamp(r.date): r for r in chosen.itertuples(index=False)}
    baseline = incumbent_metrics(oos, cmap, cfg)

    rows = []
    for rp in POLICIES:
        mt, st = evaluate(oos, cmap, cfg, rp, args.grid_sims)
        a = agg(mt)
        if not a or st.empty:
            continue
        mj = float(st["mean_jaccard"].mean())
        wp = float(st["p05_jaccard"].min())
        t1 = float(st["top1_stability"].mean())
        apass = alpha_pass(a, baseline, .95)
        dd = a["mean_dd30_rate"] <= baseline["mean_dd30_rate"] + .03
        rows.append({
            **asdict(rp), **a,
            "mean_jaccard": mj,
            "worst_p05_jaccard": wp,
            "mean_top1_stability": t1,
            "mean_names_replaced_fraction": float(st["names_replaced_fraction"].mean()),
            "mean_robust_core_n": float(st["mean_robust_core_n"].mean()),
            "worst_fold": str(st.loc[st["p05_jaccard"].idxmin(), "date"]),
            "alpha_retention_passed": bool(apass),
            "stability_passed": bool(mj >= .80 and wp >= .60 and t1 >= .70),
            "dd_not_worse_3pp": bool(dd),
        })

    table = pd.DataFrame(rows)
    table["all_gates_passed"] = (
        table["alpha_retention_passed"] &
        table["stability_passed"] &
        table["dd_not_worse_3pp"]
    )
    table = table.sort_values(
        ["stability_passed", "worst_p05_jaccard", "mean_jaccard", "rank_quantile"],
        ascending=[False, False, False, False],
    )

    # Policy choice is stability-only and label-free. Among policies clearing
    # stability, prefer the strictest quantile (smallest intervention).
    stable = table[table["stability_passed"]]
    if len(stable):
        chosen_name = str(stable.sort_values(
            ["internal_sims", "rank_quantile", "mean_jaccard"],
            ascending=[True, False, False],
        ).iloc[0]["name"])
    else:
        chosen_name = str(table.iloc[0]["name"]) if len(table) else None
    rp = next((p for p in POLICIES if p.name == chosen_name), None)

    confirmation = None
    if rp:
        mt, st = evaluate(oos, cmap, cfg, rp, args.confirm_sims)
        a = agg(mt)
        mj = float(st["mean_jaccard"].mean())
        wp = float(st["p05_jaccard"].min())
        t1 = float(st["top1_stability"].mean())
        confirmation = {
            **a,
            "mean_jaccard": mj,
            "worst_p05_jaccard": wp,
            "mean_top1_stability": t1,
            "mean_names_replaced_fraction": float(st["names_replaced_fraction"].mean()),
            "mean_robust_core_n": float(st["mean_robust_core_n"].mean()),
            "worst_fold": str(st.loc[st["p05_jaccard"].idxmin(), "date"]),
            "alpha_retention_passed": bool(alpha_pass(a, baseline, .95)),
            "mean_jaccard_passed": bool(mj >= .80),
            "worst_p05_passed": bool(wp >= .60),
            "top1_passed": bool(t1 >= .70),
            "dd_not_worse_3pp": bool(a["mean_dd30_rate"] <= baseline["mean_dd30_rate"] + .03),
        }
        confirmation["all_gates_passed"] = bool(
            confirmation["alpha_retention_passed"] and
            confirmation["mean_jaccard_passed"] and
            confirmation["worst_p05_passed"] and
            confirmation["top1_passed"] and
            confirmation["dd_not_worse_3pp"]
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "robust_quantile_grid.csv", index=False)
    if rp:
        mt, st = evaluate(oos, cmap, cfg, rp, args.confirm_sims)
        mt.to_csv(out / "chosen_metrics_by_fold.csv", index=False)
        st.to_csv(out / "chosen_stability_by_fold.csv", index=False)

    summary = {
        "model": "V10.2 robust rank-quantile stability selector",
        "principle": "CRN perturbation rank quantiles certify only repeatedly selected candidates; no labels used for membership or policy choice",
        "base_selector": "compress_030_raw50",
        "baseline": baseline,
        "chosen_policy": asdict(rp) if rp else None,
        "confirmation_500": confirmation,
        "promotion_gate": bool(confirmation and confirmation["all_gates_passed"]),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
