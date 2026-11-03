"""One fixed discovery experiment, not a parameter search or live promotion.

O'Neil's published discussion motivates earnings, relative strength and
price/volume confirmation. The three available interactions below are our
own hypotheses, not his proprietary ratings. Unavailable PIT earnings and
financial quality are not synthesized from present-day provider pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import v11_4_standalone_train_walkforward as core
import v11_4_robust_research_model as baseline
from v11_4_complete_validation import load_history, partitions, EVALUATION_CUTOFF
from v11_4_forward_research_release import fit_calibration_params, apply_frozen_calibration
from v11_4_validation_metrics import evaluate_ranked, summarize_folds

EXTRA_FEATURES = ('specific_catalyst_momentum_alignment',
                  'volume_momentum_confirmation', 'extended_runup_volatility')
FEATURES = (*baseline.FEATURES, *EXTRA_FEATURES)
VERSION = 'V11.4-free-source-discovery-interactions-research-20261010'


def source_features(frame, already_safe=False):
    x = frame.copy() if already_safe else core.safe_featureize(frame)
    price = ['mom_accel','vol_accel','ret_252','volatility_60']
    events = ['nse_order_win_90d','nse_capacity_expansion_90d','nse_regulatory_90d_positive']
    if any(k not in frame for k in price+events):
        raise ValueError('Source fields for the predeclared experiment are missing')
    for k in price+events:
        vals = pd.to_numeric(frame[k],errors='coerce')
        if np.isinf(vals).any():raise ValueError('Infinite raw research signal')
    x[EXTRA_FEATURES[0]] = x[events].sum(axis=1,min_count=len(events))*x.mom_accel.clip(-1,1)
    x[EXTRA_FEATURES[1]] = x.mom_accel.clip(-1,1)*np.log1p(x.vol_accel.clip(lower=0,upper=20))
    x[EXTRA_FEATURES[2]] = x.ret_252.clip(lower=1,upper=10).sub(1)*x.volatility_60.clip(lower=0,upper=1)
    return x


class ResearchBounds(baseline.TrainingTailBounds):
    def fit(self,X,y=None):
        super().fit(X,y)
        a=np.asarray(X,dtype=float)
        for i in range(len(baseline.FEATURES),len(FEATURES)):
            self.lower_bounds_[i]=np.nanquantile(a[:,i],self.lower)
            self.upper_bounds_[i]=np.nanquantile(a[:,i],self.upper)
        return self


def make_model():
    return Pipeline([('training_tail_bounds',ResearchBounds()),
        ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler()),
        ('lr',LogisticRegression(C=core.REGULARIZATION_C,max_iter=1300,class_weight=None,
                               solver='lbfgs',random_state=31))])


def fit_score(train,cal,candidates,variant):
    features=FEATURES if variant else baseline.FEATURES
    model=make_model() if variant else baseline.make_model()
    model.fit(train[list(features)],train.y6.astype(int))
    raw_cal=model.predict_proba(cal[list(features)])[:,1]
    calibration=fit_calibration_params(raw_cal,cal.y6,train.y6)
    raw=model.predict_proba(candidates[list(features)])[:,1]
    return apply_frozen_calibration(raw,calibration),model,calibration


def freeze_ranking(frame):
    if frame.symbol.duplicated().any():
        raise ValueError('Duplicate ranking security')
    p=pd.to_numeric(frame.probability,errors='coerce')
    if not np.isfinite(p).all() or not p.between(0,1).all():
        raise ValueError('Invalid research probability')
    return frame.sort_values(['probability','symbol'],ascending=[False,True])


def freeze_recipe(output, features_path, labels_path):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    recipe={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
        'current_model_baseline':'fixed_training_tail_bounds_21input',
        'extra_features':list(EXTRA_FEATURES),'regularization_C':core.REGULARIZATION_C,
        'parameter_search_performed':False,'new_variant_count':1,
        'lookahead_label_horizon_actual_sessions':126,
        'chronology_and_calibration_unchanged':True,
        'already_examined_dates_not_independent_blind_validation':True,
        'current_provider_financials_not_inserted_into_historical_predictors':True,
        'source_features_sha256':hashlib.sha256(Path(features_path).read_bytes()).hexdigest(),
        'label_file_sha256':hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
        'signal_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'acceptance_thresholds':{'minimum_fully_labelled_dates':12,'mean_jaccard':.8,'worst_p05_jaccard':.6},
        'full_original_financial_catalyst_conditions_preserved':True,
        'required_new_financial_signals_when_PIT_data_exists':['quarterly_EPS_and_sales_acceleration',
            'operating_cash_flow_conversion','exceptional_profit_dependence',
            'order_value_relative_to_revenue','capacity_commissioned_with_earnings_timing'],
        'primary_technique_source':'https://www.williamoneil.com/about-us/legal/oneil-proprietary-rating-and-rankings',
        'promotion_requires_source_qualified_recipe_and_unseen_evaluation':True,'production_approved':False}
    path=out/'fixed_discovery_experiment_recipe.json'
    with path.open('x') as f:json.dump(recipe,f,indent=2)
    return recipe


def run(features_path,labels_path,current_path,output):
    out=Path(output)
    recipe=freeze_recipe(out,features_path,labels_path)
    # No selection or target inspection takes place before the recipe write.
    history,_=load_history(features_path,labels_path)
    history=source_features(history,already_safe=True)
    summaries={};records={}
    for variant in (False,True):
        name='current_21input' if not variant else 'fixed_24input_discovery_hypothesis'
        folds,stability,scores=[],[],[]
        for date in sorted(history.date.unique()):
            split=partitions(history,date)
            if split is None:continue
            base,cal=split
            candidates=history.loc[history.date.eq(date)&core.eligible_asof(history)].copy()
            probs,model,params=fit_score(base,cal,candidates,variant)
            ranked=freeze_ranking(candidates[['date','symbol','close','y6','y6_mature_date','integrity_y6_clean','dd30_6m']].assign(probability=probs))
            metric=evaluate_ranked(ranked,'probability',EVALUATION_CUTOFF,float(cal.y6.mean()))
            day=str(pd.Timestamp(date).date());metric.update({'date':day,'train_rows':len(base),
                'calibration_date':str(pd.Timestamp(cal.date.iloc[0]).date())})
            chosen=ranked.sort_values(['probability','symbol'],ascending=[False,True]).head(10).symbol
            for seed in core.STABILITY_SEEDS:
                alt,_,_=fit_score(core.perturbation_training(base,seed),cal,candidates,variant)
                top=candidates.assign(probability=alt).sort_values(['probability','symbol'],ascending=[False,True]).head(10).symbol
                stability.append({'date':day,'seed':seed,'jaccard':core.top10_similarity(chosen,top)})
            folds.append(metric);scores.append(ranked)
        summary=summarize_folds(folds,stability);summary['folds']=folds
        summaries[name]=summary;records[name]=folds
        if not variant and (summary['confirmed_2x_hits'] != 14 or
                            summary['fully_labelled_selections'] != 100 or
                            summary['all_scored_known_hits'] != 19 or
                            abs(summary['mean_top10_jaccard']-.859239051546744)>1e-12):
            raise ValueError('Current 21-input baseline did not reproduce the preceding validated execution')
        pd.concat(scores,ignore_index=True).to_parquet(out/(name+'_retrospective_scores_PRIVATE.parquet'),index=False)
        pd.DataFrame(stability).to_csv(out/(name+'_stability.csv'),index=False)
        print(json.dumps({'experiment_variant_complete':name,
            'confirmed_2x_hits':summary.get('confirmed_2x_hits'),
            'mean_top10_jaccard':summary.get('mean_top10_jaccard'),
            'worst_fold_p05_jaccard':summary.get('worst_fold_p05_jaccard')}),flush=True)
    a,b=[{f['date']:f for f in records[k] if f['selected_clean_mature_outcomes']==10} for k in records]
    paired=sorted(a.keys()&b.keys())
    dif=np.array([b[d]['top10_precision']-a[d]['top10_precision'] for d in paired])
    ci=np.quantile(np.random.default_rng(20261010).choice(dif,size=(4000,len(dif)),replace=True).mean(axis=1),[.025,.975]).tolist() if len(dif)>=2 else [None,None]
    result={'scope':'ONE_FIXED_RETROSPECTIVE_RESEARCH_ABLATION_NOT_NEW_BLIND_EVIDENCE',
        'model_version':VERSION,'results':summaries,
        'same_fully_labelled_paired_dates':{'date_count':len(paired),'dates':paired,
            'current_21input_hits':sum(a[d]['top10_doublers'] for d in paired),
            'fixed_24input_hits':sum(b[d]['top10_doublers'] for d in paired),
            'mean_precision_difference':float(dif.mean()) if len(dif) else None,
            'paired_date_block_difference_95':ci},
        'new_independent_blind_outcomes':0,'full_financial_model_trained':False,
        'recipe_sha256':hashlib.sha256((out/'fixed_discovery_experiment_recipe.json').read_bytes()).hexdigest(),
        'accepted_frozen_model_or_source_inputs_overwritten':False,'production_approved':False}
    result['current_21input_baseline_reproduced']=True
    # Run the candidate on the original current source for an inspectable demo.
    # This separate package is never substituted for the independent freeze.
    current=source_features(pd.read_parquet(current_path));date=current.date.iloc[0]
    base,cal=partitions(history,date)
    prob,model,params=fit_score(base,cal,current,True)
    demo=current[['date','symbol','close']].assign(research_probability=prob)
    demo.to_parquet(out/'candidate_current_demo_PRIVATE.parquet',index=False)
    joblib.dump({'model':model,'calibration':params,'features':list(FEATURES),'model_version':VERSION,
        'source_qualified_full_model':False,'recipe':recipe},out/'candidate_model_PRIVATE.joblib')
    result['current_candidate_rows_scored']=len(demo)
    (out/'discovery_signal_research_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--current',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.features,a.labels,a.current,a.output)


if __name__=='__main__':main()
