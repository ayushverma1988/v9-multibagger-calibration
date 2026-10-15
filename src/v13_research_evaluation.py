"""Predeclared small experiments and complete-cohort prospective diagnostics.

All experiments are development diagnostics, never unseen promotion evidence.
The simple baseline is a maturity-purged, smoothed prior event rate.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss,average_precision_score
from v12_public_history import bounds

GROUPS={
 'market':['relative_strength63','log_ret63','volume_confirmation','volatility63'],
 'market_financial':['relative_strength63','log_ret63','volume_confirmation','quarter_sales_yoy','sales_growth_acceleration','quarter_pat_yoy','annual_CFO_PAT','dated_debt_equity'],
 'market_financial_catalyst':['relative_strength63','log_ret63','volume_confirmation','quarter_sales_yoy','sales_growth_acceleration','quarter_pat_yoy','annual_CFO_PAT','dated_debt_equity','commissioned_document_count90','utilisation_disclosed_pct','disclosed_order_value_due12_to_sales'],
 'market_owner_earnings':['relative_strength63','log_ret63','volume_confirmation','owner_PAT_yoy','owner_PAT_margin','reported_diluted_EPS_yoy_same_document','exceptional_items_to_abs_PBT'],
 'market_cash_quality':['relative_strength63','log_ret63','volume_confirmation','annual_CFO_PAT','annual_cash_conversion_change','accruals_to_average_assets','incremental_ROCE_proxy','annual_EBIT_proxy_interest_coverage']}

EXPERIMENT_INPUT_CONTRACT='V13_REQUIRED_RECONCILED_OWNER_CASH_INPUTS_20261015_V3'

def experiment_group_ready(name,usable):
    """A named financial experiment must actually contain its named evidence."""
    usable=set(usable)
    if name=='market_owner_earnings':return bool(usable&{'owner_PAT_yoy','owner_PAT_margin'})
    if name=='market_cash_quality':return bool(usable&{'annual_CFO_PAT','annual_cash_conversion_change','accruals_to_average_assets'})
    if name.endswith('catalyst'):return bool(usable&set(GROUPS[name][-3:]))
    return True

def prior_rates(data,date):
    result={}
    low,_=bounds(date)
    for h in (6,12):
        q=data[data.date.ge(low)&data.date.lt(pd.Timestamp(date))&data['mature_date_'+str(h)].lt(pd.Timestamp(date))&data['hit2_'+str(h)].isin([0,1])]
        result['baseline_p'+str(h)]=float((q['hit2_'+str(h)].sum()+1)/(len(q)+2)) if len(q) else None
    return result

def calibration(y,p):
    rows=[]
    for a,b in zip([0,.05,.1,.2,.4,.6,.8],[.05,.1,.2,.4,.6,.8,1.00001]):
        k=(p>=a)&(p<b)
        if k.any():rows.append({'lower':a,'upper':min(b,1),'rows':int(k.sum()),'predicted':float(p[k].mean()),'observed':float(y[k].mean())})
    return rows

def metrics(q,ycol,pcol):
    z=q[q[ycol].isin([0,1])&q[pcol].notna()]
    if z.empty:return {'known':0,'brier':None,'PR_AUC':None,'calibration':[]}
    y=z[ycol].to_numpy(float);p=z[pcol].to_numpy(float)
    return {'known':len(z),'observed_rate':float(y.mean()),'brier':float(brier_score_loss(y,p)),
      'PR_AUC':float(average_precision_score(y,p)) if y.sum() else None,'calibration':calibration(y,p)}

def prospective_report(observations):
    result={'scope':'ALL_REGISTERED_SELECTED_AND_UNSELECTED','accuracy_approved':False,'groups':[],
      'frozen_probability_target':'DIVIDEND_ADJUSTED_CLOSING_2X; DAILY_LOW_LOSS30_BEFORE_2X_OR_DEADLINE',
      'additional_price_only_and_any_horizon_events_are_not_relabelled_frozen_probabilities':True}
    if observations.empty:return result
    d=observations.copy()
    for c in ('verified_hit2_6','verified_hit2_12','verified_loss30_12'):
        if c not in d:d[c]=np.nan
    for c in ('financial_tier','market_regime','liquidity_tier'):
        if c not in d:d[c]='UNKNOWN'
        d[c]=d[c].fillna('UNKNOWN')
    for h in (6,12):
        ycol='verified_hit2_'+str(h);mature='six_month_mature' if h==6 else 'twelve_month_mature'
        for axis in ('financial_tier','market_regime','liquidity_tier'):
            for (role,value),q in d.groupby(['model_role',axis],dropna=False):
                known=q[q[mature].eq(True)&q.independent_outcome_verified.eq(True)]
                if ycol not in known:known=known.assign(**{ycol:np.nan})
                picked=q.selected_verified.fillna(False)|q.selected_discovery.fillna(False)
                selected_known=known[known.selected_verified.fillna(False)|known.selected_discovery.fillna(False)]
                result['groups'].append({'months':h,'model_role':role,'axis':axis,'group':str(value),
                  'registered':len(q),'selected':int(picked.sum()),'unknown_or_immature':len(q)-len(known),
                  'selected_precision':float(selected_known[ycol].mean()) if selected_known[ycol].isin([0,1]).any() else None,
                  **metrics(known,ycol,'p'+str(h))})
    result['confirmed_early_2x']=int(d.get('confirmed_early_2x',pd.Series(False,index=d.index)).fillna(False).sum())
    result['confirmed_risk_events']=int(d.get('confirmed_risk_event',pd.Series(False,index=d.index)).fillna(False).sum())
    loss=d[d.twelve_month_mature.eq(True)&d.independent_outcome_verified.eq(True)]
    result['loss_risk_metrics']={role:metrics(q,'verified_loss30_12','loss12') for role,q in loss.groupby('model_role')} if 'verified_loss30_12' in loss else {}
    result['risk_groups']=[]
    for axis in ('financial_tier','market_regime','liquidity_tier'):
        for (role,value),q in d.groupby(['model_role',axis],dropna=False):
            known=q[q.twelve_month_mature.eq(True)&q.independent_outcome_verified.eq(True)]
            result['risk_groups'].append({'model_role':role,'axis':axis,'group':str(value),'registered':len(q),
              'unknown_or_immature':len(q)-len(known),**metrics(known,'verified_loss30_12','loss12')})
    result['additional_verified_events']={}
    for field in ('confirmed_price_only_2x','confirmed_price_only_loss30','confirmed_any_horizon_loss30'):
        values=d[field] if field in d else pd.Series(None,index=d.index,dtype=object)
        result['additional_verified_events'][field]={'confirmed':int(values.eq(True).sum()),'unknown_records':int(values.isna().sum()),'no_matching_frozen_probability_assigned':True}
    result['paired_readiness']=paired_readiness(d)
    return result

def paired_readiness(observations):
    """Expose missing prospective proof now, without invented future outcomes."""
    if 'cohort_id' not in observations:return []
    d=observations[observations.cohort_id.notna()]
    rows=[]
    for cohort,q in d.groupby('cohort_id'):
        a=q[q.model_role.eq('champion')].set_index('isin');b=q[q.model_role.eq('challenger')].set_index('isin')
        both=a.index.intersection(b.index);registered=a.index.union(b.index)
        aa=a.loc[both];bb=b.loc[both]
        from v13_pair_proof import entry_match,same_metadata,baseline_match
        clock=aa.registered_at_utc.eq(bb.registered_at_utc)&aa.source_asof.eq(bb.source_asof)&aa.symbol.eq(bb.symbol)&same_metadata(aa,bb)
        date_entry=aa.get('entry_verified',pd.Series(False,index=aa.index)).eq(True)&bb.get('entry_verified',pd.Series(False,index=bb.index)).eq(True)&aa.get('first_entry_date',pd.Series(None,index=aa.index)).notna()&aa.get('first_entry_date',pd.Series(None,index=aa.index)).eq(bb.get('first_entry_date',pd.Series(None,index=bb.index)))
        entry=entry_match(aa,bb)
        mature=aa.twelve_month_mature.eq(True)&bb.twelve_month_mature.eq(True)&aa.independent_outcome_verified.eq(True)&bb.independent_outcome_verified.eq(True)
        baseline=baseline_match(aa,bb,12)
        rows.append({'cohort_id':cohort,'registered_stock_pairs':len(registered),'missing_partner_records':len(registered)-len(both),
          'same_clock_same_names_pairs':int(clock.sum()),'verified_same_entry_pairs':int((clock&entry).sum()),
          'verified_same_entry_date_pairs':int((clock&date_entry).sum()),
          'entry_price_and_both_source_hashes_required':True,
          'pairs_with_frozen_p12_baseline':int((clock&baseline).sum()),'independently_verified_mature_pairs':int((clock&entry&mature).sum()),
          'historical_baselines_or_partners_never_retrofitted':True})
    return rows

def paired_evaluation_report(observations):
    """Fair paired clocks/universes and unknown denominator, for both horizons.

    Original unpaired cohorts cannot be made paired retrospectively. Compare
    verified outcomes only on the same names; report missing and immature
    pairs alongside results rather than silently shrinking the cohort.
    """
    result={'scope':'IMMUTABLE_PAIRED_COHORTS_ALL_SELECTED_AND_UNSELECTED','accuracy_approved':False,'cohorts':[],'comparisons':[]}
    if observations.empty or 'cohort_id' not in observations:return result
    d=observations[observations.cohort_id.notna()]
    keys=['cohort_id','isin']
    a=d[d.model_role.eq('champion')];b=d[d.model_role.eq('challenger')]
    if a.duplicated(keys).any() or b.duplicated(keys).any():raise ValueError('Duplicate paired evaluation identity')
    joined=a.merge(b,on=keys,how='outer',suffixes=('_a','_b'),indicator=True,validate='1:1')
    for cohort,q in joined.groupby('cohort_id'):
        from v13_pair_proof import entry_match,same_metadata,baseline_match,sides
        aa,bb=sides(q)
        entries=entry_match(aa,bb)
        clocks=q._merge.eq('both')&q.source_asof_a.eq(q.source_asof_b)&q.registered_at_utc_a.eq(q.registered_at_utc_b)&q.symbol_a.eq(q.symbol_b)&same_metadata(aa,bb)
        result['cohorts'].append({'cohort_id':cohort,'registered_stock_pairs':len(q),'same_clock_same_names':bool(clocks.all()),
          'comparison_contract':q.comparison_contract_a.iloc[0] if 'comparison_contract_a' in q else 'V13_LEGACY_PAIR_20261012',
          'source_asof':q.source_asof_a.dropna().iloc[0] if q.source_asof_a.notna().any() else None,
          'registered_at_utc':q.registered_at_utc_a.dropna().iloc[0] if q.registered_at_utc_a.notna().any() else None,
          'unpaired_originals_excluded':True})
        for h in (6,12):
            yc='verified_hit2_'+str(h);m='six_month_mature' if h==6 else 'twelve_month_mature'
            good=clocks&entries&q[m+'_a'].eq(True)&q[m+'_b'].eq(True)&q.independent_outcome_verified_a.eq(True)&q.independent_outcome_verified_b.eq(True)
            good&=q[yc+'_a'].isin([0,1])&q[yc+'_a'].eq(q[yc+'_b'])&q.first_entry_date_a.eq(q.first_entry_date_b)
            for axis in ('financial_tier','market_regime','liquidity_tier'):
                col=axis+'_a'
                if col not in q:q[col]='UNKNOWN'
                for group,ix in q.groupby(col,dropna=False).groups.items():
                    z=q.loc[ix];k=z.loc[good.loc[ix]];y=k[yc+'_a'].to_numpy(float)
                    row={'cohort_id':cohort,'months':h,'axis':axis,'group':str(group),'registered_pairs':len(z),
                      'verified_pairs':len(k),'unknown_or_immature_pairs':len(z)-len(k)}
                    for suffix,label in (('a','champion'),('b','challenger')):
                        pred=k['p'+str(h)+'_'+suffix].to_numpy(float)
                        selected=z.selected_verified_a.fillna(False)|z.selected_discovery_a.fillna(False) if suffix=='a' else z.selected_verified_b.fillna(False)|z.selected_discovery_b.fillna(False)
                        selected_known=selected.loc[k.index]
                        row[label]={'selected_registered':int(selected.sum()),'brier':float(brier_score_loss(y,pred)) if len(k) else None,
                          'selected_verified_outcomes':int(selected_known.sum()),'selected_precision':float(y[selected_known.to_numpy()].mean()) if selected_known.any() else None}
                    row['paired_brier_gain']=row['champion']['brier']-row['challenger']['brier'] if len(k) else None
                    baseline_col='baseline_p'+str(h)+'_a'
                    baseline_known=baseline_match(aa,bb,h).loc[k.index]
                    base=k.loc[baseline_known]
                    row['baseline']={'verified_pairs_with_frozen_baseline':len(base),'verified_pairs_missing_baseline':len(k)-len(base),
                      'brier':float(brier_score_loss(base[yc+'_a'],base[baseline_col])) if len(base) else None,
                      'status':'FROZEN_PAST_ONLY_BASELINE' if len(base) else 'UNKNOWN_OR_IMMATURE_NOT_INVENTED'}
                    row['challenger_vs_baseline_brier_gain']=float(brier_score_loss(base[yc+'_a'],base[baseline_col])-brier_score_loss(base[yc+'_a'],base['p'+str(h)+'_b'])) if len(base) else None
                    result['comparisons'].append(row)
    return result

def experiments(data,asof,output):
    """Three fixed feature groups, three latest matured monthly holdouts/horizon.

    Fit/imputation only uses labels mature BEFORE the test decision. No tuning
    or automatic winner deployment. Missing catalyst observations can block a
    group; they never become zero commissionings or fabricated profitability.
    """
    d=data.copy();low,_=bounds(asof);d=d[d.date.ge(low)]
    if 'history_start_used' in d:d=d[pd.to_datetime(d.history_start_used,errors='coerce').ge(low)]
    report={'asof':asof,'input_contract':EXPERIMENT_INPUT_CONTRACT,'four_year_start':str(low.date()),'historical_results_previously_examined':True,
      'promotion_evidence':False,'groups_predeclared':GROUPS,'folds':[]}
    for h in (6,12):
        y='hit2_'+str(h);m='mature_date_'+str(h)
        if y not in d or m not in d:continue
        candidates=sorted(d.loc[d[m].le(pd.Timestamp(asof))&d[y].isin([0,1]),'date'].unique())[-3:]
        for date in candidates:
            train=d[d.date.lt(date)&d[m].lt(date)&d[y].isin([0,1])]
            test=d[d.date.eq(date)&d[m].le(pd.Timestamp(asof))&d[y].isin([0,1])]
            if train.empty or test.empty:continue
            p=float((train[y].sum()+1)/(len(train)+2))
            base=test.assign(prediction=p)
            report['folds'].append({'months':h,'date':str(pd.Timestamp(date).date()),'group':'prior_event_rate','train_rows':len(train),**metrics(base,y,'prediction')})
            for name,cols in GROUPS.items():
                usable=[c for c in cols if c in train and train[c].notna().sum()>=100]
                # A catalyst experiment needs measured execution features.
                if name.endswith('catalyst') and not experiment_group_ready(name,usable):
                    report['folds'].append({'months':h,'date':str(pd.Timestamp(date).date()),'group':name,'status':'BLOCKED_MISSING_QUANTIFIED_CATALYSTS'});continue
                if name in {'market_owner_earnings','market_cash_quality'} and not experiment_group_ready(name,usable):
                    report['folds'].append({'months':h,'date':str(pd.Timestamp(date).date()),'group':name,'status':'BLOCKED_MISSING_DATED_FINANCIAL_INPUTS'});continue
                if len(train)<500 or train[y].sum()<10 or not usable:continue
                pipe=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=.03,max_iter=1000,random_state=31))
                pipe.fit(train[usable],train[y].astype(int));scored=test.assign(prediction=pipe.predict_proba(test[usable])[:,1])
                report['folds'].append({'months':h,'date':str(pd.Timestamp(date).date()),'group':name,'features':usable,'train_rows':len(train),
                  'train_maturity_max':str(train[m].max()),**metrics(scored,y,'prediction')})
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False));return report
