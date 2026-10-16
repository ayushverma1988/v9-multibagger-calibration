"""Reproducible chronological redevelopment experiment; no model promotion."""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import brier_score_loss,average_precision_score
import v13_model as model
import v12_coherent_hazard_model as coherent
import v13_redevelopment as rd

def metrics(d):
    result={}
    for target,prob in [('hit2_6','p6'),('hit2_12','p12'),(coherent.LOSS,'loss12')]:
        k=d[d[target].isin([0,1])&d[prob].notna()]
        if k.empty:continue
        bins=[]
        for lo in np.arange(0,1,.2):
            b=k[k[prob].ge(lo)&k[prob].lt(lo+.2+1e-12)]
            if len(b):bins.append({'lower':float(lo),'rows':len(b),'predicted':float(b[prob].mean()),'observed':float(b[target].mean())})
        result[prob]={'rows':len(k),'events':int(k[target].sum()),'brier':float(brier_score_loss(k[target],k[prob])),
          'average_precision':float(average_precision_score(k[target],k[prob])) if k[target].sum() else None,
          'predicted':float(k[prob].mean()),'observed':float(k[target].mean()),'bins':bins}
    return result

def run(inputs,current,output,asof):
    root=Path(inputs);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=root/'challenger_matrix.parquet'
    data=rd.attach(pd.read_parquet(source),root/'history'/'securities')
    data=model.prepare(data)
    dates=sorted(data.loc[data.mature_date_12.le(pd.Timestamp(asof)),'date'].unique())[-3:]
    protocol={'version':rd.VERSION,'asof':asof,'matrix_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'holdout_dates':[str(pd.Timestamp(d).date()) for d in dates],
      'historical_results_previously_examined':True,'unseen_mature_outcomes':0,
      'survivor_bias_current_listings':True,'adjusted_price_vintage_bias_possible':True,
      'historical_earnings_not_backfilled_from_current_rows':True,
      'calibration_strengths_predeclared':[25,10],'ranking_weights':rd.WEIGHTS,
      'production_accuracy_approved':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    outputs=[];clocks=[]
    for date in dates:
        test=data[data.date.eq(date)&data.mature_date_12.le(pd.Timestamp(asof))].copy()
        for variant in ('baseline','features_only','redeveloped'):
            heads,clock=rd.fit_heads(data,date,variant)
            pred=model.probabilities(heads,test)
            if not {'p6','p12','loss12','joint12'}<=pred.keys():
                clocks.append({'date':str(date),'variant':variant,'blocked':'INSUFFICIENT_MATURE_TRAINING_ROWS'});continue
            d=test.copy()
            for k,v in pred.items():d[k]=v
            d,audit=rd.select(d);d['variant']=variant;outputs.append(d)
            clocks.append({'date':str(date),'variant':variant,'heads':clock})
            print('Evaluated',str(pd.Timestamp(date).date()),variant,len(d),flush=True)
    historical=pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()
    summary={**protocol,'historical':{},'training_clocks':clocks,
      'finite_historical_evidence_rows':{k:int(data[k].notna().sum()) for k in rd.EVIDENCE_FEATURES}}
    for variant,d in historical.groupby('variant') if len(historical) else []:
        chosen=d[d.research_selected] if variant=='baseline' else d[d.redevelopment_selected]
        summary['historical'][variant]={'all_rows':metrics(d),'selected_rows':metrics(chosen),
          'selected':len(chosen),'selected_median_return63':float(chosen.rd_return63.median()) if len(chosen) else None,
          'selected_extended':int(chosen.redevelopment_extension_status.eq('EXTENDED').sum()),
          'coherence_failures':int(((d.p6>d.p12+1e-12)|(d.joint12+d.loss12>1+1e-12)).sum())}
    if len(historical):historical.to_parquet(out/'historical_scores.parquet',index=False)
    fresh=rd.attach(pd.read_parquet(current),root/'history'/'securities')
    for variant in ('baseline','features_only','redeveloped'):
        heads,clock=rd.fit_heads(data,pd.Timestamp(asof),variant)
        pred=model.probabilities(heads,fresh)
        if not {'p6','p12','loss12','joint12'}<=pred.keys():raise ValueError('Incomplete current research heads')
        d=fresh.copy()
        for k,v in pred.items():d[k]=v
        d,audit=rd.select(d);d.to_parquet(out/(variant+'_current.parquet'),index=False)
        joblib.dump(heads,out/(variant+'_research_heads.joblib'))
        summary[variant+'_current']={**audit,'training_clocks':clock,'model_sha256':hashlib.sha256((out/(variant+'_research_heads.joblib')).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--current',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',required=True);a=p.parse_args()
    run(a.inputs,a.current,a.output,a.asof)
