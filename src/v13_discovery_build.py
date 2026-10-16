"""Reconstruct source-bound features without modifying input snapshots."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import v13_discovery_diagnostics as dd
import v13_historical_discovery as hd

def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def run(matrix,current,events,history,output,coverage_end):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    h=pd.read_parquet(matrix);c=pd.read_parquet(current);e=pd.read_parquet(events);n=len(e)
    e=e[e['isin'].isin(set(h['isin'])|set(c['isin']))&e.published_ts.ge('2022-01-01')].copy()
    print('Normalize events',len(e),flush=True);e,audit=hd.normalize(e)
    audit.update(archive_events=n,archive_sha256=sha_file(events),coverage_end=coverage_end)
    e.to_parquet(out/'historical_metadata.parquet',index=False);(out/'event_audit.json').write_text(json.dumps(audit,indent=2))
    both=dd.attach(pd.concat([h.assign(frame_kind='history'),c.assign(frame_kind='current')],ignore_index=True),history)
    both=hd.attach(both,e,coverage_end)
    h=both[both.frame_kind.eq('history')].copy();c=both[both.frame_kind.eq('current')].copy()
    h.to_parquet(out/'historical_enriched.parquet',index=False);c.to_parquet(out/'current_diagnostics.parquet',index=False)
    summary={'input_hashes':{str(p):sha_file(p) for p in [matrix,current,events]},'historical_rows':len(h),'current_rows':len(c),
      'historical_finite_rows':{k:int(h[k].notna().sum()) for k in hd.FEATURES},
      'historical_positive_rows':{k:int(h[k].gt(0).sum()) for k in hd.FEATURES},
      'selected_breakout_reasons':c[c.redevelopment_selected].dd_breakout_reason.value_counts().to_dict(),
      'all_breakout_states':c.dd_breakout_state.value_counts().to_dict()}
    (out/'build_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True);return summary
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['matrix','current','events','history','output','coverage-end']:p.add_argument('--'+key,required=True)
    a=p.parse_args();run(a.matrix,a.current,a.events,a.history,a.output,a.coverage_end)
