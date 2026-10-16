"""Protect registered forecasts and fixed heads during each collector run."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
from v13_lifecycle import validate_registry

class FrozenGuard:
    def __init__(self,root):
        self.root=Path(root);self.started=pd.Timestamp(datetime.now(timezone.utc))
        paths=list(self.root.glob('registries/*/*.json'))+list(self.root.glob('paired_cohorts/*.json'))
        self.immutable={str(p.relative_to(self.root)):p.read_bytes() for p in paths}
        self.heads={name:(self.root/name).read_bytes() for name in ['active_heads.joblib','active_model.json','challenger/research_heads.joblib','challenger/freeze.json'] if (self.root/name).exists()}
        self.outputs={name:(self.root/name).read_bytes() for name in ['prospective_observations.parquet','champion_latest_watchlist.json','challenger_latest_watchlist.json'] if (self.root/name).exists()}
        p=self.root/'prospective_observations.parquet'
        self.before=pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def check(self,observed,promote=False):
        errors=[]
        for rel,raw in self.immutable.items():
            p=self.root/rel
            if not p.exists() or p.read_bytes()!=raw:errors.append('immutable registered file changed: '+rel)
        for rel,raw in self.heads.items():
            if promote and rel in {'active_heads.joblib','active_model.json'}:continue
            p=self.root/rel
            if not p.exists() or p.read_bytes()!=raw:errors.append('fixed head changed: '+rel)
        for p in self.root.glob('registries/*/*.json'):
            try:
                obj=validate_registry(p)
                if str(p.relative_to(self.root)) not in self.immutable and pd.Timestamp(obj['registered_at_utc'])<self.started:
                    raise ValueError('New registration backdated before collector observation')
            except Exception as exc:errors.append(str(exc))
        if not self.before.empty:
            keys=['registry_sha256','isin']
            # Only forecast fields, not legitimate new outcome observations.
            columns=list(self.before.columns[:self.before.columns.get_loc('status')])
            if 'information_cutoff_utc' in self.before:columns=list(dict.fromkeys(columns+['information_cutoff_utc']))
            try:
                if observed.duplicated(keys).any():raise ValueError('Duplicate forecast identity')
                chosen=self.before[keys].merge(observed,how='left',on=keys,validate='1:1')
                pd.testing.assert_frame_equal(self.before[columns].sort_values(keys).reset_index(drop=True),chosen[columns].sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=True)
            except Exception as exc:errors.append('Frozen forecast values or cohort changed: '+str(exc)[:200])
        if errors:
            # Restore originals before the failure checkpoint is saved. Keep
            # source collection progress; never archive overwritten forecasts.
            for rel,raw in {**self.immutable,**self.heads,**self.outputs}.items():
                p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            result={'passed':False,'originals_restored':True,'errors':errors}
            out=self.root/'evidence';out.mkdir(exist_ok=True);(out/'frozen_integrity.json').write_text(json.dumps(result,indent=2))
            raise ValueError('Frozen integrity check failed; originals restored: '+'; '.join(errors)[:400])
        result={'passed':True,'original_forecast_records':len(self.before),'immutable_registered_files':len(self.immutable),
          'fixed_heads_unchanged':not promote,'promotion_separately_gated':bool(promote),'checked_at_utc':datetime.now(timezone.utc).isoformat()}
        out=self.root/'evidence';out.mkdir(exist_ok=True);(out/'frozen_integrity.json').write_text(json.dumps(result,indent=2));return result
