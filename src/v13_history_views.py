"""Select whole identity-matched histories; conflicts are never spliced."""
from pathlib import Path
import pandas as pd

def histories(folder,wanted=None):
    groups={};audit={'files':0,'quarantined':{},'redundant_files':0}
    fields=['date','symbol','isin','open','high','low','close','adj_close','volume']
    for path in sorted(Path(folder).glob('*.parquet')):
        bars=pd.read_parquet(path);audit['files']+=1
        if bars.empty or not set(fields).issubset(bars):continue
        identities=set(zip(bars['isin'],bars['symbol']))
        if len(identities)!=1:
            for key in identities:
                if wanted is None or key in wanted:audit['quarantined'][key]='MIXED_HISTORY_IDENTITY'
            continue
        key=next(iter(identities))
        if wanted is not None and key not in wanted:continue
        bars=bars.copy();bars['date']=pd.to_datetime(bars.date)
        if bars.date.duplicated().any():audit['quarantined'][key]='DUPLICATE_HISTORY_DATES';continue
        groups.setdefault(key,[]).append((path,bars))
    chosen={}
    for key,options in groups.items():
        if key in audit['quarantined']:continue
        options.sort(key=lambda x:(-len(x[1]),str(x[0])))
        path,bars=options[0];reference=bars.set_index('date');valid=True
        for other_path,other in options[1:]:
            if not set(other.date).issubset(set(bars.date)):valid=False;break
            a=reference.loc[other.date,fields[1:]].reset_index(drop=True);b=other[fields[1:]].reset_index(drop=True)
            try:pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
            except AssertionError:valid=False;break
        if not valid:audit['quarantined'][key]='CONFLICTING_HISTORY_VINTAGES';continue
        chosen[key]=(path,bars);audit['redundant_files']+=len(options)-1
    return chosen,audit
