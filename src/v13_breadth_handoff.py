"""Reconstruct an on-time public input bundle when a scheduled runner starts late.

The witness is GitHub's server-side workflow creation clock, read by the
workflow from the exact push that contains this immutable receipt. A git
author timestamp, an HTTP Last-Modified header or a later review is not proof.
"""
from datetime import datetime, timezone
from pathlib import Path
import base64, hashlib, json, re, tempfile
import pandas as pd
from v13_breadth_capture import get, MAX_BYTES
from v13_breadth_fallback import validate_receipt, SOURCE_BASE
from v13_breadth_primary import rows_in_original

PUBLIC = {'nse_original': ('nse_source_session.zip',10_000_000),
          'nse_calendar': ('nse_holiday_calendar.json',2_000_000),
          'nse_corporate_actions': ('nse_corporate_actions.json',5_000_000)}

def recover(packet, witness, commit, out, fetch=get, clock=lambda:datetime.now(timezone.utc)):
    if not re.fullmatch(r'[0-9a-f]{40}',str(commit)):raise ValueError('Missing witnessed push')
    if packet.get('contract')!='V13_PUBLIC_CAPTURE_HANDOFF_1':raise ValueError('Unexpected handoff contract')
    r=packet['receipt'];day=r['source_asof']
    if not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',day):raise ValueError('Invalid session day')
    start=pd.Timestamp(day,tz='Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
    end=start.normalize()+pd.Timedelta(hours=22)
    seen=pd.Timestamp(witness);actual=pd.Timestamp(clock())
    if seen.tzinfo is None or actual.tzinfo is None or not start<=seen<=end or seen>actual:raise ValueError('Push not witnessed inside session window')
    out=Path(out);out.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        folder=Path(tmp)
        for key,(name,limit) in PUBLIC.items():
            original=packet['originals_b64'][key]
            if len(original)>4*((limit+2)//3):raise ValueError('Oversized public original')
            raw=base64.b64decode(original,validate=True)
            if len(raw)!=r[key]['bytes'] or hashlib.sha256(raw).hexdigest()!=r[key]['sha256']:raise ValueError('Wrong public original: '+key)
            (folder/name).write_bytes(raw)
        rows_in_original((folder/'nse_source_session.zip').read_bytes(),day)
        for obj,year,name in [(r,int(day[:4]),'nse_adjusted.parquet')]+([(r['previous_year'],int(day[:4])-1,'nse_adjusted_previous.parquet')] if r.get('previous_year') else []):
            rev=obj['publisher_revision']
            if not re.fullmatch(r'[a-f0-9]{40}',rev):raise ValueError('Invalid public revision')
            url=SOURCE_BASE+rev+'/prices_adjusted/nse_'+str(year)+'.parquet?download=true'
            if obj['source_url']!=url:raise ValueError('Unexpected public data URL')
            # Earlier actual observation is inseparable from the exact bytes.
            clocks=[v['first_retrieved_utc'] for v in [obj,*(r[k] for k in PUBLIC)]]
            if any(pd.Timestamp(t).tzinfo is None or pd.Timestamp(t)>seen for t in clocks):raise ValueError('Observation later than push witness')
            raw=fetch(url,MAX_BYTES)
            if len(raw)!=obj['bytes'] or hashlib.sha256(raw).hexdigest()!=obj['sha256']:raise ValueError('Immutable public data changed')
            (folder/name).write_bytes(raw)
        validate_receipt(r,folder/'nse_adjusted.parquet',day)
        # Keep original receipt times and actual later verification separate.
        provenance={'source_asof':day,'witness_push_sha':commit,'github_run_created_utc':seen.isoformat(),
                    'reconstructed_utc':pd.Timestamp(clock()).isoformat(),'entry_or_outcome_certified':False}
        (folder/'public_handoff_witness.json').write_text(json.dumps(provenance,indent=2)+'\n')
        (folder/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
        out.mkdir(parents=True,exist_ok=True)
        if (out/'receipt.json').exists():raise ValueError('Will not replace an existing complete receipt')
        for src in sorted(folder.iterdir(),key=lambda p:p.name=='receipt.json'):src.replace(out/src.name)
    return provenance

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    for name in ['packet','witness','commit','out']:ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    print(json.dumps(recover(json.loads(Path(args.packet).read_text()),args.witness,args.commit,args.out)))
