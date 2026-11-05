"""Apply the owner's approved compact replacement; verify before retiring data."""
import argparse,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path

SOURCE='9aa4f8abb4563aa03f8925ff59883b6787e5618d4932902d2ed24c2f3060dfb3'
CHAMPION='c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571'
CHALLENGER='f6facda12b9fcc757a15cb9e7cdd54b84ebaabe3dff0213b313f3f6d6471f291'

def run(output):
    from huggingface_hub import HfApi
    from v13_archive_codec import archive_download,sha_file
    from v13_archive_cleanup import ARCHIVE,OWNER,value
    from v13_checkpoint import restore,save
    from v13_checkpoint_writes import spec
    from v13_storage_migrate import commit
    from v13_working_archive import FORMAT,identities,retire,migration_matches
    out=Path(output);out.mkdir(parents=True,exist_ok=True);token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if api.whoami().get('name')!=OWNER or not api.repo_info(ARCHIVE,repo_type='dataset').private:raise ValueError('Private archive ownership mismatch')
    refs=api.list_repo_refs(ARCHIVE,repo_type='dataset')
    if [r.name for r in refs.branches]!=['main'] or refs.tags:raise ValueError('Unreviewed archive refs')
    raw=Path(archive_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token,force_download=True)).read_bytes()
    source=json.loads(raw);before=api.repo_info(ARCHIVE,repo_type='dataset',expand=['usedStorage','sha'])
    receipt={'format':'V13-approved-working-core-replacement-1','authorized_date_IST':'2026-11-05',
      'started_at_utc':datetime.now(timezone.utc).isoformat(),'provider_used_storage_before':value(before,'used_storage'),
      'source_manifest_sha256':hashlib.sha256(raw).hexdigest(),'completed':False}
    def checkpoint():(out/'core_cutover_receipt.json').write_text(json.dumps(receipt,indent=2))
    checkpoint()
    if source.get('format')!=FORMAT:
        if hashlib.sha256(raw).hexdigest()!=SOURCE:raise ValueError('A newer legacy checkpoint needs its own verified compact source')
        if not restore(out/'state'):raise ValueError('Source working state could not be restored')
        root=out/'state';(root/'remote_manifest.json').write_bytes(raw)
        if sha_file(root/'active_heads.joblib')!=CHAMPION or sha_file(root/'challenger/research_heads.joblib')!=CHALLENGER:
            raise ValueError('Frozen model identity mismatch')
        original=identities(source)
        attrs=Path(archive_download(ARCHIVE,'.gitattributes',repo_type='dataset',token=token)).read_text();old_attrs=attrs
        for pattern in ('v13/working_core/**','v13/state/**'):
            rule=pattern+' filter=lfs diff=lfs merge=lfs -text'
            if rule not in attrs:attrs+='\n'+rule+'\n'
        additions=[]
        if attrs!=old_attrs:
            path=out/'attributes';path.write_text(attrs);additions.append(spec('.gitattributes',path))
        readme=out/'README.md';readme.write_text('''# V13 compact working archive

This private archive uses checkpoint format 4 and a verified compressed working
core. The owner approved retiring older raw source documents, response copies
and complete snapshot versions on 5 November 2026. Models, extracted working
tables, cached histories and frozen forecast records remain unchanged.

Small date indexes and dated facts/forecasts inside the working tables preserve
the retained date-wise history. Original source hashes and a retired-reference
inventory remain as provenance; they do not imply the original bodies remain.
The working archive targets one decimal GB. It will preserve the last verified
checkpoint and local recovery rather than silently discard facts if a future
core exceeds that budget. Use compatible V13 format-4 readers and writers.
''');additions.append(spec('README.md',readme))
        commit(api,api.repo_info(ARCHIVE,repo_type='dataset').sha,adds=additions,message='Prepare owner-approved compact working archive storage')
        published=save(root);receipt['archive_receipt']=published;checkpoint()
        current=Path(archive_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token,force_download=True)).read_bytes();data=json.loads(current)
        if not migration_matches(raw,data):raise ValueError('Working replacement changed original state bytes')
        if not restore(out/'independent_restart'):raise ValueError('New remote checkpoint could not restart')
        for name,(sha,size) in original.items():
            p=out/'independent_restart'/name
            if p.stat().st_size!=size or sha_file(p)!=sha:raise ValueError('Replacement changed an original working member')
        receipt.update(original_working_files_preserved=len(original),all_original_working_bytes_verified=True,
          champion_sha256=CHAMPION,challenger_sha256=CHALLENGER,new_manifest_sha256=hashlib.sha256(current).hexdigest())
        for package in (out/'v13_core_staging').glob('V13_Working_Archive_*.zip'):
            shutil.copyfile(package,out/package.name)
        checkpoint();shutil.rmtree(out/'independent_restart');shutil.rmtree(out/'state');shutil.rmtree(out/'v13_core_staging',ignore_errors=True)
    else:
        receipt['resumed_already_published_working_core']=True
    # retire() independently restarts again immediately before deletion, keeps
    # only the active core and lightweight dates, purges obsolete LFS objects,
    # then independently restarts once more before reporting completion.
    receipt['retirement']=retire(out/'core_retirement_receipt.json',purge=True)
    receipt.update(completed=True,one_gb_current_tree_verified=True,
      completed_at_utc=datetime.now(timezone.utc).isoformat(),models_and_frozen_forecast_bytes_changed=False)
    checkpoint();print(json.dumps(receipt,indent=2),flush=True);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
