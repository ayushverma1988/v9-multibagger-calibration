"""Compact the existing archive, with verified readback before source deletion."""
import argparse,hashlib,json,os,re,shutil,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from collections import defaultdict
from collections import deque
from v13_archive_cleanup import ARCHIVE,OWNER,record_file,value
from v13_archive_codec import encode_file,decode_file,sha_file,archive_download,physical_spec
from v13_storage_profile import batch_actions,download_action
from v13_checkpoint_writes import spec,temporary,status

MAX_PACK=64*1024**2
MAX_MEMBERS=400
PACKS_PER_TRANSACTION=20

def retry_seconds(exc,attempt):
    headers=getattr(getattr(exc,'response',None),'headers',{}) or {}
    headers={k.lower():v for k,v in headers.items()}
    value=headers.get('retry-after')
    if value:
        try:return max(1.,float(value))
        except (ValueError,TypeError):
            try:return max(1.,(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
            except (ValueError,TypeError):pass
    if status(exc)==429:
        match=re.search(r';\s*t=(\d+)',headers.get('ratelimit',''))
        if match:return max(1.,float(match.group(1)))
        # This repository has a separate 128-commit/hour limit. Without a
        # provider reset time, wait a full hour instead of repeating commits.
        return 3600. if re.search(r'128|commit.{0,30}(hour|limit)|hour.{0,30}commit',str(exc),re.I) else 60.
    return min(2**attempt,30.)

def publish_progress(result):
    if not os.environ.get('GITHUB_TOKEN'):return
    import requests,base64
    repo=os.environ['GITHUB_REPOSITORY']
    if repo!='ayushverma1988/v9-multibagger-calibration':raise ValueError('Unexpected progress repository')
    path='operational/v13_storage_migration_health.json';url=f'https://api.github.com/repos/{repo}/contents/{path}'
    headers={'Authorization':'Bearer '+os.environ['GITHUB_TOKEN'],'Accept':'application/vnd.github+json'}
    report={**result,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'workflow_run':os.environ.get('GITHUB_RUN_ID')}
    for attempt in range(3):
        try:response=requests.get(url+'?ref=main',headers=headers,timeout=(10,60))
        except requests.RequestException:return
        body={'branch':'main','message':'Record V13 lossless storage conversion progress',
          'content':base64.b64encode(json.dumps(report,indent=2).encode()).decode()}
        if response.status_code==200:body['sha']=response.json()['sha']
        elif response.status_code!=404:return
        try:saved=requests.put(url,headers=headers,json=body,timeout=(10,60))
        except requests.RequestException:return
        if saved.status_code in {200,201}:return
        if saved.status_code not in {409,422}:return

def catalog_rows(group,pack,raw_bytes,raw_sha,encoded_sha,encoded_bytes):
    offset=0;rows=[]
    for item in group:
        rows.append({'original_sha256':item['sha'],'original_bytes':item['bytes'],'offset':offset,
          'pack_path':pack,'pack_sha256':encoded_sha,'pack_bytes':encoded_bytes,
          'pack_original_sha256':raw_sha,'pack_original_bytes':raw_bytes})
        offset+=item['bytes']
    if offset!=raw_bytes:raise ValueError('Compact pack member coverage differs from original bytes')
    return rows

def verified_pack_members(source,rows,folder):
    decoded=Path(folder)/'readback.raw';first=rows[0]
    decode_file(source,decoded,first['pack_original_sha256'],first['pack_original_bytes'])
    with decoded.open('rb') as f:
        for row in rows:
            f.seek(row['offset']);raw=f.read(row['original_bytes'])
            if len(raw)!=row['original_bytes'] or hashlib.sha256(raw).hexdigest()!=row['original_sha256']:
                raise ValueError('Remote compact pack differs from an original source object')
    decoded.unlink()

def prepare_batch(batch,folder,download_pool,api):
    """Only read and encode here; the single writer publishes in order."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    actions=batch_actions(api,[r['physical'] for r in batch])
    def download(item):
        p=folder/(item['sha']+'.source');physical=item['physical']
        download_action(actions[physical['lfs_sha256']],p,physical['lfs_sha256'],physical['bytes'])
        q=folder/(item['sha']+'.original');decode_file(p,q,item['sha'],item['bytes']);p.unlink();return q
    inputs=list(download_pool.map(download,batch));raw_pack=folder/'pack.raw'
    with raw_pack.open('wb') as dest:
        for source in inputs:
            with source.open('rb') as f:shutil.copyfileobj(f,dest,1024*1024)
            source.unlink()
    size=raw_pack.stat().st_size;encoded=folder/'pack.zst';stats=encode_file(raw_pack,encoded)
    pack_sha=stats['sha256'];pack_path=f'v13/compact_packs/{pack_sha[:2]}/{pack_sha}.zst'
    rows=catalog_rows(batch,pack_path,size,stats['original_sha256'],pack_sha,stats['bytes'])
    verified_pack_members(encoded,rows,folder);raw_pack.unlink()
    return {'folder':folder,'batch':batch,'size':size,'encoded':encoded,'stats':stats,'pack_sha':pack_sha,'pack_path':pack_path,'rows':rows}

def commit(api,expected,adds=(),deletes=(),message='V13 lossless archive compaction',on_wait=None):
    from huggingface_hub import CommitOperationAdd,CommitOperationDelete
    for attempt in range(8):
        operations=[CommitOperationAdd(path_in_repo=s['remote'],path_or_fileobj=s['file']) for s in adds]
        operations += [CommitOperationDelete(path_in_repo=p,is_folder=False) for p in deletes]
        try:return api.create_commit(ARCHIVE,repo_type='dataset',parent_commit=expected,operations=operations,commit_message=message).oid
        except Exception as exc:
            if not temporary(exc):raise
            current=api.repo_info(ARCHIVE,repo_type='dataset').sha
            paths=[s['remote'] for s in adds]+list(deletes);infos=[]
            for start in range(0,len(paths),500):
                infos.extend(api.get_paths_info(ARCHIVE,repo_type='dataset',paths=paths[start:start+500]))
            actual={x.path:x for x in infos};ok=not any(p in actual for p in deletes)
            for s in adds:
                item=actual.get(s['remote']);lfs=value(item,'lfs')
                ok &= bool(item and item.size==s['bytes'] and (value(lfs,'sha256')==s['sha256'] if lfs else item.blob_id==s['git_blob']))
            if ok:return current
            if current!=expected:raise ValueError('Another writer changed the archive during compaction') from None
            if attempt==7:raise
            seconds=retry_seconds(exc,attempt)
            if status(exc)==429:
                report={'phase':'WAIT_PROVIDER_RATE_LIMIT','retry_after_seconds':seconds,
                  'retry_at_utc':(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat()}
                print(json.dumps(report),flush=True)
                if on_wait:on_wait(report)
            while seconds>0:
                chunk=min(30.,seconds);time.sleep(chunk);seconds-=chunk

def publish_prepared(api,expected,prepared,indexes,write,download_pool):
    """Stage many packs, verify all bytes remotely, then switch their catalogs."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from v13_checkpoint_writes import committed
    staged=[spec(p['pack_path'],p['encoded']) for p in prepared]
    expected=write(api,expected,adds=staged,message='Stage bounded batch of lossless source packs before readback')
    actions=batch_actions(api,[{'lfs_sha256':p['pack_sha'],'bytes':p['stats']['bytes']} for p in prepared])
    def readback(p):
        target=p['folder']/'remote.zst'
        download_action(actions[p['pack_sha']],target,p['pack_sha'],p['stats']['bytes'])
        verified_pack_members(target,p['rows'],p['folder']);target.unlink()
    # No source deletions or catalog publication can precede this barrier.
    list(download_pool.map(readback,prepared))
    mapping=[];changed={};paths=[]
    for p in prepared:
        prefix=p['prefix'];catalog=p['folder']/'catalog.parquet'
        pq.write_table(pa.Table.from_pylist(p['rows']),catalog,compression='zstd',compression_level=9)
        sha=sha_file(catalog);remote=f'v13/storage_catalog/{prefix}/{sha}.parquet'
        mapping.append(spec(remote,catalog))
        index=changed.setdefault(prefix,json.loads(json.dumps(indexes[prefix])))
        index['batches'].append({'path':remote,'sha256':sha,'bytes':catalog.stat().st_size})
        paths.extend(path for item in p['batch'] for path in item['paths'])
    for prefix,index in changed.items():
        target=prepared[0]['folder']/(prefix+'-index.json');target.write_text(json.dumps(index,sort_keys=True))
        mapping.append(spec(f'v13/storage_catalog/{prefix}/index.json',target))
    if len(mapping)+len(paths)>25000:raise ValueError('Bounded compact transaction exceeds supported operation count')
    expected=write(api,expected,adds=mapping,deletes=paths,
      message='Atomically publish verified compact catalogs and remove superseded source copies')
    if not committed(api,ARCHIVE,mapping):raise ValueError('Published catalog bytes differ from verified upload')
    indexes.update(changed)
    return expected,paths

def run(plan,out,max_groups=None):
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.errors import EntryNotFoundError
    import pyarrow as pa
    import pyarrow.parquet as pq
    from v13_archive_catalog import prefix_rows,fetch_member
    from v13_checkpoint import restore
    from v13_storage_index import POLICY
    raw=Path(plan).read_bytes()
    if hashlib.sha256(raw).hexdigest()!='61b014c3233140d6d633d2e7307b2d840c6dbf820bd566fdcca7b49b48b91697':raise ValueError('Unreviewed original inventory')
    inventory=json.loads(raw);original=next(r for r in inventory['repositories'] if r['repo_id']==ARCHIVE)
    retired={r['path'] for r in inventory['archive_dependencies']['delete_candidates']}
    originals=[r for r in original['files'] if r['path'] not in retired]
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if api.whoami().get('name')!=OWNER or not api.repo_info(ARCHIVE,repo_type='dataset').private:raise ValueError('Private archive ownership mismatch')
    refs=api.list_repo_refs(ARCHIVE,repo_type='dataset')
    if [r.name for r in refs.branches]!=['main'] or refs.tags:raise ValueError('Extra archive refs require review')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);work=out/'batch';work.mkdir(exist_ok=True)
    latest=Path(archive_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token)).read_bytes()
    if hashlib.sha256(latest).hexdigest()!='2f3e347ab9d0e9e902de8bab2769a1e3188db8077f55712fcaf97b90c7a6568f':
        raise ValueError('Current checkpoint advanced; refresh compaction inventory instead of overwriting it')
    data=json.loads(latest);active=data.get('derived_state_bundle',{}).get('object_path')
    if not restore(out/'before_state'):raise ValueError('Initial V13 restore failed')
    champion=sha_file(out/'before_state'/'active_heads.joblib')
    if champion!='c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571':raise ValueError('Frozen champion identity mismatch')
    before_hashes={name:sha_file(out/'before_state'/name) for name in
      ('active_heads.joblib','challenger/research_heads.joblib','prospective_observations.parquet')}
    shutil.rmtree(out/'before_state')
    expected=api.repo_info(ARCHIVE,repo_type='dataset').sha
    attrs=Path(archive_download(ARCHIVE,'.gitattributes',repo_type='dataset',token=token)).read_text();old_attrs=attrs
    for pattern in ('v13/compact_packs/**','v13/storage_catalog/**','v13/storage_index/**'):
        rule=pattern+' filter=lfs diff=lfs merge=lfs -text'
        if rule not in attrs:attrs+='\n'+rule+'\n'
    attrs_file=work/'attributes';attrs_file.write_text(attrs)
    if attrs!=old_attrs:expected=commit(api,expected,adds=[spec('.gitattributes',attrs_file)],message='Track compact V13 data packs as binary archive objects')
    # The reviewed inventory already binds every immutable original hash.
    # Query only already-migrated aliases during resume, rather than repeating
    # a 400,000-file recursive listing before the first pack can be converted.
    current={r['path']:r for r in originals}
    result={'format':'V13-storage-migration-receipt-1','started_at_utc':datetime.now(timezone.utc).isoformat(),
      'original_logical_tree_bytes':sum(r['bytes'] or 0 for r in originals),'converted_original_objects':0,
      'converted_original_bytes':0,'compact_pack_bytes':0,'deleted_uncompressed_paths':0,
      'original_manifest_sha256':hashlib.sha256(latest).hexdigest(),'fixed_champion_sha256':champion,'completed':False,'phase':'START'}
    def receipt():(out/'storage_migration_receipt.json').write_text(json.dumps(result,indent=2))
    def waiting(report):
        saved=result['phase'];result.update(report);receipt();publish_progress(result);result['phase']=saved
    def write(*args,**kwargs):
        head=commit(*args,**kwargs,on_wait=waiting)
        result.pop('retry_after_seconds',None);result.pop('retry_at_utc',None)
        return head
    receipt();publish_progress(result);groups={}
    for item in originals:
        path=item['path'];match=re.fullmatch(r'v13/(?:objects/|object_shards/[0-9a-f]{2}/[0-9a-f]{2}/|packed_objects/[0-9a-f]{2}/[0-9a-f]{2}/)([0-9a-f]{64})\.bin',path)
        if not match or path==active:continue
        sha=match.group(1)
        if item['bytes']==0 and item['lfs_sha256'] is None:
            if sha!=hashlib.sha256(b'').hexdigest():raise ValueError('Empty original file has another content identity')
            result['zero_byte_original_files_retained']=result.get('zero_byte_original_files_retained',0)+1
            continue
        if item['lfs_sha256']!=sha:raise ValueError('Original canonical object has another LFS identity')
        if item['bytes']>MAX_PACK:continue  # Large objects use exact in-place encoding below.
        group=groups.setdefault(sha,{'sha':sha,'bytes':item['bytes'],'paths':[]})
        if group['bytes']!=item['bytes']:raise ValueError('Conflicting original hash lengths')
        group['paths'].append(path)
    prefixes=defaultdict(list)
    for item in groups.values():prefixes[item['sha'][:2]].append(item)
    completed_batches=0;plans=[];indexes={};retained_packs={}
    result['phase']='PLAN_REMAINING_ORIGINALS'
    for prefix,items in sorted(prefixes.items()):
        known=prefix_rows(ARCHIVE,prefix,token,refresh=True)
        if known:
            index_path=f'v13/storage_catalog/{prefix}/index.json'
            indexes[prefix]=json.loads(Path(archive_download(ARCHIVE,index_path,repo_type='dataset',token=token)).read_text())
        else:indexes[prefix]={'format':'V13-compressed-object-catalog-1','prefix':prefix,'batches':[]}
        known_paths=[p for item in items if item['sha'] in known for p in item['paths']]
        existing_known=set()
        for start in range(0,len(known_paths),500):
            existing_known.update(r.path for r in api.get_paths_info(ARCHIVE,repo_type='dataset',paths=known_paths[start:start+500]))
        todo=[];duplicate_paths=[]
        for item in sorted(items,key=lambda r:r['sha']):
            if item['sha'] in known:
                row=known[item['sha']]
                if row['original_bytes']!=item['bytes']:raise ValueError('Resumed catalog differs from original length')
                fetch_member(ARCHIVE,item['sha'],token,cache=work/'resume')
                duplicate_paths.extend(p for p in item['paths'] if p in existing_known)
                retained_packs[row['pack_path']]=row['pack_bytes']
                for p in item['paths']:current.pop(p,None)
                result['converted_original_objects']+=1;result['converted_original_bytes']+=item['bytes']
                result['deleted_uncompressed_paths']+=len(item['paths']);continue
            physical=next((current[p] for p in item['paths'] if p in current),None)
            if physical is None:raise ValueError('Original source is missing without a verified compact catalog')
            item['physical']=physical;todo.append(item)
        for start in range(0,len(duplicate_paths),500):
            expected=write(api,expected,deletes=duplicate_paths[start:start+500],message='Finish verified duplicate compact aliases')
        while todo:
            batch=[];size=0
            while todo and len(batch)<MAX_MEMBERS and (not batch or size+todo[0]['bytes']<=MAX_PACK):
                item=todo.pop(0);batch.append(item);size+=item['bytes']
            plans.append((prefix,batch))
        if (work/'resume').exists():shutil.rmtree(work/'resume')
    result['compact_pack_bytes']=sum(retained_packs.values())
    result['remaining_batches']=len(plans);result['phase']='PACK_ORIGINAL_DATA';receipt();publish_progress(result)
    downloads=ThreadPoolExecutor(max_workers=48);preparations=ThreadPoolExecutor(max_workers=3)
    pending=deque();next_plan=0;checked_prefixes=set()
    def submit_next():
        nonlocal next_plan
        if next_plan>=len(plans):return
        prefix,batch=plans[next_plan];folder=out/'prepared'/f'{next_plan:05d}'
        pending.append((prefix,preparations.submit(prepare_batch,batch,folder,downloads,api)));next_plan+=1
    for _ in range(3):submit_next()
    try:
        while pending:
            prepared=[]
            for _ in range(PACKS_PER_TRANSACTION):
                if not pending:break
                prefix,future=pending.popleft();item=future.result();item['prefix']=prefix
                prepared.append(item);submit_next()
            expected,paths=publish_prepared(api,expected,prepared,indexes,write,downloads)
            for item in prepared:
                prefix=item['prefix'];batch=item['batch'];batch_work=item['folder'];stats=item['stats']
                if prefix not in checked_prefixes:
                    prefix_rows(ARCHIVE,prefix,token,refresh=True)
                    fetched=fetch_member(ARCHIVE,batch[0]['sha'],token,cache=batch_work/'catalog-check')
                    if not fetched or sha_file(fetched)!=batch[0]['sha']:raise ValueError('Published catalog cannot restore an original source')
                    checked_prefixes.add(prefix)
                retained_packs[item['pack_path']]=stats['bytes']
                result['converted_original_objects']+=len(batch);result['converted_original_bytes']+=item['size']
                result['compact_pack_bytes']+=stats['bytes'];completed_batches+=1
                shutil.rmtree(batch_work)
            for path in paths:current.pop(path,None)
            result['deleted_uncompressed_paths']+=len(paths)
            result['completed_batches_this_attempt']=completed_batches
            result['remaining_batches']=len(plans)-completed_batches;result['head']=expected;receipt()
            # Keep old physical LFS history bounded while processing the pool.
            if completed_batches%100==0 or not pending:
                api.super_squash_history(ARCHIVE,repo_type='dataset',branch='main',
                  commit_message='Preserve verified dated data while retiring superseded uncompressed storage history')
                expected=api.repo_info(ARCHIVE,repo_type='dataset').sha
                result['intermediate_history_prunes']=result.get('intermediate_history_prunes',0)+1
                result['head']=expected;receipt()
            print(json.dumps({k:result[k] for k in ['converted_original_objects','converted_original_bytes','compact_pack_bytes','completed_batches_this_attempt']}),flush=True)
            publish_progress(result)
            if max_groups and completed_batches>=max_groups:return result
    finally:
        for _,future in pending:future.cancel()
        preparations.shutdown(wait=True,cancel_futures=True);downloads.shutdown(wait=True,cancel_futures=True)

    # Preserve every dated manifest's original decoded bytes; compress metadata
    # and any individual object larger than the bounded source-pack size in place.
    result['phase']='COMPRESS_DATED_METADATA';receipt();publish_progress(result)
    metadata_adds=[];metadata_checks=[];metadata_physical_bytes=0
    def flush_metadata():
        nonlocal expected,metadata_adds,metadata_checks,metadata_physical_bytes
        if not metadata_adds:return
        expected=write(api,expected,adds=metadata_adds,message='Compress and index a verified batch of preserved dated V13 data')
        for path,digest in metadata_checks:
            verified=archive_download(ARCHIVE,path,repo_type='dataset',token=token,cache_dir=work/'metadata-check',force_download=True)
            if sha_file(verified)!=digest:raise ValueError('Compressed snapshot readback differs from original bytes')
            if (work/'metadata-check').exists():shutil.rmtree(work/'metadata-check')
        # Encoded staging bytes are needed only until the remote readback passes.
        for item in metadata_adds:
            source=Path(item['file'])
            if source.suffix=='.zst':source.unlink(missing_ok=True)
        shutil.rmtree(work/'metadata-index',ignore_errors=True)
        shutil.rmtree(work/'date-staging',ignore_errors=True)
        metadata_adds=[];metadata_checks=[];metadata_physical_bytes=0
        result['head']=expected;receipt();publish_progress(result)
    for number,item in enumerate(originals):
        path=item['path']
        if not (path=='v13/state/latest_manifest.json' or path.startswith('v13/state/snapshots/') and path.endswith('.json')
          or path.endswith('.bin') and item['bytes']>MAX_PACK and path!=active):continue
        # Bound disk, including source, decoded bytes and staged replacements.
        if metadata_adds and metadata_physical_bytes+item['bytes']>1024**3:flush_metadata()
        original_path=archive_download(ARCHIVE,path,repo_type='dataset',token=token,cache_dir=work/'metadata-source')
        if item['lfs_sha256'] and sha_file(original_path)!=item['lfs_sha256']:raise ValueError('Original metadata bytes changed')
        logical=spec(path,original_path);physical=physical_spec(logical)
        if physical!=logical:
            from v13_storage_index import encoding_index
            mapping=encoding_index([logical],[physical],work/'metadata-index')
            metadata_adds.append(physical)
            if mapping:metadata_adds.append(spec(mapping[0],mapping[1]))
            metadata_checks.append((path,logical['sha256']));metadata_physical_bytes+=physical['bytes']
            result['metadata_files_compressed']=result.get('metadata_files_compressed',0)+1
            result['metadata_original_bytes']=result.get('metadata_original_bytes',0)+logical['bytes']
            result['metadata_stored_bytes']=result.get('metadata_stored_bytes',0)+physical['bytes']
        if path.startswith('v13/state/snapshots/'):
            key=Path(path).stem;stamp=key.split('_')[0]
            date=datetime.strptime(stamp,'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
            index={'format':'V13-dated-data-index-1','source_asof':None,'date_scope':'ARCHIVE_SAVED_DATE',
              'archive_saved_at_utc':date.isoformat(),'snapshot':path,'manifest_sha256':logical['sha256'],
              'full_daily_data_copy_created':False,'minimum_retention_days':730}
            p=work/'date-staging'/(key+'.json');p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(index,sort_keys=True))
            remote='v13/date_index/'+date.astimezone(ZoneInfo('Asia/Kolkata')).strftime('%Y/%m/%d')+'/'+key+'.json'
            metadata_adds.append(spec(remote,p))
        if len(metadata_adds)>=80:flush_metadata()
        if (work/'metadata-source').exists():shutil.rmtree(work/'metadata-source')
    flush_metadata()
    p=work/'policy.json';p.write_text(json.dumps(POLICY,sort_keys=True,indent=2))
    expected=write(api,expected,adds=[spec('v13/storage_policy.json',p)],message='Record two-year lossless V13 storage policy')
    after_latest=Path(archive_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token,force_download=True)).read_bytes()
    if after_latest!=latest:raise ValueError('Frozen checkpoint manifest bytes changed during compaction')
    if not restore(out/'after_state'):raise ValueError('Fresh compact archive restore failed')
    for name in ('active_heads.joblib','challenger/research_heads.joblib','prospective_observations.parquet'):
        if before_hashes[name]!=sha_file(out/'after_state'/name):raise ValueError('Frozen state changed during compaction')
    api.super_squash_history(ARCHIVE,repo_type='dataset',branch='main',commit_message='Retain all dated V13 data in verified lossless compressed storage')
    files=[record_file(r) for r in api.list_repo_tree(ARCHIVE,repo_type='dataset',recursive=True) if isinstance(r,RepoFile)]
    live_oids={r['lfs_sha256'] for r in files if r.get('lfs_sha256')}
    old_oids={r['lfs_sha256'] for r in originals if r.get('lfs_sha256')}-live_oids
    obsolete=[r for r in api.list_lfs_files(ARCHIVE,repo_type='dataset') if r.oid in old_oids]
    if obsolete:api.permanently_delete_lfs_files(ARCHIVE,obsolete,repo_type='dataset',rewrite_history=False)
    result.update(completed=True,fresh_restart_verified=True,all_original_logical_bytes_verified=True,
      frozen_model_and_forecast_bytes_unchanged=True,all_dated_manifests_retained=True,
      current_tree_stored_bytes=sum(r['bytes'] or 0 for r in files),current_files=len(files),
      obsolete_uncompressed_lfs_objects_purged=len(obsolete),history_squashed=True,
      completed_at_utc=datetime.now(timezone.utc).isoformat(),head=api.repo_info(ARCHIVE,repo_type='dataset').sha)
    info=api.repo_info(ARCHIVE,repo_type='dataset',expand=['usedStorage','sha'])
    result['provider_used_storage_after']=value(info,'used_storage');receipt();publish_progress(result);print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);p.add_argument('--max-groups',type=int)
    a=p.parse_args();run(a.plan,a.out,a.max_groups)
