"""Safely finish retired-byte reclamation and measure lossless compression."""
import argparse,hashlib,json,os,time,zipfile,threading
from pathlib import Path
from collections import Counter
from v13_archive_cleanup import ARCHIVE,OWNER,value,root_fingerprint
_downloads=threading.local()

def batch_actions(api,files):
    import requests
    objects=[{'oid':r['lfs_sha256'],'size':r['bytes']} for r in files]
    response=requests.post('https://huggingface.co/datasets/'+ARCHIVE+'.git/info/lfs/objects/batch',
      headers={'Authorization':'Bearer '+os.environ['HF_ARCHIVE_TOKEN'],
        'Accept':'application/vnd.git-lfs+json','Content-Type':'application/vnd.git-lfs+json'},
      json={'operation':'download','transfers':['basic'],'objects':objects},timeout=(15,120))
    response.raise_for_status();data=response.json()
    result={}
    for item in data['objects']:
        if item.get('error'):raise ValueError('LFS batch download rejected a required object')
        action=item['actions']['download'];result[item['oid']]=action
    if set(result)!={r['lfs_sha256'] for r in files}:raise ValueError('Incomplete LFS download batch')
    return result

def download_action(action,path,digest,size):
    import requests
    from urllib.parse import urlparse
    url=action['href']
    if urlparse(url).scheme!='https':raise ValueError('Insecure archive transfer URL')
    if not hasattr(_downloads,'session'):_downloads.session=requests.Session()
    for attempt in range(5):
        h=hashlib.sha256();total=0
        try:
            with _downloads.session.get(url,headers=action.get('header',{}),stream=True,timeout=(15,120)) as response:
                code=response.status_code
                if code in {408,425,429,500,502,503,504}:
                    delay=response.headers.get('Retry-After','')
                    remaining=float(delay) if delay.isdigit() else min(2**attempt,30)
                    while remaining>0:chunk=min(remaining,30);time.sleep(chunk);remaining-=chunk
                    continue
                if code!=200:raise RuntimeError('Archive object download HTTP '+str(code))
                with Path(path).open('wb') as f:
                    for chunk in response.iter_content(1024*1024):
                        total+=len(chunk)
                        if total>size:raise ValueError('Archive download exceeded reviewed length')
                        h.update(chunk);f.write(chunk)
        except requests.RequestException as exc:
            if attempt==4:raise RuntimeError('Archive transfer unavailable: '+type(exc).__name__) from None
            time.sleep(min(2**attempt,15));continue
        if total!=size or h.hexdigest()!=digest:raise ValueError('Archive transfer bytes differ from reviewed LFS identity')
        return
    raise RuntimeError('Archive transfer failed after bounded retries')

def run(plan,out,purge=False):
    from huggingface_hub import HfApi
    import zstandard as zstd
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=Path(plan).read_bytes()
    if hashlib.sha256(raw).hexdigest()!='61b014c3233140d6d633d2e7307b2d840c6dbf820bd566fdcca7b49b48b91697':
        raise ValueError('Unexpected reviewed retention inventory')
    data=json.loads(raw);archive=next(r for r in data['repositories'] if r['repo_id']==ARCHIVE)
    removed={r['path'] for r in data['archive_dependencies']['delete_candidates']}
    kept=[r for r in archive['files'] if r['path'] not in removed]
    api=HfApi(token=os.environ['HF_ARCHIVE_TOKEN'])
    if api.whoami().get('name')!=OWNER:raise ValueError('Unexpected archive owner')
    info=api.repo_info(ARCHIVE,repo_type='dataset',expand=['sha','private','usedStorage'])
    if not info.private or info.sha!='ba498f2d9bc5caa38922d22c8147b503ab5905ae':
        raise ValueError('Archive changed since verified cleanup; review new state first')
    refs=api.list_repo_refs(ARCHIVE,repo_type='dataset')
    if [r.name for r in refs.branches]!=['main'] or refs.tags:raise ValueError('Unexpected additional archive references')
    before=root_fingerprint(api.list_repo_tree(ARCHIVE,repo_type='dataset',recursive=False))
    used={r['lfs_sha256'] for r in kept if r.get('lfs_sha256')}
    obsolete={r['lfs_sha256'] for r in data['archive_dependencies']['delete_candidates'] if r.get('lfs_sha256')}-used
    result={'mode':'LOSSLESS_COMPRESSION_PROFILE','used_storage_before':value(info,'used_storage'),
      'retired_lfs_objects_purged':0,'retired_lfs_bytes_purged':0,'sample':[]}
    def receipt():(out/'storage_profile.json').write_text(json.dumps(result,indent=2))
    receipt()
    if purge:
        selected=[]
        for item in api.list_lfs_files(ARCHIVE,repo_type='dataset'):
            if item.oid in obsolete:selected.append(item)
        if {x.oid for x in selected}&used:raise ValueError('Retired object shares required original data')
        if api.repo_info(ARCHIVE,repo_type='dataset').sha!=info.sha:raise ValueError('Archive changed during LFS audit')
        if selected:
            api.permanently_delete_lfs_files(ARCHIVE,selected,repo_type='dataset',rewrite_history=False)
            result['retired_lfs_objects_purged']=len(selected)
            result['retired_lfs_bytes_purged']=sum(x.size for x in selected)
        if root_fingerprint(api.list_repo_tree(ARCHIVE,repo_type='dataset',recursive=False))!=before:
            raise ValueError('Retained V13 tree changed during retired-byte reclamation')
        result['retained_tree_unchanged']=True;receipt()
    binaries=[r for r in kept if r['path'].endswith('.bin') and r.get('lfs_sha256')]
    bins=Counter()
    for r in binaries:
        key=next((str(n) for n in (1024,16384,262144,4194304,67108864) if r['bytes']<n),'large')
        bins[key]+=r['bytes']
    result['current_binary_bytes']=sum(r['bytes'] for r in binaries)
    result['current_binary_files']=len(binaries);result['size_class_bytes']=dict(bins)
    candidates=sorted(binaries,key=lambda r:hashlib.sha256(r['path'].encode()).hexdigest())
    chosen=[];budget=1024**3
    for lo,hi in [(0,1024),(1024,16384),(16384,262144),(262144,4194304),(4194304,67108864),(67108864,512*1024**2)]:
        for r in [r for r in candidates if lo<=r['bytes']<hi][:12]:
            if r['bytes']<=budget:chosen.append(r);budget-=r['bytes']
    actions=batch_actions(api,chosen)
    for item in chosen:
        path=out/'sample.bin';download_action(actions[item['lfs_sha256']],path,item['lfs_sha256'],item['bytes'])
        start=time.monotonic();encoded=zstd.ZstdCompressor(level=9,write_checksum=True).compress(path.read_bytes())
        if zstd.ZstdDecompressor().decompress(encoded)!=path.read_bytes():raise ValueError('Lossless compression verification failed')
        result['sample'].append({'path':item['path'],'original_bytes':item['bytes'],'zstd_bytes':len(encoded),
          'seconds':round(time.monotonic()-start,4)})
        path.unlink();receipt()
    result['sample_original_bytes']=sum(r['original_bytes'] for r in result['sample'])
    result['sample_compressed_bytes']=sum(r['zstd_bytes'] for r in result['sample'])
    after=api.repo_info(ARCHIVE,repo_type='dataset',expand=['sha','private','usedStorage'])
    result['provider_used_storage_after']=value(after,'used_storage');result['finished']=True;receipt()
    print(json.dumps({k:v for k,v in result.items() if k!='sample'},indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--purge-verified-retired-lfs',action='store_true');a=p.parse_args();run(a.plan,a.out,a.purge_verified_retired_lfs)
