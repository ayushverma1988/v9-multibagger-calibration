"""Choose the newest valid backup using streamed ZIP downloads of any size."""
import base64,hashlib,json,os,re,zipfile
from pathlib import Path
from v13_archive_codec import sha_file

def verified_recovery_floor(report):
    if report.get('latest_preserved_worker_permanently_published') is not True:return 0
    if report.get('completed') is not True or report.get('post_publication_fresh_restart_verified') is not True:
        raise ValueError('Permanent recovery floor has no completed remote restart proof')
    floor=report.get('recovery_floor')
    if type(floor) is not int or floor<=0 or floor!=report.get('source_worker_run'):
        raise ValueError('Permanent recovery floor differs from its verified worker')
    if not re.fullmatch('[0-9a-f]{64}',str(report.get('latest_manifest_sha256',''))):
        raise ValueError('Permanent recovery floor has no verified checkpoint identity')
    return floor

def select_recovery(runs,list_artifacts,download,folder='.',flushed_through=0):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);target=folder/'previous_worker.zip'
    target.unlink(missing_ok=True)
    for run in sorted(runs,key=lambda r:r['id'],reverse=True):
        if run['head_branch']!='main' or run['id']<=flushed_through:continue
        items=list_artifacts(run['id'])
        a=next((x for x in items if not x['expired'] and x['name']==f"v13-private-research-state-{run['id']}"),None)
        if a is None:continue
        download(a,target)
        if 'sha256:'+sha_file(target)!=a['digest']:
            target.unlink();raise ValueError('Previous worker artifact hash mismatch')
        with zipfile.ZipFile(target) as z:
            names=set(z.namelist());published=False
            if 'archive_receipt.json' in names:
                receipt=json.loads(z.read('archive_receipt.json'));published=bool(receipt.get('snapshot') and receipt.get('manifest_sha256'))
            pending='v13_checkpoint_recovery.json' in names
        if not published and not pending:target.unlink();continue
        result={'run_id':run['id'],'artifact_id':a['id'],'digest':a['digest'],
          'status':'REMOTE_CHECKPOINT_PUBLISHED_USE_ARCHIVE_RESTORE' if published else 'VERIFIED_RECOVERY_SELECTED'}
        if published:target.unlink()
        (folder/'previous_worker_receipt.json').write_text(json.dumps(result,indent=2));return result
    result={'status':'NO_NEWER_RECOVERY_USE_ARCHIVE','flushed_through_run':flushed_through}
    (folder/'previous_worker_receipt.json').write_text(json.dumps(result,indent=2));return result

def run():
    import requests
    from v13_archive_retention import download_artifact
    repo=os.environ['GITHUB_REPOSITORY']
    if repo!='ayushverma1988/v9-multibagger-calibration':raise ValueError('Unexpected worker repository')
    headers={'Authorization':'Bearer '+os.environ['GITHUB_TOKEN'],'Accept':'application/vnd.github+json'}
    def get(path):
        response=requests.get('https://api.github.com/repos/'+repo+path,headers=headers,timeout=(15,120))
        response.raise_for_status();return response.json()
    runs=get('/actions/workflows/v13_research_worker.yml/runs?status=completed&per_page=8')['workflow_runs']
    response=requests.get('https://api.github.com/repos/'+repo+'/contents/operational/v13_storage_migration_health.json?ref=main',
      headers=headers,timeout=(15,120))
    floor=int(os.environ.get('V13_RECOVERY_FLUSHED_THROUGH_RUN','0'))
    if response.status_code!=404:
        response.raise_for_status()
        report=json.loads(base64.b64decode(response.json()['content']))
        floor=max(floor,verified_recovery_floor(report))
    result=select_recovery(runs,lambda run:get(f'/actions/runs/{run}/artifacts')['artifacts'],
      lambda a,path:download_artifact(a['id'],a['digest'],path),
      flushed_through=floor)
    print(json.dumps(result));return result

if __name__=='__main__':run()
