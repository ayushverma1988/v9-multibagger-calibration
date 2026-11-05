"""Create and restart-check a <=1 GB working core; retain the evidence archive."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import zipfile
from v13_archive_codec import sha_file

TARGET_BYTES = 1_000_000_000


def build_core(root, manifest_raw, output, source_asof, source_worker, level=12, target_bytes=TARGET_BYTES):
    import zstandard as zstd
    from v13_core_restore import restore_package
    data = json.loads(manifest_raw);root = Path(root);output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    active = [r for r in data['files'] if not r['raw_source']]
    raw_count = sum(r['raw_source'] for r in data['files'])
    del data
    if len({r['path'] for r in active}) != len(active):
        raise ValueError('Duplicate working member')
    packet = output/'v13_state.tar.zst'
    with packet.open('wb') as destination:
        with zstd.ZstdCompressor(level=level, write_checksum=True).stream_writer(destination, closefd=False) as encoded:
            with tarfile.open(fileobj=encoded, mode='w|') as tar:
                info = tarfile.TarInfo('source_manifest.json');info.size = len(manifest_raw);info.mode = 0o644
                tar.addfile(info, io.BytesIO(manifest_raw))
                for i, item in enumerate(active):
                    rel = Path(item['path'])
                    if rel.is_absolute() or '..' in rel.parts:
                        raise ValueError('Unsafe checkpoint member')
                    source = root/rel
                    if source.stat().st_size != item['bytes'] or sha_file(source) != item['sha256']:
                        raise ValueError('Working checkpoint differs from source manifest')
                    info = tarfile.TarInfo('state/'+item['path']);info.size = item['bytes'];info.mode = 0o644
                    with source.open('rb') as handle:
                        tar.addfile(info, handle)
                    if i and i%4000 == 0:
                        print(json.dumps({'core_files_encoded':i, 'total_state_files':len(active)}), flush=True)
    receipt = {'format':'V13-compact-working-core-1', 'source_asof':source_asof, 'source_worker_run':source_worker,
        'source_manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(), 'source_manifest_bytes':len(manifest_raw),
        'state_files':len(active), 'state_uncompressed_bytes':sum(r['bytes'] for r in active),
        'payload_sha256':sha_file(packet), 'payload_bytes':packet.stat().st_size,
        'codec':'zstandard', 'compression_level':level, 'target_bytes':target_bytes,
        'raw_source_references_retained_in_manifest':raw_count,
        'raw_source_originals_included_in_core':False, 'historical_evidence_archive_deleted':False,
        'numeric_precision_changed':False, 'frozen_forecasts_or_model_bytes_changed':False}
    name=f'V13_Compact_Working_Core_{source_asof}.zip';package = output/name
    restore_source = Path(__file__).with_name('v13_core_restore.py').read_text()
    readme = f'''V13 compact working checkpoint — source {source_asof}

Contains EVERY {len(active):,} non-raw working-state file from the source checkpoint,
including both models, frozen forecast ledger, financial/master tables, cached
histories and retained derived research records. Every file is checked against
its original SHA-256 and length. Numeric values and precision are unchanged.

The full original source manifest is included inside the compressed payload.
The {raw_count:,} raw-evidence references remain recorded there, but their source
documents and responses are NOT included in this working package. They remain
in the original private archive. This package is not the complete historical
evidence archive and creating it does not reduce that archive's total usage.
Further evidence audits or source refreshes may still need that archive/network.

Restore to a NEW directory using Python 3.12 with zstandard==0.25.0 installed:
  python restore_core.py {name} --out v13_state

Use the existing V13 source and requirements_v13_storage.txt to run the model.
This package neither retrains the model nor executes a new stock-selection run.
The original dates, data gaps, frozen probabilities and flags remain intact.
'''
    with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.write(packet, 'v13_state.tar.zst')
        archive.writestr('core_receipt.json', json.dumps(receipt, indent=2))
        archive.writestr('restore_core.py', restore_source)
        archive.writestr('README.txt', readme)
    receipt['package_file'] = name;receipt['package_bytes'] = package.stat().st_size
    receipt['package_sha256'] = sha_file(package);receipt['under_one_gb'] = package.stat().st_size <= target_bytes
    if not receipt['under_one_gb']:
        (output/'core_build_receipt.json').write_text(json.dumps(receipt, indent=2))
        raise ValueError('Measured working package exceeds requested byte target')
    receipt['fresh_restore'] = restore_package(package, output/'fresh_verified_state')
    receipt['fresh_working_restart_files_verified'] = True
    (output/'core_build_receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def run(worker, output):
    from huggingface_hub import HfApi
    from v13_archive_cleanup import ARCHIVE, OWNER
    from v13_archive_codec import archive_download
    from v13_checkpoint import materialize_bundle
    from v13_worker_artifact import read_manifest
    output=Path(output);output.mkdir(parents=True, exist_ok=True)
    data, receipt, raw = read_manifest(worker)
    if receipt.get('remote_publication_confirmed') is False:
        raise ValueError('Compact core requires a published checkpoint')
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if api.whoami().get('name')!=OWNER or not api.repo_info(ARCHIVE,repo_type='dataset').private:
        raise ValueError('Private archive ownership mismatch')
    before = api.repo_info(ARCHIVE,repo_type='dataset').sha
    fetch=lambda path:archive_download(ARCHIVE,path,repo_type='dataset',token=token,cache_dir=output/'cache')
    current=Path(fetch('v13/state/latest_manifest.json')).read_bytes()
    if hashlib.sha256(current).hexdigest()!=receipt['manifest_sha256']:
        raise ValueError('Latest checkpoint advanced; choose its new worker instead')
    root=output/'original_state';bundle_bytes=data['derived_state_bundle']['bytes']
    materialize_bundle(data['derived_state_bundle'],data,root,fetch)
    del data,current
    source=json.loads((root/'latest_run.json').read_text())
    result=build_core(root,raw,output,source['asof'],source.get('workflow_run') or os.environ.get('SOURCE_WORKER_RUN'))
    if api.repo_info(ARCHIVE,repo_type='dataset').sha!=before:
        raise ValueError('Archive changed while creating read-only working core')
    result.update(archive_writes=False, source_archive_head_unchanged=True, original_restart_bundle_bytes=bundle_bytes)
    (output/'core_build_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.worker,a.out)
