import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from v13_archive_codec import encode_file,decode_file,envelope,logical_download,physical_spec

class LosslessArchiveCodec(unittest.TestCase):
    def test_standard_zstd_and_decoder_restore_exact_original(self):
        import zstandard as zstd
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);raw=(b'{"asof":"2026-11-02","EPS":1.2300000000000002,"missing":null}\n')*10000
            (p/'raw').write_bytes(raw);r=encode_file(p/'raw',p/'encoded')
            self.assertLess(r['bytes'],len(raw)//10)
            with (p/'encoded').open('rb') as f,zstd.ZstdDecompressor().stream_reader(f,read_across_frames=True) as reader:
                self.assertEqual(reader.read(),raw)
            decode_file(p/'encoded',p/'decoded',hashlib.sha256(raw).hexdigest(),len(raw))
            self.assertEqual((p/'decoded').read_bytes(),raw)

    def test_corruption_and_wrong_original_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'raw').write_bytes(b'source evidence '*10000);encode_file(p/'raw',p/'encoded')
            with self.assertRaises(ValueError):decode_file(p/'encoded',p/'decoded','a'*64)
            raw=bytearray((p/'encoded').read_bytes());raw[-1]^=1;(p/'encoded').write_bytes(raw)
            with self.assertRaises(Exception):decode_file(p/'encoded',p/'decoded')
            self.assertFalse((p/'decoded').exists())

    def test_legacy_files_and_canonical_hash_guard(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'raw').write_bytes(b'exact frozen forecasts'*1000)
            self.assertEqual(logical_download(p/'raw','original.json'),str(p/'raw'))
            encode_file(p/'raw',p/'encoded')
            with self.assertRaisesRegex(ValueError,'canonical name'):
                logical_download(p/'encoded','a'*64+'.bin',p/'cache')

    def test_upload_metadata_is_physical_but_decodes_to_logical_bytes(self):
        from v13_checkpoint_writes import spec
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);raw=b'unchanged source data '*10000;(p/'raw').write_bytes(raw)
            digest=hashlib.sha256(raw).hexdigest();s=spec('v13/object_shards/'+digest[:2]+'/'+digest[2:4]+'/'+digest+'.bin',p/'raw')
            with patch.dict(os.environ,{'V13_COMPACT_ARCHIVE':'1','V13_STORAGE_STAGING_DIR':str(p/'uploads')}):
                encoded=physical_spec(s)
            self.assertEqual(encoded['remote'],s['remote']);self.assertNotEqual(encoded['sha256'],s['sha256'])
            out=logical_download(encoded['file'],Path(s['remote']).name,p/'cache')
            self.assertEqual(Path(out).read_bytes(),raw)

    def test_source_changed_and_small_incompressible_payloads_are_not_hidden(self):
        from v13_checkpoint_writes import spec
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'raw').write_bytes(os.urandom(4096));s=spec('v13/objects/'+'a'*64+'.bin',p/'raw')
            with patch.dict(os.environ,{'V13_COMPACT_ARCHIVE':'1','V13_STORAGE_STAGING_DIR':str(p/'uploads')}):
                self.assertEqual(physical_spec(s),s)
                (p/'raw').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'source changed'):physical_spec(s)
