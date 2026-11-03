# V13 collector checkpoint repair — 29 October 2026

Scheduled run 40770356133 failed on source-28 October while writing private
archive objects. The intermediate save after price-history collection and the
failure-handler save both received HTTP 504 from the Hub commit endpoint.
Financial collection and current-model analysis were not reached. A GitHub
artifact was retained, but it must not be described as a complete copy of all
new histories: the prior artifact configuration did not include a delta pack.

The archive used 200 object additions per commit without write retries. The
repair caps commits at 50 objects, bisects a gateway-failing batch, retries
transient single-object and atomic-manifest requests within a fixed bound,
and honors throttling headers. Each attempt creates fresh upload operations.
After a lost or failed server reply, Git/LFS content digests are checked before
declaring success or resubmitting. A private upload journal records only
confirmed object hashes, so the final save can resume earlier successful
batches. Permission and source-integrity errors remain fatal.

The snapshot and latest manifest stay in one atomic publication after objects
and the derived bundle have receipts. If saving fails, a local recovery delta
contains the previous manifest, planned manifest and every new local object,
with lengths and SHA-256 values; the remote pointer is not advanced locally.
The worker artifact now includes this recovery pack and its receipt/journal.
Successful publication removes redundant emergency bytes. There is no model
refit, forecast rescore, relaxed coverage gate or weakened source admission.

Fourteen focused local regressions pass, covering lost-success replies, fresh
operations, gateway bisection, bounded retries, throttling, permission failures,
atomic manifests, upload resume, recovery byte preservation and source changes.
Full exact-revision CI and a replacement collector must pass before recovery
is reported as complete.

## Follow-through after replacement run 40803851382

The replacement reached its 180-minute GitHub job ceiling at 05:01 UTC. Logs
show that history finished near 02:34 UTC and uploads then consumed the
remaining time. The phase marker had not advanced before the intermediate
save, so aggregate health incorrectly left the last phase as history
collection. Its journal confirms 8,250 uploaded objects, but no checkpoint
receipt or complete recovery delta was produced. This is not a completed
forecast or proof that the whole collection was preserved.

Large deltas now group new immutable bytes into deterministic packs of at
most 32 MiB and 500 members. Each member retains its exact SHA-256, byte
length and content address; an additional pack hash and offset determine its
storage location. Existing direct object addresses remain unchanged. Restart
bundles and original-document readers verify both storage and member hashes.
Packed manifests use format 2, so old writers fail closed instead of silently
dropping storage metadata. Compatible readers retain support for format 1.

An offline 8,250-object regression needs 17 packs and three commits (objects,
restart bundle, atomic snapshot/latest publication), rather than 165 object
commits alone. This measures operation reduction, not a promised live speed.
All bytes, including frozen and unselected records, round-trip exactly.

A complete recovery delta is built **before the first network write**. It
survives a hard interruption without relying on a Python exception handler.
It contains both manifests and every new object's original bytes, sufficient
to reconstruct the exact packs. An unsuccessful publication does not advance
the local or remote latest pointer. Confirmed pack uploads can be resumed.
Each save stops starting requests after its 900-second budget, retains the
delta, and fails visibly. Phase labels and aggregate upload progress now
identify checkpoint time separately from collection time.

Focused tests cover scale, full restore, original-source reads, format-1
compatibility, packed metadata preservation, tampered hashes and offsets,
lost success replies, hard interruption, deadline expiry and recovery-pack
reconstruction. Exact-revision push CI also round-trips three existing
original source objects through one private pack, without changing any
checkpoint pointer, forecast, model or source bytes. A complete production
worker and independent artifact/receipt audit remain the live release gate.

Primary engineering references:
- https://huggingface.co/docs/hub/storage-limits
- https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#create_commit

The Hub recommends smaller HTTP commits and notes that a timeout can occur
after a successful server-side commit. Its API also documents mutation of
operation objects. The repair preserves the existing private connection and
content-addressed object layout.
