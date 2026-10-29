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

Primary engineering references:
- https://huggingface.co/docs/hub/storage-limits
- https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#create_commit

The Hub recommends smaller HTTP commits and notes that a timeout can occur
after a successful server-side commit. Its API also documents mutation of
operation objects. The repair preserves the existing private connection and
content-addressed object layout.
