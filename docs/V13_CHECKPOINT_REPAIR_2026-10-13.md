# V13 checkpoint directory-limit repair — 13 October 2026

The scheduled worker 38320556615 failed on 12 October at 23:15 UTC while
committing checkpoint objects to the private archive. The service rejected
the commit because `v13/objects/` would exceed 10,000 files. The same failure
affected preceding runs 38284340001 and 38285439167. Data collection stopped
before market scoring; the zero market-row health values do not establish
zero coverage or a new empty forecast.

New object hashes use `v13/object_shards/<first-two>/<next-two>/<hash>.bin`.
Existing hashes retain their exact legacy object addresses. Both layouts
require valid SHA-256 addresses and checked byte length and digest on restore.
No existing source objects, forecast records, probabilities or checkpoints
are deleted, migrated or backdated. The latest manifest still advances only
after all referenced new objects have been committed.

Regression tests simulate a full 10,000-file legacy directory, save 401 new
objects in multiple upload batches, and restore every referenced byte. Other
tests check immutable forecast reuse, duplicate-content addresses, mixed
layout reading, invalid shard addresses and corrupted bytes.

Operational deployment and live-run results are recorded in the monitoring
report. Collector recovery does not establish forecast accuracy, financial
completeness or independently verified mature outcomes.
