# V13 prospective single-source breadth expansion

The current paper scorer can use one timely, hash-bound adjusted-price source
when no second provider history exists or fewer than 126 comparable sessions
are available. The source must have the current universe's exact symbol and
ISIN, a valid current series, internally consistent adjustment factors, and
an exact match to the source-day original NSE OHLCV. An observed provider
conflict is still blocked. A one-session-lagged adjusted file may be extended
only by the exact NSE current session and only when the timely source-day
action directory has no entry for that symbol. These tests do not certify a
complete historical no-action interval.

Historical EQ/BE/BZ/SM/ST series changes for the same symbol and ISIN can be
retained for scoring. They are explicitly flagged as unverified transitions;
multiple ISINs, ambiguous dates, unsupported series and mismatched current
exchange identities are held. A source-day action is held. The view is
temporary: it does not write provider history, alter a checkpoint, or certify
an entry, outcome, return, price loss or doubling. Each new registry row and
dual-RSI output retains the source, reference-comparison and transition status.

The current prospective paper scorer also accepts positive-turnover names
below the former INR 25 lakh median-turnover floor if all existing minimum
history, gap, jump and recent-session checks pass. Such names are labelled
`BELOW_25_LAKH_OUTSIDE_TRAINING_BAND`; the frozen heads were trained under the
former floor, so their calibration and trading liquidity are unverified for
these rows. For the original-provider path, each newly added low-turnover row
also needs a hash-bound receipt captured after the source close and by the
source-day cutoff. Its actual receipt clock is frozen with a new registry row.
This does not change the historical training or frozen heads. Both
strict RSI screens continue to use the same fixed head, identity, observed
financial-weakness and source-conflict rules.

The bounded alternative-financial queues are enlarged prospectively: NSE
master requests 480/120 seconds, Yahoo statements 960/450 seconds, and the
missing-web queue 640/180 seconds. Existing source rate limits, 403/429 stops,
actual retrieval dates and unknown/contradictory fact handling remain in force.
The budgets are maximums, not promises of downloaded or verified facts.

Only sources actually captured by the real information cutoff can support a
new cohort. Post-cutoff implementation diagnostics are not forecasts, are not
used to enlarge an already registered cohort, and are not promotion evidence.
The original full-ledger and elapsed-market-time rules continue unchanged.
