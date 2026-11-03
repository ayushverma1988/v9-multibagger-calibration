# Dated inputs, BSE transfer and prospective evaluation

This change adds source collection and evaluation around the immutable V12.1 research freeze. It does not approve an accurate 2× stock model. All training observations remain inside 9 October 2022–9 October 2026. Fiscal dates never substitute for publication times. Current provider financial tables are not historical training inputs.

## Dated financials and catalysts

`v12_dated_source_collection.py` mines whole public NSE main/SME catalogs. Responses, original URLs, first-retrieval clocks and SHA-256 hashes are retained. Integrated financial results use verified pagination: request page 1 corresponds to response page 0. Repeated page IDs fail visibly.

`v12_dated_training_inputs.py` normalizes maximum broadcast/dissemination/revision timestamps. Original XBRL extraction requires a matching issuer/basis, an original document or primary catalog ISIN, finite INR facts and exact economic periods. A contradictory context is rejected individually; a clean quarter cannot be reclassified as annual YTD. Missing concepts, old periods, conflicting identities and changed ISINs remain errors. Exact facts are not automatically independent rendered-statement reconciliation or proof of immutable historical revisions.

The completed source control run selected 128 companies by SHA-256 from the full official public NSE list. It requested 2,142 filings, parsed 1,915, and recovered 22,223 facts across 115 issuers. This is a control sample, not full financial coverage of 3,182 companies. Its dated facts produced partial financial predictors for 1,400/56,261 market-panel rows across 64 issuers. Other financial inputs remain UNKNOWN.

772,280 collected announcement rows yielded 752,293 dated canonical metadata records. Counts describe observed disclosures. They do not establish earnings impact or complete feed absence. `v12_dated_catalyst_documents.py` reads a public hash sample of original attachments; 22 quantified candidates were recovered, but no complete earnings causal chains were verified.

## One separately registered experiment

`v12_dated_enrichment_experiment.py` adds quarterly sales/PAT growth, PAT margin, annual CFO/PAT, debt/equity and observed order/capacity disclosure counts to the 19 market features. Its only numerical missing-value treatment is a training-only median plus an explicit missingness indicator; a feature entirely absent in training is dropped for that fold. The original source matrix retains UNKNOWN/NaN. Numerical treatment does not pass a financial rule.

The 70/30 regularized logistic/shallow boosting recipe, strict maturity/calibration clocks and 2× targets remain unchanged. There is one registered experiment and no parameter search. On the same 19 original six-month test dates it records 11/190 hits versus 10/190; the same four twelve-month dates remain 3/40. Twelve-month Brier scoring is slightly worse. The 30% precision goal fails. The experiment is not promoted and does not replace the frozen weights or current forecasts. Its larger fully known selection-date summary has 11/200 at six months; this must not be compared as if its denominator equaled the earlier 190.

## BSE-only source-reference transfer

`v12_bse_transfer.py` checks `.BO` symbol, venue, INR currency, equity type, issuer name, daily OHLC and adjusted close. History is trimmed before indicator warmup. NSE market reference distributions provide the stock-relative features. Historical transfer fits use strictly earlier NSE labels; today's weights never score historical outcomes, and BSE labels never train these heads.

The public matched BSE-exclusive reference cohort contains 1,679 requests and 1,653 usable histories, with 1,482,408 valid bars. Liquidity/history rules leave 1,191 security-date rows across 151 securities. Retrospective results are 7/182 confirmed six-month opportunities and 2/38 twelve-month opportunities. Five six-month outcomes and three twelve-month outcomes remain UNKNOWN after the secondary-price sensitivity check; they remain in the denominators. Unknown selections are never replaced.

`v12_BSE_secondary_price_check.py` reads a second free provider's visible history for 63 predetermined public controls. 59 pages were usable; 2,863/2,889 common stock-date closes agreed within ₹0.011 and 26 disagreed. Both provider identity binding and action representation need primary verification. Delayed history is not guaranteed live tick data. Current BSE-exclusive membership does not certify historical BSE-only membership or a full active/delisted universe.

## Prospective registration

`v12_prospective_registry.py` seals the exact original forecast file, weights, recipe, entry/target/cost/completeness protocol and executable adjudication code. It rejects altered weights/protocols, backdated/stale registrations, future/unclocked prices, mismatched currencies/venues/identities and incomplete outcome horizons. A 2× hit before maturity remains pending. Pre-target loss risk excludes losses after a successful target exit.

The final registration has 639 forecasts, 10 source-checked research selections and zero conviction selections. There are zero matured unseen outcomes. The actual first verified post-registration trading open is still pending. Calendar six/twelve-month deadlines are based on that actual entry date, never the October 9 quote date. The initial observation-chain record contains 1,278 PENDING_ENTRY horizon rows. Entry verification and source evidence must be supplied; a scheduler or independent human assessor is not deployed by this PR.

## Commands

Run from the repository with `PYTHONPATH=src` and the pinned requirements. Existing evidence paths below refer to the accompanying saved source archive; new output directories are deliberate.

```bash
python -m v12_dated_source_collection --kind catalogs --output dated_catalogs --asof 2026-10-09
python -m v12_dated_training_inputs --kind collect --catalogs dated_catalogs --universe complete_public_NSE_request_universe.parquet --companies 128 --output dated_financial_originals
python -m v12_dated_training_inputs --kind join --panel monthly_four_year_features_and_outcomes.parquet --financial dated_primary_financial_facts.parquet --catalogs dated_catalogs --output dated_training_inputs
python -m v12_dated_catalyst_documents --catalogs dated_catalogs --output dated_catalyst_documents
python -m v12_dated_enrichment_experiment --panel monthly_four_year_features_and_outcomes.parquet --matrix dated_financial_catalyst_training_input_matrix.parquet --output NEW_EXPERIMENT_DIRECTORY
python -m v12_dated_source_collection --kind bse --reference BSE_latest_quotes_source_audit_PRIVATE.parquet --output BSE_four_year_full_public
python -m v12_bse_transfer --history BSE_four_year_full_public --nse-panel monthly_four_year_features_and_outcomes.parquet --output BSE_transfer
python -m v12_prospective_registry --kind evaluate --output EXISTING_FINAL_REGISTRY --bars VERIFIED_SOURCE_OBSERVATIONS.parquet
```

The 128-company sample can be expanded using `--companies`; source coverage must be measured again. Reusing an existing prospective registration requires its exact pinned adjudication code. Outcomes need real future sessions. Full financial/catalyst coverage, historical membership/action verification and an accurate independently confirmed predictor remain unresolved. Software tests validate contracts, not investment returns.
