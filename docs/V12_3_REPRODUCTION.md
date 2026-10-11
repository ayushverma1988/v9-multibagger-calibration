# V12.3 selective reliability research

This is development research, not an approved investment model. Keep the 9 October 2022–9 October 2026 boundary and no earlier indicator warmup. The V12.1 frozen weights and final prospective registration remain unchanged.

## Evidence and source cohort

The prior `Multibagger_V12_1_Frozen_Model_And_Evidence_2026-10-10.zip` supplies original NSE histories and the full public request universe. `Multibagger_Dated_Inputs_BSE_Prospective_Evidence_2026-10-10.zip` supplies dated catalogues, authoritative financial facts, event metadata, the original monthly panel, BSE histories, and registered forecasts. Verify the archives against the new evidence manifest. Use `dated_financial_validated_snapshot`, not obsolete normalized outputs from the original-response directory.

The expanded financial cohort is the first 512 SHA256-sorted symbols from the full public NSE universe, with all their dated catalogue filings. Some symbols have no catalogue documents. This remains a sample, not full NSE/BSE coverage. Neither historical winners nor model selections choose outbound requests.

## Reproduction

Use Python 3.12 with `requirements_v12_frozen.txt` and `requirements_v11_4_systematic.txt`. Set `PYTHONPATH=src` from the repository root. The paths below refer to extracted evidence directories.

```bash
python src/v12_dated_training_inputs.py --kind collect --catalogs OLD/dated_catalogs --universe UPSTREAM/v12_four_year_history/complete_public_NSE_request_universe.parquet --output NEW/financial --companies 512 --asof 2026-10-09
python src/v12_3_point_in_time.py --panel OLD/upstream_NSE_four_year_panel/monthly_four_year_features_and_outcomes.parquet --facts NEW/financial/dated_primary_financial_facts.parquet --old-matrix OLD/dated_training_inputs/dated_financial_catalyst_training_input_matrix.parquet --output NEW/joined
python src/v12_3_selective_evaluation.py --panel OLD/upstream_NSE_four_year_panel/monthly_four_year_features_and_outcomes.parquet --matrix NEW/joined/training_inputs.parquet --history UPSTREAM/v12_four_year_history --bse-panel OLD/BSE_chronological_validation/BSE_four_year_panel_and_outcomes.parquet --protocol config/v12_3_registered_protocol.json --output NEW/evaluation
python src/v12_3_selective_evaluation.py --panel OLD/upstream_NSE_four_year_panel/monthly_four_year_features_and_outcomes.parquet --matrix NEW/joined/training_inputs.parquet --history OLD/BSE_four_year_full_public --scores NEW/evaluation/BSE_transfer_development_scores.parquet --conflicts OLD/BSE_secondary_price_validation/BSE_secondary_price_comparisons.parquet --output NEW/BSE_evaluation
python src/v12_3_catalyst_evidence.py --events OLD/dated_training_inputs/dated_primary_event_metadata.parquet --output NEW/catalysts --per-year 64
python src/v12_3_official_monitor.py --date 2026-10-09 --output NEW/monitor_sources --registry OLD/V12_1_prospective_registry_final_20261010
python -m unittest discover -s tests
```

Copy original raw caches into source directories before collection to avoid downloading identical documents. Cache reuse checks hashes and retrieval clocks. Do not replace successful original bytes. Model fitting writes an exclusive run receipt and refuses to overwrite it. The saved heads are explicitly unpromoted research weights.

`add_materiality` in `v12_3_catalyst_evidence.py` produces a separate review file linking gross contract amounts to annual revenue known at the disclosure date. It is not a profit forecast. Repeated amounts are not summed, tax-inclusive amounts remain labelled, and execution durations beyond twelve months remain visible.

## Interpretation

The five-field research gate uses PAT margin, not OPM. It does not replace historical valuation, recurring-earnings reconciliation, promoter/pledge history, or verified incremental earnings. Missing values never pass. Banks and insurers need separate financial definitions before extending claims to those sectors.

The public current-listing universe still implies survivor bias. Delistings, historical identity changes, corporate actions and executable exits are incompletely verified. BSE transfer trains only on earlier NSE data; all unavailable BSE financial/catalyst inputs remain unknown. Existing secondary-price conflicts quarantine fixed BSE selections without replacements.

Calendar-followup comparisons retain unknown selected outcomes in the precision denominator. Known-only return summaries are labelled. Fixed-slot cohorts reserve unused slots as zero-interest cash; an unresolved held position makes the cohort return unknown. Overlapping cohort averages are not portfolio CAGR. The moving-block interval is descriptive, only partly addresses repeated time exposure, and cannot establish independence across repeated issuers. Short tests return no interval.

The official monitor verifies raw daily OHLC, positive volume, ISIN, trade counts and lot sizes. It does not invent adjustment factors, certify the first actual entry without further evidence, or modify the registered outcome protocol. The GitHub snapshot workflow is manual and unmerged. A separate ChatGPT task, V12 Prospective Data Watch, was enabled on 10 October 2026 for weekdays around 19:00 Asia/Kolkata, starting 12 October 2026. No scheduled capture has run yet. It is instructed to preserve completed official whole-market responses, backfill missing completed sessions and report coverage failures without changing forecasts. A scheduled task is not proof of a successful capture or of outcome maturity. The older V10 schedule does not constitute V12 prospective monitoring.

A separate reviewer can reproduce source hashes, dates and outcome arithmetic from these files. CI and separately implemented arithmetic are not independent investment validation. Real future outcomes still require actual six- and twelve-month horizons.

## Separate future comparison registration

After the fixed development run, register all four NSE variants together before a new market session. This adds a new research registration and does not replace V12.1 or approve trades. All current NSE forecasts are retained, with the exact membership for each variant and each horizon, including zero selections. Use the original frozen adjudication protocol for entry, adjustments, full elapsed horizons, costs and UNKNOWN outcomes. No BSE prospective registration is claimed.

```bash
python src/v12_3_prospective_comparison.py --scores NEW/evaluation/chronological_development_scores.parquet --model NEW/evaluation/UNPROMOTED_research_heads.joblib --run-receipt NEW/evaluation/model_run_receipt.json --protocol config/v12_3_registered_protocol.json --original-registry OLD/V12_1_prospective_registry_final_20261010 --output NEW/prospective_comparison
```

`validate_comparison` verifies the stored registration and adjudication-protocol hashes. The separate registration has its own actual UTC decision time; never inherit the earlier V12.1 decision time. The whole-market collection can support both registrations, but collecting raw prices does not supply verified adjustment factors, first-entry proof, or mature future outcomes.

## Current BSE research inference

The saved four-year BSE histories ended on 8 October, so their monthly panel did not contain the 9 October decision. The new adapter fetches recent one-month Yahoo .BO charts for the whole saved public universe. It requires exact provider identity, BSE/INR/timezone, a completed 9 October bar, at least three overlapping sessions, matching OHLC/adjusted-close within ₹0.011 and exact volume. Any unexplained revision rejects that issuer; it is not silently corrected. The adapter appends verified overlapping-source history, computes the original features against the same current NSE reference, and uses the saved unpromoted NSE heads without fitting. All unavailable BSE financial and catalyst fields remain UNKNOWN. This is free end-of-day research coverage, not a primary BSE feed, proof of live latency or independent corporate-action verification.

```bash
python src/v12_3_BSE_current_refresh.py --reference OLD/BSE_four_year_full_public/public_BSE_exclusive_request_universe.parquet --history OLD/BSE_four_year_full_public --nse-scores NEW/evaluation/chronological_development_scores.parquet --model NEW/evaluation/UNPROMOTED_research_heads.joblib --output NEW/BSE_current
```

This current inference is separate from the already scored historical BSE comparison. Historical selections and performance remain unchanged. Six additional tests cover output-directory reuse, current-bar absence, identity, adjustments, volume conflicts and source-preserving appends.
