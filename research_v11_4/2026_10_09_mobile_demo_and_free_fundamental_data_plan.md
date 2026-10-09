# V11.4 — Mobile demo and free-fundamental-source workflow

Verified on 2026-10-09 IST, source-only research branch. No promotion of historical fundamentals to training.

## A real demo exists already

* Original verified market date: **2026-10-08**.
* Original model: **V11.4 standalone research frozen 2026-10-08**, technical price/volume + NSE catalyst 22 features. The historical 5Y/7Y company fundamentals are **not in** the frozen model.
* Real successful source and private-filing seal: [GitHub run 37812061580](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37812061580).
* 2,102 original NSE market candidates, 1,662 integrity-clean, 12 companies with Wilder RSI(14)>80, 10 frozen model picks privately archived. The 12 RSI matches are not necessarily the 10 model selections.
* Sign into Hugging Face as the archive owner and open [private V11.4 October 8 forward observation](https://huggingface.co/datasets/ayushverma1988/v10-multibagger-archive/tree/main/v11_4/forward_observations/2026-10-08).
* Inspect `verified_forward_top10.csv`, `four_screener_status_original_top10_NO_RERANK.csv`, and `forward_observation_manifest.json`. The real names, prices, provisional calibrated six-month scores and rule coverage are in that **private** archive, **not** GitHub logs or public artifacts.
* Never treat source-cutoff same-day close as an executable quoted buying price; 6-month outcomes are not yet known. Use only for demonstration and paper watch.

## Next **fresh** verified research scoring

* Official Indian cash market closes at 15:30 IST on Friday, 2026-10-09.
* Default-branch [V11.4 forward scheduler](https://github.com/ayushverma1988/v9-multibagger-calibration/blob/main/.github/workflows/v11_4_scheduled_forward_research.yml) specifies 13:25 UTC = **18:55 IST on weekdays**.
* Actual schedule triggers can be delayed by GitHub. A scheduled workflow is **not proof** that forward scoring ran. Check `daily-forward` conclusion = success and private new Hugging Face day folder; `schedule-contract` alone is insufficient.
* Holidays, missing NSE event catalog, stale market prices or duplicate private observation fail closed.
* Next stage of source recovery can continue in parallel without modifying frozen forward picks.

## Screener.in and free alternative sources

Official [Screener guide](https://support.screener.in/article/28-export-screen-results): Screener has no official public API; bulk screen CSV export requires Premium. Individual company pages and financial sheets may be viewed manually, subject to the site's [terms](https://www.screener.in/guides/terms/). Third-party open source GitHub clients reading public pages do not confer data redistribution, scraping, or licensing permission.

If a user legitimately obtains a CSV with `NSE Code` and requested fundamental columns, the strict **optional, offline** adapter is:

```bash
python src/v11_4_manual_screener_research_import.py \
  --csv /path/to/authorized_company_screen.csv \
  --asof 2026-10-09 \
  --output locally_imported_external_research
```

The parser checks exact NSE ticker, unique identity, percentage decimal scale and absent financial fields. It produces a **research-only**, timestamped CSV with an original source hash. It never fetches Screener, uploads private credentials, claims historic availability, joins names to securities, or changes V11.4 predictions/training. A present-day 5/7-year ratio is not a verifiable 2019 market-date fact.

Alternate free, authentic long-history path: continue reading original published NSE annual XBRL using verified FY2022 FourD parser, then reconciling filing-date, fiscal context, statement mode, and original hash. Existing recovered data: **151** strict FY2022/FY2023 company pairs from a 1,276-stock 2023 universe (11.83%), so 70%-for-12-fold training gate is still NOT MET.

[Current research readiness workflow](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37877485406) passed all 8 manual source import tests and original private model-seal proof. Its public artifact contains readiness metadata only.

## Acceptance criteria for demo vs production

* Research demo: **AVAILABLE NOW**, using original October 8 private frozen observation.
* Today's new close: **eligible to run after market close; scheduled attempt 18:55 IST**, not guaranteed successful or immediately published.
* Four-fundamental-screen ML training / reliable 2x predictive claims: **NOT READY**. Required PIT sourcing and genuine forward validation remain outstanding.
