# Intraday Signature-Conditioned Latent-SDE Alpha

Research code for 30-minute, sector-neutral return ranking on U.S. large caps from
one-minute OHLCV bars (Massive/Polygon aggregates). A neural network maps
path-signature features to the coefficients of a linear latent SDE. The SDE gives
an exact Gaussian or jump-mixture forecast of each stock's sector-excess return,
used for ranking and for uncertainty estimates. No quotes, trades or order-book
data are used, so costs are assumptions and no result is an executable-PnL claim.

## Workflow

| Step | Script | Notes |
|---|---|---|
| 1. Configure | `project_settings.py` | Universe, dates, model, evaluation settings |
| 2. Download | `download_data.py` | Resumable; retries on rate limits (needs `MASSIVE_API_KEY` in `.env`) |
| 3. Check data | `check_universe_data.py` | Missing files, days lost to gaps, suggested ticker drops |
| 4. Sanity tests | `test_linear_sde.py` | Van Loan vs quadrature/Monte Carlo, likelihoods vs scipy, gradients |
| 5. Train | `train_model.py` | One run; validation-only while `EVALUATE_TEST = False` |
| 6. Tune | `tune_sde.py` | Staged search (NLL weight, diffusion rank, capacity, optimiser, tails) |
| 7. Baseline | `train_xgboost.py` | Grouped XGBoost ranker on the same inputs, splits and labels |
| 8. Analysis | `signal_horizon_analysis.py`, `execution_research.py` | IC decay, persistence, reversal benchmark; cost-aware execution |

Core library: `minute_alpha.py`. Universes and sectors: `universe.py`. Data
client: `massive_download.py`. Install `requirements.txt`.

## Data and universe

- `UNIVERSE = "sp100"`: the S&P 100 as of the sample start, reconstructed from
  the current list by undoing later index changes, so names are not selected for
  having grown into the index. GOOG is dropped in favour of GOOGL.
  `"mega25"` is the original 25-name universe.
- Sectors use GICS after the March 2023 reclassification, which puts V and MA
  in Financials.
- Each regular session is expanded to a common minute grid. A missing bar carries
  the last price forward with zero volume, and `has_trade` and
  `staleness_minutes` mark that state. A ticker with no bars on a day removes that
  day for all tickers. Sessions that end before the close (a day still in
  progress at download time) are dropped. 13:00 early closes are kept.

## Features, target and model

- **Inputs (30-minute window, causal through the close of minute t):** a
  sector-relative price path (sector median shrunk to the universe median), log
  volume, range, realised volatility, VWAP deviation, close-location value, 5- and
  15-minute relative returns, and trading-state counters.
- **Representation:** a depth-2 signature of a selective lead-lag path. It has 13
  channels, giving 91 coordinates: level one plus antisymmetric areas. Added to
  it are trading-state summaries and cross-ticker price-price and price-volume
  Lévy areas, PCA-compressed on training windows only. For large universes the
  PCA is fitted on a deterministic subsample of windows.
- **Target:** the log return from the open at t+1 to the open at t+31, minus its
  sector mean.
- **Latent SDE.** Given the context c,

  dZ = (A(c) Z + b(c)) dt + G(c) dW,   A = skew − diag(softplus) (stable),   y = wᵀZ₁ + q + ε.

  - The mean comes from a matrix exponential.
  - The covariance ∫₀¹ e^{As} G Gᵀ e^{Aᵀs} ds is exact via Van Loan's block
    exponential, so no Monte Carlo is needed.
  - `SDE_DIFFUSION_RANK = 0` gives the deterministic ablation.
- **Tails (`SDE_JUMP_MODE`):**
  - `"merton"`: adds a compound-Poisson jump with context-dependent intensity and
    jump variance. Its likelihood is the exact truncated Poisson mixture.
  - `"student_t"`: a Student-t law, as a control.
  - `"none"`: Gaussian.
- **Loss:** a differentiable Spearman surrogate (correlation with normal-score
  ranks) plus `SDE_NLL_WEIGHT` × the predictive NLL. Targets are standardised by
  a training-only robust scale.
- **Ensemble:** seeds are z-scored within (time, sector) and averaged. Predictive
  laws are combined as an exact equal-weight mixture.

## Evaluation conventions

- Chronological whole-day split: 70% train, 15% validation, 15% test.
  Train, validation and test are all sampled every 5 minutes.
- **Sector-neutral IC.** Scores are demeaned within sector in the loss, in the
  per-epoch validation IC used for checkpoint selection, and in every reported
  Rank IC (`sector_neutral_objective`). This matches the within-sector target
  and the sector-neutral portfolio.
- Daily IC is the mean over a day's cross-sections. Confidence intervals come
  from a day-block bootstrap, which accounts for overlapping intraday labels.
- **The test split stays sealed** (`EVALUATE_TEST = False`) until hyperparameters
  are frozen. It is then evaluated once.
- **Distribution diagnostics** (`sde_distribution_metrics.json`):
  - NLL against a per-symbol homoskedastic baseline;
  - PIT and coverage up to 99.9%;
  - Mincer-Zarnowitz regression;
  - for Merton runs, jump intensity by time and symbol, and posterior jump
    probabilities.
- **Benchmarks:** the XGBoost ranker (paired daily-IC comparison) and the simple
  5-minute reversal factor. A model should beat both.

## Execution (research backtest)

- **Positions:** sector-neutral, rebalanced only on the non-overlapping 30-minute
  grid. Fixed sector gross budgets, partial smoothing toward the target, and a
  sector-level turnover buffer. Flat at every close.
- **Costs:** charged on one-way turnover at an assumed rate in bp.
- **Optional profitability gate:** a regularised logistic model fitted on
  validation.
- **`execution_research.py`:**
  - checks a fast numpy simulator against the pipeline;
  - runs a placebo test of the gate (random gates with the same acceptance
    rate);
  - evaluates an exact sector-neutral mean-variance optimiser with an L1
    turnover penalty, which acts as a per-name no-trade band.

## Current findings (validation only, 25-name universe)

- **Ranking.** The SDE and deterministic variants rank about equally well. The
  model's 30-minute IC, scored every 5 minutes, is about 0.01. That is no
  better than a 5-minute reversal factor.
- **Distribution.** The SDE's added value is a calibrated predictive
  distribution. With Merton jumps, coverage is close to nominal up to 99.9%, and
  NLL improves significantly on the baseline.
- **Horizon and persistence.** Alpha decays within 30 minutes. The score's
  autocorrelation at a 30-minute lag is about 0.02, so turnover dominates and
  the break-even one-way cost is only about 0.2–0.6 bp.
- These figures supersede the first published version of this repository. That
  version scored every 30 minutes, selected checkpoints on all-name IC, and
  inspected the test segment repeatedly.

## Data policy

Vendor bars, prediction files, checkpoints and artifacts are not committed
(see `.gitignore`). Supply your own Massive access to reproduce.
