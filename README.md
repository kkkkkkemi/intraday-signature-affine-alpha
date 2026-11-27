# Intraday Signature-Conditioned Latent-SDE Alpha

Research code for 30-minute, sector-neutral return ranking on U.S. large caps from
one-minute OHLCV bars (Massive/Polygon aggregates). A neural network maps
path-signature features to the coefficients of a linear latent SDE. The SDE gives
an exact Gaussian or jump-mixture forecast of each stock's sector-excess return,
used for ranking and for uncertainty estimates. No quotes, trades or order-book
data are used, so costs are assumptions and no result is an executable-PnL claim.

**Status:** finished. The configuration in `project_settings.py` was frozen on
validation and the sealed test split was scored once. **Bottom line:** the
30-minute cross-sectional signal is small (validation Rank IC 0.87%) and did not
hold up on the test split (0.24%, not significant); break-even costs are far
below realistic costs, so this is not a tradable strategy. The model's
predictive distribution, however, stays calibrated out of sample. See
[Results](#results).

## Workflow

| Step | Script | Notes |
|---|---|---|
| 1. Configure | `project_settings.py` | Universe, dates, model, evaluation settings |
| 2. Download | `download_data.py` | Universe plus SPY / sector ETFs (`DATA_DIR/etf`); resumable; stitches ticker changes (BK→BNY) |
| 3. Check data | `check_universe_data.py` | Missing files, days lost to gaps, suggested ticker drops |
| 4. Sanity tests | `test_linear_sde.py` | Van Loan vs quadrature/Monte Carlo, likelihoods vs scipy, gradients |
| 5. Train | `train_model.py` | One run; validation-only while `EVALUATE_TEST = False` |
| 6. Baseline | `train_xgboost.py` | Grouped XGBoost ranker on the same inputs, splits and labels |
| 7. Filter and execution | `filter_execution_analysis.py` | Holding-period filter vs re-forecasting: information, turnover and cost on the 5-minute grid |

Core library: `minute_alpha.py`. Universes and sectors: `universe.py`. Data
client: `massive_download.py`. Install `requirements.txt` (`xgboost` is only
needed for the baseline).

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
- **v2 context inputs (endpoint values, plus window means for the flow
  features):**
  - volume surprise: per-minute and cumulative, against the same minute over
    the prior 20 sessions;
  - trade count and average trade size;
  - return since open, sector-relative overnight gap, prior-day return and
    volatility, intraday range position, and deviation from the session VWAP;
  - SPY 5/15/30-minute returns, and the own-sector SPDR return relative to SPY.

  A future-perturbation test checks that every feature uses only data up to
  minute t.
- **Representation:** a depth-2 signature of a selective lead-lag path. It has 13
  channels, giving 91 coordinates: level one plus antisymmetric areas. Added to
  it are trading-state summaries and cross-ticker price-price and price-volume
  Lévy areas, PCA-compressed on training windows only. For large universes the
  PCA is fitted on a deterministic subsample of windows.
- **Target:** the log return from the open at t+1 to the open at t+31, minus its
  sector mean.  With `SDE_HORIZON_STEPS = K` the cumulative sector-excess
  returns at every 30/K minutes of the holding period are modelled jointly.
- **Latent SDE.** Given the context c,

  dZ = (A(c) Z + b(c)) dt + G(c) dW,   A = skew − diag(softplus) (stable),   y = wᵀZ₁ + q + ε.

  - The mean comes from a matrix exponential.
  - The covariance ∫₀¹ e^{As} G Gᵀ e^{Aᵀs} ds is exact via Van Loan's block
    exponential, so no Monte Carlo is needed.
  - `SDE_DIFFUSION_RANK = 0` gives the deterministic ablation.
  - **Ranking is identical to the ODE.** In a linear SDE the diffusion and the
    zero-mean jumps do not move the mean, so the SDE only adds a predictive
    distribution on top of the ODE score.
- **Multi-horizon law and holding-period filter.** With the observation
  y(τ) = wᵀ(Z_τ − Z₀) + qτ + σ_B B_τ + ε on τ_k = k/K, the means and the full
  cross-covariance Cov(y_j, y_k) = wᵀP_j(e^{Aᵀh})^{k−j}w + σ_B² min(τ_j, τ_k)
  follow exact recursions (Van Loan step covariance). Merton jumps arrive
  independently in each sub-interval, so the joint law is a finite Gaussian
  mixture over jump-count vectors. After m sub-intervals of the holding period
  the forecast of the remaining return is updated by exact Gaussian conditioning
  within each mixture component, with the components re-weighted by the
  likelihood of the observed path (a Kalman update that does not mistake an
  early jump for drift). The update uses the causal partial return to the close
  of the observation minute; see the look-ahead note in the results.
- **Identity and interaction:**
  - a per-stock embedding (`STOCK_EMBEDDING_DIM`);
  - transformer layers across the stocks of each cross-section
    (`CROSS_ATTENTION_LAYERS`).

  Both feed only the context that conditions the SDE coefficients, and setting
  either to 0 gives the ablation.
- **Tails (`SDE_JUMP_MODE`):**
  - `"merton"`: adds a compound-Poisson jump with context-dependent intensity and
    jump variance. Its likelihood is the exact truncated Poisson mixture.
  - `"student_t"`: a Student-t law, as a control.
  - `"none"`: Gaussian.
- **Loss:** a differentiable Spearman surrogate (correlation with normal-score
  ranks) plus `SDE_NLL_WEIGHT` × the predictive NLL. Targets are standardised by
  a training-only robust scale.
- **Regularisation and ablations:** AdamW weight decay, dropout on the
  conditioning context and inside the attention layers, early stopping on
  validation IC, and `DROP_FEATURE_GROUPS`, which zeroes named v2 feature groups
  (`day_constant`, `volume_flow`, `session`, `etf`) while keeping the input
  dimension fixed.
- **Ensemble:** seeds are z-scored within (time, sector) and averaged. Predictive
  laws are combined as an exact equal-weight mixture.

## Evaluation conventions

- Chronological whole-day split: 70% train, 15% validation, 15% test.
  Train, validation and test are all sampled every 5 minutes.
- Each epoch logs Rank IC on a fixed subsample of training windows next to
  the validation IC, to tell under- from over-fitting.
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
- **Benchmark:** the XGBoost ranker on the same inputs, with a paired daily-IC
  comparison.

## Execution (research backtest)

- **Positions:** sector-neutral, rebalanced only on the non-overlapping 30-minute
  grid. Fixed sector gross budgets, partial smoothing toward the target, and a
  sector-level turnover buffer. Flat at every close.
- **Costs:** charged on one-way turnover at an assumed rate in bp.
- **Optional profitability gate:** a regularised logistic model fitted on
  validation.

## Results

Universe: point-in-time S&P 100 (100 names). Sessions: train 2024-12-03 to
2026-04-21 (345 days), validation 2026-04-22 to 2026-08-06 (74), test
2026-08-07 to 2026-11-20 (75). Frozen configuration, chosen on validation only:
no stock embedding or attention, v2 feature groups switched off (they did not
raise validation IC), lr 1e-4, weight decay 1e-3, context dropout 0.1, rank-4
latent SDE with Merton jumps and a 6-step multi-horizon likelihood, three seeds.
Every interval below is a 95% day-block bootstrap interval.

**Ranking (30-minute sector-neutral daily Rank IC)**

| Model | Validation | Test |
|---|---|---|
| Latent SDE ensemble | 0.87% [0.22%, 1.56%] | 0.24% [−0.40%, 0.87%] |
| XGBoost ranker, same inputs | 0.50% [−0.08%, 1.08%] | 0.57% [0.08%, 1.06%] |
| XGBoost − SDE (paired, test) | | +0.33% [−0.24%, 0.90%] |

- Validation IC was the same, within seed noise, for every architecture tried
  (with and without stock embedding, cross-stock attention and the v2 features).
  The best epoch was always the first or second; after that, training-subset IC
  kept rising while validation IC did not.
- On the test split the SDE's IC is not significantly different from zero, and
  the two models are not significantly different from each other.
- The multi-horizon likelihood left ranking unchanged: per-seed validation IC
  matched the endpoint-only model to within 0.0001.

**Predictive distribution (test split)**

- NLL 1.621 against 1.680 for a per-symbol homoskedastic Gaussian baseline:
  an improvement of 0.059 [0.044, 0.080] nats per label.
- Central-interval coverage at 50 / 80 / 90 / 95 / 99 / 99.9%: 54.3 / 81.6 /
  90.1 / 94.4 / 98.3 / 99.6%. The PIT KS statistic is 0.033.

**Term structure and holding-period filter (causal)**

- Validation Rank IC of the cumulative forecast is highest at 5 minutes and
  falls with the horizon (1.05% at 5 min, 0.79% at 30 min). Most of the
  predictability sits in the first 5 to 10 minutes.
- After observing 5 to 25 minutes of the holding period, the remaining return is
  ranked as follows:

  | | Validation | Test |
  |---|---|---|
  | Original forecast (not updated) | ≈ 0.2 to 0.35% | ≈ −0.35 to −0.03% |
  | Filter update | 0.40 to 0.71% | 0.80 to 1.25% |
  | Re-run model at the observation time | 0.68 to 1.04% | 0.22 to 0.44% |

  On the test split the update beats both alternatives at every observation
  time; for example, after 15 minutes the filter exceeds the re-run model by
  +0.87% [0.19%, 1.57%]. On validation it does not.
- The update is close to a short-term reversal of the observed path: the causal
  partial return alone has an IC of −1.0 to −1.4% on test.
- The filter's conditional variance is calibrated: z-score standard deviation
  1.01 to 1.03, and 90% coverage 90.4 to 91.2% on test.
- **Look-ahead note.** A first version conditioned on the partial return to the
  open of the next minute. That price is also the execution price and the
  first price of the remaining return, so bid-ask bounce leaked into it. It
  roughly doubled the apparent reversal (validation partial-return IC −1.1%
  against −0.6% when causal) and made the filter look significant on
  validation. All reported numbers use the causal version.

**Economics**

- **30-minute strategy** (sector-neutral, flat at the close). Annualised test
  Sharpe at zero cost is −2.0 for the SDE and +1.2 for XGBoost; both are
  negative at 0.6 bp one-way.
- **5-minute decisions** (filter, re-run model, or reversal, with partial
  adjustment). On validation the best break-even one-way cost is 0.27 bp, below
  the ~0.2 bp half-spread of the most liquid names plus fees and impact. On
  test, gross PnL with linear weights is about zero or negative, even though the
  filter's rank IC is positive. The rank effect does not carry over to
  magnitude-weighted positions.
- **Why.** With break-even cost ≈ 1.25 · IC · σ / √(2(1 − ρ)), σ ≈ 18 bp per 30
  minutes and almost no signal persistence (ρ ≈ 0), even a doubled IC would
  break even only around 0.3 bp. Minute-bar OHLCV on S&P 100 names does not
  carry enough cross-sectional information for an intraday strategy after
  costs.

**What the SDE adds.** It does not improve ranking, and cannot, since its mean
is the ODE's. What it adds is a calibrated predictive law with jumps, an exact
term structure across the holding period, and a calibrated filter for the
remaining return.

## Data policy

Vendor bars, prediction files, checkpoints and artifacts are not committed
(see `.gitignore`). Supply your own Massive access to reproduce.
