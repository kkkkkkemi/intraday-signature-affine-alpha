# Intraday Signature-Affine Alpha

This project now studies intraday U.S. equity alpha from Massive minute OHLCV
aggregates. It does **not** claim to model the limit order book unless a separate
L1/L2 source is introduced.

The main experiment combines selective path signatures with a neural-network-
parameterised affine latent drift and a linear readout. Its conditional mean is
computed analytically, so the current training and inference code does not draw
stochastic paths. `train_xgboost.py` provides an independent grouped-ranking
baseline on the same engineered inputs and chronological split. See
[`RESULTS.md`](RESULTS.md) for the matched comparison and its limitations.

In the latest local 75-day test, mean daily Rank IC was 0.0227 for the
signature-affine ensemble and 0.0089 for the XGBoost reference. The paired
95% day-bootstrap interval for their difference includes zero, so this is
an observed comparison, not a statistically established model advantage.

This repository contains research code and aggregate findings. It does not
include vendor minute bars, LOBSTER files, credentials, per-row predictions, or
trained checkpoints. Supply your own authorised data access to reproduce a run.

## Scope and execution convention

- Input: regular-session, one-minute OHLCV and VWAP aggregates from Massive.
- Features: intraday time, cumulative and multi-scale sector-relative returns,
  log volume, relative range, rolling realised volatility, close-to-VWAP
  deviation, Close Location Value (CLV), plus a real-bar indicator and
  price-staleness counter for missing aggregates.
- Signal timing: all features through the close of minute `t`.
- Execution target: enter at the open of `t+1` and mark at the open of the next
  rebalance, `t + holding_minutes + 1`. Consecutive accepted targets therefore
  form one continuous intraday position. Training/IC target: the corresponding
  sector-neutral forward excess return, transformed to a cross-sectional
  normal-score rank.
- Model: each ticker's selective lead-lag signature—time once, continuous market
  variables in lead-lag form, and discrete trading-state summaries—plus all
  PCA-compressed price-price and price-volume cross-ticker contexts
  condition stable affine latent drift and a linear latent readout. The terminal
  conditional-mean alpha is computed by a differentiable matrix exponential, with
  no Brownian Monte Carlo at training, validation, or inference.
- Sampling: feature lookback and holding period are 30 minutes. Training uses a
  five-minute sliding stride with overlapping paths and labels; validation and
  final OOS use a strict 30-minute stride, so their sleeves are non-overlapping.
- Split: complete trading days are assigned chronologically as 70% training,
  15% validation, and 15% final OOS test.
- Evaluation: consecutive target-weight rebalancing with a configurable **per
  one-way turnover** cost assumption. A validation-fitted probability gate may skip an
  optional rebalance when its estimated incremental net-PnL probability is below
  the configured threshold. OOS reports a zero-friction path whose gate is fitted
  at 0 bp and a friction-aware path whose gate is fitted at the configured cost
  (currently 0.6 bp); their position
  paths are intentionally allowed to differ. The default intraday policy closes
  the final modelled target of every session, so unmodelled overnight/early-session
  gaps are excluded.
  Without historical bid/ask quotes, no reported result is an executable-PnL
  claim.

## Quick start

1. Copy `.env.example` to a new local file named `.env`, then put the Massive key
   after `MASSIVE_API_KEY=`. The downloader reads this file automatically.
2. Edit `project_settings.py` if you want a different ticker universe, dates, or
   model settings.
3. Run `download_data.py` directly from your normal Python IDE or double-click it.
   It downloads and stores the minute bars. Check your current Massive plan's
   rate and history limits before running a large download.
4. Once downloading finishes, run `train_model.py` the same way. It trains and
   evaluates the model from the locally stored bars.
5. Run `train_xgboost.py` for the comparison baseline after installing
   `requirements_xgboost.txt`. See [`XGBOOST_README.md`](XGBOOST_README.md).

The script is executed by the Python environment that launches it. This avoids the
PowerShell interpreter/path mismatch; choose your existing configured Python
interpreter in the IDE you normally use. `massive_download.py`,
`run_minute_alpha.py`, and `run_project.py` remain available as optional advanced
command-line entry points.

## GPU runtime

If CUDA is available, the model runs on GPU automatically. The default settings in
`project_settings.py` use pinned memory, two data-loader workers, and TF32 on an
RTX 3090; they retain deterministic FP32 matrix-exponential evaluation.
`GPU_AMP_ENABLED` is an opt-in speed experiment, but should be enabled only after
comparing its validation IC and OOS output against an FP32 run.

Outputs are written to the directory set by `OUTPUT_DIR` in `project_settings.py`
(the current configuration uses a `continuous_z1` suffix). The folder name is
historical; inspect `project_settings.py` and each run's `run_config.json` for
the actual cost and threshold values:

- `run_config.json` and `feature_scaler.json` record the run specification;
- `relative_return_benchmark.json` records the fixed sector map, members, and
  shrinkage rule used to form causal relative-price increments;
- `cross_ticker_area_features.json` records the exact 900 raw cross-ticker area
  coordinates for the 25-stock default universe;
- `signature_feature_spec.json` records the 91-dimensional selective signature,
  seven-dimensional discrete-state context, and three multiscale directional
  context values;
- `cross_ticker_area_pca.json` and `cross_ticker_area_pca.npz` record independent,
  training-only standardisation and PCA transforms; their retained dimensions are
  the `PRICE_PRICE_AREA_PCA_COMPONENTS` and `PRICE_VOLUME_AREA_PCA_COMPONENTS`
  values of that run;
- `target_calibration.json` records the forward sector-neutral training target
  and normal-score rank transform;
- `rank_execution_rule.json` records the rank-selection and probability-gate
  portfolio rule;
- `profitability_gate_0bps.json` and `profitability_gate_<cost>bps.json` record the
  validation-fitted regularised logistic gates; the matching
  `profitability_gate_validation_actions_<cost>bps.parquet` files record their
  validation action labels and fitted probabilities. The unqualified
  `profitability_gate.json` is an alias for the configured-cost gate;
- `return_score_calibration.json` and `expected_alpha_gate.json` explicitly
  record that score-to-return calibration and an expected-alpha gate are disabled;
- `seed_<n>_best_signature_affine_mean.pt` is each seed's validation-selected
  checkpoint; `ensemble_summary.json` records their equal-weight ensemble;
- `ensemble_validation_predictions.parquet`,
  `ensemble_validation_daily_rank_ic.csv`, and
  `ensemble_validation_block_bootstrap.json` audit the averaged validation score;
- `oos_predictions.parquet` contains every test prediction and its exact entry/exit label;
- `oos_holding_period_returns.parquet` and `oos_metrics.json` contain the cost-sensitive OOS evaluation.
- `missing_bar_diagnostics.csv` records observed versus carried-forward minute
  cells for each symbol; review it before interpreting results.
- `seed_<n>_best_validation_daily_rank_ic.csv` and
  `seed_<n>_best_validation_block_bootstrap.json` audit each individual checkpoint
  using day-block bootstrap.
- `oos_daily_rank_ic.csv` and the `rank_ic_block_bootstrap` section of
  `oos_metrics.json` report the OOS daily-IC confidence interval and significance flag.

## Data constraints

Massive Basic supplies historical one-minute OHLCV aggregates, not historical
quotes, trades, order messages, or L2 depth. Consequently this code cannot compute
OFI, queue position, or true spread crossing. Earlier LOBSTER sample data and
derived files remain local only; they are not part of this public workflow.

Missing minute aggregates are not dropped from the input path: each regular session
is expanded to a common minute grid, then a missing price is carried forward and
volume is set to zero. The `has_trade` and `staleness_minutes` features make that
state explicit. This is a feature-construction convention, not an assumed fill. A
label, trade weight, and return are emitted only when both the next-minute entry and
the horizon exit have genuine source bars.

Rebalances are anchored to the actual regular-session minute number (09:30 is
minute zero). A missing eligible signal can skip a sleeve, but cannot shorten the
holding period or create overlapping positions.

At each minute, a stock's return is measured against its contemporaneous sector
median, shrunk toward the universe median of real bars:

\[
\tilde r_{i,t}=r_{i,t}-\left[\lambda_{s,t}\operatorname{median}_{j\in s}(r_{j,t})
+(1-\lambda_{s,t})\operatorname{median}_{j\in\mathcal U}(r_{j,t})\right],
\qquad \lambda_{s,t}=\frac{n_{s,t}}{n_{s,t}+5}.
\]

Only real bars contribute to the medians and to \(n_{s,t}\). This is causal,
removes the technology-heavy universe's mechanical influence on the common
benchmark, and keeps sparse sectors stable. The residuals are accumulated into the
relative-price path. The model also forms selected antisymmetric level-two areas: one
relative-price/relative-price coordinate for every unordered ticker pair and one
relative-price/relative-log-volume coordinate for every ordered distinct pair.
For 25 stocks this is \(\binom{25}{2}+25\times24=900\) explicit cross-ticker
coordinates, rather than a full signature over all ticker-feature channels.

The endpoint multi-scale context contains
\(r_{5m}=R_t-R_{t-5}\), \(r_{15m}=R_t-R_{t-15}\), and the velocity difference
\(r_{5m}/5-r_{15m}/15\); it does not redundantly lead-lag expand these delayed
versions of the relative-price path. CLV is computed only on a real bar as

\[
\operatorname{CLV}_t=\frac{2C_t-H_t-L_t}{H_t-L_t+10^{-12}},
\]

and is zero on a missing or zero-range bar. It is a close-location/bar-pressure
proxy, not observed order-flow or bid/ask aggressor data.

The individual path retains `t_dim` once and lead-lag expands relative price, log
volume, relative range, realised volatility, VWAP deviation, and CLV. It therefore
has \(1+2\times6=13\) signature-path channels. The model retains 13 level-one
coordinates and \({13\choose2}=78\) independent antisymmetric areas, for a
91-coordinate representation. At depth two the discarded symmetric coordinates
are algebraically determined by level one. `has_trade` and `staleness_minutes`
are summarised instead by seven state values: absolute end time; mean, last, and
transition rate of `has_trade`; and mean, maximum, and last staleness. The terminal
5-minute and 15-minute relative returns plus their per-minute velocity difference
form a separate three-coordinate directional context. The 300
price-price areas are PCA-fitted on training windows only, then compressed to the
configured number of frozen components. The 600 price-volume areas are
independently PCA-fitted on training windows only and likewise use the configured
number of components for validation and OOS inference.

## Sector-neutral normal-score rank training and profitability-gated execution

The raw executable holding return is retained for realised PnL. The training,
validation, early-stopping, and OOS Rank-IC target is instead the forward
within-sector excess return:

\[
y^{\mathrm{sector}}_{i,t}=r^H_{i,t}-\operatorname{mean}_{j\in s(i)}r^H_{j,t}.
\]

At least two valid entry/exit labels are required within a sector; a singleton
sector such as XOM's energy sleeve is excluded from the rank target and from the
sector-neutral portfolio. Each valid target vector is hard-ranked and converted to
normal scores \(z_i=\Phi^{-1}((\operatorname{rank}(y_i)-0.5)/n)\). The neural
model then uses the differentiable Spearman surrogate:

\[
\mathcal L=-\operatorname{Corr}_{CS}(\hat y,z).
\]

Three independently trained models (`42`, `43`, and `44`) retain their own
validation-selected checkpoints. Before equal averaging, each seed's score is
z-scored within each signal-time/sector, so arbitrary neural-score scales cannot
determine ensemble weight. Each eligible sector receives a fixed gross budget

\[
b_s=\frac{n_s}{\sum_{u\in\mathcal S_{\rm eligible}}n_u},
\]

where \(n_s\) is its eligible symbol count. An inactive sector leaves its budget
as cash; the other sectors are never re-normalised. The model score is used only
for ordering. In particular, it is not calibrated to a return forecast and its
absolute magnitude has no execution interpretation.

Let \(w^*_t\) denote this desired signal target and \(w_{t-1}\) the current
portfolio. The proposed target first moves only a fraction \(\lambda\) toward
the signal, then retains the whole sector sleeve if its proposed one-way turnover
is below buffer \(\tau\):

\[
\widetilde w_t=\operatorname{Buffer}_{\tau}\left(
w_{t-1}+\lambda(w^*_t-w_{t-1})\right).
\]

The default settings use \(\lambda=0.5\) and \(\tau=0.02\) per sector sleeve.
Within an eligible sector, only scores beyond the configured z-score threshold
are selected: \(z_i\geq z_{\rm threshold}\) for longs and
\(z_i\leq-z_{\rm threshold}\) for shorts. The buffer is sleeve-level rather
than stock-level, preserving industry neutrality. For an unchanged target, the
proposed turnover is zero and no rebalance cost is charged. A switch from zero
to a full-gross sleeve has one-way turnover of about 1; complete replacement can
approach 2. The default `flatten_daily` policy closes each session's final
modelled target, including closing turnover in realised PnL. The optional
`carry_to_next_rebalance` comparison marks the resulting overnight/early-session
bridge return separately. There is no score-to-return calibration and no
expected-alpha gate. Instead, two strongly regularised logistic gates are fitted
on the frozen validation rank predictions: one under \(c=0\) and one under
\(c=1\) bp.
For every optional proposed rebalance, it predicts whether executing the action
is better than retaining the current eligible target for the next holding period:

\[
\Delta\Pi_t=(\widetilde w_t-w_{t-1})^\top R_{t:t+H}
-c\lVert\widetilde w_t-w_{t-1}\rVert_1.
\]

The gate is fed only ex-ante action summaries: score strength and dispersion,
candidate and prior exposure, proposed turnover, selected-symbol and active-sector
counts, and a final-session flag. It executes an optional rebalance only when

\[
P(\Delta\Pi_t>0\mid x_t)\geq p_{\min}.
\]

The default is \(p_{\min}=0.55\). The 0 bp gate produces the no-friction
portfolio path; the configured-cost gate produces the friction-aware path. A missing future
label forces a position to zero at its known prior endpoint and is not treated as
an optional prediction. The gates do not alter rank-model weights, and OOS returns
are never used to fit their coefficients or select their threshold. Their
validation fits are deliberately small and strongly regularised because intraday
rows within a day are dependent.

## Affine latent mean model

For each ticker-specific context \(c_i\)—its own signature concatenated with the
shared cross-ticker areas—the latent mean follows

\[
\frac{dm_t}{dt}=A_\theta(c)m_t+b_\theta(c),
\]

where the learned drift matrix has the stable form \(A=\mathrm{skew}-D\), with
positive diagonal decay \(D\). The alpha is

\[
\alpha=w^\top m_1+q.
\]

The initial latent state is encoded from a five-minute beginning-of-window anchor:
time, log-volume, relative range, realised volatility, trading availability, and
staleness are averaged over that anchor. It is augmented only by whole-window
direction-free statistics: relative-price quadratic variation, absolute VWAP
deviation, volume dispersion, average range and realised volatility, trading
availability, and maximum staleness. Signed terminal price, signed VWAP deviation,
and path trend are deliberately excluded from this encoder. The readout \(w,q\) is global, so signature context can affect alpha only through the affine
latent dynamics. The implementation uses an augmented matrix exponential to obtain
\(m_1\) exactly.
This is a conditional-mean model, not a claim to have fitted a full return
distribution. Additive diffusion and analytic uncertainty supervision are a
possible future extension.

For validation and OOS reporting, minute-level Rank IC is first averaged within
each trading day. The pipeline then resamples whole trading days with replacement
10,000 times to form a day-block-bootstrap 95% confidence interval for mean daily
IC. `significant_positive_95` is true only when that lower confidence bound exceeds
zero; this preserves intraday dependence rather than treating overlapping minutes
as independent observations.

