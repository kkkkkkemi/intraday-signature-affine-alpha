# XGBoost comparison

Run `train_xgboost.py` from the same Python IDE/interpreter used for
`train_model.py`. Install `requirements_xgboost.txt` in that interpreter first.
The script resolves `minute_data` and artifact paths relative to its own
location, so the IDE working directory does not matter.
The existing signature-affine model and its artifacts are not changed. Baseline
outputs are saved to `artifacts/xgboost_comparison/`.

The baseline uses the same minute-bar feature engineering, chronological day
split, 30-minute labels, training/validation/test sample strides, train-only
scaler and train-only cross-ticker PCA. Inputs are the fixed selective path
signature, trading-state and multiscale summaries, cross-ticker PCA components,
and the raw inputs to the neural model's initial-state encoder. It has no ticker
identifier. XGBoost hyperparameters are at the top of `train_xgboost.py`.

Each signal time is one XGBoost ranking group. The objective is `rank:pairwise`,
with the seed's tree count chosen by validation mean daily Rank IC. Seed scores
are standardised within time and sector before averaging, as with the neural
ensemble. The training losses differ, so this compares model families rather
than isolating architecture alone.

Read `comparison.json` for the paired OOS daily Rank IC comparison and the
ungated and validation-gated portfolios run through the same execution
settings. A paired
comparison is written only when the saved neural validation and test prediction
files have identical signal keys and labels. `oos_metrics_ungated.json` contains
the XGBoost portfolio without a profitability gate. `oos_metrics.json` contains
the XGBoost portfolio with validation-fitted gates at zero and the configured
one-way cost. `feature_importance_gain.csv` reports tree gain importance; it is
descriptive and does not establish economic causality.
