# Research results and interpretation

The latest local matched comparison uses 25 U.S. equities, 30-minute features and
holding periods, and a chronological 70/15/15 split. Both models use the same
signal times and forward labels. The final segment contains 75 trading days.

| Metric | Signature-conditioned affine model | XGBoost baseline |
| --- | ---: | ---: |
| Mean daily OOS Rank IC | 0.0227 | 0.0089 |
| 95% day-bootstrap CI for mean Rank IC | [0.0058, 0.0399] | [-0.0100, 0.0284] |
| Ungated mean return per holding period, 0 bp cost | +0.525 bp | -0.045 bp |
| Ungated mean return per holding period, 0.6 bp one-way cost | +0.140 bp | -0.433 bp |

The paired day-bootstrap interval for *XGBoost minus affine-model* mean Rank IC
is [-0.0343, 0.0070]. It includes zero: the observed difference between the
models is not statistically established at the 95% level. The XGBoost model is
a reference baseline, not a claim that boosting is generally inferior.

These are research backtests on minute OHLCV aggregates. Costs are assumed,
because historical bid/ask quotes and execution queue information are absent.
The same final segment was inspected repeatedly during project development,
so its effective independence from research decisions is limited. Neither the
reported IC nor the simulated portfolio return is evidence of live profitability.

Raw minute bars, LOBSTER files, prediction rows, weights, charts, and trained
checkpoints are excluded from this public repository. Users with their own data
access can reproduce the pipeline locally using the scripts and configuration.
