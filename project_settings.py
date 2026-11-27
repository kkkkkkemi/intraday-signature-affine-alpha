"""Edit this file to configure the Massive minute-bar experiment.

Both download_data.py and train_model.py read this file, so neither workflow needs
PowerShell arguments or PowerShell environment variables.
"""

from __future__ import annotations



from universe import universe

# Research universe:
#   "sp100"  - S&P 100 as of the sample start (100 names; see universe.py)
#   "mega25" - the original 25 mega caps (results under the old artifact folders)
UNIVERSE = "sp100"
TICKERS = list(universe(UNIVERSE))
# Each universe has its own data folder so files always share one download window.
DATA_DIR = {"sp100": "minute_data_sp100", "mega25": "minute_data"}[UNIVERSE]
# Earlier 25-name runs (kept for reference):
#   artifacts/linsde_r4_nll0p1_n25_sig91_state7_ens42_43_44_pp_pca12_pv_pca16_w30_h30_tr5_ev30   (Gaussian)
#   artifacts/linsde_r4_merton_nll0p1_clip20_n25_sig91_state7_ens42_43_44_pp_pca12_pv_pca16_w30_h30_tr5_ev30 (Merton)
#   artifacts/sp100_linsde_r4_merton_nll0p1_clip20_ens42_43_44_pp_pca12_pv_pca16_w30_h30_tr5_ev5 (first sp100 run, v1 features)
# Architecture, optimiser, horizon and ablation tags are appended at the end of this file.
OUTPUT_DIR = f"artifacts/{UNIVERSE}_v2feat_linsde_r4_merton_nll0p1_clip20_ens42_43_44_w30_h30_tr5_ev5"
# Massive Basic serves roughly the most recent two years of minute aggregates.
# The window is pinned (it is the one used for the 2026-11-23 sp100 download) so
# that re-running download_data.py to retry failed tickers fetches the same dates.
# The final, still-open session of 2026-11-23 is removed automatically by
# minute_alpha.drop_incomplete_sessions.  For a fresh download, move both dates.
START_DATE = "2024-12-03"
END_DATE = "2026-11-23"

# Model and evaluation settings. Missing minute aggregates are represented in the
# feature path; targets still require real entry and exit bars.
WINDOW_MINUTES = 30
HOLDING_MINUTES = 30
# Split only on whole trading days: 70% train / 15% validation / 15% final OOS.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
BATCH_SIZE = 16
# The train/validation IC gap opens within a few epochs, so stop early.
MAX_EPOCHS = 30
PATIENCE = 5
WARMUP_EPOCHS = 0
# Minimum labelled names per cross-section (~60% of the universe).
MIN_LABELLED_SYMBOLS = {"sp100": 60, "mega25": 15}[UNIVERSE]
# The contemporaneous sector median is shrunk toward the universe median using
# lambda = n_real_sector_bars / (n_real_sector_bars + this value).
SECTOR_MEDIAN_SHRINKAGE_COUNT = 5.0
# Selective model: 91-dimensional level-two signature (time plus six continuous
# channels, including CLV), 7 trading-state summaries, and three explicit
# multi-scale relative-momentum context values. The two raw cross-ticker area
# groups are each PCA-fitted on training windows only: 300 price-price -> 12
# components, and 600 price-volume -> 20 components.
LATENT_DIM = 16
AFFINE_HIDDEN_DIM = 32
PRICE_PRICE_AREA_PCA_COMPONENTS = 12
PRICE_VOLUME_AREA_PCA_COMPONENTS = 16
# Calculated independently within each sector. Higher values select fewer,
# more-extreme names; 1.0 is the highest threshold that always retains a valid
# two-stock sleeve (z = +1 and -1).
Z_SCORE_THRESHOLD = 0.75
# This is a 30-minute intraday-alpha study. Close the final modelled target of
# every session rather than assuming unmodelled overnight/early-session gaps can
# be forecast or safely marked from minute OHLCV data. The optional comparison
# mode is "carry_to_next_rebalance".
SESSION_END_POLICY = "flatten_daily"
# Keep sector budgets fixed in proportion to eligible sector sizes. The budget of
# an inactive sector remains cash rather than being redistributed to others.
# Each active sleeve only moves this fraction toward its new target; skip a whole
# sector rebalance when its proposed one-way turnover is below the buffer.
TARGET_WEIGHT_SMOOTHING_FRACTION = 0.5
SECTOR_REBALANCE_BUFFER_TURNOVER = 0.02
# The post-ranking gate predicts whether executing this rebalance, rather than
# retaining the current eligible weights, has positive *incremental* net PnL.
# It is fitted only on frozen validation predictions and uses this fixed cost.
PROFITABILITY_GATE_COST_BPS = 0.6
PROFITABILITY_GATE_MIN_PROBABILITY = 0.45
PROFITABILITY_GATE_L2 = 1.0
# Training samples use overlapping 30-minute feature/label windows at five-minute
# offsets.  Validation and OOS are now scored at the same 5-minute stride, so
# checkpoint selection and reported Rank IC use every 5-minute cross-section
# (~6x more than the old 30-minute stride, which over-rewarded noise at the 12
# fixed daily times).  Overlapping labels are handled by averaging IC within each
# day before the day-block bootstrap.  Portfolio evaluation still trades only on
# the non-overlapping 30-minute rebalance grid, so backtests are unchanged.
TRAIN_SAMPLE_STRIDE_MINUTES = 5
EVALUATION_SAMPLE_STRIDE_MINUTES = 5
BOOTSTRAP_ITERATIONS = 10_000
SEED = 42
ENSEMBLE_SEEDS = (42, 43, 44)

# GPU runtime. RTX 3090 supports CUDA and TF32. Leave AMP off until one complete
# FP32 run has been checked for matrix-exponential numerical stability; then enable it.
GPU_DATA_LOADER_WORKERS = 2
GPU_PIN_MEMORY = True
GPU_ALLOW_TF32 = True
GPU_AMP_ENABLED = False
DETERMINISTIC = True

# Linear latent SDE layer: dZ = (A(c)Z + b(c))dt + G(c)dW with G(c) of rank
# SDE_DIFFUSION_RANK.  The predictive mean and variance of the 30-minute
# sector-excess return are computed exactly (matrix exponential + Van Loan), so
# no Monte Carlo is used.  0 = deterministic affine-mean model (ablation).
SDE_DIFFUSION_RANK = 4
SDE_DIFFUSION_FLOOR = 1e-4
# Total loss = rank loss + SDE_NLL_WEIGHT * Gaussian NLL.  Rank IC still selects
# checkpoints, so ablation runs use the same selection rule.
SDE_NLL_WEIGHT = 0.1
# Standardised targets are clipped at +/- this many robust-scale units in the NLL.
# Jump models need the tails, so use a wide clip.  Compare NLL only between runs
# that use the same clip (re-run the Gaussian model with 20 for a like-for-like test).
SDE_TARGET_CLIP = 20.0
# Ungated one-way cost grid used to compare rank-score vs uncertainty-scaled sizing.
SDE_COST_CURVE_BPS = (0.0, 0.3, 0.6, 1.0, 1.5, 2.0)
# Heavy-tailed component over the 30-minute holding period:
#   "none"      Gaussian (diffusion + observation noise)
#   "merton"    + compound-Poisson jumps, N ~ Poisson(lambda(c)), xi ~ N(0, s_J(c)^2);
#               exact Poisson-mixture likelihood truncated at SDE_JUMP_MAX_COUNT
#   "student_t" Student-t with a global learned dof (low-cost control)
SDE_JUMP_MODE = "merton"
SDE_JUMP_MAX_COUNT = 8
# Initial jump intensity per 30-minute period and jump std (standardised units).
SDE_JUMP_INITIAL_INTENSITY = 0.05
SDE_JUMP_INITIAL_STD = 2.0
# Multi-horizon likelihood: fit the cumulative sector-excess return at
# K equally spaced points of the holding period (every HOLDING_MINUTES / K
# minutes) jointly, with exact SDE cross-covariances.  1 = endpoint only.
# It also enables the holding-period filter diagnostics (horizon_filter_*.json):
# after observing part of the holding period, does conditioning on it improve
# the forecast of the remaining return?
SDE_HORIZON_STEPS = 6
# Truncation of the per-holding-period jump count in the multi-horizon mixture
# (3 jumps -> 84 mixture components for K = 6).
SDE_HORIZON_JUMP_MAX_COUNT = 3
# Also apply the rank loss at every intermediate horizon (averaged with the final one).
SDE_HORIZON_RANK_ALL = False

# Tuning discipline: while False, train_model.py never scores the test split and
# skips the gate / OOS evaluation.  Set True only for the final frozen run.
# FINAL RUN: the configuration in this file is frozen (chosen on validation only:
# v1 architecture without embedding/attention, v2 feature groups off, lr 1e-4,
# weight decay 1e-3, context dropout 0.1, 6-step multi-horizon Merton SDE).
# Setting this to True trains once more with the same seeds and scores the
# sealed test split exactly once; the run goes to a separate "_final" directory.
EVALUATE_TEST = True

# v2 inputs: volume surprise, trade counts/size, session context, and SPY /
# sector-ETF returns (ETF bars live in DATA_DIR/etf; download_data.py fetches them).
USE_ETF_CONTEXT = True
# Architecture: per-stock embedding and transformer layers across the stocks of
# each cross-section (0 disables either, for ablations).
STOCK_EMBEDDING_DIM = 0
CROSS_ATTENTION_LAYERS = 0
ATTENTION_DIM = 64
ATTENTION_HEADS = 4
# Training windows scored each epoch to log train-vs-validation Rank IC.
TRAIN_IC_MONITOR_WINDOWS = 2000
# Regularisation (AdamW weight decay, dropout on the SDE conditioning context and
# inside the cross-stock attention).
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
CONTEXT_DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1
# v2 feature groups to switch off (set to zero), for ablations; choose from
# "day_constant", "volume_flow", "session", "etf" (see minute_alpha.FEATURE_GROUPS).
DROP_FEATURE_GROUPS: tuple[str, ...] = ("day_constant","volume_flow", "session", "etf")
# Tag the run directory with the early-stopping / regularisation / ablation
# settings so a new configuration never overwrites an earlier run.
OUTPUT_DIR += (
    f"_emb{STOCK_EMBEDDING_DIM}_att{CROSS_ATTENTION_LAYERS}_lr{LEARNING_RATE:g}"
    f"_p{PATIENCE}_wd{WEIGHT_DECAY:g}_do{CONTEXT_DROPOUT:g}"
)
if SDE_HORIZON_STEPS > 1:
    OUTPUT_DIR += f"_hz{SDE_HORIZON_STEPS}" + ("_hzrank" if SDE_HORIZON_RANK_ALL else "")
if DROP_FEATURE_GROUPS:
    OUTPUT_DIR += "_drop-" + "+".join(DROP_FEATURE_GROUPS)
if EVALUATE_TEST:
    OUTPUT_DIR += "_final"
