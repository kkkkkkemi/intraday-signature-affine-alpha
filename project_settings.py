"""Edit this file to configure the Massive minute-bar experiment.

Both download_data.py and train_model.py read this file, so neither workflow needs
PowerShell arguments or PowerShell environment variables.
"""

from __future__ import annotations

from datetime import date, timedelta


TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
    "AMD", "INTC", "ORCL", "CRM", "NFLX", "TSLA",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "WMT", "COST", "HD", "JNJ", "UNH", "XOM",
]
DATA_DIR = "minute_data"
OUTPUT_DIR = "artifacts/signature_affine_n25_sig91_state7_ms3_clv_sector_shrink5_target_sector_mean_normalscore_ens42_43_44_z0anchor5scale_pp_pca12_pv_pca16_continuous_z1_gate1bp_m0_w30_h30_tr5_ev30"
START_DATE = (date.today() - timedelta(days=720)).isoformat()
END_DATE = date.today().isoformat()

# Model and evaluation settings. Missing minute aggregates are represented in the
# feature path; targets still require real entry and exit bars.
WINDOW_MINUTES = 30
HOLDING_MINUTES = 30
# Split only on whole trading days: 70% train / 15% validation / 15% final OOS.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
BATCH_SIZE = 16
MAX_EPOCHS = 60
PATIENCE = 10
WARMUP_EPOCHS = 3
MIN_LABELLED_SYMBOLS = 15
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
# offsets. Validation and final OOS retain strictly non-overlapping 30-minute
# sleeves.
TRAIN_SAMPLE_STRIDE_MINUTES = 5
EVALUATION_SAMPLE_STRIDE_MINUTES = 30
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
