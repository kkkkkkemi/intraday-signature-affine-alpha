"""Train the signature-conditioned affine latent mean model from minute bars.

Run this file directly from the same Python environment that contains torch,
signatory, pandas, and pyarrow.  Edit project_settings.py for settings.
"""

from __future__ import annotations

from pathlib import Path

from minute_alpha import RunConfig, run_pipeline
from project_settings import (
    AFFINE_HIDDEN_DIM,
    BATCH_SIZE,
    BOOTSTRAP_ITERATIONS,
    DATA_DIR,
    DETERMINISTIC,
    ENSEMBLE_SEEDS,
    GPU_ALLOW_TF32,
    GPU_AMP_ENABLED,
    GPU_DATA_LOADER_WORKERS,
    GPU_PIN_MEMORY,
    HOLDING_MINUTES,
    LATENT_DIM,
    MAX_EPOCHS,
    MIN_LABELLED_SYMBOLS,
    OUTPUT_DIR,
    PATIENCE,
    PRICE_PRICE_AREA_PCA_COMPONENTS,
    PRICE_VOLUME_AREA_PCA_COMPONENTS,
    PROFITABILITY_GATE_COST_BPS,
    PROFITABILITY_GATE_L2,
    PROFITABILITY_GATE_MIN_PROBABILITY,
    SEED,
    SECTOR_MEDIAN_SHRINKAGE_COUNT,
    SECTOR_REBALANCE_BUFFER_TURNOVER,
    SESSION_END_POLICY,
    TRAIN_SAMPLE_STRIDE_MINUTES,
    TICKERS,
    TRAIN_RATIO,
    TARGET_WEIGHT_SMOOTHING_FRACTION,
    VALIDATION_RATIO,
    WARMUP_EPOCHS,
    WINDOW_MINUTES,
    EVALUATION_SAMPLE_STRIDE_MINUTES,
    Z_SCORE_THRESHOLD,
)


PROJECT_DIR = Path(__file__).resolve().parent


def _project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_DIR / path)


def main() -> None:
    run_pipeline(
        RunConfig(
            data_dir=_project_path(DATA_DIR),
            output_dir=_project_path(OUTPUT_DIR),
            window_minutes=WINDOW_MINUTES,
            holding_minutes=HOLDING_MINUTES,
            train_ratio=TRAIN_RATIO,
            val_ratio=VALIDATION_RATIO,
            latent_dim=LATENT_DIM,
            affine_hidden_dim=AFFINE_HIDDEN_DIM,
            price_price_pca_components=PRICE_PRICE_AREA_PCA_COMPONENTS,
            price_volume_pca_components=PRICE_VOLUME_AREA_PCA_COMPONENTS,
            sector_median_shrinkage_count=SECTOR_MEDIAN_SHRINKAGE_COUNT,
            z_threshold=Z_SCORE_THRESHOLD,
            session_end_policy=SESSION_END_POLICY,
            target_smoothing_fraction=TARGET_WEIGHT_SMOOTHING_FRACTION,
            sector_rebalance_buffer_turnover=SECTOR_REBALANCE_BUFFER_TURNOVER,
            profitability_gate_cost_bps=PROFITABILITY_GATE_COST_BPS,
            profitability_gate_min_probability=PROFITABILITY_GATE_MIN_PROBABILITY,
            profitability_gate_l2=PROFITABILITY_GATE_L2,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            warmup_epochs=WARMUP_EPOCHS,
            min_labelled_symbols=MIN_LABELLED_SYMBOLS,
            train_sample_stride_minutes=TRAIN_SAMPLE_STRIDE_MINUTES,
            evaluation_sample_stride_minutes=EVALUATION_SAMPLE_STRIDE_MINUTES,
            bootstrap_iterations=BOOTSTRAP_ITERATIONS,
            symbols=tuple(TICKERS),
            data_loader_workers=GPU_DATA_LOADER_WORKERS,
            pin_memory=GPU_PIN_MEMORY,
            amp_enabled=GPU_AMP_ENABLED,
            allow_tf32=GPU_ALLOW_TF32,
            deterministic=DETERMINISTIC,
            seed=SEED,
            ensemble_seeds=ENSEMBLE_SEEDS,
        )
    )


if __name__ == "__main__":
    main()
