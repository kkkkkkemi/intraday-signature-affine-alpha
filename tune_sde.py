"""Staged hyperparameter search for the latent-SDE model (validation only).

Run directly from the same Python environment as train_model.py.

* Every run uses project_settings.py as its base, plus FIXED, plus one entry of the
  chosen STAGE.  The test split is never scored (evaluate_test=False).
* Finished runs are skipped, so the script can be stopped and restarted.
* After each run, and at the end, tuning/summary.csv is rewritten with all runs
  found under artifacts/tuning plus EXTRA_RUN_DIRS.

Workflow: run stage "A", copy the winning value into FIXED, run stage "B", and so
on.  Prefer the simpler setting when two configurations differ by less than the
seed-to-seed spread of validation IC (column seed_ic_std).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import project_settings as ps
from minute_alpha import RunConfig, run_pipeline

PROJECT_DIR = Path(__file__).resolve().parent
TUNING_DIR = PROJECT_DIR / "artifacts" / "tuning"

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
STAGE = "A"
# Winners of earlier stages go here, e.g. {"sde_nll_weight": 0.3}.
FIXED: dict[str, object] = {}
STAGES: dict[str, list[dict[str, object]]] = {
    # Ranking vs distribution trade-off.  0 would leave the diffusion untrained.
    "A": [{"sde_nll_weight": weight} for weight in (0.03, 0.1, 0.3, 1.0)],
    # Rank of the latent diffusion loading.
    "B": [{"sde_diffusion_rank": rank} for rank in (2, 4, 8)],
    # Capacity of the latent state and coefficient networks.
    "C": [{"latent_dim": dim, "affine_hidden_dim": 2 * dim} for dim in (8, 16, 32)],
    # Optimiser.
    "D": [
        {"learning_rate": lr, "weight_decay": wd}
        for lr in (3e-4, 1e-3)
        for wd in (1e-4, 1e-3)
    ],
    # Heavy-tail specification at the chosen settings (Merton vs controls).
    "E": [{"sde_jump_mode": mode} for mode in ("none", "student_t", "merton")],
}
# Existing runs to include in the summary table (validation files only).
EXTRA_RUN_DIRS = (
    "artifacts/linsde_r4_nll0p1_n25_sig91_state7_ens42_43_44_pp_pca12_pv_pca16_w30_h30_tr5_ev30",
    "artifacts/linsde_r4_merton_nll0p1_clip20_n25_sig91_state7_ens42_43_44_pp_pca12_pv_pca16_w30_h30_tr5_ev30",
    "artifacts/signature_affine_n25_sig91_state7_ms3_clv_sector_shrink5_target_sector_mean_normalscore_ens42_43_44_z0anchor5scale_pp_pca12_pv_pca16_continuous_z1_gate1bp_m0_w30_h30_tr5_ev30",
)

SHORT_NAMES = {
    "sde_nll_weight": "nll",
    "sde_diffusion_rank": "rank",
    "latent_dim": "ld",
    "affine_hidden_dim": "hd",
    "learning_rate": "lr",
    "weight_decay": "wd",
    "sde_jump_mode": "jump",
    "sde_target_clip": "clip",
}


def base_config() -> RunConfig:
    """Same construction as train_model.py, in validation-only mode."""
    return RunConfig(
        data_dir=str(PROJECT_DIR / ps.DATA_DIR),
        output_dir=str(TUNING_DIR / "unused"),
        window_minutes=ps.WINDOW_MINUTES,
        holding_minutes=ps.HOLDING_MINUTES,
        train_ratio=ps.TRAIN_RATIO,
        val_ratio=ps.VALIDATION_RATIO,
        latent_dim=ps.LATENT_DIM,
        affine_hidden_dim=ps.AFFINE_HIDDEN_DIM,
        price_price_pca_components=ps.PRICE_PRICE_AREA_PCA_COMPONENTS,
        price_volume_pca_components=ps.PRICE_VOLUME_AREA_PCA_COMPONENTS,
        sector_median_shrinkage_count=ps.SECTOR_MEDIAN_SHRINKAGE_COUNT,
        z_threshold=ps.Z_SCORE_THRESHOLD,
        session_end_policy=ps.SESSION_END_POLICY,
        target_smoothing_fraction=ps.TARGET_WEIGHT_SMOOTHING_FRACTION,
        sector_rebalance_buffer_turnover=ps.SECTOR_REBALANCE_BUFFER_TURNOVER,
        profitability_gate_cost_bps=ps.PROFITABILITY_GATE_COST_BPS,
        profitability_gate_min_probability=ps.PROFITABILITY_GATE_MIN_PROBABILITY,
        profitability_gate_l2=ps.PROFITABILITY_GATE_L2,
        batch_size=ps.BATCH_SIZE,
        max_epochs=ps.MAX_EPOCHS,
        patience=ps.PATIENCE,
        warmup_epochs=ps.WARMUP_EPOCHS,
        min_labelled_symbols=ps.MIN_LABELLED_SYMBOLS,
        train_sample_stride_minutes=ps.TRAIN_SAMPLE_STRIDE_MINUTES,
        evaluation_sample_stride_minutes=ps.EVALUATION_SAMPLE_STRIDE_MINUTES,
        bootstrap_iterations=ps.BOOTSTRAP_ITERATIONS,
        symbols=tuple(ps.TICKERS),
        data_loader_workers=ps.GPU_DATA_LOADER_WORKERS,
        pin_memory=ps.GPU_PIN_MEMORY,
        amp_enabled=ps.GPU_AMP_ENABLED,
        allow_tf32=ps.GPU_ALLOW_TF32,
        deterministic=ps.DETERMINISTIC,
        seed=ps.SEED,
        ensemble_seeds=tuple(ps.ENSEMBLE_SEEDS),
        sde_diffusion_rank=ps.SDE_DIFFUSION_RANK,
        sde_diffusion_floor=ps.SDE_DIFFUSION_FLOOR,
        sde_nll_weight=ps.SDE_NLL_WEIGHT,
        sde_target_clip=ps.SDE_TARGET_CLIP,
        sde_cost_curve_bps=tuple(ps.SDE_COST_CURVE_BPS),
        sde_jump_mode=ps.SDE_JUMP_MODE,
        sde_jump_max_count=ps.SDE_JUMP_MAX_COUNT,
        sde_jump_initial_intensity=ps.SDE_JUMP_INITIAL_INTENSITY,
        sde_jump_initial_std=ps.SDE_JUMP_INITIAL_STD,
        evaluate_test=False,
    )


def run_name(config: RunConfig) -> str:
    parts = [
        f"n{len(config.symbols) if config.symbols else 0}",
        f"jump-{config.sde_jump_mode}",
        f"nll-{config.sde_nll_weight:g}",
        f"rank-{config.sde_diffusion_rank}",
        f"ld-{config.latent_dim}",
        f"hd-{config.affine_hidden_dim}",
        f"lr-{config.learning_rate:g}",
        f"wd-{config.weight_decay:g}",
        f"clip-{config.sde_target_clip:g}",
    ]
    return "__".join(parts)


def collect(run_dir: Path) -> dict[str, object] | None:
    """Validation-only summary of one finished run."""
    ensemble_path = run_dir / "ensemble_summary.json"
    if not ensemble_path.exists():
        return None
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    seeds = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("seed_*_training_summary.json"))
    ]
    seed_ic = np.array([seed["best_validation_sector_neutral_rank_ic"] for seed in seeds], dtype=float)
    bootstrap = ensemble["ensemble_validation_rank_ic_block_bootstrap"]
    record: dict[str, object] = {
        "run": run_dir.name,
        "n_symbols": len(config.get("symbols") or []),
        "eval_stride": config.get("evaluation_sample_stride_minutes"),
        "jump": config.get("sde_jump_mode", "none") if config.get("sde_diffusion_rank", 0) else "ode",
        "nll_weight": config.get("sde_nll_weight"),
        "rank": config.get("sde_diffusion_rank", 0),
        "latent_dim": config.get("latent_dim"),
        "hidden_dim": config.get("affine_hidden_dim"),
        "lr": config.get("learning_rate"),
        "wd": config.get("weight_decay"),
        "clip": config.get("sde_target_clip"),
        "seed_ic_mean": float(seed_ic.mean()) if len(seed_ic) else np.nan,
        "seed_ic_std": float(seed_ic.std(ddof=1)) if len(seed_ic) > 1 else np.nan,
        "ensemble_ic": float(bootstrap["mean_daily_rank_ic"]),
        "ensemble_ic_ci_low": float(bootstrap["ci_95_lower"]),
        "ensemble_ic_ci_high": float(bootstrap["ci_95_upper"]),
        "best_epochs": "/".join(str(seed["best_epoch"]) for seed in seeds),
        "seed_val_nll_mean": float(np.nanmean([seed.get("best_validation_nll", np.nan) for seed in seeds])) if seeds else np.nan,
    }
    distribution_path = run_dir / "sde_distribution_metrics.json"
    if distribution_path.exists():
        validation = json.loads(distribution_path.read_text(encoding="utf-8"))["validation"]
        coverage = validation["central_interval_coverage"]
        record.update(
            {
                "val_nll": validation["mean_nll"],
                "cov50": coverage.get("50%"),
                "cov90": coverage.get("90%"),
                "cov99": coverage.get("99%"),
                "pit_ks": validation["pit_ks_statistic"],
                "nll_gain_vs_baseline": validation.get("nll_improvement_vs_baseline", {}).get("mean_daily"),
                "nll_gain_ci_low": validation.get("nll_improvement_vs_baseline", {}).get("ci_95_lower"),
                "jump_lambda": validation.get("jumps", {}).get("mean_intensity_per_period"),
                "jump_var_share": validation.get("jumps", {}).get("mean_jump_share_of_total_variance"),
            }
        )
    return record


def write_summary() -> pd.DataFrame:
    run_dirs = [path for path in TUNING_DIR.glob("*") if path.is_dir()]
    run_dirs += [PROJECT_DIR / extra for extra in EXTRA_RUN_DIRS]
    records = [record for record in (collect(path) for path in run_dirs if path.exists()) if record is not None]
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table
    table = table.sort_values("seed_ic_mean", ascending=False)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TUNING_DIR / "summary.csv", index=False)
    return table


def main() -> None:
    if STAGE not in STAGES:
        raise ValueError(f"STAGE must be one of {sorted(STAGES)}")
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    base = replace(base_config(), **FIXED)
    configs = [replace(base, **overrides) for overrides in STAGES[STAGE]]
    print(f"Stage {STAGE}: {len(configs)} configurations, fixed={FIXED}, validation only.")
    for index, config in enumerate(configs, start=1):
        name = run_name(config)
        output_dir = TUNING_DIR / name
        if (output_dir / "ensemble_summary.json").exists():
            print(f"[{index}/{len(configs)}] {name}: already finished, skipping.")
            continue
        print(f"\n[{index}/{len(configs)}] {name}", flush=True)
        run_pipeline(replace(config, output_dir=str(output_dir)))
        write_summary()

    table = write_summary()
    columns = [
        "n_symbols", "eval_stride", "jump", "nll_weight", "rank", "latent_dim", "lr", "wd", "clip",
        "seed_ic_mean", "seed_ic_std", "ensemble_ic", "ensemble_ic_ci_low", "best_epochs",
        "val_nll", "cov50", "cov90", "cov99", "pit_ks", "nll_gain_vs_baseline", "jump_lambda",
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n=== Validation summary (sorted by mean seed IC) ===")
        print(table[[column for column in columns if column in table.columns]].to_string(index=False))
    print(
        f"\nWritten to {TUNING_DIR / 'summary.csv'}.  Compare only rows with the same n_symbols and "
        "eval_stride; NLL / coverage only with the same clip; IC differences below seed_ic_std are noise."
    )


if __name__ == "__main__":
    main()
