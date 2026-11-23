"""Signal horizon, persistence, smoothing and reversal analysis (validation only).

Run directly from the same Python environment as train_model.py after editing the
settings block.  It loads a finished run's checkpoints, re-scores the validation
split every 5 minutes, and never reads the test split.

1. IC decay: Rank IC of the ensemble score against sector-excess returns over
   (0, H] for H = 5..120 minutes, and over the disjoint intervals
   (0,30], (30,60], (60,90], (90,120].  Alpha that survives past 30 minutes can
   be held longer, which lowers turnover.
2. Persistence: rank autocorrelation of the score at lags of 5..120 minutes.
3. Smoothing: an EMA of the 5-minute scores for several half-lives, reporting the
   IC / persistence trade-off, then the cost curve of the existing rule and of
   the L1 optimizer with kappa around 1 - rho on the smoothed score.
4. Reversal: IC and persistence of -(past 5/15/30-minute sector-relative return),
   the cross-sectional correlation between the model score and reversal, and the
   model's IC after removing the reversal factors.
5. IC by time of day, as evidence for or against a time-of-day trading gate.

Outputs: <RUN_DIR>/signal_horizon_validation/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import execution_research as er
from minute_alpha import (
    FEATURE_COLUMNS,
    CrossSectionalMinuteDataset,
    CrossTickerAreaPCA,
    SignatureConditionedAffineMeanModel,
    apply_feature_scaler,
    average_seed_prediction_frames,
    engineer_minute_features,
    load_minute_bars,
    make_prediction_frame,
    split_by_session,
)
from project_settings import DATA_DIR, OUTPUT_DIR

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
RUN_DIR = OUTPUT_DIR
SCORE_STRIDE_MINUTES = 5
CUMULATIVE_HORIZONS = (5, 15, 30, 60, 90, 120)
INTERVALS = ((0, 30), (30, 60), (60, 90), (90, 120))
AUTOCORRELATION_LAGS = (5, 10, 15, 30, 60, 90, 120)
EMA_HALF_LIVES = (0, 10, 20, 30, 60)          # minutes; 0 = raw score
BOOTSTRAP_ITERATIONS = 5_000
SEED = 2026


# --------------------------------------------------------------------------- #
# Labels and factors from the raw feature frame
# --------------------------------------------------------------------------- #
def forward_sector_excess(frame: pd.DataFrame, start: int, end: int) -> pd.Series:
    """Sector-excess log return from open(t+1+start) to open(t+1+end).

    ``start=0, end=H`` reproduces the pipeline's ``rank_target_return`` for a
    holding period H (same validity rules and >=2-name sector requirement).
    """
    grouped = frame.groupby(["symbol", "session"], sort=False)
    entry_price = grouped["raw_open"].shift(-(start + 1))
    exit_price = grouped["raw_open"].shift(-(end + 1))
    entry_time = grouped["timestamp"].shift(-(start + 1))
    exit_time = grouped["timestamp"].shift(-(end + 1))
    valid = (
        (entry_time - frame["timestamp"] == pd.Timedelta(minutes=start + 1))
        & (exit_time - frame["timestamp"] == pd.Timedelta(minutes=end + 1))
        & entry_price.gt(0)
        & exit_price.gt(0)
    )
    raw = pd.Series(np.where(valid, np.log(exit_price / entry_price), np.nan), index=frame.index)
    keys = [frame["session"], frame["timestamp"], frame["sector"]]
    mean = raw.groupby(keys, sort=False).transform("mean")
    count = raw.groupby(keys, sort=False).transform("count")
    return (raw - mean).where(count.ge(2))


def reversal_factors(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["symbol", "session"], sort=False)["relative_cum_ret"]
    return pd.DataFrame(
        {
            "reversal_5m": -frame["relative_return_5m"],
            "reversal_15m": -frame["relative_return_15m"],
            "reversal_30m": -grouped.diff(30),
        },
        index=frame.index,
    )


# --------------------------------------------------------------------------- #
# Fast cross-sectional statistics on [T, N] panels
# --------------------------------------------------------------------------- #
def _row_ranks(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(axis=1).to_numpy()


def rowwise_rank_correlation(a: np.ndarray, b: np.ndarray, min_names: int = 5) -> np.ndarray:
    """Spearman correlation per row over names where both are finite."""
    mask = np.isfinite(a) & np.isfinite(b)
    ra = _row_ranks(np.where(mask, a, np.nan))
    rb = _row_ranks(np.where(mask, b, np.nan))
    count = mask.sum(axis=1)
    ra = np.where(mask, ra, 0.0)
    rb = np.where(mask, rb, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = ra.sum(axis=1) / count
        mb = rb.sum(axis=1) / count
        da = np.where(mask, ra - ma[:, None], 0.0)
        db = np.where(mask, rb - mb[:, None], 0.0)
        corr = (da * db).sum(axis=1) / np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    corr[count < min_names] = np.nan
    return corr


def daily_ic_summary(per_row: np.ndarray, sessions: np.ndarray) -> dict[str, float]:
    daily = pd.Series(per_row).groupby(sessions).mean().dropna().to_numpy()
    if len(daily) < 2:
        return {"mean_daily_ic": float("nan"), "ci_95_lower": float("nan"), "ci_95_upper": float("nan"), "days": len(daily)}
    rng = np.random.default_rng(SEED)
    boot = daily[rng.integers(0, len(daily), size=(BOOTSTRAP_ITERATIONS, len(daily)))].mean(axis=1)
    return {
        "mean_daily_ic": float(daily.mean()),
        "ci_95_lower": float(np.quantile(boot, 0.025)),
        "ci_95_upper": float(np.quantile(boot, 0.975)),
        "days": int(len(daily)),
    }


def lagged_autocorrelation(score: np.ndarray, sessions: np.ndarray, lag_rows: int) -> float:
    if lag_rows <= 0 or lag_rows >= len(score):
        return float("nan")
    same_session = sessions[lag_rows:] == sessions[:-lag_rows]
    corr = rowwise_rank_correlation(score[:-lag_rows], score[lag_rows:])
    return float(np.nanmean(corr[same_session]))


def ema_within_session(score: np.ndarray, sessions: np.ndarray, half_life_rows: float) -> np.ndarray:
    """Causal EMA per symbol, reset at each session; missing scores keep the state."""
    if half_life_rows <= 0:
        return score.copy()
    decay = 0.5 ** (1.0 / half_life_rows)
    out = np.full_like(score, np.nan)
    state = np.full(score.shape[1], np.nan)
    for t in range(len(score)):
        if t == 0 or sessions[t] != sessions[t - 1]:
            state = np.full(score.shape[1], np.nan)
        current = score[t]
        fresh = np.isfinite(current)
        update = fresh & np.isfinite(state)
        state = np.where(update, decay * state + (1.0 - decay) * current, state)
        state = np.where(fresh & ~np.isfinite(state), current, state)
        out[t] = np.where(fresh, state, np.nan)
    return out


def residualise(score: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    """Per-row OLS of score ranks on factor ranks; returns residuals (nan where missing)."""
    out = np.full_like(score, np.nan)
    ranked_score = _row_ranks(score)
    ranked_factors = [_row_ranks(factor) for factor in factors]
    for t in range(len(score)):
        mask = np.isfinite(ranked_score[t])
        for factor in ranked_factors:
            mask &= np.isfinite(factor[t])
        if mask.sum() < len(factors) + 3:
            continue
        design = np.column_stack([np.ones(mask.sum())] + [factor[t, mask] for factor in ranked_factors])
        coefficients, *_ = np.linalg.lstsq(design, ranked_score[t, mask], rcond=None)
        out[t, mask] = ranked_score[t, mask] - design @ coefficients
    return out


# --------------------------------------------------------------------------- #
# Model scoring
# --------------------------------------------------------------------------- #
def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def load_validation_frame(config: dict) -> pd.DataFrame:
    """Raw (unscaled) validation feature frame, split exactly as in training."""
    bars = load_minute_bars(_resolve(DATA_DIR), config.get("symbols"))
    features = engineer_minute_features(
        bars,
        int(config["holding_minutes"]),
        sector_median_shrinkage_count=float(config.get("sector_median_shrinkage_count", 5.0)),
    )
    _train, validation, _test = split_by_session(features, float(config["train_ratio"]), float(config["val_ratio"]))
    del _train, _test, features
    return validation


def score_validation(run_dir: Path, config: dict, validation_scaled: pd.DataFrame) -> pd.DataFrame:
    import torch

    dataset = CrossSectionalMinuteDataset(
        validation_scaled, int(config["window_minutes"]), int(config["min_labelled_symbols"]), SCORE_STRIDE_MINUTES
    )
    archive = np.load(run_dir / "cross_ticker_area_pca.npz")
    pca = CrossTickerAreaPCA(
        price_price_mean=archive["price_price_mean"],
        price_price_scale=archive["price_price_scale"],
        price_price_components=archive["price_price_components"],
        price_price_explained_variance_ratio=np.zeros(archive["price_price_components"].shape[0]),
        price_volume_mean=archive["price_volume_mean"],
        price_volume_scale=archive["price_volume_scale"],
        price_volume_components=archive["price_volume_components"],
        price_volume_explained_variance_ratio=np.zeros(archive["price_volume_components"].shape[0]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = tuple(config["ensemble_seeds"])
    frames = []
    for seed in seeds:
        model = SignatureConditionedAffineMeanModel(
            input_channels=len(FEATURE_COLUMNS),
            signature_depth=int(config["signature_depth"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dim=int(config["affine_hidden_dim"]),
            num_symbols=len(dataset.symbols),
            cross_area_pca=pca,
            diffusion_rank=int(config.get("sde_diffusion_rank", 0)),
            diffusion_floor=float(config.get("sde_diffusion_floor", 1e-4)),
            jump_mode=str(config.get("sde_jump_mode", "none")),
        ).to(device)
        state = torch.load(run_dir / f"seed_{seed}_best_signature_affine_mean.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        frames.append(make_prediction_frame(model, dataset, device, False, device.type == "cuda"))
        print(f"  scored seed {seed}: {len(frames[-1]):,} rows", flush=True)
    return average_seed_prediction_frames(frames, seeds)


# --------------------------------------------------------------------------- #
# Analysis on aligned panels
# --------------------------------------------------------------------------- #
def build_panels(scores: pd.DataFrame, validation_raw: pd.DataFrame) -> dict[str, object]:
    """Align ensemble scores, forward labels and reversal factors on a [T, N] grid."""
    keys = ["session", "timestamp", "symbol"]
    scores = scores.copy()
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True)
    scores["session"] = scores["session"].astype(str)
    raw = validation_raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["session"] = raw["session"].astype(str)
    labels = {f"cum_{h}": forward_sector_excess(raw, 0, h) for h in CUMULATIVE_HORIZONS}
    labels.update({f"int_{a}_{b}": forward_sector_excess(raw, a, b) for a, b in INTERVALS})
    raw = pd.concat([raw[keys], pd.DataFrame(labels), reversal_factors(raw)], axis=1)
    merged = scores[keys + ["prediction", "rank_target_return"]].merge(raw, on=keys, how="left")
    check = merged[["rank_target_return", "cum_30"]].dropna()
    label_check = float((check["rank_target_return"] - check["cum_30"]).abs().max()) if len(check) else float("nan")

    times = merged[["session", "timestamp"]].drop_duplicates().sort_values(["session", "timestamp"]).reset_index(drop=True)
    symbols = sorted(merged["symbol"].unique())
    row = pd.MultiIndex.from_frame(times)
    panels: dict[str, np.ndarray] = {}
    for column in ["prediction"] + list(labels) + ["reversal_5m", "reversal_15m", "reversal_30m"]:
        pivot = merged.pivot_table(index=["session", "timestamp"], columns="symbol", values=column, aggfunc="first")
        panels[column] = pivot.reindex(index=row, columns=symbols).to_numpy(dtype=float)
    return {
        "panels": panels,
        "sessions": times["session"].to_numpy(),
        "timestamps": times["timestamp"].to_numpy(),
        "symbols": symbols,
        "label_check_max_abs_diff": label_check,
    }


def ic_decay(score: np.ndarray, data: dict) -> dict[str, object]:
    panels, sessions = data["panels"], data["sessions"]
    return {
        "cumulative": {
            f"(0,{h}]": daily_ic_summary(rowwise_rank_correlation(score, panels[f"cum_{h}"]), sessions)
            for h in CUMULATIVE_HORIZONS
        },
        "incremental": {
            f"({a},{b}]": daily_ic_summary(rowwise_rank_correlation(score, panels[f"int_{a}_{b}"]), sessions)
            for a, b in INTERVALS
        },
    }


def persistence(score: np.ndarray, sessions: np.ndarray) -> dict[str, float]:
    return {
        f"lag_{lag}m": lagged_autocorrelation(score, sessions, lag // SCORE_STRIDE_MINUTES)
        for lag in AUTOCORRELATION_LAGS
    }


def ic_by_time_of_day(score: np.ndarray, data: dict) -> dict[str, dict[str, float]]:
    per_row = rowwise_rank_correlation(score, data["panels"]["cum_30"])
    bucket = (
        pd.to_datetime(pd.Series(data["timestamps"]), utc=True)
        .dt.tz_convert("America/New_York")
        .dt.floor("30min")
        .dt.strftime("%H:%M")
        .to_numpy()
    )
    result = {}
    for name in sorted(set(bucket)):
        mask = bucket == name
        result[name] = daily_ic_summary(per_row[mask], data["sessions"][mask])
    return result


def smoothed_execution(
    scores: pd.DataFrame, smoothed: np.ndarray, data: dict, run_dir: Path, kappas: tuple[float, ...]
) -> dict[str, object]:
    """Cost curves of the baseline rule and the L1 optimizer on a smoothed score."""
    frame = scores.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["session"] = frame["session"].astype(str)
    lookup = pd.DataFrame(smoothed, columns=data["symbols"])
    lookup["session"] = pd.Series(data["sessions"]).astype(str).to_numpy()
    lookup["timestamp"] = pd.to_datetime(pd.Series(data["timestamps"]), utc=True).to_numpy()
    long = lookup.melt(id_vars=["session", "timestamp"], var_name="symbol", value_name="smoothed")
    frame = frame.merge(long, on=["session", "timestamp", "symbol"], how="left")
    frame["prediction"] = frame["smoothed"].fillna(frame["prediction"])
    settings = er._execution_settings(run_dir)
    target_scale, baseline_variance = er._target_scale_and_baseline(run_dir)
    base_panel = er.build_panel(frame, settings, target_scale, "none", baseline_variance)
    baseline = er.baseline_curve(base_panel, settings)
    target_gross = float(baseline.loc[baseline["cost_bps"] == 0.0, "mean_gross_exposure"].iloc[0])
    result = {
        "rho_at_rebalance": er.signal_persistence(base_panel)["mean_consecutive_rank_autocorrelation"],
        "baseline": {
            "breakeven_bps": er.breakeven_cost(baseline),
            "sharpe_by_cost": {f"{c:g}": round(float(s), 3) for c, s in zip(baseline["cost_bps"], baseline["annualized_sharpe"])},
            "turnover_per_unit_gross": float(baseline["turnover_per_unit_gross"].iloc[0]),
        },
        "optimizer": {},
    }
    variance_source = "sde" if (target_scale is not None and "predicted_variance_std" in frame.columns) else "homoskedastic"
    panel = er.build_panel(frame, settings, target_scale, variance_source, baseline_variance)
    for kappa in kappas:
        curve = pd.DataFrame.from_records(
            [
                {"cost_bps": cost, **er.summarise(panel, er.simulate(panel, er.optimizer_policy(panel, kappa, cost, er.GRINOLD_IC), cost), target_gross)}
                for cost in er.COSTS_BPS
            ]
        )
        result["optimizer"][f"kappa={kappa:g}"] = {
            "variance_source": variance_source,
            "breakeven_bps": er.breakeven_cost(curve),
            "sharpe_by_cost": {f"{c:g}": round(float(s), 3) for c, s in zip(curve["cost_bps"], curve["annualized_sharpe"])},
            "turnover_per_unit_gross_at_design_cost": float(
                curve.loc[np.isclose(curve["cost_bps"], er.DESIGN_COST_BPS), "turnover_per_unit_gross"].iloc[0]
            ),
        }
    return result


def _kappa_grid(rho: float) -> tuple[float, ...]:
    centre = min(max(1.0 - rho, 0.02), 1.0)
    return tuple(sorted({round(value, 3) for value in (0.0, 0.5 * centre, centre, min(2.0 * centre, 1.0), 1.0)}))


def run_analysis(scores: pd.DataFrame, data: dict, run_dir: Path | None) -> dict[str, object]:
    panels, sessions = data["panels"], data["sessions"]
    score = panels["prediction"]
    report: dict[str, object] = {"label_check_max_abs_diff_vs_pipeline_30m": data["label_check_max_abs_diff"]}

    print("IC decay and persistence of the raw score...", flush=True)
    report["model"] = {"ic_decay": ic_decay(score, data), "persistence": persistence(score, sessions)}
    report["ic_by_time_of_day_30m"] = ic_by_time_of_day(score, data)

    print("Reversal factors...", flush=True)
    reversal = {}
    for name in ("reversal_5m", "reversal_15m", "reversal_30m"):
        factor = panels[name]
        reversal[name] = {
            "ic_30m": daily_ic_summary(rowwise_rank_correlation(factor, panels["cum_30"]), sessions),
            "ic_60m": daily_ic_summary(rowwise_rank_correlation(factor, panels["cum_60"]), sessions),
            "persistence_lag_30m": lagged_autocorrelation(factor, sessions, 30 // SCORE_STRIDE_MINUTES),
            "mean_cross_sectional_corr_with_model_score": float(np.nanmean(rowwise_rank_correlation(score, factor))),
        }
    residual = residualise(score, [panels["reversal_5m"], panels["reversal_15m"], panels["reversal_30m"]])
    reversal["model_score_after_removing_reversal"] = {
        "ic_30m": daily_ic_summary(rowwise_rank_correlation(residual, panels["cum_30"]), sessions),
        "ic_60m": daily_ic_summary(rowwise_rank_correlation(residual, panels["cum_60"]), sessions),
        "persistence_lag_30m": lagged_autocorrelation(residual, sessions, 30 // SCORE_STRIDE_MINUTES),
    }
    report["reversal"] = reversal

    print("EMA smoothing...", flush=True)
    smoothing = {}
    for half_life in EMA_HALF_LIVES:
        smoothed = ema_within_session(score, sessions, half_life / SCORE_STRIDE_MINUTES)
        entry = {
            "ic_30m": daily_ic_summary(rowwise_rank_correlation(smoothed, panels["cum_30"]), sessions),
            "ic_60m": daily_ic_summary(rowwise_rank_correlation(smoothed, panels["cum_60"]), sessions),
            "persistence_lag_30m": lagged_autocorrelation(smoothed, sessions, 30 // SCORE_STRIDE_MINUTES),
        }
        if run_dir is not None:
            print(f"  execution on EMA half-life {half_life} min...", flush=True)
            entry["execution"] = smoothed_execution(
                scores, smoothed, data, run_dir, _kappa_grid(entry["persistence_lag_30m"])
            )
        smoothing[f"half_life_{half_life}m"] = entry
    report["ema_smoothing"] = smoothing
    return report


def print_summary(report: dict[str, object]) -> None:
    def fmt(ic: dict) -> str:
        return f"{ic['mean_daily_ic']:+.4f} [{ic['ci_95_lower']:+.4f}, {ic['ci_95_upper']:+.4f}]"

    print("\n=== Signal horizon summary (validation) ===")
    print(f"30m label check vs pipeline (max abs diff): {report['label_check_max_abs_diff_vs_pipeline_30m']:.2e}")
    model = report["model"]
    print("IC, cumulative horizons:")
    for name, ic in model["ic_decay"]["cumulative"].items():
        print(f"  {name:>9}: {fmt(ic)}")
    print("IC, disjoint intervals:")
    for name, ic in model["ic_decay"]["incremental"].items():
        print(f"  {name:>9}: {fmt(ic)}")
    print("Score rank autocorrelation:", {k: round(v, 3) for k, v in model["persistence"].items()})
    print("Reversal:")
    for name, values in report["reversal"].items():
        extra = values.get("mean_cross_sectional_corr_with_model_score")
        extra_text = f" | corr with model {extra:+.3f}" if extra is not None else ""
        print(f"  {name:<38} IC30 {fmt(values['ic_30m'])} | rho30 {values['persistence_lag_30m']:+.3f}{extra_text}")
    print("EMA smoothing:")
    for name, values in report["ema_smoothing"].items():
        line = f"  {name:<16} IC30 {fmt(values['ic_30m'])} | IC60 {fmt(values['ic_60m'])} | rho30 {values['persistence_lag_30m']:+.3f}"
        if "execution" in values:
            execution = values["execution"]
            best = max(execution["optimizer"].items(), key=lambda item: item[1]["sharpe_by_cost"].get("0.6", -1e9))
            line += (
                f"\n      baseline breakeven {execution['baseline']['breakeven_bps']:.2f} bp, Sharpe {execution['baseline']['sharpe_by_cost']}"
                f"\n      best optimizer @0.6bp {best[0]}: breakeven {best[1]['breakeven_bps']:.2f} bp, Sharpe {best[1]['sharpe_by_cost']}"
            )
        print(line)
    print("IC by time of day (30m):")
    for name, ic in report["ic_by_time_of_day_30m"].items():
        print(f"  {name}: {fmt(ic)}")


def main() -> None:
    run_dir = _resolve(RUN_DIR)
    out_dir = run_dir / "signal_horizon_validation"
    out_dir.mkdir(exist_ok=True)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    scaler = json.loads((run_dir / "feature_scaler.json").read_text(encoding="utf-8"))
    print(f"Run: {run_dir.name}\nLoading bars and rebuilding the validation split...", flush=True)
    validation_raw = load_validation_frame(config)
    validation_scaled = apply_feature_scaler(validation_raw, scaler)
    print(f"Scoring validation every {SCORE_STRIDE_MINUTES} minutes with {len(config['ensemble_seeds'])} seeds...", flush=True)
    scores = score_validation(run_dir, config, validation_scaled)
    scores.to_parquet(out_dir / "validation_scores_5min.parquet", index=False)
    data = build_panels(scores, validation_raw)
    report = run_analysis(scores, data, run_dir)
    (out_dir / "signal_horizon_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
