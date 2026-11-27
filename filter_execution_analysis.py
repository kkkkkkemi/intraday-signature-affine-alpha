"""Holding-period filter vs re-forecasting: information and trading cost (validation only).

Run from the same environment as train_model.py, after a multi-horizon run
(SDE_HORIZON_STEPS > 1).  Two parts:

1. Causal re-scoring.  The filter in older runs conditioned on the partial return
   to open(t+1+tau), which is also the first price of the remaining return
   (one-tick look-ahead, bid-ask bounce).  The saved checkpoints are re-scored
   with the causal filter (partial return to the close of minute t+tau) and the
   information diagnostics are rewritten as horizon_filter_validation_causal.json.
   Skipped when the causal ensemble parquet already exists.

2. Execution comparison on the 5-minute grid.  Every 30 minutes an anchor opens a
   holding period; positions are sector-neutral, linear in the sector-demeaned
   signal, gross 1, flat at every close.  Signals at anchor + 5m:
     static              the anchor's 30-minute forecast, held unchanged (trades every 30 min)
     filtered            Kalman/Merton update of the anchor forecast given the causal partial path
     fresh_remaining     re-run model: forecast from the new window to the anchor's end
     fresh_30m / fresh_5m re-run model: 30-minute / 5-minute forecast from the new window
     filtered_plus_fresh equal-weight sum of the z-scored filtered and fresh_remaining signals
     partial_reversal    minus the causal partial return (is the update just reversal?)
   Every signal earns the same open-to-open 5-minute returns; costs are charged
   on one-way turnover.  Each signal is also run with partial adjustment
   w_t = w_{t-1} + kappa (target - w_{t-1}).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from project_settings import OUTPUT_DIR

RUN_DIR = OUTPUT_DIR
# "validation", or "oos" for the final run (EVALUATE_TEST = True), whose sealed
# test-split filter frame is written by train_model.py with the causal filter.
SPLIT = "oos"
COST_GRID_BPS = (0.0, 0.3, 0.6, 1.0, 1.5)
ADJUSTMENT_RATES = (1.0, 0.5, 0.25)
TRADING_DAYS_PER_YEAR = 252
PROJECT_DIR = Path(__file__).resolve().parent


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


# --------------------------------------------------------------------------- #
# 1. Causal re-scoring of the saved checkpoints
# --------------------------------------------------------------------------- #
def load_validation_frame(config: dict) -> pd.DataFrame:
    """Raw (unscaled) validation feature frame, split exactly as in training."""
    from minute_alpha import (
        apply_feature_drops,
        engineer_minute_features,
        load_etf_bars,
        load_minute_bars,
        split_by_session,
    )

    data_dir = _resolve(config["data_dir"])
    bars = load_minute_bars(data_dir, config.get("symbols"))
    etf_bars = load_etf_bars(data_dir) if config.get("use_etf_context", False) else None
    features = engineer_minute_features(
        bars,
        int(config["holding_minutes"]),
        sector_median_shrinkage_count=float(config.get("sector_median_shrinkage_count", 5.0)),
        etf_bars=etf_bars,
    )
    features = apply_feature_drops(features, config.get("drop_feature_groups") or ())
    _train, validation, _test = split_by_session(features, float(config["train_ratio"]), float(config["val_ratio"]))
    return validation


def rescore_causal_filter(run_dir: Path, config: dict) -> pd.DataFrame:
    import torch

    from minute_alpha import (
        FEATURE_COLUMNS,
        CrossSectionalMinuteDataset,
        CrossTickerAreaPCA,
        MultiHorizonNLL,
        SignatureConditionedAffineMeanModel,
        apply_feature_scaler,
        average_filter_frames,
        evaluate_horizon_filter,
        horizon_filter_frame,
    )

    steps = int(config["sde_horizon_steps"])
    spec = json.loads((run_dir / "sde_model_spec.json").read_text(encoding="utf-8"))
    scaler = json.loads((run_dir / "feature_scaler.json").read_text(encoding="utf-8"))
    validation = apply_feature_scaler(load_validation_frame(config), scaler)
    dataset = CrossSectionalMinuteDataset(
        validation,
        int(config["window_minutes"]),
        int(config["min_labelled_symbols"]),
        int(config["evaluation_sample_stride_minutes"]),
    )
    del validation
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
    nll = MultiHorizonNLL(
        float(spec["target_scale"]),
        float(config["sde_target_clip"]),
        str(config["sde_jump_mode"]),
        steps,
        int(config.get("sde_horizon_jump_max_count", 3)),
    ).to(device)
    frames = []
    for seed in config["ensemble_seeds"]:
        model = SignatureConditionedAffineMeanModel(
            input_channels=len(FEATURE_COLUMNS),
            signature_depth=int(config["signature_depth"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dim=int(config["affine_hidden_dim"]),
            num_symbols=len(dataset.symbols),
            cross_area_pca=pca,
            diffusion_rank=int(config["sde_diffusion_rank"]),
            diffusion_floor=float(config.get("sde_diffusion_floor", 1e-4)),
            jump_mode=str(config["sde_jump_mode"]),
            stock_embedding_dim=int(config.get("stock_embedding_dim", 0)),
            cross_attention_layers=int(config.get("cross_attention_layers", 0)),
            attention_dim=int(config.get("attention_dim", 64)),
            attention_heads=int(config.get("attention_heads", 4)),
            horizon_steps=steps,
        ).to(device)
        state = torch.load(run_dir / f"seed_{seed}_best_signature_affine_mean.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        frame = horizon_filter_frame(model, dataset, nll, int(config["holding_minutes"]), device, False, device.type == "cuda")
        frame.to_parquet(run_dir / f"seed_{seed}_validation_horizon_filter_causal.parquet", index=False)
        frames.append(frame)
        print(f"  re-scored seed {seed}: {len(frame):,} rows", flush=True)
    ensemble = average_filter_frames(frames)
    ensemble.to_parquet(run_dir / "ensemble_validation_horizon_filter_causal.parquet", index=False)
    report = {
        "note": "filter conditioned on the causal partial return (to the close of the observation minute)",
        "ensemble": evaluate_horizon_filter(
            ensemble, int(config["holding_minutes"]), int(config["bootstrap_iterations"]), int(config["seed"])
        ),
        "seeds": {
            str(seed): evaluate_horizon_filter(
                frame, int(config["holding_minutes"]), int(config["bootstrap_iterations"]), int(config["seed"])
            )
            for seed, frame in zip(config["ensemble_seeds"], frames)
        },
    }
    (run_dir / "horizon_filter_validation_causal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ensemble


def print_information_summary(report: dict) -> None:
    print("\nRemaining-return Rank IC after observing part of the holding period (causal filter, ensemble):")
    print(f"{'observed':>9} {'uncond':>8} {'filtered':>9} {'fresh':>8} {'partial':>8} {'filt-unc [95% CI]':>26} {'filt-fresh [95% CI]':>26}")
    for observed, summary in report["update_after_observing"].items():
        def ic(name: str) -> float:
            return summary[name]["mean_daily_rank_ic"]

        def paired(name: str) -> str:
            values = summary[name]
            return f"{values['mean_daily_rank_ic']:+.4f} [{values['ci_95_lower']:+.4f},{values['ci_95_upper']:+.4f}]"

        print(
            f"{observed:>9} {ic('unconditional'):8.4f} {ic('filtered'):9.4f} {ic('fresh'):8.4f} "
            f"{ic('partial_return'):8.4f} {paired('paired_filtered_minus_unconditional'):>26} "
            f"{paired('paired_filtered_minus_fresh'):>26}"
        )


# --------------------------------------------------------------------------- #
# 2. Execution comparison
# --------------------------------------------------------------------------- #
def _sector_weights(signal: np.ndarray, sector_codes: np.ndarray, sector_count: int) -> np.ndarray:
    """Sector-demeaned, gross-1 weights for a [periods, names] signal (NaN -> 0)."""
    valid = np.isfinite(signal)
    values = np.where(valid, signal, 0.0)
    one_hot = np.eye(sector_count)[sector_codes]
    sums = values @ one_hot
    counts = valid.astype(float) @ one_hot
    centred = np.where(valid, values - (sums / np.maximum(counts, 1.0))[:, sector_codes], 0.0)
    centred = np.where(counts[:, sector_codes] >= 2, centred, 0.0)
    gross = np.abs(centred).sum(axis=1, keepdims=True)
    return np.divide(centred, gross, out=np.zeros_like(centred), where=gross > 0)


def _zscore_within_sector(signal: np.ndarray, sector_codes: np.ndarray, sector_count: int) -> np.ndarray:
    out = np.full_like(signal, np.nan)
    for code in range(sector_count):
        members = sector_codes == code
        block = signal[:, members]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(block, axis=1, keepdims=True)
            std = np.nanstd(block, axis=1, keepdims=True)
        out[:, members] = (block - mean) / np.where(std > 0, std, np.nan)
    return out


def build_periods(frame: pd.DataFrame, holding_minutes: int) -> dict[str, object]:
    """Anchor-aligned [anchors, steps, names] arrays of returns and every signal."""
    steps = sum(1 for column in frame.columns if column.startswith("y_h"))
    step_minutes = holding_minutes // steps
    symbols = sorted(frame["symbol"].unique())
    sectors = frame.drop_duplicates("symbol").set_index("symbol").loc[symbols, "sector"]
    sector_names = sorted(sectors.unique())
    sector_codes = np.array([sector_names.index(sector) for sector in sectors])
    columns = [column for column in frame.columns if column.startswith(("y_h", "obs_h", "mu_h", "filter_mean_m"))]
    wide = {
        column: frame.pivot(index=["session", "timestamp"], columns="symbol", values=column).reindex(columns=symbols)
        for column in columns
    }
    keys = wide["y_h1"].index
    key_set = set(keys)
    anchors = []
    for session, times in pd.Series(keys.get_level_values("timestamp"), index=keys.get_level_values("session")).groupby(level=0):
        first = times.min()
        for timestamp in times:
            offset = (timestamp - first) / pd.Timedelta(minutes=1)
            if offset % holding_minutes != 0:
                continue
            later = [(session, timestamp + pd.Timedelta(minutes=m * step_minutes)) for m in range(1, steps)]
            if all(key in key_set for key in later):
                anchors.append((session, timestamp))
    anchor_index = pd.MultiIndex.from_tuples(anchors, names=["session", "timestamp"])

    def at(column: str, minutes: int) -> np.ndarray:
        shifted = pd.MultiIndex.from_arrays(
            [anchor_index.get_level_values(0), anchor_index.get_level_values(1) + pd.Timedelta(minutes=minutes)]
        )
        return wide[column].reindex(shifted).to_numpy(dtype=float)

    count, names = len(anchor_index), len(symbols)
    cumulative = np.zeros((count, steps + 1, names))
    for k in range(1, steps + 1):
        cumulative[:, k] = at(f"y_h{k}", 0)
    returns = cumulative[:, 1:] - cumulative[:, :-1]
    anchor_forecast = at(f"mu_h{steps}", 0)
    signals = {name: np.full((count, steps, names), np.nan) for name in (
        "static", "filtered", "fresh_remaining", "fresh_30m", "fresh_5m", "partial_reversal"
    )}
    for m in range(steps):
        minutes = m * step_minutes
        signals["static"][:, m] = anchor_forecast
        signals["fresh_30m"][:, m] = at(f"mu_h{steps}", minutes)
        signals["fresh_5m"][:, m] = at("mu_h1", minutes)
        if m == 0:
            for name in ("filtered", "fresh_remaining", "partial_reversal"):
                signals[name][:, 0] = anchor_forecast
        else:
            signals["filtered"][:, m] = at(f"filter_mean_m{m}", 0)
            signals["fresh_remaining"][:, m] = at(f"mu_h{steps - m}", minutes)
            signals["partial_reversal"][:, m] = -at(f"obs_h{m}", 0)
    flat = lambda array: array.reshape(count * steps, names)  # noqa: E731
    combined = _zscore_within_sector(flat(signals["filtered"]), sector_codes, len(sector_names)) + _zscore_within_sector(
        flat(signals["fresh_remaining"]), sector_codes, len(sector_names)
    )
    signals["filtered_plus_fresh"] = combined.reshape(count, steps, names)
    available = np.isfinite(returns) & np.isfinite(anchor_forecast)[:, None, :]
    return {
        "anchors": anchor_index,
        "returns": np.where(available, returns, 0.0),
        "available": available,
        "signals": signals,
        "sector_codes": sector_codes,
        "sector_count": len(sector_names),
        "steps": steps,
        "step_minutes": step_minutes,
    }


def simulate(periods: dict[str, object], signal: np.ndarray, kappa: float, scale_bp: float) -> pd.DataFrame:
    """Daily gross PnL (bp of gross capital) and one-way turnover for one signal."""
    anchors: pd.MultiIndex = periods["anchors"]
    steps = periods["steps"]
    returns, available = periods["returns"], periods["available"]
    names = returns.shape[2]
    target = _sector_weights(
        np.where(available, signal, np.nan).reshape(-1, names), periods["sector_codes"], periods["sector_count"]
    ).reshape(len(anchors), steps, names)
    sessions = anchors.get_level_values(0)
    times = anchors.get_level_values(1)
    records = []
    weight = np.zeros(names)
    pnl = turnover = 0.0
    for i in range(len(anchors)):
        new_session = i == 0 or sessions[i] != sessions[i - 1]
        contiguous = not new_session and times[i] - times[i - 1] == pd.Timedelta(minutes=steps * periods["step_minutes"])
        if not new_session and not contiguous:
            turnover += np.abs(weight).sum()
            weight = np.zeros(names)
        if new_session:
            if i > 0:
                turnover += np.abs(weight).sum()
                records.append((sessions[i - 1], pnl, turnover))
            weight, pnl, turnover = np.zeros(names), 0.0, 0.0
        for m in range(steps):
            new_weight = weight + kappa * (target[i, m] - weight)
            new_weight = np.where(available[i, m], new_weight, 0.0)
            turnover += np.abs(new_weight - weight).sum()
            weight = new_weight
            pnl += float(weight @ returns[i, m]) * scale_bp
    turnover += np.abs(weight).sum()
    records.append((sessions[-1], pnl, turnover))
    return pd.DataFrame(records, columns=["session", "gross_bp", "turnover"])


def summarise(daily: pd.DataFrame, periods_per_day: float) -> dict[str, object]:
    result = {
        "days": int(len(daily)),
        "gross_bp_per_day": float(daily["gross_bp"].mean()),
        "turnover_per_day": float(daily["turnover"].mean()),
        "turnover_per_5m_step": float(daily["turnover"].mean() / periods_per_day),
        "break_even_one_way_cost_bp": float(daily["gross_bp"].sum() / daily["turnover"].sum()),
    }
    for cost in COST_GRID_BPS:
        net = daily["gross_bp"] - cost * daily["turnover"]
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if net.std(ddof=1) > 0 else float("nan")
        result[f"net_{cost:g}bp"] = {"bp_per_day": float(net.mean()), "annualised_sharpe": sharpe}
    return result


def main() -> None:
    run_dir = _resolve(RUN_DIR)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if int(config.get("sde_horizon_steps", 1)) < 2:
        raise ValueError(f"{run_dir.name} is not a multi-horizon run (sde_horizon_steps < 2).")
    if SPLIT == "oos":
        if not config.get("evaluate_test", False):
            raise ValueError(f"{run_dir.name} did not score the test split (evaluate_test=False).")
        # Runs trained with the current code already use the causal filter.
        frame = pd.read_parquet(run_dir / "ensemble_oos_horizon_filter.parquet")
        if "obs_h1" not in frame.columns:
            raise ValueError("This OOS filter frame predates the causal filter; re-train with the current code.")
        report_path = run_dir / "horizon_filter_oos.json"
    else:
        causal_path = run_dir / "ensemble_validation_horizon_filter_causal.parquet"
        plain_path = run_dir / "ensemble_validation_horizon_filter.parquet"
        if causal_path.exists():
            print(f"Using {causal_path.name}")
            frame = pd.read_parquet(causal_path)
        elif plain_path.exists() and "obs_h1" in pd.read_parquet(plain_path, columns=None).columns:
            frame = pd.read_parquet(plain_path)
        else:
            print("Re-scoring the checkpoints with the causal filter...")
            frame = rescore_causal_filter(run_dir, config)
        report_path = run_dir / (
            "horizon_filter_validation_causal.json"
            if (run_dir / "horizon_filter_validation_causal.json").exists()
            else "horizon_filter_validation.json"
        )
    if report_path.exists():
        print_information_summary(json.loads(report_path.read_text(encoding="utf-8"))["ensemble"])

    spec = json.loads((run_dir / "sde_model_spec.json").read_text(encoding="utf-8"))
    scale_bp = float(spec["target_scale"]) * 1e4
    periods = build_periods(frame, int(config["holding_minutes"]))
    periods_per_day = len(periods["anchors"]) * periods["steps"] / periods["anchors"].get_level_values(0).nunique()
    print(
        f"\nExecution comparison: {len(periods['anchors']):,} holding periods, {periods['steps']} steps of "
        f"{periods['step_minutes']} min, ~{periods_per_day:.0f} decisions per day; returns in bp of gross capital "
        f"(1 standardised unit = {scale_bp:.1f} bp)."
    )
    results: dict[str, dict[str, object]] = {}
    for name, signal in periods["signals"].items():
        for kappa in ADJUSTMENT_RATES:
            if name == "static" and kappa != 1.0:
                continue
            daily = simulate(periods, signal, kappa, scale_bp)
            results[f"{name}|kappa={kappa:g}"] = summarise(daily, periods_per_day)
    output_path = run_dir / f"filter_execution_{SPLIT}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    header = f"{'signal':<22}{'kappa':>6}{'gross bp/d':>11}{'turn/d':>8}{'b/e bp':>8}" + "".join(
        f"{f'SR@{cost:g}bp':>9}" for cost in COST_GRID_BPS
    )
    print(header)
    for key, summary in results.items():
        name, kappa = key.split("|kappa=")
        print(
            f"{name:<22}{kappa:>6}{summary['gross_bp_per_day']:11.2f}{summary['turnover_per_day']:8.1f}"
            f"{summary['break_even_one_way_cost_bp']:8.3f}"
            + "".join(f"{summary[f'net_{cost:g}bp']['annualised_sharpe']:9.2f}" for cost in COST_GRID_BPS)
        )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
