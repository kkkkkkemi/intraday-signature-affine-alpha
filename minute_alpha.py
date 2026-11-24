"""Train a signature-conditioned latent linear-SDE model on minute OHLCV bars.

With ``sde_diffusion_rank=0`` the model reduces to the original deterministic
affine latent-mean model, which is kept as the ablation baseline.

Research scope
--------------
This module evaluates an intraday *price-volume* alpha.  Massive Basic minute bars
do not include quotes, order-book depth, cancellations, or order-flow imbalance;
the code deliberately does not manufacture any of those features.  A signal is
observed after bar t closes, entered at bar t+1's open, and exited after a configurable
number of minutes.  Because bid/ask quotes are unavailable, results are reported as
a configured one-way-cost assumption rather than executable PnL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.special as special
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import signatory
except ImportError as exc:  # pragma: no cover - makes the installation failure actionable
    raise ImportError("Install signatory and the project requirements before running the minute-alpha pipeline.") from exc


FEATURE_COLUMNS = (
    "t_dim",
    "relative_cum_ret",
    "relative_return_5m",
    "relative_return_15m",
    "momentum_acceleration_5v15",
    "log_volume",
    "range_rel",
    "realized_vol",
    "vwap_deviation",
    "clv",
    "has_trade",
    "staleness_minutes",
    # --- v2 context features (appended; earlier channel indices are unchanged) ---
    "volume_surprise",
    "cumulative_volume_surprise",
    "log_transactions",
    "log_avg_trade_size",
    "relative_return_since_open",
    "overnight_gap_relative",
    "prior_day_relative_return",
    "prior_day_realized_vol",
    "intraday_range_position",
    "session_vwap_deviation",
    "market_return_5m",
    "market_return_15m",
    "market_return_30m",
    "sector_etf_relative_return_5m",
    "sector_etf_relative_return_15m",
    "sector_etf_relative_return_30m",
)
# Endpoint values of the v2 features, plus window means of the per-minute flow
# features, enter the model as extra context (the signature path is unchanged).
EXTRA_CONTEXT_ENDPOINT_COLUMNS = FEATURE_COLUMNS[12:]
EXTRA_CONTEXT_MEAN_COLUMNS = ("volume_surprise", "log_transactions", "log_avg_trade_size")
# Named groups of v2 features that a run can switch off (set to zero) for ablations.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # Constant within a day: together they can act as a stock-day fingerprint.
    "day_constant": ("overnight_gap_relative", "prior_day_relative_return", "prior_day_realized_vol"),
    "volume_flow": ("volume_surprise", "cumulative_volume_surprise", "log_transactions", "log_avg_trade_size"),
    "session": ("relative_return_since_open", "intraday_range_position", "session_vwap_deviation"),
    "etf": (
        "market_return_5m", "market_return_15m", "market_return_30m",
        "sector_etf_relative_return_5m", "sector_etf_relative_return_15m", "sector_etf_relative_return_30m",
    ),
}
VOLUME_SURPRISE_LOOKBACK_SESSIONS = 20
VOLUME_SURPRISE_MIN_SESSIONS = 5
TIME_FEATURE_INDEX = FEATURE_COLUMNS.index("t_dim")
RELATIVE_PRICE_FEATURE_INDEX = FEATURE_COLUMNS.index("relative_cum_ret")
VOLUME_FEATURE_INDEX = FEATURE_COLUMNS.index("log_volume")
RANGE_FEATURE_INDEX = FEATURE_COLUMNS.index("range_rel")
REALIZED_VOLATILITY_FEATURE_INDEX = FEATURE_COLUMNS.index("realized_vol")
DIRECTIONAL_MULTISCALE_COLUMNS = (
    "relative_return_5m",
    "relative_return_15m",
    "momentum_acceleration_5v15",
)
DIRECTIONAL_MULTISCALE_INDICES = tuple(FEATURE_COLUMNS.index(column) for column in DIRECTIONAL_MULTISCALE_COLUMNS)
SIGNATURE_CONTINUOUS_COLUMNS = (
    "relative_cum_ret",
    "log_volume",
    "range_rel",
    "realized_vol",
    "vwap_deviation",
    "clv",
)
SIGNATURE_CONTINUOUS_INDICES = tuple(FEATURE_COLUMNS.index(column) for column in SIGNATURE_CONTINUOUS_COLUMNS)
HAS_TRADE_FEATURE_INDEX = FEATURE_COLUMNS.index("has_trade")
STALENESS_FEATURE_INDEX = FEATURE_COLUMNS.index("staleness_minutes")
VWAP_DEVIATION_FEATURE_INDEX = FEATURE_COLUMNS.index("vwap_deviation")
INITIAL_STATE_ANCHOR_COLUMNS = (
    "t_dim",
    "log_volume",
    "range_rel",
    "realized_vol",
    "has_trade",
    "staleness_minutes",
)
INITIAL_STATE_ANCHOR_INDICES = tuple(FEATURE_COLUMNS.index(column) for column in INITIAL_STATE_ANCHOR_COLUMNS)
INITIAL_STATE_ANCHOR_MINUTES = 5
INITIAL_STATE_PATH_STATISTICS = (
    "relative_price_quadratic_variation",
    "mean_absolute_vwap_deviation",
    "log_volume_standard_deviation",
    "mean_relative_range",
    "mean_realized_volatility",
    "mean_has_trade",
    "maximum_staleness",
)
INITIAL_STATE_CONTEXT_DIM = len(INITIAL_STATE_ANCHOR_COLUMNS) + len(INITIAL_STATE_PATH_STATISTICS)
SIGNATURE_PATH_CHANNELS = 1 + 2 * len(SIGNATURE_CONTINUOUS_COLUMNS)
SELECTIVE_SIGNATURE_DIM = SIGNATURE_PATH_CHANNELS + SIGNATURE_PATH_CHANNELS * (SIGNATURE_PATH_CHANNELS - 1) // 2
STATE_CONTEXT_DIM = 7
DIRECTIONAL_MULTISCALE_CONTEXT_DIM = len(DIRECTIONAL_MULTISCALE_COLUMNS)
EXTRA_CONTEXT_ENDPOINT_INDICES = tuple(FEATURE_COLUMNS.index(column) for column in EXTRA_CONTEXT_ENDPOINT_COLUMNS)
EXTRA_CONTEXT_MEAN_INDICES = tuple(FEATURE_COLUMNS.index(column) for column in EXTRA_CONTEXT_MEAN_COLUMNS)
EXTRA_CONTEXT_DIM = len(EXTRA_CONTEXT_ENDPOINT_INDICES) + len(EXTRA_CONTEXT_MEAN_INDICES)
CLV_DENOMINATOR_EPS = 1e-12
RAW_COLUMNS = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap"}
# Fixed GICS sectors for every supported research universe (see universe.py).
# They are part of the feature specification rather than a live vendor lookup.
# V and MA are Financials under the March 2023 GICS reclassification.
from universe import ETF_TICKERS, MARKET_ETF, SECTOR_BY_SYMBOL, SECTOR_ETF  # noqa: E402
SECTOR_MEDIAN_SHRINKAGE_COUNT = 5.0
# Cross-ticker PCA memory bound (elements of the windows x area-columns matrix).
PCA_MAX_MATRIX_ELEMENTS = 150_000_000
PCA_MIN_FIT_WINDOWS = 4_000
# Universes above this size store engineered features in float32 to save memory.
FLOAT32_FEATURE_SYMBOL_THRESHOLD = 50
JUMP_MODES = ("none", "merton", "student_t")
PREDICTIVE_PARAMETER_COLUMNS = {
    "mean": "pred_mu",
    "gaussian_variance": "pred_gvar",
    "jump_intensity": "pred_jump_intensity",
    "jump_variance": "pred_jump_var",
    "dof": "pred_dof",
}


def _inverse_softplus(value: float) -> float:
    return float(value + math.log(-math.expm1(-value)))


@dataclass(frozen=True)
class RunConfig:
    data_dir: str = "minute_data"
    output_dir: str = "artifacts/signature_affine_n25_sig91_state7_ms3_clv_sector_shrink5_target_sector_mean_normalscore_ens42_z0anchor5scale_pp_pca12_pv_pca12_gate1bp_m2bp_w30_h30_tr5_ev30"
    window_minutes: int = 30
    holding_minutes: int = 30
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 16
    latent_dim: int = 6
    affine_hidden_dim: int = 12
    signature_depth: int = 2
    price_price_pca_components: int = 12
    price_volume_pca_components: int = 12
    sector_median_shrinkage_count: float = SECTOR_MEDIAN_SHRINKAGE_COUNT
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 15
    patience: int = 3
    warmup_epochs: int = 1
    z_threshold: float = 1.0
    session_end_policy: str = "flatten_daily"
    target_smoothing_fraction: float = 0.5
    sector_rebalance_buffer_turnover: float = 0.02
    profitability_gate_cost_bps: float = 1.0
    profitability_gate_min_probability: float = 0.55
    profitability_gate_l2: float = 1.0
    min_labelled_symbols: int = 15
    train_sample_stride_minutes: int = 5
    evaluation_sample_stride_minutes: int = 30
    bootstrap_iterations: int = 10_000
    symbols: tuple[str, ...] | None = None
    data_loader_workers: int = 2
    pin_memory: bool = True
    amp_enabled: bool = False
    allow_tf32: bool = True
    deterministic: bool = True
    seed: int = 42
    ensemble_seeds: tuple[int, ...] = (42,)
    # Linear latent SDE.  0 = original deterministic affine-mean model (ablation).
    sde_diffusion_rank: int = 0
    sde_diffusion_floor: float = 1e-4
    # Total loss = rank loss + weight * Gaussian NLL of the standardised target.
    sde_nll_weight: float = 0.1
    sde_target_clip: float = 8.0
    # Ungated one-way cost grid for the mean vs uncertainty-scaled score comparison.
    sde_cost_curve_bps: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0, 1.5, 2.0)
    # Heavy-tailed holding-period component: "none", "merton" or "student_t".
    sde_jump_mode: str = "none"
    sde_jump_max_count: int = 8
    sde_jump_initial_intensity: float = 0.05
    sde_jump_initial_std: float = 2.0
    # False while tuning: the test split is never scored, and the execution /
    # gate / OOS evaluation is skipped.  Set True once for the final frozen run.
    evaluate_test: bool = True
    # Demean scores within sector in the rank loss and the validation IC, so the
    # training objective and checkpoint selection measure the within-sector
    # ranking that the ensemble and sector-neutral portfolio use.
    sector_neutral_objective: bool = True
    # Regularisation and ablation switches.
    context_dropout: float = 0.0
    attention_dropout: float = 0.1
    drop_feature_groups: tuple[str, ...] = ()
    # v2 inputs and architecture.
    use_etf_context: bool = True
    stock_embedding_dim: int = 8
    cross_attention_layers: int = 1
    attention_dim: int = 64
    attention_heads: int = 4
    # Training-set windows scored each epoch to compare train vs validation IC.
    train_ic_monitor_windows: int = 2000


def set_global_seed(seed: int, deterministic: bool, allow_tf32: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _autocast_context(device: torch.device, enabled: bool):
    """Use AMP only when explicitly enabled; matrix-exponential stability needs validation."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _to_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool) -> torch.Tensor:
    return tensor.to(device, non_blocking=non_blocking)


def _is_regular_session(timestamp: pd.Series) -> pd.Series:
    ny = pd.to_datetime(timestamp, utc=True).dt.tz_convert("America/New_York")
    minute_of_day = ny.dt.hour * 60 + ny.dt.minute
    return (ny.dt.dayofweek < 5) & (minute_of_day >= 570) & (minute_of_day < 960)


def load_minute_bars(data_dir: str | Path, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load downloader output and reject files that cannot support this study."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files in {data_path}. Run `python massive_download.py` first."
        )

    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_parquet(file)
        if "symbol" not in frame.columns:
            frame["symbol"] = file.stem.upper()
        missing = RAW_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{file.name} is missing required columns: {sorted(missing)}")
        if "transactions" not in frame.columns:
            frame["transactions"] = np.nan
        frames.append(frame[[*sorted(RAW_COLUMNS), "transactions"]].copy())

    bars = pd.concat(frames, ignore_index=True)
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "vwap", "transactions"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(axis=1) & (bars["volume"] >= 0)]
    bars["vwap"] = bars["vwap"].where(bars["vwap"] > 0, bars["close"])
    bars = bars.loc[_is_regular_session(bars["timestamp"])]
    bars = bars.drop_duplicates(["symbol", "timestamp"], keep="last").sort_values(["symbol", "timestamp"])
    if symbols is not None:
        requested_symbols = {symbol.upper() for symbol in symbols}
        available_symbols = set(bars["symbol"].unique())
        missing_symbols = sorted(requested_symbols.difference(available_symbols))
        if missing_symbols:
            raise FileNotFoundError(f"Requested symbols are missing from {data_path}: {missing_symbols}")
        bars = bars[bars["symbol"].isin(requested_symbols)]
    if bars["symbol"].nunique() < 3:
        raise ValueError("At least three symbols are required for a cross-sectional alpha experiment.")
    return bars.reset_index(drop=True)


def _expand_to_session_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """Put every selected symbol on a common minute grid without claiming fills.

    A missing aggregate is retained as an observed data state.  For model inputs,
    the last traded price is carried forward and volume is zero; ``has_trade`` and
    ``staleness_minutes`` tell the model that this is not a genuine OHLCV bar.
    Raw open/close prices remain separate and are used exclusively for labels.
    """
    symbols = sorted(bars["symbol"].unique())
    pieces: list[pd.DataFrame] = []
    observed_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap"]
    if "transactions" in bars.columns:
        observed_columns.append("transactions")

    for session, observed in bars.groupby("session", sort=True):
        observed = observed[observed_columns].drop_duplicates(["symbol", "timestamp"], keep="last").copy()
        if observed.empty:
            continue
        session_open = (
            pd.Timestamp(session, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        ).tz_convert("UTC")
        # The last bar across the whole universe identifies regular versus early
        # closes without assuming that every individual symbol traded at the close.
        session_end = observed["timestamp"].max().floor("min")
        if session_end < session_open:
            continue
        timestamps = pd.date_range(session_open, session_end, freq="min", tz="UTC")
        dense = pd.MultiIndex.from_product(
            [symbols, timestamps], names=["symbol", "timestamp"]
        ).to_frame(index=False)
        dense = dense.merge(observed, on=["symbol", "timestamp"], how="left", validate="one_to_one")
        dense["session"] = session
        dense["has_trade"] = dense["close"].notna().astype(np.float32)
        dense["raw_open"] = dense["open"].where(dense["has_trade"].eq(1.0))
        dense["raw_close"] = dense["close"].where(dense["has_trade"].eq(1.0))

        dense = dense.sort_values(["symbol", "timestamp"])
        last_trade_time = dense["timestamp"].where(dense["has_trade"].eq(1.0))
        dense["last_trade_timestamp"] = last_trade_time.groupby(dense["symbol"], sort=False).ffill()
        dense["staleness_minutes"] = (
            (dense["timestamp"] - dense["last_trade_timestamp"]).dt.total_seconds() / 60.0
        )
        dense["close"] = dense.groupby("symbol", sort=False)["close"].ffill()
        # These effective OHLCV values exist only to make a causal feature path.
        # They are never used as assumed execution prices.
        for column in ("open", "high", "low"):
            dense[column] = dense[column].where(dense["has_trade"].eq(1.0), dense["close"])
        dense["volume"] = dense["volume"].where(dense["has_trade"].eq(1.0), 0.0).fillna(0.0)
        if "transactions" in dense.columns:
            dense["transactions"] = dense["transactions"].where(dense["has_trade"].eq(1.0), 0.0).fillna(0.0)
        else:
            dense["transactions"] = np.nan
        dense["vwap"] = dense["vwap"].where(
            dense["has_trade"].eq(1.0) & dense["vwap"].gt(0), dense["close"]
        )
        pieces.append(dense)

    if not pieces:
        raise ValueError("Could not construct any regular-session minute grids from the downloaded bars.")
    return pd.concat(pieces, ignore_index=True).sort_values(["session", "timestamp", "symbol"]).reset_index(drop=True)


def sector_relative_return_feature_spec(
    symbols: Iterable[str], shrinkage_count: float
) -> dict[str, object]:
    """Describe the fixed sector benchmark used for relative-price increments."""
    ordered_symbols = sorted({str(symbol).upper() for symbol in symbols})
    missing = sorted(set(ordered_symbols).difference(SECTOR_BY_SYMBOL))
    if missing:
        raise ValueError(
            "Sector-relative returns require explicit sector labels for every selected symbol; "
            f"missing: {missing}. Add them to SECTOR_BY_SYMBOL."
        )
    sector_members = {
        sector: [symbol for symbol in ordered_symbols if SECTOR_BY_SYMBOL[symbol] == sector]
        for sector in sorted({SECTOR_BY_SYMBOL[symbol] for symbol in ordered_symbols})
    }
    return {
        "benchmark": "shrunken contemporaneous sector median of real one-minute log returns",
        "sector_labels": {symbol: SECTOR_BY_SYMBOL[symbol] for symbol in ordered_symbols},
        "sector_members": sector_members,
        "shrinkage_count": shrinkage_count,
        "sector_weight": "n_sector_real_bars / (n_sector_real_bars + shrinkage_count)",
        "relative_return": (
            "log_return_1m - (sector_weight * sector_median_return + "
            "(1 - sector_weight) * universe_median_return)"
        ),
        "real_bar_rule": "Only contemporaneous real bars enter either median and n_sector_real_bars.",
    }


def _add_v2_context_features(frame: pd.DataFrame, etf_bars: pd.DataFrame | None) -> pd.DataFrame:
    """Causal flow, session and ETF context features (all known at the close of minute t)."""
    real = frame["has_trade"].eq(1.0)
    session_keys = [frame["symbol"], frame["session"]]

    # Volume relative to the same minute over the previous sessions.
    frame["volume_surprise"] = frame["log_volume"] - _historical_same_minute_mean(frame, "log_volume")
    frame["_log_cumulative_volume"] = np.log1p(frame.groupby(session_keys, sort=False)["volume"].cumsum())
    frame["cumulative_volume_surprise"] = frame["_log_cumulative_volume"] - _historical_same_minute_mean(
        frame, "_log_cumulative_volume"
    )

    # Trade count and average trade size (zero on missing bars).
    transactions = frame["transactions"].fillna(0.0) if "transactions" in frame.columns else pd.Series(0.0, index=frame.index)
    frame["log_transactions"] = np.log1p(transactions.where(real, 0.0))
    frame["log_avg_trade_size"] = np.where(
        real & transactions.gt(0), frame["log_volume"] - np.log1p(transactions), 0.0
    )

    # Session context.
    frame["relative_return_since_open"] = frame["relative_cum_ret"]
    by_session = frame[real].groupby(["symbol", "session"], sort=True)
    daily = pd.DataFrame(
        {
            "first_open": by_session["raw_open"].first(),
            "first_time": by_session["timestamp"].first(),
            "last_close": by_session["raw_close"].last(),
            "day_relative_return": frame.groupby(["symbol", "session"], sort=True)["relative_cum_ret"].last(),
            "day_realized_vol": by_session["log_return_1m"].std(ddof=0),
        }
    ).reset_index().sort_values(["symbol", "session"])
    previous = daily.groupby("symbol", sort=False)[["last_close", "day_relative_return", "day_realized_vol"]].shift(1)
    daily["overnight_gap"] = np.log(daily["first_open"] / previous["last_close"])
    daily["prior_day_relative_return"] = previous["day_relative_return"]
    daily["prior_day_realized_vol"] = previous["day_realized_vol"]
    daily["sector"] = daily["symbol"].map(SECTOR_BY_SYMBOL)
    sector_gap = daily.groupby(["session", "sector"])["overnight_gap"].transform("median")
    universe_gap = daily.groupby("session")["overnight_gap"].transform("median")
    daily["overnight_gap_relative"] = daily["overnight_gap"] - sector_gap.fillna(universe_gap)
    merged = frame[["symbol", "session"]].merge(
        daily[["symbol", "session", "first_time", "overnight_gap_relative", "prior_day_relative_return", "prior_day_realized_vol"]],
        on=["symbol", "session"],
        how="left",
    )
    merged.index = frame.index
    known_gap = frame["timestamp"].ge(merged["first_time"])
    frame["overnight_gap_relative"] = merged["overnight_gap_relative"].where(known_gap, 0.0)
    frame["prior_day_relative_return"] = merged["prior_day_relative_return"]
    frame["prior_day_realized_vol"] = merged["prior_day_realized_vol"]

    running_high = frame.groupby(session_keys, sort=False)["close"].cummax()
    running_low = frame.groupby(session_keys, sort=False)["close"].cummin()
    span = running_high - running_low
    frame["intraday_range_position"] = np.where(span > 1e-12, (frame["close"] - running_low) / span.where(span > 1e-12, 1.0), 0.5)
    dollar_volume = (frame["vwap"] * frame["volume"]).groupby(session_keys, sort=False).cumsum()
    cumulative_volume = frame.groupby(session_keys, sort=False)["volume"].cumsum()
    session_vwap = dollar_volume / cumulative_volume.where(cumulative_volume > 0)
    frame["session_vwap_deviation"] = (frame["close"] / session_vwap - 1.0).fillna(0.0)

    etf_features = _etf_context_features(frame, etf_bars)
    for column in etf_features.columns:
        frame[column] = etf_features[column].to_numpy()
    return frame.drop(columns=["_log_cumulative_volume"])


def apply_feature_drops(frame: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    """Zero out named v2 feature groups (dimensions are kept, so runs stay comparable)."""
    groups = tuple(groups)
    unknown = sorted(set(groups).difference(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature group(s) {unknown}; choose from {sorted(FEATURE_GROUPS)}.")
    for group in groups:
        for column in FEATURE_GROUPS[group]:
            frame[column] = 0.0
    if groups:
        print(f"Feature groups switched off: {list(groups)}")
    return frame


def drop_incomplete_sessions(bars: pd.DataFrame) -> pd.DataFrame:
    """Remove sessions whose data stop before the market closed.

    A session's close is the last bar across the whole universe.  Regular early
    closes end at 13:00 New York time, so any session ending before 12:59 is
    incomplete; the final session in the data must reach 15:59, which removes a
    trading day that was still in progress when the data were downloaded (each
    ticker would otherwise carry a different number of its minutes).
    """
    if "session" not in bars.columns:
        raise ValueError("drop_incomplete_sessions needs a session column.")
    last_bar = bars.groupby("session")["timestamp"].max().dt.tz_convert("America/New_York")
    last_minute = last_bar.dt.hour * 60 + last_bar.dt.minute
    incomplete = last_minute < 12 * 60 + 59
    if len(last_minute):
        final_session = last_minute.index.max()
        incomplete.loc[final_session] = bool(incomplete.loc[final_session] or last_minute.loc[final_session] < 15 * 60 + 59)
    dropped = sorted(incomplete.index[incomplete])
    if dropped:
        print(f"Dropping incomplete session(s) (data end before the close): {dropped}")
    return bars[~bars["session"].isin(dropped)]


def load_etf_bars(data_dir: str | Path) -> pd.DataFrame | None:
    """Minute bars of the context ETFs stored in <data_dir>/etf (None if absent)."""
    etf_dir = Path(data_dir) / "etf"
    files = {path.stem.upper(): path for path in etf_dir.glob("*.parquet")} if etf_dir.exists() else {}
    if not files:
        return None
    missing = sorted(set(ETF_TICKERS).difference(files))
    if missing:
        raise FileNotFoundError(f"Context ETFs missing from {etf_dir}: {missing}. Run download_data.py.")
    frames = []
    for ticker in ETF_TICKERS:
        frame = pd.read_parquet(files[ticker], columns=["timestamp", "close"])
        frame["symbol"] = ticker
        frames.append(frame)
    etf = pd.concat(frames, ignore_index=True)
    etf["timestamp"] = pd.to_datetime(etf["timestamp"], utc=True)
    etf = etf[etf["close"] > 0]
    return etf.loc[_is_regular_session(etf["timestamp"])].reset_index(drop=True)


def _historical_same_minute_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    """Mean of ``column`` at the same session minute over the previous sessions (causal)."""
    keys = [frame["symbol"], frame["minute_of_session"]]
    shifted = frame.groupby(keys, sort=False)[column].shift(1)
    rolled = (
        shifted.groupby(keys, sort=False)
        .rolling(VOLUME_SURPRISE_LOOKBACK_SESSIONS, min_periods=VOLUME_SURPRISE_MIN_SESSIONS)
        .mean()
    )
    return rolled.reset_index(level=[0, 1], drop=True).reindex(frame.index)


def _etf_context_features(frame: pd.DataFrame, etf_bars: pd.DataFrame | None) -> pd.DataFrame:
    """SPY and own-sector ETF log returns over 5/15/30 minutes, on the stock grid."""
    columns = [f"market_return_{h}m" for h in (5, 15, 30)] + [
        f"sector_etf_relative_return_{h}m" for h in (5, 15, 30)
    ]
    if etf_bars is None:
        return pd.DataFrame(0.0, index=frame.index, columns=columns)
    grid = frame[["session", "timestamp"]].drop_duplicates().sort_values(["session", "timestamp"])
    closes = etf_bars.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
    closes = closes.reindex(grid["timestamp"].to_numpy())
    closes.index = pd.MultiIndex.from_arrays([grid["session"].to_numpy(), grid["timestamp"].to_numpy()], names=["session", "timestamp"])
    log_close = np.log(closes).groupby(level="session").ffill()
    returns = {h: log_close.groupby(level="session").diff(h) for h in (5, 15, 30)}
    keyed = frame.set_index(["session", "timestamp"]).index
    sector_etf = frame["sector"].map(SECTOR_ETF)
    out = pd.DataFrame(index=frame.index)
    for h in (5, 15, 30):
        market = returns[h][MARKET_ETF].reindex(keyed).to_numpy()
        out[f"market_return_{h}m"] = market
        sector_values = np.full(len(frame), np.nan)
        for etf in sector_etf.dropna().unique():
            members = (sector_etf == etf).to_numpy()
            sector_values[members] = returns[h][etf].reindex(keyed[members]).to_numpy()
        out[f"sector_etf_relative_return_{h}m"] = sector_values - market
    return out


def engineer_minute_features(
    bars: pd.DataFrame,
    holding_minutes: int,
    sector_median_shrinkage_count: float = SECTOR_MEDIAN_SHRINKAGE_COUNT,
    etf_bars: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create causal OHLCV features and labels valid only at real trade bars."""
    prepared = bars.copy()
    timestamp_ny = prepared["timestamp"].dt.tz_convert("America/New_York")
    prepared["session"] = timestamp_ny.dt.date.astype(str)
    prepared = drop_incomplete_sessions(prepared)
    frame = _expand_to_session_grid(prepared)
    sector_spec = sector_relative_return_feature_spec(
        frame["symbol"].unique(), sector_median_shrinkage_count
    )
    frame["sector"] = frame["symbol"].map(sector_spec["sector_labels"])
    timestamp_ny = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["minute_of_session"] = timestamp_ny.dt.hour * 60 + timestamp_ny.dt.minute - 570
    group_keys = ["symbol", "session"]
    grouped = frame.groupby(group_keys, group_keys=False, sort=False)

    frame["log_close"] = np.log(frame["close"])
    frame["log_return_1m"] = grouped["log_close"].diff()
    previous_time = grouped["timestamp"].shift(1)
    frame.loc[(frame["timestamp"] - previous_time) != pd.Timedelta(minutes=1), "log_return_1m"] = np.nan
    frame["t_dim"] = frame["minute_of_session"] / 390.0
    # Price is represented as a causal, sector-relative path.  At each
    # minute, an observed stock return is compared with a sector median shrunk
    # toward the universe median.  This prevents the technology-heavy universe
    # from defining the common return by sheer ticker count while stabilising
    # sparse sectors.  Only real bars enter either benchmark; a non-trading
    # symbol receives zero relative increment and retains explicit state inputs.
    observed_return = frame["log_return_1m"].where(frame["has_trade"].eq(1.0))
    universe_median_return = observed_return.groupby(
        [frame["session"], frame["timestamp"]], sort=False
    ).transform("median").fillna(0.0)
    sector_group_keys = [frame["session"], frame["timestamp"], frame["sector"]]
    sector_median_return = observed_return.groupby(sector_group_keys, sort=False).transform("median")
    sector_real_bar_count = observed_return.groupby(sector_group_keys, sort=False).transform("count").astype(np.float64)
    sector_weight = sector_real_bar_count / (sector_real_bar_count + sector_median_shrinkage_count)
    sector_median_return = sector_median_return.fillna(universe_median_return)
    shrunken_sector_benchmark_return = (
        sector_weight * sector_median_return + (1.0 - sector_weight) * universe_median_return
    )
    frame["relative_return_1m"] = np.where(
        frame["has_trade"].eq(1.0),
        frame["log_return_1m"].fillna(0.0) - shrunken_sector_benchmark_return,
        0.0,
    )
    frame["relative_cum_ret"] = frame.groupby(group_keys, sort=False)["relative_return_1m"].cumsum()
    frame["relative_return_5m"] = frame.groupby(group_keys, sort=False)["relative_cum_ret"].diff(5)
    frame["relative_return_15m"] = frame.groupby(group_keys, sort=False)["relative_cum_ret"].diff(15)
    frame["momentum_acceleration_5v15"] = (
        frame["relative_return_5m"] / 5.0 - frame["relative_return_15m"] / 15.0
    )
    frame["log_volume"] = np.log1p(frame["volume"])
    frame["range_rel"] = ((frame["high"] - frame["low"]) / frame["close"]).clip(lower=0.0)
    frame["realized_vol"] = grouped["log_return_1m"].transform(
        lambda values: values.rolling(5, min_periods=5).std(ddof=0)
    )
    frame["vwap_deviation"] = frame["close"] / frame["vwap"] - 1.0
    bar_range = (frame["high"] - frame["low"]).clip(lower=0.0)
    raw_clv = (2.0 * frame["close"] - frame["high"] - frame["low"]) / (bar_range + CLV_DENOMINATOR_EPS)
    frame["clv"] = np.where(
        frame["has_trade"].eq(1.0) & bar_range.gt(CLV_DENOMINATOR_EPS), raw_clv, 0.0
    )
    frame["clv"] = frame["clv"].clip(lower=-1.0, upper=1.0)

    # No same-bar fill assumption: information through close(t), enter at the
    # *raw* open(t+1), and mark the position at the raw open of the next
    # rebalance, t+holding+1. Consecutive non-overlapping signals therefore
    # join exactly, so an unchanged target is genuinely continuous rather than
    # silently omitting the close-to-next-open interval. The dense feature grid
    # never creates an executable price where the source had no bar.
    frame["entry_price"] = grouped["raw_open"].shift(-1)
    frame["exit_price"] = grouped["raw_open"].shift(-(holding_minutes + 1))
    next_time = grouped["timestamp"].shift(-1)
    exit_time = grouped["timestamp"].shift(-(holding_minutes + 1))
    valid_execution = (
        (next_time - frame["timestamp"] == pd.Timedelta(minutes=1))
        & (exit_time - frame["timestamp"] == pd.Timedelta(minutes=holding_minutes + 1))
        & frame["entry_price"].gt(0)
        & frame["exit_price"].gt(0)
    )
    frame["target_return"] = np.where(
        valid_execution,
        np.log(frame["exit_price"] / frame["entry_price"]),
        np.nan,
    )
    # The rank model is deliberately trained on forward industry-neutral alpha,
    # while target_return remains the raw executable outcome for PnL.  A sector
    # with fewer than two valid entry/exit labels cannot identify an intra-sector
    # return, so it is excluded from the rank label at that signal time.
    forward_sector_keys = [frame["session"], frame["timestamp"], frame["sector"]]
    forward_sector_mean = frame["target_return"].groupby(forward_sector_keys, sort=False).transform("mean")
    forward_sector_count = frame["target_return"].groupby(forward_sector_keys, sort=False).transform("count")
    frame["rank_target_return"] = (frame["target_return"] - forward_sector_mean).where(
        forward_sector_count.ge(2)
    )
    frame = _add_v2_context_features(frame, etf_bars)
    frame["input_available"] = frame[list(FEATURE_COLUMNS)].notna().all(axis=1)
    if frame["symbol"].nunique() > FLOAT32_FEATURE_SYMBOL_THRESHOLD:
        # Model inputs are float32 anyway; labels and prices stay float64.
        frame[list(FEATURE_COLUMNS)] = frame[list(FEATURE_COLUMNS)].astype(np.float32)
    if not frame["input_available"].any():
        raise ValueError("No usable feature paths after feature construction. Check the raw minute bars.")
    return frame.sort_values(["session", "timestamp", "symbol"]).reset_index(drop=True)


def split_by_session(frame: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on complete trading sessions, never on intraday timestamps."""
    sessions = np.array(sorted(frame["session"].unique()))
    if len(sessions) < 15:
        raise ValueError(f"Need at least 15 complete sessions; only found {len(sessions)}.")
    train_end = int(len(sessions) * train_ratio)
    val_end = int(len(sessions) * (train_ratio + val_ratio))
    if train_end < 5 or val_end <= train_end or val_end >= len(sessions):
        raise ValueError("Invalid day split. Adjust the split ratios or download more data.")
    train = frame[frame["session"].isin(sessions[:train_end])].copy()
    validation = frame[frame["session"].isin(sessions[train_end:val_end])].copy()
    test = frame[frame["session"].isin(sessions[val_end:])].copy()
    print(
        "Session split: "
        f"train={sessions[0]}..{sessions[train_end - 1]} ({train['session'].nunique()} days), "
        f"val={sessions[train_end]}..{sessions[val_end - 1]} ({validation['session'].nunique()} days), "
        f"test={sessions[val_end]}..{sessions[-1]} ({test['session'].nunique()} days)"
    )
    return train, validation, test


def fit_feature_scaler(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    scaler: dict[str, dict[str, float]] = {}
    for column in FEATURE_COLUMNS:
        median = float(train[column].median())
        scale = float(train[column].std(ddof=0))
        scaler[column] = {"median": median, "scale": scale if scale > 1e-12 else 1.0}
    return scaler


def apply_feature_scaler(frame: pd.DataFrame, scaler: dict[str, dict[str, float]]) -> pd.DataFrame:
    scaled = frame.copy()
    for column, values in scaler.items():
        scaled[column] = (scaled[column].fillna(values["median"]) - values["median"]) / values["scale"]
    return scaled


def cross_ticker_area_feature_spec(symbols: Iterable[str]) -> dict[str, object]:
    """Describe the explicit, selected level-two cross-ticker area features."""
    ordered = list(symbols)
    price_price = [
        f"area(relative_price:{left}, relative_price:{right})"
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1 :]
    ]
    price_volume = [
        f"area(relative_price:{price_symbol}, relative_log_volume:{volume_symbol})"
        for price_symbol in ordered
        for volume_symbol in ordered
        if price_symbol != volume_symbol
    ]
    return {
        "relative_price_path": (
            "Cumulative one-minute log return minus a contemporaneous sector median "
            "shrunk toward the universe median, using real bars only."
        ),
        "relative_log_volume_path": (
            "Scaled log volume minus the contemporaneous cross-sectional median, then "
            "rebased at the beginning of each feature path."
        ),
        "area_definition": "0.5 * sum_t (x_t * y_(t+1) - y_t * x_(t+1))",
        "price_price_count": len(price_price),
        "price_volume_count": len(price_volume),
        "total_cross_ticker_area_count": len(price_price) + len(price_volume),
        "price_price_areas": price_price,
        "price_volume_areas": price_volume,
    }


def selective_signature_feature_spec() -> dict[str, object]:
    """Document the non-redundant individual-path representation."""
    return {
        "time_coordinate": "t_dim (single, not lead-lag duplicated)",
        "lead_lag_continuous_channels": list(SIGNATURE_CONTINUOUS_COLUMNS),
        "lead_lag_path_channels": SIGNATURE_PATH_CHANNELS,
        "compact_signature_dimension": SELECTIVE_SIGNATURE_DIM,
        "signature_coordinates": "level one plus independent antisymmetric level-two areas",
        "state_context": {
            "absolute_time_end": 1,
            "has_trade": ["mean", "last", "transition_rate"],
            "staleness_minutes": ["mean", "max", "last"],
        },
        "state_context_dimension": STATE_CONTEXT_DIM,
        "directional_multiscale_context": {
            "aggregation": "last feature-window value",
            "channels": list(DIRECTIONAL_MULTISCALE_COLUMNS),
            "definition": "5m and 15m sector-relative returns plus their per-minute velocity difference",
        },
        "directional_multiscale_context_dimension": DIRECTIONAL_MULTISCALE_CONTEXT_DIM,
        "excluded_from_individual_signature": ["has_trade", "staleness_minutes"],
        "initial_state_encoder": {
            "initial_anchor_minutes": INITIAL_STATE_ANCHOR_MINUTES,
            "initial_anchor_channels": list(INITIAL_STATE_ANCHOR_COLUMNS),
            "whole_path_direction_free_statistics": list(INITIAL_STATE_PATH_STATISTICS),
            "excluded_directional_channels": [
                "relative_cum_ret",
                *DIRECTIONAL_MULTISCALE_COLUMNS,
                "vwap_deviation",
                "clv",
            ],
            "input_dimension": INITIAL_STATE_CONTEXT_DIM,
        },
    }


def _cross_ticker_areas_numpy(path: np.ndarray) -> np.ndarray:
    """Return selected areas for paths shaped [batch, symbols, time, features]."""
    if path.ndim != 4:
        raise ValueError("Expected cross-ticker paths with shape [batch, symbols, time, features].")
    _, num_symbols, _, channels = path.shape
    if channels != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} feature channels, received {channels}.")
    relative_price = np.swapaxes(path[..., RELATIVE_PRICE_FEATURE_INDEX], 1, 2)
    relative_price = relative_price - relative_price[:, :1, :]
    relative_volume = np.swapaxes(path[..., VOLUME_FEATURE_INDEX], 1, 2)
    relative_volume = relative_volume - np.median(relative_volume, axis=2, keepdims=True)
    relative_volume = relative_volume - relative_volume[:, :1, :]

    price_previous, price_next = relative_price[:, :-1, :], relative_price[:, 1:, :]
    volume_previous, volume_next = relative_volume[:, :-1, :], relative_volume[:, 1:, :]
    price_price_matrix = 0.5 * (
        np.matmul(np.swapaxes(price_previous, 1, 2), price_next)
        - np.matmul(np.swapaxes(price_next, 1, 2), price_previous)
    )
    price_volume_matrix = 0.5 * (
        np.matmul(np.swapaxes(price_previous, 1, 2), volume_next)
        - np.matmul(np.swapaxes(price_next, 1, 2), volume_previous)
    )
    price_i, price_j = np.triu_indices(num_symbols, k=1)
    volume_i, volume_j = np.where(~np.eye(num_symbols, dtype=bool))
    return np.concatenate(
        [price_price_matrix[:, price_i, price_j], price_volume_matrix[:, volume_i, volume_j]], axis=1
    ).astype(np.float64, copy=False)


class CrossSectionalMinuteDataset(Dataset):
    """Dense cross-sectional paths with a per-symbol execution-label mask."""

    def __init__(
        self,
        frame: pd.DataFrame,
        window_minutes: int,
        min_labelled_symbols: int,
        sample_stride_minutes: int,
    ) -> None:
        self.window_minutes = window_minutes
        self.min_labelled_symbols = min_labelled_symbols
        self.sample_stride_minutes = sample_stride_minutes
        if sample_stride_minutes < 1:
            raise ValueError("sample_stride_minutes must be positive.")
        self.symbols = sorted(frame["symbol"].unique())
        x_parts: list[np.ndarray] = []
        raw_target_parts: list[np.ndarray] = []
        rank_target_parts: list[np.ndarray] = []
        target_mask_parts: list[np.ndarray] = []
        entry_parts: list[np.ndarray] = []
        exit_parts: list[np.ndarray] = []
        timestamp_parts: list[np.ndarray] = []
        session_parts: list[np.ndarray] = []
        sample_end_indices: list[int] = []
        offset = 0

        for session, session_frame in frame.groupby("session", sort=True):
            timestamps = sorted(session_frame["timestamp"].unique())
            if len(timestamps) < window_minutes:
                continue
            index = pd.Index(timestamps, name="timestamp")
            minute_numbers = (
                session_frame.drop_duplicates("timestamp")
                .set_index("timestamp")
                .reindex(index)["minute_of_session"]
                .to_numpy(dtype=np.int64)
            )
            x = np.empty((len(index), len(self.symbols), len(FEATURE_COLUMNS)), dtype=np.float32)
            raw_target = np.zeros((len(index), len(self.symbols)), dtype=np.float32)
            rank_target = np.zeros((len(index), len(self.symbols)), dtype=np.float32)
            target_mask = np.zeros((len(index), len(self.symbols)), dtype=bool)
            entry = np.empty((len(index), len(self.symbols)), dtype=np.float32)
            exit_ = np.empty((len(index), len(self.symbols)), dtype=np.float32)
            input_available = np.ones(len(index), dtype=bool)
            for symbol_index, symbol in enumerate(self.symbols):
                symbol_frame = session_frame[session_frame["symbol"] == symbol].set_index("timestamp").reindex(index)
                if symbol_frame["symbol"].isna().any():
                    raise ValueError(f"{session}: {symbol} is absent from the constructed minute grid.")
                x[:, symbol_index, :] = symbol_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
                input_available &= symbol_frame["input_available"].to_numpy(dtype=bool)
                symbol_raw_target = symbol_frame["target_return"].to_numpy(dtype=np.float32)
                symbol_rank_target = symbol_frame["rank_target_return"].to_numpy(dtype=np.float32)
                valid_target = np.isfinite(symbol_raw_target) & np.isfinite(symbol_rank_target)
                raw_target[np.isfinite(symbol_raw_target), symbol_index] = symbol_raw_target[
                    np.isfinite(symbol_raw_target)
                ]
                rank_target[valid_target, symbol_index] = symbol_rank_target[valid_target]
                target_mask[:, symbol_index] = valid_target
                entry[:, symbol_index] = symbol_frame["entry_price"].to_numpy(dtype=np.float32)
                exit_[:, symbol_index] = symbol_frame["exit_price"].to_numpy(dtype=np.float32)

            x_parts.append(x)
            raw_target_parts.append(raw_target)
            rank_target_parts.append(rank_target)
            target_mask_parts.append(target_mask)
            entry_parts.append(entry)
            exit_parts.append(exit_)
            timestamp_parts.append(index.to_numpy())
            session_parts.append(np.repeat(session, len(index)))
            for local_end in range(window_minutes - 1, len(index)):
                local_start = local_end - window_minutes + 1
                if (
                    input_available[local_start : local_end + 1].all()
                    and target_mask[local_end].sum() >= min_labelled_symbols
                    and minute_numbers[local_end] % sample_stride_minutes == 0
                ):
                    sample_end_indices.append(offset + local_end)
            offset += len(index)

        if not sample_end_indices:
            raise ValueError(
                "No usable paths: inspect missing-bar diagnostics or lower window_minutes. "
                "The model now permits partial cross-sectional labels, but needs a fully observed input path."
            )
        self.x = np.concatenate(x_parts, axis=0)
        self.raw_targets = np.concatenate(raw_target_parts, axis=0)
        self.rank_targets = np.concatenate(rank_target_parts, axis=0)
        self.target_masks = np.concatenate(target_mask_parts, axis=0)
        self.entry_prices = np.concatenate(entry_parts, axis=0)
        self.exit_prices = np.concatenate(exit_parts, axis=0)
        self.timestamps = np.concatenate(timestamp_parts, axis=0)
        self.sessions = np.concatenate(session_parts, axis=0)
        self.sample_end_indices = np.array(sample_end_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sample_end_indices)

    def subset(self, max_windows: int) -> "CrossSectionalMinuteDataset":
        """Shallow copy restricted to evenly spaced windows (shares the arrays)."""
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        if len(self.sample_end_indices) > max_windows:
            positions = np.unique(np.linspace(0, len(self.sample_end_indices) - 1, max_windows).astype(np.int64))
            clone.sample_end_indices = self.sample_end_indices[positions]
        return clone

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        end = self.sample_end_indices[index]
        path = self.x[end - self.window_minutes + 1 : end + 1].copy()
        # Relative price is rebased within the feature window.  Absolute scaled
        # exchange-clock time is retained for the selective state context.
        path[:, :, RELATIVE_PRICE_FEATURE_INDEX] -= path[0:1, :, RELATIVE_PRICE_FEATURE_INDEX]
        path = np.transpose(path, (1, 0, 2))
        return (
            torch.from_numpy(path),
            torch.from_numpy(self.rank_targets[end]),
            torch.from_numpy(self.target_masks[end]),
            index,
        )

    def metadata(self, sample_indices: Iterable[int]) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for sample_index in sample_indices:
            end = self.sample_end_indices[int(sample_index)]
            for symbol_index, symbol in enumerate(self.symbols):
                if not self.target_masks[end, symbol_index]:
                    continue
                records.append(
                    {
                        "session": self.sessions[end],
                        "timestamp": self.timestamps[end],
                        "symbol": symbol,
                        "sector": SECTOR_BY_SYMBOL[symbol],
                        "target_return": float(self.raw_targets[end, symbol_index]),
                        "rank_target_return": float(self.rank_targets[end, symbol_index]),
                        "entry_price": float(self.entry_prices[end, symbol_index]),
                        "exit_price": float(self.exit_prices[end, symbol_index]),
                    }
                )
        return records

    def prediction_records(
        self,
        sample_indices: Iterable[int],
        prediction: np.ndarray,
        extra_columns: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, object]]:
        """Attach predictions without assuming every symbol has a valid label."""
        records: list[dict[str, object]] = []
        for batch_index, sample_index in enumerate(sample_indices):
            end = self.sample_end_indices[int(sample_index)]
            for symbol_index, symbol in enumerate(self.symbols):
                if not self.target_masks[end, symbol_index]:
                    continue
                record = {
                    "session": self.sessions[end],
                    "timestamp": self.timestamps[end],
                    "symbol": symbol,
                    "sector": SECTOR_BY_SYMBOL[symbol],
                    "target_return": float(self.raw_targets[end, symbol_index]),
                    "rank_target_return": float(self.rank_targets[end, symbol_index]),
                    "entry_price": float(self.entry_prices[end, symbol_index]),
                    "exit_price": float(self.exit_prices[end, symbol_index]),
                    "prediction": float(prediction[batch_index, symbol_index]),
                }
                for column, values in (extra_columns or {}).items():
                    record[column] = float(values[batch_index, symbol_index])
                records.append(record)
        return records

    def cross_ticker_area_matrix(
        self, batch_size: int = 256, sample_positions: np.ndarray | None = None
    ) -> np.ndarray:
        """Materialise raw selected area vectors for PCA, using this split only.

        ``sample_positions`` selects a subset of this split's windows (used to bound
        memory for large universes).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        offsets = np.arange(-self.window_minutes + 1, 1, dtype=np.int64)
        end_indices = (
            self.sample_end_indices if sample_positions is None else self.sample_end_indices[sample_positions]
        )
        batches: list[np.ndarray] = []
        for start in range(0, len(end_indices), batch_size):
            ends = end_indices[start : start + batch_size]
            paths = self.x[ends[:, None] + offsets[None, :]]
            # self.x is [time, symbols, features]; areas expect [batch, symbols, time, features].
            batches.append(_cross_ticker_areas_numpy(np.transpose(paths, (0, 2, 1, 3))))
        return np.concatenate(batches, axis=0)


@dataclass(frozen=True)
class CrossTickerAreaPCA:
    price_price_mean: np.ndarray
    price_price_scale: np.ndarray
    price_price_components: np.ndarray
    price_price_explained_variance_ratio: np.ndarray
    price_volume_mean: np.ndarray
    price_volume_scale: np.ndarray
    price_volume_components: np.ndarray
    price_volume_explained_variance_ratio: np.ndarray
    fit_window_count: int = 0
    fit_window_subsampled: bool = False


def _fit_standardized_pca(
    values: np.ndarray, components: int, feature_group: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardisation and PCA on one training-only feature group."""
    maximum_components = min(values.shape)
    if components < 1 or components > maximum_components:
        raise ValueError(f"{feature_group} PCA components must be in [1, {maximum_components}].")
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardised = ((values - mean) / scale).astype(values.dtype, copy=False)
    _left, singular_values, right_vectors = np.linalg.svd(standardised, full_matrices=False)
    explained_variance = np.square(singular_values) / max(len(standardised) - 1, 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = (
        explained_variance / total_variance if total_variance > 0.0 else np.zeros_like(explained_variance)
    )
    return (
        mean.astype(np.float32),
        scale.astype(np.float32),
        right_vectors[:components].astype(np.float32),
        explained_variance_ratio[:components].astype(np.float32),
    )


def fit_cross_ticker_area_pca(
    dataset: CrossSectionalMinuteDataset,
    price_price_components: int,
    price_volume_components: int,
) -> CrossTickerAreaPCA:
    """PCA-compress both cross-ticker area families using training windows only.

    The raw area matrix has N(N-1)/2 + N(N-1) columns, i.e. ~15k for 100 names.  When
    windows x columns exceeds ``PCA_MAX_MATRIX_ELEMENTS`` the PCA is fitted on evenly
    spaced training windows (deterministic) in float32; smaller universes keep the
    exact all-window float64 fit used before.
    """
    num_symbols = len(dataset.symbols)
    area_columns = num_symbols * (num_symbols - 1) // 2 + num_symbols * (num_symbols - 1)
    window_count = len(dataset)
    sample_positions = None
    if window_count * area_columns > PCA_MAX_MATRIX_ELEMENTS:
        keep = max(PCA_MIN_FIT_WINDOWS, int(PCA_MAX_MATRIX_ELEMENTS // area_columns))
        sample_positions = np.unique(np.linspace(0, window_count - 1, min(keep, window_count)).astype(np.int64))
        print(
            f"Cross-ticker PCA: fitting on {len(sample_positions):,} of {window_count:,} training windows "
            f"({area_columns:,} raw area columns) to bound memory."
        )
    raw_areas = dataset.cross_ticker_area_matrix(sample_positions=sample_positions)
    if sample_positions is not None:
        raw_areas = raw_areas.astype(np.float32, copy=False)
    if not np.isfinite(raw_areas).all():
        raise ValueError("Training cross-ticker areas contain non-finite values.")
    price_price_dim = num_symbols * (num_symbols - 1) // 2
    price_price_areas = raw_areas[:, :price_price_dim]
    price_volume_areas = raw_areas[:, price_price_dim:]
    (
        price_price_mean,
        price_price_scale,
        price_price_projection,
        price_price_explained_variance_ratio,
    ) = _fit_standardized_pca(price_price_areas, price_price_components, "price-price area")
    (
        price_volume_mean,
        price_volume_scale,
        price_volume_projection,
        price_volume_explained_variance_ratio,
    ) = _fit_standardized_pca(price_volume_areas, price_volume_components, "price-volume area")
    return CrossTickerAreaPCA(
        price_price_mean=price_price_mean,
        price_price_scale=price_price_scale,
        price_price_components=price_price_projection,
        price_price_explained_variance_ratio=price_price_explained_variance_ratio,
        price_volume_mean=price_volume_mean,
        price_volume_scale=price_volume_scale,
        price_volume_components=price_volume_projection,
        price_volume_explained_variance_ratio=price_volume_explained_variance_ratio,
        fit_window_count=int(len(raw_areas)),
        fit_window_subsampled=sample_positions is not None,
    )


class SignatureConditionedAffineMeanModel(nn.Module):
    """Selective compact signatures, state summaries, and cross-ticker areas.

    ``t_dim`` is a single clock coordinate and only continuous market variables
    are lead-lag expanded. The individual signature retains level one and only
    antisymmetric level-two coordinates. At level two this is information-
    equivalent to the full geometric signature, because symmetric coordinates are
    determined by level one. ``has_trade`` and ``staleness_minutes`` are instead
    summarised as a small state context.
    Price-price and price-volume areas are each standardised and projected through
    independent PCAs fitted on training windows only. Both compressed contexts are
    shared across all tickers in a sample.

    For a fixed combined context ``c`` the latent mean follows

        dm / dt = A(c) m + b(c),

    and the alpha readout is linear in ``m(1)``.  The augmented matrix exponential
    computes that endpoint exactly, so the Rank-IC objective never receives
    Brownian Monte Carlo noise.

    With ``diffusion_rank > 0`` the latent state is a linear (Ornstein-Uhlenbeck
    type) SDE with context-dependent coefficients,

        dZ_t = (A(c) Z_t + b(c)) dt + G(c) dW_t,   Z_0 = m_0,   G(c) in R^{d x r},

    so Z_1 is Gaussian.  Its mean is the ODE endpoint above and its covariance

        P(c) = int_0^1 exp(A s) G G^T exp(A^T s) ds

    is evaluated exactly with Van Loan's block matrix exponential.  The forecast
    y = w^T Z_1 + q + eps, eps ~ N(0, sigma_obs^2), therefore has the closed-form
    predictive law N(w^T m_1 + q, w^T P w + sigma_obs^2).  ``sigma_obs`` is a single
    global parameter, so every context-dependent (heteroskedastic) part of the
    predictive variance must pass through the latent diffusion.  Training,
    validation and inference remain free of Monte Carlo noise.  ``diffusion_rank=0``
    reproduces the original deterministic model and its checkpoints exactly.

    ``jump_mode`` adds a heavy-tailed component over the holding period:

    * ``"merton"``: y = w^T Z_1 + q + eps + sum_{i<=N} xi_i with
      N ~ Poisson(lambda(c)) and xi_i ~ N(0, s_J(c)^2).  The jump mean is fixed at
      zero (the target is sector-neutral), so jumps change the predictive law but
      not the mean or the ranking.  The likelihood is the exact Poisson mixture
      sum_k Pois(k; lambda) N(y; mu, v + k s_J^2), truncated at ``jump_max_count``.
    * ``"student_t"``: a Student-t law with the Gaussian variance v and a global
      learned degrees of freedom nu > 2 (a continuous normal-variance mixture),
      kept as a low-cost control for the Merton specification.
    """

    def __init__(
        self,
        input_channels: int,
        signature_depth: int,
        latent_dim: int,
        hidden_dim: int,
        num_symbols: int,
        cross_area_pca: CrossTickerAreaPCA,
        diffusion_rank: int = 0,
        diffusion_floor: float = 1e-4,
        jump_mode: str = "none",
        jump_initial_intensity: float = 0.05,
        jump_initial_std: float = 2.0,
        stock_embedding_dim: int = 0,
        cross_attention_layers: int = 0,
        attention_dim: int = 64,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        context_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= context_dropout < 1.0:
            raise ValueError("context_dropout must lie in [0, 1).")
        if stock_embedding_dim < 0 or cross_attention_layers < 0:
            raise ValueError("stock_embedding_dim and cross_attention_layers must be non-negative.")
        if cross_attention_layers > 0 and attention_dim % attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads.")
        if jump_mode not in JUMP_MODES:
            raise ValueError(f"jump_mode must be one of {JUMP_MODES}.")
        if jump_mode != "none" and diffusion_rank <= 0:
            raise ValueError("A jump or Student-t component requires diffusion_rank > 0.")
        if not jump_initial_intensity > 0.0 or not jump_initial_std > 0.0:
            raise ValueError("Initial jump intensity and jump standard deviation must be positive.")
        self.jump_mode = jump_mode
        if diffusion_rank < 0:
            raise ValueError("diffusion_rank must be non-negative.")
        if diffusion_floor < 0.0:
            raise ValueError("diffusion_floor must be non-negative.")
        self.diffusion_rank = int(diffusion_rank)
        self.diffusion_floor = float(diffusion_floor)
        if num_symbols < 3:
            raise ValueError("Cross-ticker areas require at least three symbols.")
        if signature_depth != 2:
            raise ValueError("The compact signature representation currently requires signature_depth=2.")
        self.latent_dim = latent_dim
        self.signature_depth = signature_depth
        self.num_symbols = num_symbols
        if input_channels != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected {len(FEATURE_COLUMNS)} input channels, received {input_channels}.")
        self.signature_path_channels = SIGNATURE_PATH_CHANNELS
        self.signature_dim = SELECTIVE_SIGNATURE_DIM
        self.state_context_dim = STATE_CONTEXT_DIM
        self.directional_multiscale_context_dim = DIRECTIONAL_MULTISCALE_CONTEXT_DIM
        self.price_price_area_dim = num_symbols * (num_symbols - 1) // 2
        self.price_volume_area_dim = num_symbols * (num_symbols - 1)
        self.cross_area_dim = self.price_price_area_dim + self.price_volume_area_dim
        if (
            cross_area_pca.price_price_mean.shape != (self.price_price_area_dim,)
            or cross_area_pca.price_price_scale.shape != (self.price_price_area_dim,)
            or cross_area_pca.price_volume_mean.shape != (self.price_volume_area_dim,)
            or cross_area_pca.price_volume_scale.shape != (self.price_volume_area_dim,)
        ):
            raise ValueError("Cross-ticker standardisation statistics do not match the selected raw area dimensions.")
        if (
            cross_area_pca.price_price_components.ndim != 2
            or cross_area_pca.price_price_components.shape[1] != self.price_price_area_dim
        ):
            raise ValueError("Cross-ticker PCA components do not match the price-price area dimension.")
        if (
            cross_area_pca.price_volume_components.ndim != 2
            or cross_area_pca.price_volume_components.shape[1] != self.price_volume_area_dim
        ):
            raise ValueError("Cross-ticker PCA components do not match the price-volume area dimension.")
        self.price_price_pca_components = cross_area_pca.price_price_components.shape[0]
        self.price_volume_pca_components = cross_area_pca.price_volume_components.shape[0]
        self.base_context_dim = (
            self.signature_dim
            + self.state_context_dim
            + self.directional_multiscale_context_dim
            + EXTRA_CONTEXT_DIM
            + self.price_price_pca_components
            + self.price_volume_pca_components
        )
        # Optional per-stock identity and cross-stock interaction.  Both only add to
        # the context that conditions A(c), b(c), G(c) and the jump head.
        self.stock_embedding_dim = int(stock_embedding_dim)
        self.cross_attention_layers = int(cross_attention_layers)
        self.stock_embedding = (
            nn.Embedding(num_symbols, self.stock_embedding_dim) if self.stock_embedding_dim > 0 else None
        )
        if self.stock_embedding is not None:
            nn.init.normal_(self.stock_embedding.weight, std=0.1)
        if self.cross_attention_layers > 0:
            self.attention_input = nn.Sequential(
                nn.LayerNorm(self.base_context_dim + self.stock_embedding_dim),
                nn.Linear(self.base_context_dim + self.stock_embedding_dim, attention_dim),
            )
            self.cross_attention = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=attention_dim,
                    nhead=attention_heads,
                    dim_feedforward=2 * attention_dim,
                    dropout=attention_dropout,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=self.cross_attention_layers,
                enable_nested_tensor=False,
            )
            self.attention_output_dim = attention_dim
        else:
            self.attention_input = None
            self.cross_attention = None
            self.attention_output_dim = 0
        self.context_dim = self.base_context_dim + self.stock_embedding_dim + self.attention_output_dim
        # Dropout on the full conditioning context (training only); regularises the
        # drift, diffusion and jump heads together without changing their layout.
        self.context_dropout = nn.Dropout(context_dropout)
        pair_i, pair_j = torch.triu_indices(num_symbols, num_symbols, offset=1)
        ordered_i, ordered_j = torch.where(~torch.eye(num_symbols, dtype=torch.bool))
        signature_i, signature_j = torch.triu_indices(
            self.signature_path_channels, self.signature_path_channels, offset=1
        )
        self.register_buffer("price_pair_i", pair_i, persistent=False)
        self.register_buffer("price_pair_j", pair_j, persistent=False)
        self.register_buffer("price_volume_i", ordered_i, persistent=False)
        self.register_buffer("price_volume_j", ordered_j, persistent=False)
        self.register_buffer("signature_pair_i", signature_i, persistent=False)
        self.register_buffer("signature_pair_j", signature_j, persistent=False)
        self.register_buffer("price_price_area_mean", torch.from_numpy(cross_area_pca.price_price_mean), persistent=True)
        self.register_buffer("price_price_area_scale", torch.from_numpy(cross_area_pca.price_price_scale), persistent=True)
        self.register_buffer(
            "price_price_area_components", torch.from_numpy(cross_area_pca.price_price_components), persistent=True
        )
        self.register_buffer("price_volume_area_mean", torch.from_numpy(cross_area_pca.price_volume_mean), persistent=True)
        self.register_buffer("price_volume_area_scale", torch.from_numpy(cross_area_pca.price_volume_scale), persistent=True)
        self.register_buffer(
            "price_volume_area_components", torch.from_numpy(cross_area_pca.price_volume_components), persistent=True
        )
        self.initial_state_encoder = nn.Sequential(
            nn.Linear(INITIAL_STATE_CONTEXT_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.coefficient_network = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, hidden_dim),
            nn.SiLU(),
            # raw drift matrix, diagonal decay, and affine drift intercept b(c)
            nn.Linear(hidden_dim, latent_dim * latent_dim + 2 * latent_dim),
        )
        # The readout is global and linear.  Signature information can affect
        # alpha only through the affine latent dynamics, not through a direct
        # signature-to-alpha shortcut.
        self.readout = nn.Linear(latent_dim, 1)
        if self.diffusion_rank > 0:
            # A separate head keeps diffusion_rank=0 state_dicts identical to the
            # original deterministic model.
            self.diffusion_network = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim * self.diffusion_rank),
            )
            # Start with a small diffusion so early training is driven by the rank
            # objective; the NLL term then grows the volatility it can explain.
            nn.init.normal_(self.diffusion_network[-1].weight, std=1e-3)
            nn.init.constant_(self.diffusion_network[-1].bias, 0.1 / math.sqrt(latent_dim * self.diffusion_rank))
            # Global homoskedastic observation noise, log-parameterised.
            self.log_observation_std = nn.Parameter(torch.tensor(0.0))
        if self.jump_mode == "merton":
            # Context-dependent jump intensity lambda(c) and jump variance s_J(c)^2.
            self.jump_network = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),
            )
            nn.init.normal_(self.jump_network[-1].weight, std=1e-3)
            with torch.no_grad():
                self.jump_network[-1].bias.copy_(
                    torch.tensor(
                        [
                            _inverse_softplus(jump_initial_intensity),
                            _inverse_softplus(jump_initial_std**2),
                        ]
                    )
                )
        elif self.jump_mode == "student_t":
            # nu = 2 + exp(theta) keeps the variance finite; start at nu = 8.
            self.log_dof_minus_two = nn.Parameter(torch.tensor(math.log(6.0)))

    @staticmethod
    def _selective_lead_lag(path: torch.Tensor) -> torch.Tensor:
        """Keep time once; lead-lag only continuous market variables."""
        time = path[:, :, TIME_FEATURE_INDEX : TIME_FEATURE_INDEX + 1]
        continuous = path[:, :, SIGNATURE_CONTINUOUS_INDICES]
        lead_time = torch.repeat_interleave(time, repeats=2, dim=1)[:, :-1, :]
        lead_continuous = torch.repeat_interleave(continuous, repeats=2, dim=1)[:, :-1, :]
        lag_continuous = torch.repeat_interleave(continuous, repeats=2, dim=1)[:, 1:, :]
        return torch.cat([lead_time, lead_continuous, lag_continuous], dim=-1)

    def _compact_signature(self, path: torch.Tensor) -> torch.Tensor:
        """Keep level one plus upper-triangular antisymmetric level-two areas."""
        full_signature = signatory.signature(self._selective_lead_lag(path), depth=2)
        level_one = full_signature[:, : self.signature_path_channels]
        level_two = full_signature[:, self.signature_path_channels :].view(
            -1, self.signature_path_channels, self.signature_path_channels
        )
        antisymmetric_area = 0.5 * (level_two - level_two.transpose(-1, -2))
        return torch.cat(
            [level_one, antisymmetric_area[:, self.signature_pair_i, self.signature_pair_j]], dim=1
        )

    @staticmethod
    def _state_context(path: torch.Tensor) -> torch.Tensor:
        """Summarise discrete trading-state channels without signature areas."""
        time_end = path[:, -1, TIME_FEATURE_INDEX : TIME_FEATURE_INDEX + 1]
        has_trade = path[:, :, HAS_TRADE_FEATURE_INDEX]
        staleness = path[:, :, STALENESS_FEATURE_INDEX]
        transition_rate = (has_trade[:, 1:] - has_trade[:, :-1]).abs().gt(1e-6).to(path.dtype).mean(dim=1)
        return torch.stack(
            [
                time_end.squeeze(1),
                has_trade.mean(dim=1),
                has_trade[:, -1],
                transition_rate,
                staleness.mean(dim=1),
                staleness.max(dim=1).values,
                staleness[:, -1],
            ],
            dim=1,
        )

    @staticmethod
    def _extra_context(path: torch.Tensor) -> torch.Tensor:
        """v2 features: endpoint values plus window means of the flow features."""
        return torch.cat(
            [
                path[:, -1, list(EXTRA_CONTEXT_ENDPOINT_INDICES)],
                path[:, :, list(EXTRA_CONTEXT_MEAN_INDICES)].mean(dim=1),
            ],
            dim=1,
        )

    def _augment_context(self, context: torch.Tensor, batch_size: int, num_symbols: int) -> torch.Tensor:
        """Append the stock embedding and the cross-stock attention output."""
        pieces = [context]
        if self.stock_embedding is not None:
            identity = self.stock_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
            pieces.append(identity.reshape(batch_size * num_symbols, -1))
        if self.cross_attention is not None:
            tokens = torch.cat(pieces, dim=1).view(batch_size, num_symbols, -1)
            attended = self.cross_attention(self.attention_input(tokens))
            pieces.append(attended.reshape(batch_size * num_symbols, -1))
        return torch.cat(pieces, dim=1)

    @staticmethod
    def _directional_multiscale_context(path: torch.Tensor) -> torch.Tensor:
        """Expose explicit short/medium relative momentum without redundant signatures."""
        return path[:, -1, DIRECTIONAL_MULTISCALE_INDICES]

    def _initial_state(self, path: torch.Tensor) -> torch.Tensor:
        """Encode an initial anchor and whole-path scale/liquidity, never path direction."""
        if path.shape[1] < INITIAL_STATE_ANCHOR_MINUTES:
            raise ValueError(
                f"Initial-state anchor requires at least {INITIAL_STATE_ANCHOR_MINUTES} path minutes."
            )
        initial_anchor = path[:, :INITIAL_STATE_ANCHOR_MINUTES, INITIAL_STATE_ANCHOR_INDICES].mean(dim=1)
        relative_price = path[:, :, RELATIVE_PRICE_FEATURE_INDEX]
        relative_price_increment = relative_price[:, 1:] - relative_price[:, :-1]
        whole_path_statistics = torch.stack(
            [
                relative_price_increment.square().sum(dim=1),
                path[:, :, VWAP_DEVIATION_FEATURE_INDEX].abs().mean(dim=1),
                path[:, :, VOLUME_FEATURE_INDEX].std(dim=1, unbiased=False),
                path[:, :, RANGE_FEATURE_INDEX].mean(dim=1),
                path[:, :, REALIZED_VOLATILITY_FEATURE_INDEX].mean(dim=1),
                path[:, :, HAS_TRADE_FEATURE_INDEX].mean(dim=1),
                path[:, :, STALENESS_FEATURE_INDEX].max(dim=1).values,
            ],
            dim=1,
        )
        summary = torch.cat([initial_anchor, whole_path_statistics], dim=1)
        return self.initial_state_encoder(summary)

    def _cross_ticker_areas(self, path: torch.Tensor) -> torch.Tensor:
        """Compute selected price-price and price-volume areas for the universe.

        Every path is rebased before the discrete Stratonovich area is calculated.
        Volume is additionally centred at each minute across the universe, which
        removes the common intraday volume profile before cross interactions are
        formed.  No observations after the signal time enter this calculation.
        """
        relative_price = path[..., RELATIVE_PRICE_FEATURE_INDEX].transpose(1, 2)
        relative_price = relative_price - relative_price[:, :1, :]
        relative_volume = path[..., VOLUME_FEATURE_INDEX].transpose(1, 2)
        relative_volume = relative_volume - relative_volume.median(dim=2, keepdim=True).values
        relative_volume = relative_volume - relative_volume[:, :1, :]

        price_previous, price_next = relative_price[:, :-1, :], relative_price[:, 1:, :]
        volume_previous, volume_next = relative_volume[:, :-1, :], relative_volume[:, 1:, :]
        price_price_matrix = 0.5 * (
            torch.bmm(price_previous.transpose(1, 2), price_next)
            - torch.bmm(price_next.transpose(1, 2), price_previous)
        )
        price_volume_matrix = 0.5 * (
            torch.bmm(price_previous.transpose(1, 2), volume_next)
            - torch.bmm(price_next.transpose(1, 2), volume_previous)
        )
        price_price = price_price_matrix[:, self.price_pair_i, self.price_pair_j]
        price_volume = price_volume_matrix[:, self.price_volume_i, self.price_volume_j]
        return torch.cat([price_price, price_volume], dim=1)

    def _project_cross_ticker_areas(self, path: torch.Tensor) -> torch.Tensor:
        raw_areas = self._cross_ticker_areas(path)
        price_price = (raw_areas[:, : self.price_price_area_dim] - self.price_price_area_mean) / self.price_price_area_scale
        price_volume = (
            raw_areas[:, self.price_price_area_dim :] - self.price_volume_area_mean
        ) / self.price_volume_area_scale
        compressed_price_price = price_price @ self.price_price_area_components.transpose(0, 1)
        compressed_price_volume = price_volume @ self.price_volume_area_components.transpose(0, 1)
        return torch.cat([compressed_price_price, compressed_price_volume], dim=1)

    def _coefficients(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the combined-context-conditioned stable drift matrix and intercept."""
        batch_size = context.shape[0]
        latent_dim = self.latent_dim
        values = self.coefficient_network(context)
        matrix_end = latent_dim * latent_dim
        decay_end = matrix_end + latent_dim

        raw_matrix = values[:, :matrix_end].view(batch_size, latent_dim, latent_dim)
        raw_decay = values[:, matrix_end:decay_end]
        drift_intercept = values[:, decay_end:]

        # A(c) = skew(c) - D(c) has a negative-definite symmetric part.  This
        # makes its matrix exponential stable while preserving context-dependent
        # rotations and decay rates in the latent state.
        skew = 0.25 * (raw_matrix - raw_matrix.transpose(-1, -2))
        decay = torch.nn.functional.softplus(raw_decay) + 0.05
        drift_matrix = skew - torch.diag_embed(decay)
        return drift_matrix, drift_intercept

    @property
    def has_diffusion(self) -> bool:
        return self.diffusion_rank > 0

    def _diffusion_matrix(self, context: torch.Tensor) -> torch.Tensor:
        """Context-conditioned low-rank diffusion loading G(c), shape [n, d, r]."""
        return self.diffusion_network(context).view(-1, self.latent_dim, self.diffusion_rank)

    @staticmethod
    def van_loan_covariance(drift_matrix: torch.Tensor, diffusion_covariance: torch.Tensor) -> torch.Tensor:
        """Exact P = int_0^1 exp(A s) Q exp(A^T s) ds via one block matrix exponential.

        With C = [[-A, Q], [0, A^T]] and exp(C) = [[F11, F12], [0, F22]],
        F22 = exp(A^T) and P = F22^T F12 (Van Loan, 1978).
        """
        n, d, _ = drift_matrix.shape
        block = torch.zeros(n, 2 * d, 2 * d, device=drift_matrix.device, dtype=drift_matrix.dtype)
        block[:, :d, :d] = -drift_matrix
        block[:, :d, d:] = diffusion_covariance
        block[:, d:, d:] = drift_matrix.transpose(-1, -2)
        exponential = torch.matrix_exp(block)
        covariance = exponential[:, d:, d:].transpose(-1, -2) @ exponential[:, :d, d:]
        # Remove the O(machine-epsilon) asymmetry left by floating point.
        return 0.5 * (covariance + covariance.transpose(-1, -2))

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """Conditional-mean alpha score, shape [batch, symbols]."""
        mean, _, _ = self._latent_endpoint(path, need_covariance=False)
        return mean

    def forward_predictive(self, path: torch.Tensor) -> dict[str, torch.Tensor]:
        """All predictive-law parameters, each [batch, symbols].

        Keys: ``mean``, ``gaussian_variance`` (diffusion + observation noise),
        ``total_variance``; plus ``jump_intensity`` and ``jump_variance`` (Merton)
        or ``dof`` (Student-t).
        """
        if not self.has_diffusion:
            raise RuntimeError("forward_predictive requires diffusion_rank > 0.")
        mean, gaussian_variance, context = self._latent_endpoint(path, need_covariance=True)
        assert gaussian_variance is not None
        result = {"mean": mean, "gaussian_variance": gaussian_variance, "total_variance": gaussian_variance}
        if self.jump_mode == "merton":
            raw = self.jump_network(context).float()
            intensity = torch.nn.functional.softplus(raw[:, 0]).view_as(mean) + 1e-6
            jump_variance = torch.nn.functional.softplus(raw[:, 1]).view_as(mean) + 1e-4
            result["jump_intensity"] = intensity
            result["jump_variance"] = jump_variance
            result["total_variance"] = gaussian_variance + intensity * jump_variance
        elif self.jump_mode == "student_t":
            dof = 2.0 + torch.exp(self.log_dof_minus_two.float())
            result["dof"] = dof.expand_as(mean)
        return result

    def forward_distribution(self, path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predictive mean and total variance, each [batch, symbols]."""
        parameters = self.forward_predictive(path)
        return parameters["mean"], parameters["total_variance"]

    def _latent_endpoint(
        self, path: torch.Tensor, need_covariance: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if path.ndim != 4:
            raise ValueError("Expected path with shape [batch, symbols, time, features].")
        batch_size, num_symbols, path_length, input_channels = path.shape
        if num_symbols != self.num_symbols:
            raise ValueError(f"Expected {self.num_symbols} symbols, received {num_symbols}.")
        if input_channels != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected {len(FEATURE_COLUMNS)} feature channels, received {input_channels}.")

        individual_path = path.reshape(batch_size * num_symbols, path_length, input_channels)
        signature = self._compact_signature(individual_path)
        state_context = self._state_context(individual_path)
        directional_multiscale_context = self._directional_multiscale_context(individual_path)
        cross_context = self._project_cross_ticker_areas(path)
        context = torch.cat(
            [
                signature,
                state_context,
                directional_multiscale_context,
                self._extra_context(individual_path),
                cross_context.unsqueeze(1)
                .expand(-1, num_symbols, -1)
                .reshape(
                    batch_size * num_symbols,
                    self.price_price_pca_components + self.price_volume_pca_components,
                ),
            ],
            dim=1,
        )
        context = self.context_dropout(self._augment_context(context, batch_size, num_symbols))
        initial_state = self._initial_state(individual_path)
        drift_matrix, drift_intercept = self._coefficients(context)

        # Matrix exponentials are deliberately evaluated in fp32 even when the
        # optional outer AMP context is enabled.
        drift_matrix = drift_matrix.float()
        drift_intercept = drift_intercept.float()
        initial_state = initial_state.float()
        augmented = torch.zeros(
            batch_size * num_symbols,
            self.latent_dim + 1,
            self.latent_dim + 1,
            device=path.device,
            dtype=torch.float32,
        )
        augmented[:, : self.latent_dim, : self.latent_dim] = drift_matrix
        augmented[:, : self.latent_dim, self.latent_dim] = drift_intercept
        transition = torch.matrix_exp(augmented)
        mean_endpoint = (
            torch.bmm(transition[:, : self.latent_dim, : self.latent_dim], initial_state.unsqueeze(-1)).squeeze(-1)
            + transition[:, : self.latent_dim, self.latent_dim]
        )
        mean = self.readout(mean_endpoint).view(batch_size, num_symbols)
        if not need_covariance:
            return mean, None, context

        loading = self._diffusion_matrix(context).float()
        diffusion_covariance = loading @ loading.transpose(-1, -2)
        if self.diffusion_floor > 0.0:
            diffusion_covariance = diffusion_covariance + self.diffusion_floor * torch.eye(
                self.latent_dim, device=path.device, dtype=torch.float32
            )
        latent_covariance = self.van_loan_covariance(drift_matrix, diffusion_covariance)
        readout_weight = self.readout.weight.float().view(1, self.latent_dim, 1)
        diffusion_variance = (readout_weight.transpose(1, 2) @ latent_covariance @ readout_weight).view(
            batch_size, num_symbols
        )
        observation_variance = torch.exp(2.0 * self.log_observation_std.float())
        variance = diffusion_variance.clamp_min(0.0) + observation_variance
        return mean, variance, context


class GaussianPredictiveNLL(nn.Module):
    """Masked Gaussian negative log-likelihood of the standardised sector-excess return.

    The target is divided by a fixed train-split scale so that the model's mean and
    variance are O(1).  Standardised targets are clipped at +/- ``target_clip`` so a
    handful of event-driven jumps cannot dominate the variance head.
    """

    def __init__(self, target_scale: float, target_clip: float) -> None:
        super().__init__()
        if not target_scale > 0.0:
            raise ValueError("target_scale must be positive.")
        if not target_clip > 0.0:
            raise ValueError("target_clip must be positive.")
        self.target_scale = float(target_scale)
        self.target_clip = float(target_clip)

    def forward(
        self, mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = target_mask.to(dtype=mean.dtype)
        standardised = (target / self.target_scale).clamp(-self.target_clip, self.target_clip)
        variance = variance.clamp_min(1e-6)
        nll = 0.5 * (torch.log(2.0 * math.pi * variance) + (standardised - mean).square() / variance)
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)


class PredictiveNLL(nn.Module):
    """Masked NLL of the standardised target under the model's predictive law.

    ``mode="none"``: Gaussian N(mu, v).
    ``mode="merton"``: exact Poisson mixture sum_k Pois(k; lambda) N(mu, v + k s_J^2),
    truncated at ``max_jumps`` (the omitted Poisson mass is reported by the
    diagnostics).
    ``mode="student_t"``: Student-t with variance v and degrees of freedom nu.
    """

    def __init__(self, target_scale: float, target_clip: float, mode: str = "none", max_jumps: int = 8) -> None:
        super().__init__()
        if mode not in JUMP_MODES:
            raise ValueError(f"mode must be one of {JUMP_MODES}.")
        if not target_scale > 0.0 or not target_clip > 0.0:
            raise ValueError("target_scale and target_clip must be positive.")
        if max_jumps < 1:
            raise ValueError("max_jumps must be at least 1.")
        self.target_scale = float(target_scale)
        self.target_clip = float(target_clip)
        self.mode = mode
        self.max_jumps = int(max_jumps)

    def per_label_nll(self, parameters: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        y = (target.float() / self.target_scale).clamp(-self.target_clip, self.target_clip)
        mean = parameters["mean"].float()
        variance = parameters["gaussian_variance"].float().clamp_min(1e-6)
        squared_error = (y - mean).square()
        if self.mode == "none":
            return 0.5 * (torch.log(2.0 * math.pi * variance) + squared_error / variance)
        if self.mode == "merton":
            intensity = parameters["jump_intensity"].float().clamp_min(1e-8)
            jump_variance = parameters["jump_variance"].float()
            counts = torch.arange(self.max_jumps + 1, device=y.device, dtype=y.dtype).view(-1, *([1] * y.ndim))
            component_variance = variance.unsqueeze(0) + counts * jump_variance.unsqueeze(0)
            log_poisson = counts * torch.log(intensity).unsqueeze(0) - intensity.unsqueeze(0) - torch.lgamma(counts + 1.0)
            log_normal = -0.5 * (
                torch.log(2.0 * math.pi * component_variance) + squared_error.unsqueeze(0) / component_variance
            )
            return -torch.logsumexp(log_poisson + log_normal, dim=0)
        dof = parameters["dof"].float()
        scale_squared = variance * (dof - 2.0) / dof
        return -(
            torch.lgamma(0.5 * (dof + 1.0))
            - torch.lgamma(0.5 * dof)
            - 0.5 * torch.log(dof * math.pi * scale_squared)
            - 0.5 * (dof + 1.0) * torch.log1p(squared_error / (dof * scale_squared))
        )

    def forward(
        self, parameters: dict[str, torch.Tensor], target: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = target_mask.to(dtype=torch.float32)
        nll = self.per_label_nll(parameters, target)
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def fit_sde_target_scale(dataset: "CrossSectionalMinuteDataset") -> float:
    """Robust train-only scale of the sector-excess forward return (1.4826 x MAD)."""
    ends = dataset.sample_end_indices
    values = dataset.rank_targets[ends][dataset.target_masks[ends]].astype(np.float64)
    if values.size < 10:
        raise ValueError("Too few training labels to fit the SDE target scale.")
    scale = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    if not scale > 0.0:
        scale = float(values.std())
    if not scale > 0.0:
        raise ValueError("Training sector-excess returns have zero dispersion.")
    return scale


def per_symbol_baseline_variance(dataset: "CrossSectionalMinuteDataset", target_scale: float, target_clip: float) -> dict[str, float]:
    """Train-only, homoskedastic per-symbol variance of the standardised target (benchmark)."""
    ends = dataset.sample_end_indices
    targets = np.clip(dataset.rank_targets[ends] / target_scale, -target_clip, target_clip)
    masks = dataset.target_masks[ends]
    baseline: dict[str, float] = {}
    for symbol_index, symbol in enumerate(dataset.symbols):
        values = targets[masks[:, symbol_index], symbol_index]
        baseline[symbol] = float(np.mean(np.square(values))) if values.size else float("nan")
    return baseline


def _predictive_components(frame: pd.DataFrame) -> list[dict[str, np.ndarray]]:
    """Exact law parameters per ensemble member (or one member for a seed frame)."""
    seed_tags = sorted({column.split("__seed")[1] for column in frame.columns if column.startswith("pred_mu__seed")})
    suffixes = [f"__seed{tag}" for tag in seed_tags] if seed_tags else [""]
    components: list[dict[str, np.ndarray]] = []
    for suffix in suffixes:
        component: dict[str, np.ndarray] = {}
        for column in PREDICTIVE_PARAMETER_COLUMNS.values():
            if f"{column}{suffix}" in frame.columns:
                component[column] = frame[f"{column}{suffix}"].to_numpy(dtype=float)
        if "pred_mu" not in component or "pred_gvar" not in component:
            component = {}
        components.append(component)
    if not components or not components[0]:
        # Older runs: only the first two moments were saved; use a moment-matched Gaussian.
        return [
            {
                "pred_mu": frame["predicted_mean_std"].to_numpy(dtype=float),
                "pred_gvar": frame["predicted_variance_std"].to_numpy(dtype=float),
            }
        ]
    return components


def _component_logpdf_cdf(
    y: np.ndarray, component: dict[str, np.ndarray], jump_mode: str, max_jumps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Log density, CDF and (Merton only) log density of the no-jump term."""
    mu = component["pred_mu"]
    variance = np.maximum(component["pred_gvar"], 1e-12)
    if jump_mode == "merton" and "pred_jump_intensity" in component:
        intensity = np.maximum(component["pred_jump_intensity"], 1e-12)
        jump_variance = component["pred_jump_var"]
        counts = np.arange(max_jumps + 1)[:, None]
        component_variance = variance[None, :] + counts * jump_variance[None, :]
        log_poisson = counts * np.log(intensity)[None, :] - intensity[None, :] - special.gammaln(counts + 1.0)
        log_normal = stats.norm.logpdf(y[None, :], loc=mu[None, :], scale=np.sqrt(component_variance))
        joint = log_poisson + log_normal
        weights = np.exp(log_poisson)
        cdf = (weights * stats.norm.cdf(y[None, :], loc=mu[None, :], scale=np.sqrt(component_variance))).sum(axis=0)
        return special.logsumexp(joint, axis=0), cdf, joint[0]
    if jump_mode == "student_t" and "pred_dof" in component:
        dof = component["pred_dof"]
        scale = np.sqrt(variance * (dof - 2.0) / dof)
        return stats.t.logpdf(y, dof, loc=mu, scale=scale), stats.t.cdf(y, dof, loc=mu, scale=scale), None
    scale = np.sqrt(variance)
    return stats.norm.logpdf(y, loc=mu, scale=scale), stats.norm.cdf(y, loc=mu, scale=scale), None


def evaluate_predictive_distribution(
    frame: pd.DataFrame,
    target_scale: float,
    target_clip: float,
    baseline_variance_by_symbol: dict[str, float] | None = None,
    jump_mode: str = "none",
    max_jumps: int = 8,
) -> dict[str, object]:
    """Calibration, sharpness and jump diagnostics for the predictive law.

    The density of an ensemble frame is the exact equal-weight mixture of its
    seeds' laws (Gaussian, Merton Poisson mixture or Student-t).  Reports NLL
    against a homoskedastic per-symbol Gaussian benchmark, PIT calibration,
    central-interval coverage including the 99% and 99.9% tails, and a
    Mincer-Zarnowitz regression of squared errors on predicted total variance.
    Merton runs also get intensity summaries and posterior jump probabilities.
    """
    required = {"rank_target_return", "predicted_mean_std", "predicted_variance_std", "symbol", "session"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing distribution columns: {sorted(missing)}")
    y = np.clip(frame["rank_target_return"].to_numpy(dtype=float) / target_scale, -target_clip, target_clip)
    mu = frame["predicted_mean_std"].to_numpy(dtype=float)
    var = np.maximum(frame["predicted_variance_std"].to_numpy(dtype=float), 1e-12)
    residual = y - mu
    z = residual / np.sqrt(var)

    components = _predictive_components(frame)
    evaluated = [_component_logpdf_cdf(y, component, jump_mode, max_jumps) for component in components]
    log_density = special.logsumexp(np.stack([item[0] for item in evaluated]), axis=0) - np.log(len(evaluated))
    pit = np.clip(np.mean(np.stack([item[1] for item in evaluated]), axis=0), 0.0, 1.0)
    nll = -log_density

    coverage = {}
    for level in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
        coverage[f"{level * 100:g}%"] = float(np.mean(np.abs(pit - 0.5) <= level / 2.0))
    pit_histogram = np.histogram(pit, bins=10, range=(0.0, 1.0))[0] / len(pit)
    design = np.column_stack([np.ones_like(var), var])
    (mz_intercept, mz_slope), *_ = np.linalg.lstsq(design, residual**2, rcond=None)
    mz_r2 = float(np.corrcoef(var, residual**2)[0, 1] ** 2) if np.std(var) > 0 else float("nan")
    result: dict[str, object] = {
        "observations": int(len(y)),
        "jump_mode": jump_mode,
        "density": "exact equal-weight mixture over seeds" if len(components) > 1 else "single model",
        "target_units": "sector-excess forward return / train-only robust scale, clipped",
        "target_scale": float(target_scale),
        "target_clip": float(target_clip),
        "fraction_targets_at_clip": float(np.mean(np.abs(y) >= target_clip)),
        "mean_nll": float(nll.mean()),
        "standardised_residual_mean": float(z.mean()),
        "standardised_residual_std": float(z.std()),
        "central_interval_coverage": coverage,
        "pit_histogram_10_bins": [float(value) for value in pit_histogram],
        "pit_ks_statistic": float(stats.kstest(pit, "uniform").statistic),
        "mincer_zarnowitz": {"intercept": float(mz_intercept), "slope": float(mz_slope), "r2": mz_r2},
        "spearman_predicted_std_vs_abs_residual": float(
            stats.spearmanr(np.sqrt(var), np.abs(residual)).correlation
        ),
        "spearman_predicted_std_vs_abs_target": float(stats.spearmanr(np.sqrt(var), np.abs(y)).correlation),
        "warning": "PIT/coverage pool dependent intraday observations; treat as descriptive diagnostics.",
    }

    if jump_mode == "merton" and all("pred_jump_intensity" in component for component in components):
        intensity = np.mean(np.stack([component["pred_jump_intensity"] for component in components]), axis=0)
        jump_std = np.sqrt(np.mean(np.stack([component["pred_jump_var"] for component in components]), axis=0))
        no_jump_log = special.logsumexp(np.stack([item[2] for item in evaluated]), axis=0) - np.log(len(evaluated))
        posterior_jump = 1.0 - np.exp(np.minimum(no_jump_log - log_density, 0.0))
        truncated_mass = np.max(
            [float(stats.poisson.sf(max_jumps, component["pred_jump_intensity"]).max()) for component in components]
        )
        jump_share = intensity * jump_std**2 / var
        timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("America/New_York")
        bucket = timestamps.dt.strftime("%H:%M")
        abs_y = np.abs(y)
        result["jumps"] = {
            "mean_intensity_per_period": float(intensity.mean()),
            "intensity_quantiles_5_50_95": [float(value) for value in np.quantile(intensity, [0.05, 0.5, 0.95])],
            "mean_jump_std": float(jump_std.mean()),
            "mean_jump_share_of_total_variance": float(jump_share.mean()),
            "mean_posterior_jump_probability": float(posterior_jump.mean()),
            "posterior_jump_probability_by_abs_target": {
                "|y|<1": float(posterior_jump[abs_y < 1].mean()) if (abs_y < 1).any() else float("nan"),
                "1<=|y|<3": float(posterior_jump[(abs_y >= 1) & (abs_y < 3)].mean())
                if ((abs_y >= 1) & (abs_y < 3)).any()
                else float("nan"),
                "3<=|y|<5": float(posterior_jump[(abs_y >= 3) & (abs_y < 5)].mean())
                if ((abs_y >= 3) & (abs_y < 5)).any()
                else float("nan"),
                "|y|>=5": float(posterior_jump[abs_y >= 5].mean()) if (abs_y >= 5).any() else float("nan"),
            },
            "spearman_intensity_vs_abs_target": float(stats.spearmanr(intensity, abs_y).correlation),
            "spearman_intensity_vs_posterior_jump": float(stats.spearmanr(intensity, posterior_jump).correlation),
            "mean_intensity_by_time_et": {
                key: float(value) for key, value in pd.Series(intensity).groupby(bucket.to_numpy()).mean().items()
            },
            "mean_intensity_by_symbol": {
                key: float(value)
                for key, value in pd.Series(intensity).groupby(frame["symbol"].to_numpy()).mean().sort_values().items()
            },
            "max_truncated_poisson_mass": truncated_mass,
        }
    if jump_mode == "student_t" and all("pred_dof" in component for component in components):
        result["student_t_dof_by_member"] = [float(np.median(component["pred_dof"])) for component in components]

    if baseline_variance_by_symbol is not None:
        baseline_var = np.maximum(frame["symbol"].map(baseline_variance_by_symbol).to_numpy(dtype=float), 1e-12)
        baseline_nll = 0.5 * (np.log(2.0 * np.pi * baseline_var) + y**2 / baseline_var)
        daily = (
            pd.DataFrame({"session": frame["session"].to_numpy(), "gain": baseline_nll - nll})
            .groupby("session")["gain"]
            .mean()
        )
        rng = np.random.default_rng(0)
        values = daily.to_numpy()
        boot = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
        result["baseline"] = "homoskedastic per-symbol train variance, zero mean, Gaussian"
        result["baseline_mean_nll"] = float(baseline_nll.mean())
        result["nll_improvement_vs_baseline"] = {
            "mean_daily": float(values.mean()),
            "ci_95_lower": float(np.quantile(boot, 0.025)),
            "ci_95_upper": float(np.quantile(boot, 0.975)),
            "days": int(len(values)),
        }
    return result


class CrossSectionalRankLoss(nn.Module):
    """Pearson correlation with hard-rank normal-score targets (Spearman surrogate).

    With ``sector_ids`` the prediction is first demeaned within each sector over
    the valid names.  The target is already a within-sector excess return, so a
    sector-level offset in the score cannot predict it; removing it makes the
    loss measure the within-sector ranking that the ensemble and the
    sector-neutral portfolio actually use.
    """

    def __init__(self, min_labelled_symbols: int, sector_ids: torch.Tensor | None = None) -> None:
        super().__init__()
        self.min_labelled_symbols = min_labelled_symbols
        if sector_ids is not None:
            one_hot = torch.nn.functional.one_hot(sector_ids.long()).to(torch.float32)
            self.register_buffer("sector_one_hot", one_hot, persistent=False)
        else:
            self.sector_one_hot = None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        mask = target_mask.to(dtype=prediction.dtype)
        if self.sector_one_hot is not None:
            one_hot = self.sector_one_hot.to(dtype=prediction.dtype)
            sector_sum = (prediction * mask) @ one_hot
            sector_count = (mask @ one_hot).clamp_min(1.0)
            prediction = prediction - (sector_sum / sector_count) @ one_hot.transpose(0, 1)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        prediction = (prediction - (prediction * mask).sum(dim=1, keepdim=True) / count) * mask
        normal_score_target = torch.zeros_like(target)
        for batch_index in range(target.shape[0]):
            valid_indices = torch.nonzero(target_mask[batch_index], as_tuple=False).squeeze(1)
            valid_count = len(valid_indices)
            if valid_count == 0:
                continue
            values = target[batch_index, valid_indices]
            order = torch.argsort(values, stable=True)
            ranks = torch.empty(valid_count, dtype=target.dtype, device=target.device)
            ranks[order] = torch.arange(valid_count, dtype=target.dtype, device=target.device)
            probabilities = (ranks + 0.5) / valid_count
            normal_score_target[batch_index, valid_indices] = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
        normal_score_target = (
            normal_score_target
            - (normal_score_target * mask).sum(dim=1, keepdim=True) / count
        ) * mask
        numerator = (prediction * normal_score_target).sum(dim=1)
        denominator = torch.sqrt(
            (prediction.square().sum(dim=1) * normal_score_target.square().sum(dim=1)).clamp_min(1e-12)
        )
        valid = target_mask.sum(dim=1) >= self.min_labelled_symbols
        if not valid.any():
            raise RuntimeError("A training batch has fewer valid execution labels than configured.")
        return -(numerator[valid] / denominator[valid]).mean()


def _rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) < 3 or np.allclose(prediction, prediction[0]) or np.allclose(target, target[0]):
        return float("nan")
    return float(stats.spearmanr(prediction, target).correlation)


def _sector_demean(values: np.ndarray, sector_ids: np.ndarray) -> np.ndarray:
    """Subtract each sector's mean from a 1-D score vector (valid names only)."""
    result = np.asarray(values, dtype=float).copy()
    for sector in np.unique(sector_ids):
        members = sector_ids == sector
        result[members] -= result[members].mean()
    return result


def _daily_rank_ic_from_prediction_frame(
    frame: pd.DataFrame, target_column: str = "rank_target_return", sector_neutral: bool = True
) -> pd.Series:
    """Equal-weight each session after Rank IC against the requested forward target.

    With ``sector_neutral`` (default) the score is demeaned within each sector at
    each signal time first, matching the within-sector target and the
    sector-neutral portfolio.  For scores that are already standardised within
    sector (the ensemble) this is a no-op.
    """
    if target_column not in frame.columns:
        raise ValueError(f"Prediction frame is missing Rank-IC target column: {target_column}")
    scores = frame["prediction"]
    if sector_neutral and "sector" in frame.columns:
        scores = scores - scores.groupby([frame["session"], frame["timestamp"], frame["sector"]]).transform("mean")
    frame = frame.assign(_ic_score=scores.to_numpy())
    records: list[tuple[str, float]] = []
    for (session, _timestamp), group in frame.groupby(["session", "timestamp"], sort=True):
        value = _rank_ic(group["_ic_score"].to_numpy(), group[target_column].to_numpy())
        if np.isfinite(value):
            records.append((str(session), value))
    if not records:
        return pd.Series(dtype=float, name="daily_rank_ic")
    minute_ic = pd.DataFrame(records, columns=["session", "rank_ic"])
    return minute_ic.groupby("session", sort=True)["rank_ic"].mean().rename("daily_rank_ic")


def block_bootstrap_mean_ic(daily_ic: pd.Series, iterations: int, seed: int) -> dict[str, object]:
    """Day-block bootstrap for the mean IC, preserving intraday dependence."""
    values = daily_ic.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return {
            "days": int(len(values)),
            "mean_daily_rank_ic": float("nan"),
            "iterations": int(iterations),
            "ci_95_lower": float("nan"),
            "ci_95_upper": float("nan"),
            "bootstrap_fraction_mean_le_zero": float("nan"),
            "significant_positive_95": False,
        }
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive.")
    rng = np.random.default_rng(seed)
    sampled_means = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    lower, upper = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "days": int(len(values)),
        "mean_daily_rank_ic": float(values.mean()),
        "iterations": int(iterations),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "bootstrap_fraction_mean_le_zero": float((sampled_means <= 0.0).mean()),
        "significant_positive_95": bool(lower > 0.0),
    }


def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    non_blocking: bool,
    sector_neutral: bool = True,
) -> tuple[float, pd.Series]:
    model.eval()
    records: list[tuple[str, float]] = []
    if not isinstance(loader.dataset, CrossSectionalMinuteDataset):
        raise TypeError("Validation loader must contain CrossSectionalMinuteDataset.")
    dataset = loader.dataset
    symbol_sector = np.array([SECTOR_BY_SYMBOL[symbol] for symbol in dataset.symbols])
    with torch.no_grad():
        for path, target, target_mask, indices in loader:
            path = _to_device(path, device, non_blocking)
            with _autocast_context(device, amp_enabled):
                prediction = model(path)
            prediction = prediction.cpu().numpy()
            target_np = target.numpy()
            mask_np = target_mask.numpy()
            for batch_index, sample_index in enumerate(indices.tolist()):
                valid = mask_np[batch_index]
                scores = prediction[batch_index][valid]
                if sector_neutral:
                    scores = _sector_demean(scores, symbol_sector[valid])
                value = _rank_ic(scores, target_np[batch_index][valid])
                if np.isfinite(value):
                    end = dataset.sample_end_indices[int(sample_index)]
                    records.append((str(dataset.sessions[end]), value))
    if not records:
        return float("nan"), pd.Series(dtype=float, name="daily_rank_ic")
    minute_ic = pd.DataFrame(records, columns=["session", "rank_ic"])
    daily_ic = minute_ic.groupby("session", sort=True)["rank_ic"].mean().rename("daily_rank_ic")
    return float(minute_ic["rank_ic"].mean()), daily_ic


def make_prediction_frame(
    model: nn.Module,
    dataset: CrossSectionalMinuteDataset,
    device: torch.device,
    amp_enabled: bool,
    non_blocking: bool,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=non_blocking)
    records: list[dict[str, object]] = []
    model.eval()
    has_diffusion = bool(getattr(model, "has_diffusion", False))
    with torch.no_grad():
        for path, _target, _target_mask, indices in loader:
            path = _to_device(path, device, non_blocking)
            with _autocast_context(device, amp_enabled):
                if has_diffusion:
                    parameters = model.forward_predictive(path)
                    prediction = parameters["mean"]
                else:
                    parameters, prediction = None, model(path)
            prediction = prediction.float().cpu().numpy()
            extra = None
            if parameters is not None:
                # In the SDE model the score is also the calibrated predictive mean
                # (standardised units).  ``predicted_*`` are the first two moments;
                # ``pred_*`` are the exact law parameters used by the diagnostics.
                extra = {
                    "predicted_mean_std": prediction,
                    "predicted_variance_std": parameters["total_variance"].float().cpu().numpy(),
                }
                for key, column in PREDICTIVE_PARAMETER_COLUMNS.items():
                    if key in parameters:
                        extra[column] = parameters[key].float().cpu().numpy()
            records.extend(dataset.prediction_records(indices.tolist(), prediction, extra))
    return pd.DataFrame.from_records(records).sort_values(["session", "timestamp", "symbol"]).reset_index(drop=True)


def _validation_nll(
    model: nn.Module,
    loader: DataLoader,
    nll_function: "PredictiveNLL",
    device: torch.device,
    amp_enabled: bool,
    non_blocking: bool,
) -> float:
    """Label-weighted mean predictive NLL on a frozen split (SDE model only)."""
    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for path, target, target_mask, _ in loader:
            path = _to_device(path, device, non_blocking)
            target = _to_device(target, device, non_blocking)
            target_mask = _to_device(target_mask, device, non_blocking)
            with _autocast_context(device, amp_enabled):
                parameters = model.forward_predictive(path)
            labels = float(target_mask.sum().item())
            total += float(nll_function(parameters, target, target_mask).item()) * labels
            count += labels
    return total / count if count > 0 else float("nan")


def _scheduled_rebalances(frame: pd.DataFrame, holding_minutes: int) -> pd.DataFrame:
    """Return exchange-clock rebalance keys for intraday OOS evaluation."""
    if holding_minutes < 1:
        raise ValueError("holding_minutes must be positive.")
    required = {"session", "timestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing schedule columns: {sorted(missing)}")
    signal_times = frame[["session", "timestamp"]].drop_duplicates().sort_values(["session", "timestamp"])
    timestamp_ny = pd.to_datetime(signal_times["timestamp"], utc=True).dt.tz_convert("America/New_York")
    signal_times["minute_of_session"] = timestamp_ny.dt.hour * 60 + timestamp_ny.dt.minute - 570
    return signal_times.loc[
        signal_times["minute_of_session"] % holding_minutes == 0,
        ["session", "timestamp"],
    ]


def _fixed_symbol_count_sector_budgets(frame: pd.DataFrame) -> dict[str, float]:
    """Return immutable gross budgets proportional to eligible sector size.

    The denominator is fixed from the whole split's symbol universe, not the
    sectors passing a particular timestamp's signal threshold. An inactive
    sector therefore leaves cash rather than levering every other sector.
    """
    required = {"symbol", "sector"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing sector-budget columns: {sorted(missing)}")
    membership = frame[["symbol", "sector"]].drop_duplicates()
    inconsistent = membership.groupby("symbol", sort=False)["sector"].nunique()
    if (inconsistent > 1).any():
        symbols = ", ".join(inconsistent.index[inconsistent > 1].tolist())
        raise ValueError(f"Each symbol must have one sector label; inconsistent labels: {symbols}")
    counts = membership.groupby("sector", sort=True)["symbol"].nunique()
    eligible_counts = counts[counts.ge(2)]
    total_symbols = int(eligible_counts.sum())
    if total_symbols < 2:
        raise ValueError("At least one multi-symbol sector is required for sector-neutral weights.")
    return {sector: float(count / total_symbols) for sector, count in eligible_counts.items()}


def _sector_neutral_weights(
    predictions: pd.Series,
    sectors: pd.Series,
    threshold: float,
    sector_budgets: dict[str, float],
) -> pd.Series:
    """Use fixed symbol-count sector budgets with zero net per active sector.

    The neural score is ordinal and is free to undergo arbitrary level shifts.
    Weight magnitudes must therefore use within-sector z-scores, never raw
    scores: a sector whose *entire* score vector is negative still has a valid
    high-score long leg.
    """
    if not predictions.index.equals(sectors.index):
        raise ValueError("Predictions and sector labels must share an index.")
    weights = pd.Series(0.0, index=predictions.index)
    active_sleeves: list[tuple[str, pd.Series, pd.Series]] = []
    for sector, sector_scores in predictions.groupby(sectors, sort=True):
        if sector not in sector_budgets:
            continue
        score_std = sector_scores.std(ddof=0)
        if score_std <= 1e-12:
            continue
        z_score = (sector_scores - sector_scores.mean()) / score_std
        positive_z = z_score[z_score >= threshold]
        negative_z = z_score[z_score <= -threshold]
        if not positive_z.empty and not negative_z.empty:
            active_sleeves.append((sector, positive_z, negative_z))
    for sector, positive_z, negative_z in active_sleeves:
        sector_gross = sector_budgets[sector]
        weights.loc[positive_z.index] = 0.5 * sector_gross * positive_z / positive_z.sum()
        weights.loc[negative_z.index] = -0.5 * sector_gross * negative_z.abs() / negative_z.abs().sum()
    return weights


def _smoothed_sector_buffered_target(
    prior_weights: pd.Series,
    desired_weights: pd.Series,
    sector_by_symbol: pd.Series,
    smoothing_fraction: float,
    sector_buffer_turnover: float,
) -> tuple[pd.Series, dict[str, float]]:
    """Move toward a neutral desired target without breaking sleeve neutrality.

    A stock-by-stock buffer can leave one leg of a sector updated while the
    other is stale, creating accidental sector beta. The buffer therefore acts
    on each complete sector sleeve's one-way turnover.
    """
    if not prior_weights.index.equals(desired_weights.index) or not prior_weights.index.equals(sector_by_symbol.index):
        raise ValueError("Prior weights, desired weights, and sectors must share an index.")
    smoothed_weights = prior_weights + smoothing_fraction * (desired_weights - prior_weights)
    target_weights = prior_weights.copy()
    buffered_sector_count = 0
    rebalanced_sector_count = 0
    for _sector, symbol_index in sector_by_symbol.groupby(sector_by_symbol, sort=True).groups.items():
        sector_turnover = float((smoothed_weights.loc[symbol_index] - prior_weights.loc[symbol_index]).abs().sum())
        if sector_turnover <= 1e-15:
            continue
        if sector_turnover < sector_buffer_turnover:
            buffered_sector_count += 1
            continue
        target_weights.loc[symbol_index] = smoothed_weights.loc[symbol_index]
        rebalanced_sector_count += 1
    return target_weights, {
        "smoothed_one_way_turnover": float((smoothed_weights - prior_weights).abs().sum()),
        "executed_one_way_turnover": float((target_weights - prior_weights).abs().sum()),
        "buffered_sector_count": float(buffered_sector_count),
        "rebalanced_sector_count": float(rebalanced_sector_count),
    }


PROFITABILITY_GATE_FEATURE_NAMES = (
    "score_strength",
    "prediction_dispersion",
    "candidate_gross_exposure",
    "prior_gross_exposure",
    "optional_turnover",
    "selected_symbol_count",
    "active_sector_count",
    "is_session_final",
)


@dataclass(frozen=True)
class ProfitabilityGate:
    """Regularised logistic model for a rebalance's incremental profitability."""

    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    cost_bps: float
    l2: float
    observations: int
    positive_rate: float
    iterations: int

    def predict_probability(self, feature_values: dict[str, float]) -> float:
        values = np.asarray([feature_values[name] for name in self.feature_names], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Profitability-gate features must be finite at inference time.")
        standardized = (values - self.feature_mean) / self.feature_scale
        logit = float(np.clip(self.intercept + np.dot(standardized, self.coefficients), -35.0, 35.0))
        return float(1.0 / (1.0 + math.exp(-logit)))


def _profitability_gate_features(
    predictions: pd.Series,
    label_available: pd.Series,
    candidate_weights: pd.Series,
    prior_for_target: pd.Series,
    proposed_weights: pd.Series,
    optional_turnover: float,
    sector_by_symbol: pd.Series,
    is_session_final: bool,
) -> dict[str, float]:
    """Summarise one proposed action using only information known at its signal time."""
    valid_predictions = predictions[label_available].dropna()
    active_sector_count = int(
        candidate_weights.abs().groupby(sector_by_symbol, sort=False).sum().gt(1e-15).sum()
    )
    return {
        "score_strength": float(abs((candidate_weights * predictions.fillna(0.0)).sum())),
        "prediction_dispersion": float(valid_predictions.std(ddof=0)) if len(valid_predictions) > 1 else 0.0,
        "candidate_gross_exposure": float(candidate_weights.abs().sum()),
        "prior_gross_exposure": float(prior_for_target.abs().sum()),
        "optional_turnover": float(optional_turnover),
        "selected_symbol_count": float(candidate_weights.abs().gt(1e-15).sum()),
        "active_sector_count": float(active_sector_count),
        "is_session_final": float(is_session_final),
    }


def _scheduled_trade_frame(prediction_frame: pd.DataFrame, holding_minutes: int) -> pd.DataFrame:
    """Return only exchange-clock rebalance rows with their original ordering."""
    required = {"sector", "symbol", "target_return", "prediction", "entry_price", "exit_price"}
    missing = required.difference(prediction_frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing rows required for portfolio evaluation: {sorted(missing)}")
    traded = (
        prediction_frame.merge(
            _scheduled_rebalances(prediction_frame, holding_minutes),
            on=["session", "timestamp"],
            how="inner",
        )
        .sort_values(["session", "timestamp", "symbol"])
        .reset_index(drop=True)
    )
    if traded.empty:
        raise ValueError("No scheduled OOS rebalances are available for cost evaluation.")
    return traded


def _portfolio_symbol_metadata(traded: pd.DataFrame) -> tuple[pd.Index, pd.Series, dict[str, float]]:
    """Build the fixed symbol and sector reference objects for a split."""
    symbols = pd.Index(sorted(traded["symbol"].unique()))
    sector_by_symbol = (
        traded[["symbol", "sector"]]
        .drop_duplicates()
        .set_index("symbol")["sector"]
        .reindex(symbols)
    )
    if sector_by_symbol.isna().any():
        raise ValueError("Every tradable symbol must have a sector label.")
    return symbols, sector_by_symbol, _fixed_symbol_count_sector_budgets(traded)


def _profitability_gate_training_frame(
    prediction_frame: pd.DataFrame,
    holding_minutes: int,
    z_threshold: float,
    cost_bps: float,
    session_end_policy: str,
    target_smoothing_fraction: float,
    sector_rebalance_buffer_turnover: float,
) -> pd.DataFrame:
    """Create validation-only labels for the incremental rebalance decision.

    The reference path always accepts the rank-selected target.  At every
    signal, the label compares that proposed action with holding the current
    eligible target through the same next holding period.  Forced exits caused
    by a missing future label occur in both alternatives and are excluded from
    the incremental decision label.
    """
    if cost_bps < 0.0:
        raise ValueError("profitability-gate cost must be non-negative.")
    if session_end_policy not in {"carry_to_next_rebalance", "flatten_daily"}:
        raise ValueError("session_end_policy must be 'carry_to_next_rebalance' or 'flatten_daily'.")
    traded = _scheduled_trade_frame(prediction_frame, holding_minutes)
    symbols, sector_by_symbol, sector_budgets = _portfolio_symbol_metadata(traded)
    grouped_periods = list(traded.groupby(["session", "timestamp"], sort=False))
    cost_rate = cost_bps / 10_000.0
    prior_weights = pd.Series(0.0, index=symbols)
    records: list[dict[str, object]] = []
    for period_index, ((session, timestamp), group) in enumerate(grouped_periods):
        by_symbol = group.set_index("symbol")
        predictions = by_symbol["prediction"].reindex(symbols)
        candidate_weight_by_row = _sector_neutral_weights(
            group["prediction"], group["sector"], z_threshold, sector_budgets
        )
        candidate_weights = pd.Series(
            candidate_weight_by_row.to_numpy(dtype=float), index=group["symbol"].to_numpy()
        ).reindex(symbols).fillna(0.0)
        asset_returns = np.expm1(by_symbol["target_return"].reindex(symbols)).fillna(0.0)
        label_available = by_symbol["target_return"].reindex(symbols).notna()
        is_session_final = (
            period_index + 1 == len(grouped_periods)
            or grouped_periods[period_index + 1][0][0] != session
        )
        flatten_after_period = period_index + 1 == len(grouped_periods) or (
            session_end_policy == "flatten_daily" and is_session_final
        )
        prior_for_target = prior_weights.where(label_available, 0.0)
        candidate_weights = candidate_weights.where(label_available, 0.0)
        proposed_weights, _controls = _smoothed_sector_buffered_target(
            prior_for_target,
            candidate_weights,
            sector_by_symbol,
            target_smoothing_fraction,
            sector_rebalance_buffer_turnover,
        )
        optional_turnover = float((proposed_weights - prior_for_target).abs().sum())
        if optional_turnover > 1e-15:
            terminal_turnover_difference = (
                float(proposed_weights.abs().sum() - prior_for_target.abs().sum())
                if flatten_after_period
                else 0.0
            )
            incremental_gross_return = float(((proposed_weights - prior_for_target) * asset_returns).sum())
            incremental_net_return = incremental_gross_return - cost_rate * (
                optional_turnover + terminal_turnover_difference
            )
            features = _profitability_gate_features(
                predictions,
                label_available,
                candidate_weights,
                prior_for_target,
                proposed_weights,
                optional_turnover,
                sector_by_symbol,
                is_session_final,
            )
            records.append(
                {
                    "session": session,
                    "timestamp": timestamp,
                    **features,
                    "incremental_gross_return": incremental_gross_return,
                    "incremental_net_return": incremental_net_return,
                    "profitable_rebalance": int(incremental_net_return > 0.0),
                }
            )
        prior_weights = pd.Series(0.0, index=symbols) if flatten_after_period else proposed_weights
    if not records:
        raise ValueError("No optional validation rebalances are available to fit the profitability gate.")
    return pd.DataFrame.from_records(records)


def _binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(probabilities, method="average")
    return float((ranks[labels.astype(bool)].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def fit_profitability_gate(
    validation_predictions: pd.DataFrame,
    holding_minutes: int,
    z_threshold: float,
    cost_bps: float,
    min_probability: float,
    l2: float,
    session_end_policy: str,
    target_smoothing_fraction: float,
    sector_rebalance_buffer_turnover: float,
    max_iterations: int = 100,
) -> tuple[ProfitabilityGate, pd.DataFrame, dict[str, object]]:
    """Fit a strongly regularised logistic gate on frozen validation predictions."""
    if not 0.0 < min_probability < 1.0:
        raise ValueError("profitability-gate minimum probability must lie in (0, 1).")
    if l2 < 0.0:
        raise ValueError("profitability-gate L2 regularisation must be non-negative.")
    training_frame = _profitability_gate_training_frame(
        validation_predictions,
        holding_minutes,
        z_threshold,
        cost_bps,
        session_end_policy,
        target_smoothing_fraction,
        sector_rebalance_buffer_turnover,
    )
    feature_names = PROFITABILITY_GATE_FEATURE_NAMES
    x_raw = training_frame.loc[:, feature_names].to_numpy(dtype=float)
    labels = training_frame["profitable_rebalance"].to_numpy(dtype=float)
    finite = np.isfinite(x_raw).all(axis=1) & np.isfinite(labels)
    x_raw = x_raw[finite]
    labels = labels[finite]
    training_frame = training_frame.loc[finite].reset_index(drop=True)
    if len(labels) < 20:
        raise ValueError("At least 20 validation rebalances are required for the profitability gate.")
    feature_mean = x_raw.mean(axis=0)
    feature_scale = x_raw.std(axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    x = (x_raw - feature_mean) / feature_scale
    positive_rate = float(labels.mean())
    clipped_rate = float(np.clip(positive_rate, 1e-4, 1.0 - 1e-4))
    coefficients = np.zeros(x.shape[1], dtype=float)
    intercept = math.log(clipped_rate / (1.0 - clipped_rate))
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        logits = np.clip(intercept + x @ coefficients, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = probability - labels
        gradient_intercept = float(residual.mean())
        gradient_coefficients = x.T @ residual / len(labels) + l2 * coefficients
        curvature = np.clip(probability * (1.0 - probability), 1e-6, None)
        hessian = np.empty((x.shape[1] + 1, x.shape[1] + 1), dtype=float)
        hessian[0, 0] = float(curvature.mean())
        hessian[0, 1:] = (x.T @ curvature) / len(labels)
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = (x.T * curvature) @ x / len(labels) + l2 * np.eye(x.shape[1])
        gradient = np.concatenate(([gradient_intercept], gradient_coefficients))
        try:
            step = np.linalg.solve(hessian + 1e-8 * np.eye(len(gradient)), gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian + 1e-6 * np.eye(len(gradient)), gradient, rcond=None)[0]
        current_loss = float(np.mean(np.logaddexp(0.0, logits) - labels * logits) + 0.5 * l2 * coefficients.dot(coefficients))
        step_scale = 1.0
        while step_scale >= 1.0 / 128.0:
            candidate_intercept = intercept - step_scale * step[0]
            candidate_coefficients = coefficients - step_scale * step[1:]
            candidate_logits = np.clip(candidate_intercept + x @ candidate_coefficients, -35.0, 35.0)
            candidate_loss = float(
                np.mean(np.logaddexp(0.0, candidate_logits) - labels * candidate_logits)
                + 0.5 * l2 * candidate_coefficients.dot(candidate_coefficients)
            )
            if candidate_loss <= current_loss + 1e-12:
                intercept = candidate_intercept
                coefficients = candidate_coefficients
                break
            step_scale *= 0.5
        if np.max(np.abs(step_scale * step)) < 1e-7:
            break
    gate = ProfitabilityGate(
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=float(intercept),
        cost_bps=float(cost_bps),
        l2=float(l2),
        observations=int(len(labels)),
        positive_rate=positive_rate,
        iterations=iterations,
    )
    training_frame["predicted_profitability_probability"] = [
        gate.predict_probability(dict(zip(feature_names, row, strict=True)))
        for row in training_frame.loc[:, feature_names].to_numpy(dtype=float)
    ]
    fitted_probability = training_frame["predicted_profitability_probability"].to_numpy(dtype=float)
    diagnostics = {
        "fit_split": "validation_predictions_only_after_rank-checkpoint_selection",
        "target": "incremental 30-minute simple PnL after fixed one-way cost, versus retaining current eligible target weights",
        "cost_bps": float(cost_bps),
        "min_probability": float(min_probability),
        "method": "L2-regularised logistic regression on ex-ante candidate-action summaries",
        "feature_names": list(feature_names),
        "observations": int(len(labels)),
        "positive_rate": positive_rate,
        "in_sample_brier_score": float(np.mean((fitted_probability - labels) ** 2)),
        "in_sample_auc": _binary_auc(labels.astype(int), fitted_probability),
        "iterations": int(iterations),
        "l2": float(l2),
        "warning": "These are in-sample diagnostics for the validation-fitted gate. OOS is the only untouched evaluation of the full decision rule.",
    }
    return gate, training_frame, diagnostics


def evaluate_cost_sensitivity(
    prediction_frame: pd.DataFrame,
    holding_minutes: int,
    z_threshold: float,
    session_end_policy: str = "flatten_daily",
    target_smoothing_fraction: float = 0.5,
    sector_rebalance_buffer_turnover: float = 0.02,
    profitability_gate: ProfitabilityGate | None = None,
    profitability_gates_by_cost: dict[float, ProfitabilityGate] | None = None,
    profitability_gate_min_probability: float = 0.55,
    costs_bps: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0),
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    """Evaluate a target-weight strategy with an optional profitability gate.

    A signal creates a desired portfolio weight with fixed, symbol-count sector
    budgets. The target moves only part way toward that desired weight, and a
    complete sector sleeve is left unchanged when its proposed one-way turnover
    is below the buffer. There is no score-to-return calibration. When supplied,
    a validation-fitted gate decides whether to execute the proposed rebalance
    from its estimated probability of positive incremental net PnL.
    Transaction cost is charged on
    ``sum(abs(target_t - target_{t-1}))``, not on an assumed full exit and
    re-entry. Under ``carry_to_next_rebalance`` positions persist over the close
    and their overnight/early-session bridge return is marked explicitly.
    ``flatten_daily`` remains available as a comparison. A ticker with no valid
    label for the next holding period is forcibly liquidated at the previous
    interval's known endpoint; it cannot be carried through an unpriced period.
    """
    frame = prediction_frame.copy()
    # Anchor to the exchange clock rather than the count of usable observations.
    # A skipped signal is a target of zero; the liquidation cost is assessed via
    # the same consecutive-target turnover calculation.
    traded = _scheduled_trade_frame(frame, holding_minutes)
    if session_end_policy not in {"carry_to_next_rebalance", "flatten_daily"}:
        raise ValueError("session_end_policy must be 'carry_to_next_rebalance' or 'flatten_daily'.")
    if not 0.0 < target_smoothing_fraction <= 1.0:
        raise ValueError("target_smoothing_fraction must lie in (0, 1].")
    if sector_rebalance_buffer_turnover < 0.0:
        raise ValueError("sector_rebalance_buffer_turnover must be non-negative.")
    if not 0.0 < profitability_gate_min_probability < 1.0:
        raise ValueError("profitability-gate minimum probability must lie in (0, 1).")
    if profitability_gates_by_cost is not None:
        missing_gate_costs = [float(cost) for cost in costs_bps if float(cost) not in profitability_gates_by_cost]
        if missing_gate_costs:
            raise ValueError(
                "Missing profitability gate(s) for reported cost scenario(s): "
                f"{missing_gate_costs}"
            )

    symbols, sector_by_symbol, sector_budgets = _portfolio_symbol_metadata(traded)
    candidate_weight_parts: list[pd.Series] = []
    for (session, timestamp), group in traded.groupby(["session", "timestamp"], sort=False):
        candidate_weights = _sector_neutral_weights(
            group["prediction"], group["sector"], z_threshold, sector_budgets
        )
        candidate_weight_parts.append(candidate_weights)
    traded["candidate_weight"] = pd.concat(candidate_weight_parts).sort_index().reindex(traded.index)
    traded["asset_simple_return"] = np.expm1(traded["target_return"])
    traded["candidate_gross_return"] = traded["candidate_weight"] * traded["asset_simple_return"]
    gross = traded.groupby(["session", "timestamp"], as_index=False)["candidate_gross_return"].sum()
    gross = gross.rename(columns={"candidate_gross_return": "candidate_gross_portfolio_return"})
    candidate_exposure = traded.groupby(["session", "timestamp"], as_index=False)["candidate_weight"].agg(
        lambda weights: weights.abs().sum()
    ).rename(columns={"candidate_weight": "candidate_gross_exposure"})
    gross = gross.merge(candidate_exposure, on=["session", "timestamp"], how="left")

    metrics: dict[str, dict[str, object]] = {}
    periods_per_year = 252.0 * (len(gross) / gross["session"].nunique())
    grouped_periods = list(traded.groupby(["session", "timestamp"], sort=False))
    for cost in costs_bps:
        if cost < 0.0:
            raise ValueError("Cost sensitivity values must be non-negative.")
        scenario_gate = (
            profitability_gates_by_cost[float(cost)]
            if profitability_gates_by_cost is not None
            else profitability_gate
        )
        cost_rate = cost / 10_000.0
        prior_weights = pd.Series(0.0, index=symbols)
        previous_exit_prices: pd.Series | None = None
        previous_session: object | None = None
        scenario_records: list[dict[str, object]] = []
        for period_index, ((session, timestamp), group) in enumerate(grouped_periods):
            by_symbol = group.set_index("symbol")
            predictions = by_symbol["prediction"].reindex(symbols)
            candidate_weights = by_symbol["candidate_weight"].reindex(symbols).fillna(0.0)
            asset_returns = by_symbol["asset_simple_return"].reindex(symbols).fillna(0.0)
            entry_prices = by_symbol["entry_price"].reindex(symbols)
            exit_prices = by_symbol["exit_price"].reindex(symbols)
            label_available = by_symbol["target_return"].reindex(symbols).notna()
            is_session_final = (
                period_index + 1 == len(grouped_periods)
                or grouped_periods[period_index + 1][0][0] != session
            )
            flatten_after_period = period_index + 1 == len(grouped_periods) or (
                session_end_policy == "flatten_daily" and is_session_final
            )
            # Each selected OOS row has both an entry and an exit source bar.
            # Rows absent from this group have no full next holding-period label.
            # The previous position finishes at this rebalance boundary, so it is
            # safely liquidated there; it must not be smoothed into an unmarked
            # future interval merely because the current label is unavailable.
            forced_liquidation_weights = prior_weights.where(~label_available, 0.0)
            forced_unavailable_liquidation_turnover = float(forced_liquidation_weights.abs().sum())
            prior_for_target = prior_weights.where(label_available, 0.0)
            candidate_weights = candidate_weights.where(label_available, 0.0)
            proposed_weights, execution_controls = _smoothed_sector_buffered_target(
                prior_for_target,
                candidate_weights,
                sector_by_symbol,
                target_smoothing_fraction,
                sector_rebalance_buffer_turnover,
            )
            proposed_turnover = execution_controls["executed_one_way_turnover"]
            gate_action_available = bool(proposed_turnover > 1e-15)
            gate_probability = float("nan")
            gate_accepted = True
            if scenario_gate is not None and gate_action_available:
                gate_features = _profitability_gate_features(
                    predictions,
                    label_available,
                    candidate_weights,
                    prior_for_target,
                    proposed_weights,
                    proposed_turnover,
                    sector_by_symbol,
                    is_session_final,
                )
                gate_probability = scenario_gate.predict_probability(gate_features)
                gate_accepted = bool(gate_probability >= profitability_gate_min_probability)
            target_weights = proposed_weights if gate_accepted else prior_for_target
            bridge_portfolio_return = 0.0
            is_cross_session_bridge = previous_session is not None and session != previous_session
            if session_end_policy == "carry_to_next_rebalance" and is_cross_session_bridge and previous_exit_prices is not None:
                bridge_asset_returns = entry_prices / previous_exit_prices - 1.0
                invalid_bridge = (
                    prior_weights.abs().gt(1e-12)
                    & (~np.isfinite(bridge_asset_returns) | entry_prices.le(0) | previous_exit_prices.le(0))
                )
                if invalid_bridge.any():
                    missing_symbols = ", ".join(invalid_bridge.index[invalid_bridge].tolist())
                    raise ValueError(
                        "Cannot mark a carried position because entry/exit prices are missing for: "
                        f"{missing_symbols}"
                    )
                bridge_portfolio_return = float((prior_weights * bridge_asset_returns.fillna(0.0)).sum())
            transaction_turnover = forced_unavailable_liquidation_turnover + float(
                (target_weights - prior_for_target).abs().sum()
            )
            rebalance_executed = bool(transaction_turnover > 1e-15)
            end_liquidation_turnover = 0.0
            if flatten_after_period:
                end_liquidation_turnover = float(target_weights.abs().sum())
                transaction_turnover += end_liquidation_turnover
                next_weights = pd.Series(0.0, index=symbols)
            else:
                next_weights = target_weights
            gross_portfolio_return = float((target_weights * asset_returns).sum())
            total_gross_portfolio_return = bridge_portfolio_return + gross_portfolio_return
            net_return = total_gross_portfolio_return - transaction_turnover * cost_rate
            scenario_records.append(
                {
                    "session": session,
                    "timestamp": timestamp,
                    "candidate_available": bool(candidate_weights.abs().sum() > 0.0),
                    "rebalance_executed": rebalance_executed,
                    "profitability_gate_action_available": gate_action_available,
                    "profitability_gate_probability": gate_probability,
                    "profitability_gate_accepted": gate_accepted,
                    "active_position": bool(target_weights.abs().sum() > 0.0),
                    "bridge_portfolio_return": bridge_portfolio_return,
                    "cross_session_bridge_portfolio_return": (
                        bridge_portfolio_return if is_cross_session_bridge else 0.0
                    ),
                    "gross_portfolio_return": gross_portfolio_return,
                    "total_gross_portfolio_return": total_gross_portfolio_return,
                    "target_gross_exposure": float(target_weights.abs().sum()),
                    "candidate_one_way_turnover": proposed_turnover,
                    "forced_unavailable_liquidation_turnover": forced_unavailable_liquidation_turnover,
                    "smoothed_one_way_turnover": execution_controls["smoothed_one_way_turnover"],
                    "buffered_sector_count": execution_controls["buffered_sector_count"],
                    "rebalanced_sector_count": execution_controls["rebalanced_sector_count"],
                    "transaction_turnover": transaction_turnover,
                    "end_liquidation_turnover": end_liquidation_turnover,
                    "net_return": net_return,
                }
            )
            prior_weights = next_weights
            previous_exit_prices = exit_prices
            previous_session = session

        scenario = pd.DataFrame.from_records(scenario_records)
        net = scenario["net_return"]
        # Include the initial NAV in the running peak. Without this explicit
        # 1.0 baseline, a loss at the start of OOS can be omitted from MDD if
        # no later observation exceeds the starting capital.
        equity = np.concatenate(([1.0], np.cumprod(1.0 + net.to_numpy(dtype=float))))
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        metrics[f"{cost:g} bps"] = {
            "mean_return_per_holding_period": float(net.mean()),
            "annualized_sharpe": float(net.mean() / (net.std(ddof=1) + 1e-12) * math.sqrt(periods_per_year)),
            "annualized_return_approx": float(net.mean() * periods_per_year),
            "max_drawdown_pct": float(drawdown.min() * 100.0),
            "holding_periods": int(len(net)),
            "one_way_cost_bps": float(cost),
            "active_trade_rate": float(scenario["active_position"].mean()),
            "rebalance_execution_rate": float(scenario["rebalance_executed"].mean()),
            "profitability_gate_action_rate": float(scenario["profitability_gate_action_available"].mean()),
            "profitability_gate_acceptance_rate": float(
                scenario.loc[scenario["profitability_gate_action_available"], "profitability_gate_accepted"].mean()
            ) if scenario["profitability_gate_action_available"].any() else float("nan"),
            "mean_profitability_gate_probability": float(
                scenario.loc[scenario["profitability_gate_action_available"], "profitability_gate_probability"].mean()
            ) if scenario_gate is not None and scenario["profitability_gate_action_available"].any() else float("nan"),
            "mean_gross_exposure": float(scenario["target_gross_exposure"].mean()),
            "mean_one_way_turnover": float(scenario["transaction_turnover"].mean()),
            "total_one_way_turnover": float(scenario["transaction_turnover"].sum()),
            "total_forced_unavailable_liquidation_turnover": float(
                scenario["forced_unavailable_liquidation_turnover"].sum()
            ),
            "mean_smoothed_one_way_turnover": float(scenario["smoothed_one_way_turnover"].mean()),
            "mean_buffered_sector_count": float(scenario["buffered_sector_count"].mean()),
            "mean_rebalanced_sector_count": float(scenario["rebalanced_sector_count"].mean()),
            "mean_bridge_portfolio_return": float(scenario["bridge_portfolio_return"].mean()),
            "total_cross_session_bridge_return": float(scenario["cross_session_bridge_portfolio_return"].sum()),
            "total_end_liquidation_turnover": float(scenario["end_liquidation_turnover"].sum()),
            "session_end_policy": session_end_policy,
        }
        gross = gross.merge(
            scenario.rename(
                columns={
                    "active_position": f"active_position_{cost:g}bps",
                    "candidate_available": f"candidate_available_{cost:g}bps",
                    "rebalance_executed": f"rebalance_executed_{cost:g}bps",
                    "profitability_gate_action_available": f"profitability_gate_action_available_{cost:g}bps",
                    "profitability_gate_probability": f"profitability_gate_probability_{cost:g}bps",
                    "profitability_gate_accepted": f"profitability_gate_accepted_{cost:g}bps",
                    "bridge_portfolio_return": f"bridge_portfolio_return_{cost:g}bps",
                    "cross_session_bridge_portfolio_return": f"cross_session_bridge_portfolio_return_{cost:g}bps",
                    "gross_portfolio_return": f"gross_portfolio_return_{cost:g}bps",
                    "total_gross_portfolio_return": f"total_gross_portfolio_return_{cost:g}bps",
                    "target_gross_exposure": f"gross_exposure_{cost:g}bps",
                    "candidate_one_way_turnover": f"candidate_one_way_turnover_{cost:g}bps",
                    "forced_unavailable_liquidation_turnover": f"forced_unavailable_liquidation_turnover_{cost:g}bps",
                    "smoothed_one_way_turnover": f"smoothed_one_way_turnover_{cost:g}bps",
                    "buffered_sector_count": f"buffered_sector_count_{cost:g}bps",
                    "rebalanced_sector_count": f"rebalanced_sector_count_{cost:g}bps",
                    "transaction_turnover": f"one_way_turnover_{cost:g}bps",
                    "end_liquidation_turnover": f"end_liquidation_turnover_{cost:g}bps",
                    "net_return": f"net_return_{cost:g}bps",
                }
            ),
            on=["session", "timestamp"],
            how="left",
        )

    daily_rank_ic = _daily_rank_ic_from_prediction_frame(frame)
    metrics["signal_diagnostics"] = {
        "mean_daily_rank_ic": float(daily_rank_ic.mean()),
        "daily_rank_ic_ir": float(daily_rank_ic.mean() / (daily_rank_ic.std(ddof=1) + 1e-12)),
        "daily_rank_ic_positive_rate": float((daily_rank_ic > 0).mean()),
        "candidate_trade_rate_before_cost_gate": float((gross["candidate_gross_exposure"] > 0.0).mean()),
        "mean_candidate_gross_exposure": float(gross["candidate_gross_exposure"].mean()),
        "rank_signal_selection": "within-sector z-score threshold; no score-to-return calibration",
        "profitability_gate": (
            {
                "enabled": True,
                "cost_bps": (
                    [float(cost) for cost in costs_bps]
                    if profitability_gates_by_cost is not None
                    else float(profitability_gate.cost_bps)
                ),
                "min_probability": float(profitability_gate_min_probability),
                "target": "positive incremental net PnL from executing the proposed rebalance",
            }
            if profitability_gate is not None or profitability_gates_by_cost is not None
            else {"enabled": False}
        ),
        "cost_scenarios_share_identical_target_path": profitability_gates_by_cost is None,
        "execution_model": (
            "consecutive intraday target weights with daily terminal flattening"
            if session_end_policy == "flatten_daily"
            else "consecutive target weights with marked between-rebalance bridge returns"
        ),
        "session_end_policy": session_end_policy,
        "sector_budget_rule": "fixed gross budgets proportional to eligible sector symbol counts; inactive budgets remain cash",
        "sector_budget_by_name": sector_budgets,
        "target_smoothing_fraction": float(target_smoothing_fraction),
        "sector_rebalance_buffer_turnover": float(sector_rebalance_buffer_turnover),
    }
    return metrics, gross


def write_missing_bar_diagnostics(features: pd.DataFrame, output_dir: Path) -> None:
    """Persist observed-versus-filled bar counts for auditability."""
    summary = (
        features.groupby("symbol", as_index=False)
        .agg(
            sessions=("session", "nunique"),
            grid_minutes=("timestamp", "size"),
            observed_bars=("has_trade", "sum"),
            labelled_samples=("target_return", lambda values: int(values.notna().sum())),
            max_staleness_minutes=("staleness_minutes", "max"),
        )
        .sort_values("symbol")
    )
    summary["missing_bars"] = summary["grid_minutes"] - summary["observed_bars"]
    summary["missing_bar_rate"] = summary["missing_bars"] / summary["grid_minutes"].clip(lower=1)
    summary.to_csv(output_dir / "missing_bar_diagnostics.csv", index=False)


@dataclass(frozen=True)
class SeedTrainingResult:
    """Validation-selected run and its uncalibrated validation/test scores."""

    seed: int
    checkpoint_path: Path
    best_epoch: int
    best_validation_ic: float
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame | None


def _format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def _train_one_seed(
    config: RunConfig,
    seed: int,
    output_dir: Path,
    train_set: CrossSectionalMinuteDataset,
    validation_set: CrossSectionalMinuteDataset,
    test_set: CrossSectionalMinuteDataset | None,
    cross_area_pca: CrossTickerAreaPCA,
    device: torch.device,
    use_amp: bool,
    use_pinned_memory: bool,
    workers: int,
    sde_target_scale: float | None = None,
) -> SeedTrainingResult:
    """Train one independently initialised model and retain its validation checkpoint."""
    set_global_seed(seed, config.deterministic, config.allow_tf32)
    model = SignatureConditionedAffineMeanModel(
        input_channels=len(FEATURE_COLUMNS),
        signature_depth=config.signature_depth,
        latent_dim=config.latent_dim,
        hidden_dim=config.affine_hidden_dim,
        num_symbols=len(train_set.symbols),
        cross_area_pca=cross_area_pca,
        diffusion_rank=config.sde_diffusion_rank,
        diffusion_floor=config.sde_diffusion_floor,
        jump_mode=config.sde_jump_mode,
        jump_initial_intensity=config.sde_jump_initial_intensity,
        jump_initial_std=config.sde_jump_initial_std,
        stock_embedding_dim=config.stock_embedding_dim,
        cross_attention_layers=config.cross_attention_layers,
        attention_dim=config.attention_dim,
        attention_heads=config.attention_heads,
        attention_dropout=config.attention_dropout,
        context_dropout=config.context_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sector_ids = None
    if config.sector_neutral_objective:
        sector_names = sorted({SECTOR_BY_SYMBOL[symbol] for symbol in train_set.symbols})
        sector_ids = torch.tensor([sector_names.index(SECTOR_BY_SYMBOL[symbol]) for symbol in train_set.symbols])
    loss_function = CrossSectionalRankLoss(config.min_labelled_symbols, sector_ids).to(device)
    nll_function: PredictiveNLL | None = None
    if model.has_diffusion:
        if sde_target_scale is None:
            raise ValueError("The SDE model needs a train-only target scale.")
        nll_function = PredictiveNLL(
            sde_target_scale, config.sde_target_clip, config.sde_jump_mode, config.sde_jump_max_count
        ).to(device)
    best_validation_nll = float("nan")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # pragma: no cover - compatibility with older PyTorch releases
        grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=workers,
        pin_memory=use_pinned_memory,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_pinned_memory,
        persistent_workers=workers > 0,
    )
    if len(train_loader) == 0:
        raise ValueError("Training set is smaller than one batch; lower --batch-size or download more sessions.")
    train_monitor_loader = (
        DataLoader(
            train_set.subset(config.train_ic_monitor_windows),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_pinned_memory,
        )
        if config.train_ic_monitor_windows > 0
        else None
    )

    best_path = output_dir / f"seed_{seed}_best_signature_affine_mean.pt"
    best_validation_ic = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    seed_start = time.perf_counter()
    epoch_durations: list[float] = []
    batch_count = len(train_loader)
    progress_every = max(1, batch_count // 10)
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        nll_losses: list[float] = []
        epoch_start = time.perf_counter()
        for batch_index, (path, target, target_mask, _) in enumerate(train_loader, start=1):
            path = _to_device(path, device, use_pinned_memory)
            target = _to_device(target, device, use_pinned_memory)
            target_mask = _to_device(target_mask, device, use_pinned_memory)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp):
                if nll_function is not None:
                    parameters = model.forward_predictive(path)
                    prediction = parameters["mean"]
                    rank_loss = loss_function(prediction, target, target_mask)
                    nll_loss = nll_function(parameters, target, target_mask)
                    loss = rank_loss + config.sde_nll_weight * nll_loss
                    nll_losses.append(float(nll_loss.item()))
                else:
                    prediction = model(path)
                    rank_loss = loss_function(prediction, target, target_mask)
                    loss = rank_loss
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            losses.append(float(rank_loss.item()))
            if batch_index % progress_every == 0 or batch_index == batch_count:
                elapsed = time.perf_counter() - epoch_start
                left = elapsed / batch_index * (batch_count - batch_index)
                print(
                    f"  seed {seed} epoch {epoch:03d}: batch {batch_index}/{batch_count} "
                    f"({100 * batch_index / batch_count:.0f}%) | {_format_duration(elapsed)} elapsed, "
                    f"~{_format_duration(left)} left in training pass",
                    end="\r",
                    flush=True,
                )

        train_seconds = time.perf_counter() - epoch_start
        print(" " * 100, end="\r", flush=True)
        validation_start = time.perf_counter()
        validation_ic, validation_daily_ic = _validate(
            model,
            validation_loader,
            device,
            use_amp,
            use_pinned_memory,
            sector_neutral=config.sector_neutral_objective,
        )
        validation_nll = float("nan")
        sde_message = ""
        if nll_function is not None:
            validation_nll = _validation_nll(
                model, validation_loader, nll_function, device, use_amp, use_pinned_memory
            )
            sde_message = (
                f" | train NLL {np.mean(nll_losses):.5f} | validation NLL {validation_nll:.5f}"
                f" | obs std {float(torch.exp(model.log_observation_std).item()):.4f}"
            )
            if model.jump_mode == "student_t":
                sde_message += f" | dof {2.0 + float(torch.exp(model.log_dof_minus_two).item()):.2f}"
        train_ic_message = ""
        if train_monitor_loader is not None:
            train_ic, _ = _validate(
                model, train_monitor_loader, device, use_amp, use_pinned_memory,
                sector_neutral=config.sector_neutral_objective,
            )
            train_ic_message = f" | train-subset Rank IC {train_ic:.5f}"
        validation_seconds = time.perf_counter() - validation_start
        epoch_durations.append(time.perf_counter() - epoch_start)
        print(
            f"Seed {seed} | Epoch {epoch:03d} | train normal-score rank loss {np.mean(losses):.5f}{train_ic_message} | "
            f"validation sector-neutral Rank IC {validation_ic:.5f}{sde_message}"
        )
        # Upper bound on what is left for this seed: the epoch cap, or early
        # stopping if validation IC does not improve for `patience` more epochs.
        improved_now = epoch > config.warmup_epochs and not np.isnan(validation_ic) and validation_ic > best_validation_ic
        stale_after = 0 if improved_now else (stale_epochs + 1 if epoch > config.warmup_epochs else 0)
        if epoch <= config.warmup_epochs:
            remaining = min(config.max_epochs - epoch, config.warmup_epochs - epoch + config.patience)
        else:
            remaining = min(config.max_epochs - epoch, config.patience - stale_after)
        print(
            f"  time: epoch {_format_duration(epoch_durations[-1])} (train {_format_duration(train_seconds)}, "
            f"validation {_format_duration(validation_seconds)}) | seed elapsed "
            f"{_format_duration(time.perf_counter() - seed_start)} | at most {max(remaining, 0)} more epoch(s), "
            f"<= {_format_duration(max(remaining, 0) * float(np.mean(epoch_durations)))} if no further improvement"
        )
        # Checkpoints are still selected on validation Rank IC, so the SDE and
        # deterministic (diffusion_rank=0) runs are selected by the same rule.
        if epoch > config.warmup_epochs and not np.isnan(validation_ic) and validation_ic > best_validation_ic:
            best_validation_ic = validation_ic
            best_validation_nll = validation_nll
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
            validation_bootstrap = block_bootstrap_mean_ic(validation_daily_ic, config.bootstrap_iterations, seed)
            validation_bootstrap["seed"] = seed
            validation_bootstrap["epoch"] = epoch
            (output_dir / f"seed_{seed}_best_validation_block_bootstrap.json").write_text(
                json.dumps(validation_bootstrap, indent=2), encoding="utf-8"
            )
            validation_daily_ic.rename_axis("session").reset_index().to_csv(
                output_dir / f"seed_{seed}_best_validation_daily_rank_ic.csv", index=False
            )
            print(
                f"  Seed {seed} checkpoint | daily-IC 95% CI "
                f"[{validation_bootstrap['ci_95_lower']:.5f}, {validation_bootstrap['ci_95_upper']:.5f}] | "
                f"significant_positive_95={validation_bootstrap['significant_positive_95']}"
            )
        elif epoch > config.warmup_epochs:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Seed {seed} early stopping after {epoch} epochs.")
                break

    if not best_path.exists():
        raise RuntimeError(f"Seed {seed} saved no validation checkpoint; inspect validation Rank IC and training data.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation_predictions = make_prediction_frame(model, validation_set, device, use_amp, use_pinned_memory)
    test_predictions = (
        make_prediction_frame(model, test_set, device, use_amp, use_pinned_memory) if test_set is not None else None
    )
    validation_predictions.to_parquet(output_dir / f"seed_{seed}_validation_predictions.parquet", index=False)
    if test_predictions is not None:
        test_predictions.to_parquet(output_dir / f"seed_{seed}_oos_predictions.parquet", index=False)
    (output_dir / f"seed_{seed}_training_summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "checkpoint": best_path.name,
                "best_epoch": best_epoch,
                "best_validation_sector_neutral_rank_ic": best_validation_ic,
                "diffusion_rank": config.sde_diffusion_rank,
                "jump_mode": config.sde_jump_mode,
                "best_validation_nll": best_validation_nll,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del train_loader, validation_loader, optimizer, loss_function, grad_scaler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return SeedTrainingResult(
        seed=seed,
        checkpoint_path=best_path,
        best_epoch=best_epoch,
        best_validation_ic=best_validation_ic,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
    )


def _sector_standardised_scores(frame: pd.DataFrame) -> pd.Series:
    """Normalise each seed's scores by signal time and sector before ensembling."""
    required = {"session", "timestamp", "sector", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing ensemble columns: {sorted(missing)}")
    group_keys = [frame["session"], frame["timestamp"], frame["sector"]]
    mean = frame["prediction"].groupby(group_keys, sort=False).transform("mean")
    std = frame["prediction"].groupby(group_keys, sort=False).transform("std", ddof=0)
    return ((frame["prediction"] - mean) / std.where(std > 1e-12, 1.0)).astype(float)


def average_seed_prediction_frames(frames: list[pd.DataFrame], seeds: tuple[int, ...]) -> pd.DataFrame:
    """Equal-average per-sector standardised scores from validation-selected seeds."""
    if len(frames) != len(seeds) or not frames:
        raise ValueError("Need one non-empty prediction frame for each ensemble seed.")
    keys = ["session", "timestamp", "symbol"]
    result = frames[0].sort_values(keys).reset_index(drop=True).copy()
    score_columns: list[str] = []
    for seed, frame in zip(seeds, frames, strict=True):
        candidate = frame.sort_values(keys).reset_index(drop=True)
        if not result[keys].equals(candidate[keys]):
            raise ValueError("Seed prediction frames have inconsistent samples or ticker ordering.")
        column = f"prediction_seed_{seed}"
        result[column] = _sector_standardised_scores(candidate).to_numpy()
        score_columns.append(column)
    result["prediction"] = result[score_columns].mean(axis=1)
    distribution_columns = {"predicted_mean_std", "predicted_variance_std"}
    if all(distribution_columns.issubset(frame.columns) for frame in frames):
        # Equal-weight Gaussian mixture of the seeds' predictive laws:
        # mean = avg(mu_k); variance = avg(var_k) + avg(mu_k^2) - mean^2.
        means = np.column_stack(
            [frame.sort_values(keys)["predicted_mean_std"].to_numpy(dtype=float) for frame in frames]
        )
        variances = np.column_stack(
            [frame.sort_values(keys)["predicted_variance_std"].to_numpy(dtype=float) for frame in frames]
        )
        mixture_mean = means.mean(axis=1)
        result["predicted_mean_std"] = mixture_mean
        result["predicted_variance_std"] = variances.mean(axis=1) + (means**2).mean(axis=1) - mixture_mean**2
        result["predicted_epistemic_variance_std"] = means.var(axis=1)
        # Keep each seed's exact law parameters so diagnostics can evaluate the
        # equal-weight mixture density, not only its first two moments.
        for seed, frame in zip(seeds, frames, strict=True):
            ordered = frame.sort_values(keys)
            for column in PREDICTIVE_PARAMETER_COLUMNS.values():
                if column in ordered.columns:
                    result[f"{column}__seed{seed}"] = ordered[column].to_numpy(dtype=float)
        for column in PREDICTIVE_PARAMETER_COLUMNS.values():
            if column in result.columns:
                result = result.drop(columns=column)
    return result


def run_pipeline(config: RunConfig) -> dict[str, dict[str, object]]:
    if config.sector_median_shrinkage_count < 0.0:
        raise ValueError("sector_median_shrinkage_count must be non-negative.")
    if config.profitability_gate_cost_bps < 0.0:
        raise ValueError("profitability_gate_cost_bps must be non-negative.")
    if not 0.0 < config.profitability_gate_min_probability < 1.0:
        raise ValueError("profitability_gate_min_probability must lie in (0, 1).")
    if config.profitability_gate_l2 < 0.0:
        raise ValueError("profitability_gate_l2 must be non-negative.")
    set_global_seed(config.seed, config.deterministic, config.allow_tf32)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    bars = load_minute_bars(config.data_dir, config.symbols)
    relative_return_spec = sector_relative_return_feature_spec(
        bars["symbol"].unique(), config.sector_median_shrinkage_count
    )
    (output_dir / "relative_return_benchmark.json").write_text(
        json.dumps(relative_return_spec, indent=2), encoding="utf-8"
    )
    etf_bars = load_etf_bars(config.data_dir) if config.use_etf_context else None
    if config.use_etf_context and etf_bars is None:
        raise FileNotFoundError(
            f"use_etf_context=True but no ETF bars in {Path(config.data_dir) / 'etf'}; run download_data.py "
            "or set USE_ETF_CONTEXT = False."
        )
    features = engineer_minute_features(
        bars,
        config.holding_minutes,
        sector_median_shrinkage_count=config.sector_median_shrinkage_count,
        etf_bars=etf_bars,
    )
    del etf_bars
    features = apply_feature_drops(features, config.drop_feature_groups)
    print(
        "Relative-price benchmark: sector median shrunk to universe median "
        f"with k={config.sector_median_shrinkage_count:g}; "
        f"{len(relative_return_spec['sector_members'])} fixed sectors."
    )
    write_missing_bar_diagnostics(features, output_dir)
    del bars
    train_raw, validation_raw, test_raw = split_by_session(features, config.train_ratio, config.val_ratio)
    # Free each intermediate frame as soon as it is no longer needed: for a
    # 100-name universe the feature table alone is several GB.
    del features
    scaler = fit_feature_scaler(train_raw)
    (output_dir / "feature_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    train = apply_feature_scaler(train_raw, scaler)
    del train_raw
    validation = apply_feature_scaler(validation_raw, scaler)
    del validation_raw
    test = apply_feature_scaler(test_raw, scaler) if config.evaluate_test else None
    del test_raw

    train_set = CrossSectionalMinuteDataset(
        train, config.window_minutes, config.min_labelled_symbols, config.train_sample_stride_minutes
    )
    del train
    validation_set = CrossSectionalMinuteDataset(
        validation, config.window_minutes, config.min_labelled_symbols, config.evaluation_sample_stride_minutes
    )
    del validation
    test_set = (
        CrossSectionalMinuteDataset(
            test, config.window_minutes, config.min_labelled_symbols, config.evaluation_sample_stride_minutes
        )
        if config.evaluate_test
        else None
    )
    del test
    if not config.evaluate_test:
        print("evaluate_test=False: the test split is not scored and no OOS evaluation is run.")
    if not (train_set.symbols == validation_set.symbols) or (
        test_set is not None and train_set.symbols != test_set.symbols
    ):
        raise ValueError("Train, validation, and test must contain the same ordered ticker universe.")
    target_calibration_spec = {
        "fit_split": "train_only",
        "rank_target": "forward 30-minute return minus contemporaneous valid-sector mean return",
        "raw_execution_target": "raw 30-minute log return retained for realised PnL only",
        "excluded_rank_labels": "sectors with fewer than two valid forward returns at a signal time",
        "rank_transform": "hard cross-sectional rank transformed to normal scores",
        "training_objective": "Pearson correlation with normal-score rank target (differentiable Spearman surrogate)",
        "execution_use": "rank-only: score magnitude is not converted to an expected-return forecast",
    }
    (output_dir / "target_calibration.json").write_text(
        json.dumps(target_calibration_spec, indent=2), encoding="utf-8"
    )
    cross_area_spec = cross_ticker_area_feature_spec(train_set.symbols)
    (output_dir / "cross_ticker_area_features.json").write_text(
        json.dumps(cross_area_spec, indent=2), encoding="utf-8"
    )
    signature_spec = selective_signature_feature_spec()
    (output_dir / "signature_feature_spec.json").write_text(
        json.dumps(signature_spec, indent=2), encoding="utf-8"
    )
    cross_area_pca = fit_cross_ticker_area_pca(
        train_set,
        price_price_components=config.price_price_pca_components,
        price_volume_components=config.price_volume_pca_components,
    )
    np.savez(
        output_dir / "cross_ticker_area_pca.npz",
        price_price_mean=cross_area_pca.price_price_mean,
        price_price_scale=cross_area_pca.price_price_scale,
        price_price_components=cross_area_pca.price_price_components,
        price_volume_mean=cross_area_pca.price_volume_mean,
        price_volume_scale=cross_area_pca.price_volume_scale,
        price_volume_components=cross_area_pca.price_volume_components,
    )
    pca_summary = {
        "fit_split": "train_only",
        "fit_window_count": int(cross_area_pca.fit_window_count),
        "fit_window_subsampled": bool(cross_area_pca.fit_window_subsampled),
        "raw_area_features": cross_area_spec["total_cross_ticker_area_count"],
        "pca_input_price_price_features": cross_area_spec["price_price_count"],
        "retained_price_price_components": int(config.price_price_pca_components),
        "price_price_explained_variance_ratio": [
            float(value) for value in cross_area_pca.price_price_explained_variance_ratio
        ],
        "price_price_cumulative_explained_variance_ratio": float(
            cross_area_pca.price_price_explained_variance_ratio.sum()
        ),
        "pca_input_price_volume_features": cross_area_spec["price_volume_count"],
        "retained_price_volume_components": int(config.price_volume_pca_components),
        "price_volume_explained_variance_ratio": [
            float(value) for value in cross_area_pca.price_volume_explained_variance_ratio
        ],
        "price_volume_cumulative_explained_variance_ratio": float(
            cross_area_pca.price_volume_explained_variance_ratio.sum()
        ),
    }
    (output_dir / "cross_ticker_area_pca.json").write_text(json.dumps(pca_summary, indent=2), encoding="utf-8")
    print(
        f"Dataset: {len(train_set):,} train / {len(validation_set):,} validation / "
        f"{(len(test_set) if test_set is not None else 0):,} test windows (scored={test_set is not None}); "
        f"{len(train_set.symbols)} common symbols; train stride={config.train_sample_stride_minutes} minutes, "
        f"validation/test stride={config.evaluation_sample_stride_minutes} minutes; "
        f"selective signature={SELECTIVE_SIGNATURE_DIM} + state context={STATE_CONTEXT_DIM} + "
        f"multi-scale context={DIRECTIONAL_MULTISCALE_CONTEXT_DIM}; "
        f"cross-ticker areas={cross_area_spec['price_price_count']} price-price→"
        f"{config.price_price_pca_components} PCA components "
        f"(train explained variance={pca_summary['price_price_cumulative_explained_variance_ratio']:.1%}) + "
        f"{cross_area_spec['price_volume_count']} price-volume→{config.price_volume_pca_components} PCA components "
        f"(train explained variance={pca_summary['price_volume_cumulative_explained_variance_ratio']:.1%})."
    )

    sde_target_scale: float | None = None
    sde_baseline_variance: dict[str, float] | None = None
    if config.sde_diffusion_rank > 0:
        if config.sde_nll_weight < 0.0:
            raise ValueError("sde_nll_weight must be non-negative.")
        if config.sde_jump_mode not in JUMP_MODES:
            raise ValueError(f"sde_jump_mode must be one of {JUMP_MODES}.")
        sde_target_scale = fit_sde_target_scale(train_set)
        sde_baseline_variance = per_symbol_baseline_variance(train_set, sde_target_scale, config.sde_target_clip)
        sde_spec = {
            "model": "linear latent SDE dZ = (A(c)Z + b(c))dt + G(c)dW, exact Gaussian endpoint via matrix exponentials",
            "covariance_method": "Van Loan block matrix exponential",
            "diffusion_rank": config.sde_diffusion_rank,
            "diffusion_floor": config.sde_diffusion_floor,
            "observation_noise": "single global homoskedastic parameter",
            "loss": "rank loss + sde_nll_weight * Gaussian NLL of standardised sector-excess return",
            "sde_nll_weight": config.sde_nll_weight,
            "target_scale_fit_split": "train_only",
            "target_scale_method": "1.4826 * MAD of sector-excess forward return",
            "target_scale": sde_target_scale,
            "target_clip_in_scale_units": config.sde_target_clip,
            "baseline_variance_by_symbol": sde_baseline_variance,
            "checkpoint_selection": "validation Rank IC (same rule as the deterministic model)",
            "jump_mode": config.sde_jump_mode,
            "jump_specification": {
                "none": "Gaussian predictive law",
                "merton": "y = w'Z_1 + q + eps + sum_{i<=N} xi_i, N ~ Poisson(lambda(c)), xi ~ N(0, s_J(c)^2); "
                "exact Poisson-mixture likelihood truncated at jump_max_count",
                "student_t": "Student-t with the Gaussian variance and a global learned dof",
            }[config.sde_jump_mode],
            "jump_max_count": config.sde_jump_max_count,
            "jump_initial_intensity": config.sde_jump_initial_intensity,
            "jump_initial_std": config.sde_jump_initial_std,
        }
        (output_dir / "sde_model_spec.json").write_text(json.dumps(sde_spec, indent=2), encoding="utf-8")
        print(
            f"Linear latent SDE: rank={config.sde_diffusion_rank}, jumps={config.sde_jump_mode}, "
            f"NLL weight={config.sde_nll_weight:g}, "
            f"train target scale={sde_target_scale:.3e}"
        )

    training_seeds = tuple(dict.fromkeys(config.ensemble_seeds))
    if not training_seeds:
        raise ValueError("ensemble_seeds must contain at least one training seed.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pinned_memory = config.pin_memory and device.type == "cuda"
    use_amp = config.amp_enabled and device.type == "cuda"
    workers = max(0, config.data_loader_workers)
    print(
        f"Runtime: device={device}, workers={workers}, pinned_memory={use_pinned_memory}, "
        f"TF32={config.allow_tf32 and device.type == 'cuda'}, AMP={use_amp}, deterministic={config.deterministic}, "
        f"ensemble_seeds={list(training_seeds)}"
    )
    seed_results = [
        _train_one_seed(
            config,
            seed,
            output_dir,
            train_set,
            validation_set,
            test_set,
            cross_area_pca,
            device,
            use_amp,
            use_pinned_memory,
            workers,
            sde_target_scale,
        )
        for seed in training_seeds
    ]
    validation_predictions = average_seed_prediction_frames(
        [result.validation_predictions for result in seed_results], training_seeds
    )
    validation_predictions.to_parquet(output_dir / "ensemble_validation_predictions.parquet", index=False)
    test_predictions: pd.DataFrame | None = None
    if config.evaluate_test:
        test_predictions = average_seed_prediction_frames(
            [result.test_predictions for result in seed_results], training_seeds
        )
        test_predictions.to_parquet(output_dir / "ensemble_oos_predictions_uncalibrated.parquet", index=False)
    ensemble_validation_daily_ic = _daily_rank_ic_from_prediction_frame(validation_predictions)
    ensemble_validation_bootstrap = block_bootstrap_mean_ic(
        ensemble_validation_daily_ic, config.bootstrap_iterations, config.seed
    )
    ensemble_validation_daily_ic.rename_axis("session").reset_index().to_csv(
        output_dir / "ensemble_validation_daily_rank_ic.csv", index=False
    )
    (output_dir / "ensemble_validation_block_bootstrap.json").write_text(
        json.dumps(ensemble_validation_bootstrap, indent=2), encoding="utf-8"
    )
    ensemble_summary = {
        "method": "equal average of per-signal, per-sector z-scored predictions from validation-selected seeds",
        "seeds": [
            {
                "seed": result.seed,
                "checkpoint": result.checkpoint_path.name,
                "best_epoch": result.best_epoch,
                "best_validation_sector_neutral_rank_ic": result.best_validation_ic,
            }
            for result in seed_results
        ],
        "ensemble_validation_sector_neutral_rank_ic": float(ensemble_validation_daily_ic.mean()),
        "ensemble_validation_rank_ic_block_bootstrap": ensemble_validation_bootstrap,
    }
    (output_dir / "ensemble_summary.json").write_text(json.dumps(ensemble_summary, indent=2), encoding="utf-8")
    print(
        "Ensemble validation checkpoint | "
        f"sector-neutral daily Rank IC={ensemble_validation_daily_ic.mean():.5f} | "
        f"95% CI [{ensemble_validation_bootstrap['ci_95_lower']:.5f}, "
        f"{ensemble_validation_bootstrap['ci_95_upper']:.5f}]"
    )
    if not config.evaluate_test:
        if sde_target_scale is not None:
            write_sde_diagnostics(
                config,
                output_dir,
                validation_predictions,
                None,
                sde_target_scale,
                sde_baseline_variance,
                include_oos=False,
                include_sizing=False,
            )
        print("Validation-only run finished (no gate fitting, no OOS evaluation).")
        return {"validation_only": ensemble_summary}
    score_calibration_spec = {
        "enabled": False,
        "reason": "Rank-only execution: model-score magnitude is not interpreted as an expected return.",
    }
    (output_dir / "return_score_calibration.json").write_text(
        json.dumps(score_calibration_spec, indent=2), encoding="utf-8"
    )
    gate_costs = tuple(sorted({0.0, config.profitability_gate_cost_bps}))
    profitability_gates_by_cost: dict[float, ProfitabilityGate] = {}
    profitability_gate_specs: dict[str, dict[str, object]] = {}
    for gate_cost_bps in gate_costs:
        gate, gate_validation_actions, gate_diagnostics = fit_profitability_gate(
            validation_predictions,
            config.holding_minutes,
            config.z_threshold,
            gate_cost_bps,
            config.profitability_gate_min_probability,
            config.profitability_gate_l2,
            config.session_end_policy,
            config.target_smoothing_fraction,
            config.sector_rebalance_buffer_turnover,
        )
        cost_key = f"{gate_cost_bps:g} bps"
        profitability_gates_by_cost[gate_cost_bps] = gate
        gate_validation_actions.to_parquet(
            output_dir / f"profitability_gate_validation_actions_{gate_cost_bps:g}bps.parquet", index=False
        )
        gate_spec = {
            **gate_diagnostics,
            "intercept": gate.intercept,
            "feature_mean": {
                name: float(value)
                for name, value in zip(gate.feature_names, gate.feature_mean, strict=True)
            },
            "feature_scale": {
                name: float(value)
                for name, value in zip(gate.feature_names, gate.feature_scale, strict=True)
            },
            "coefficients": {
                name: float(value)
                for name, value in zip(gate.feature_names, gate.coefficients, strict=True)
            },
        }
        profitability_gate_specs[cost_key] = gate_spec
        (output_dir / f"profitability_gate_{gate_cost_bps:g}bps.json").write_text(
            json.dumps(gate_spec, indent=2), encoding="utf-8"
        )
    # Keep the unqualified files as aliases for the friction-aware primary gate.
    primary_gate_spec = profitability_gate_specs[f"{config.profitability_gate_cost_bps:g} bps"]
    (output_dir / "profitability_gate.json").write_text(
        json.dumps(primary_gate_spec, indent=2), encoding="utf-8"
    )
    rank_execution_rule = {
        "fit_split": "rank checkpoints selected on validation; profitability gate fitted only to frozen validation predictions",
        "signal_selection": "within-sector z-score threshold; no score-to-return calibration",
        "z_threshold": config.z_threshold,
        "profitability_gate": {
            "decision_rule": "execute an optional proposed rebalance only when P(incremental net PnL > 0) >= min_probability",
            "cost_specific_paths": {
                f"{cost:g} bps": {"gate_fit_cost_bps": cost, "min_probability": config.profitability_gate_min_probability}
                for cost in gate_costs
            },
            "min_probability": config.profitability_gate_min_probability,
            "l2": config.profitability_gate_l2,
        },
        "execution_model": (
            "consecutive intraday target weights; zero cost for an unchanged weight; daily terminal flattening"
            if config.session_end_policy == "flatten_daily"
            else "consecutive target weights; zero cost for an unchanged weight; marked between-rebalance bridge returns"
        ),
        "session_end_policy": config.session_end_policy,
        "sector_budget_rule": "fixed gross budgets proportional to eligible sector symbol counts; inactive budgets remain cash",
        "target_smoothing_fraction": config.target_smoothing_fraction,
        "sector_rebalance_buffer_turnover": config.sector_rebalance_buffer_turnover,
        "cost_sensitivity_rule": "0 bp and configured-cost scenarios each use a gate fitted under their own cost, producing intentionally distinct no-friction and friction-aware target paths",
    }
    (output_dir / "rank_execution_rule.json").write_text(
        json.dumps(rank_execution_rule, indent=2), encoding="utf-8"
    )
    expected_alpha_gate = {
        "enabled": False,
        "reason": "The former score-to-return expected-alpha gate is replaced by a validation-fitted incremental-profitability probability gate.",
    }
    (output_dir / "expected_alpha_gate.json").write_text(
        json.dumps(expected_alpha_gate, indent=2), encoding="utf-8"
    )
    print(
        "Cost-specific profitability-gated execution | "
        f"sector z-threshold={config.z_threshold:.2f}, "
        f"smoothing={config.target_smoothing_fraction:.2f}, "
        f"sector turnover buffer={config.sector_rebalance_buffer_turnover:.3f}, "
        f"paths={list(gate_costs)} bps, "
        f"p_min={config.profitability_gate_min_probability:.2f}"
    )
    predictions = test_predictions.copy()
    predictions.to_parquet(output_dir / "oos_predictions.parquet", index=False)
    metrics, period_returns = evaluate_cost_sensitivity(
        predictions,
        config.holding_minutes,
        config.z_threshold,
        session_end_policy=config.session_end_policy,
        target_smoothing_fraction=config.target_smoothing_fraction,
        sector_rebalance_buffer_turnover=config.sector_rebalance_buffer_turnover,
        profitability_gates_by_cost=profitability_gates_by_cost,
        profitability_gate_min_probability=config.profitability_gate_min_probability,
        costs_bps=gate_costs,
    )
    oos_daily_ic = _daily_rank_ic_from_prediction_frame(predictions)
    oos_bootstrap = block_bootstrap_mean_ic(
        oos_daily_ic, config.bootstrap_iterations, config.seed + 1
    )
    metrics["rank_ic_block_bootstrap"] = oos_bootstrap
    metrics["rank_execution_rule"] = rank_execution_rule
    metrics["profitability_gates"] = profitability_gate_specs
    oos_daily_ic.rename_axis("session").reset_index().to_csv(
        output_dir / "oos_daily_rank_ic.csv", index=False
    )
    period_returns.to_parquet(output_dir / "oos_holding_period_returns.parquet", index=False)
    (output_dir / "oos_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        "\nOOS evaluation (validation-fitted profitability gate; "
        f"session policy={config.session_end_policy}):"
    )
    for cost, values in metrics.items():
        print(f"  {cost}: {values}")
    if sde_target_scale is not None:
        write_sde_diagnostics(
            config,
            output_dir,
            validation_predictions,
            predictions,
            sde_target_scale,
            sde_baseline_variance,
        )
    return metrics


def uncertainty_scaled_score_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Alternative execution scores built from the SDE predictive law.

    ``rank_score``: the ensemble's sector-standardised score (current default).
    ``predictive_mean``: the calibrated Gaussian mean.
    ``mean_over_std``: predictive mean divided by predictive standard deviation, i.e.
    a per-name forecast information ratio; within-sector z-scoring then sizes
    low-uncertainty names more heavily.
    ``rank_score_over_std``: the rank score divided by predictive standard deviation.
    """
    std = np.sqrt(np.maximum(frame["predicted_variance_std"].to_numpy(dtype=float), 1e-12))
    variants = {
        "rank_score": frame["prediction"].to_numpy(dtype=float),
        "predictive_mean": frame["predicted_mean_std"].to_numpy(dtype=float),
        "mean_over_std": frame["predicted_mean_std"].to_numpy(dtype=float) / std,
        # Best ranking (seed-standardised ensemble score) scaled by SDE uncertainty.
        "rank_score_over_std": frame["prediction"].to_numpy(dtype=float) / std,
    }
    result: dict[str, pd.DataFrame] = {}
    for name, score in variants.items():
        candidate = frame.copy()
        candidate["prediction"] = score
        result[name] = candidate
    return result


def write_sde_diagnostics(
    config: RunConfig,
    output_dir: Path,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame | None,
    target_scale: float,
    baseline_variance: dict[str, float] | None,
    include_oos: bool = True,
    file_suffix: str = "",
    include_sizing: bool = True,
) -> dict[str, object]:
    """Distribution calibration plus an ungated mean-vs-uncertainty sizing cost curve.

    ``include_oos=False`` evaluates validation only, for tuning without touching OOS.
    """
    splits: list[tuple[str, pd.DataFrame]] = [("validation", validation_predictions)]
    if include_oos:
        if test_predictions is None:
            raise ValueError("include_oos=True needs test predictions.")
        splits.append(("oos", test_predictions))
    distribution = {
        split_name: evaluate_predictive_distribution(
            frame,
            target_scale,
            config.sde_target_clip,
            baseline_variance,
            jump_mode=config.sde_jump_mode,
            max_jumps=config.sde_jump_max_count,
        )
        for split_name, frame in splits
    }
    (output_dir / f"sde_distribution_metrics{file_suffix}.json").write_text(
        json.dumps(distribution, indent=2), encoding="utf-8"
    )

    if not include_sizing:
        for split_name, _frame in splits:
            split_distribution = distribution[split_name]
            print(
                f"\nSDE predictive distribution ({split_name}): NLL={split_distribution['mean_nll']:.4f}, "
                f"coverage={split_distribution['central_interval_coverage']}"
            )
        return {"distribution": distribution}
    costs = tuple(sorted(set(float(cost) for cost in config.sde_cost_curve_bps)))
    comparison: dict[str, object] = {
        "note": (
            "Ungated execution for every score so the comparison isolates the score. "
            "Validation results are the basis for choosing a score; OOS is reported once."
        ),
        "costs_bps": list(costs),
    }
    for split_name, frame in splits:
        split_result: dict[str, object] = {}
        print(f"  Sizing comparison on {split_name}: 3 scores x {len(costs)} costs (ungated)...", flush=True)
        for score_name, scored in uncertainty_scaled_score_frames(frame).items():
            daily_ic = _daily_rank_ic_from_prediction_frame(scored)
            ungated, _ = evaluate_cost_sensitivity(
                scored,
                config.holding_minutes,
                config.z_threshold,
                session_end_policy=config.session_end_policy,
                target_smoothing_fraction=config.target_smoothing_fraction,
                sector_rebalance_buffer_turnover=config.sector_rebalance_buffer_turnover,
                costs_bps=costs,
            )
            split_result[score_name] = {
                "rank_ic_block_bootstrap": block_bootstrap_mean_ic(
                    daily_ic, config.bootstrap_iterations, config.seed + 7
                ),
                "cost_curve": {
                    key: {
                        "annualized_sharpe": values.get("annualized_sharpe"),
                        "mean_return_per_holding_period": values.get("mean_return_per_holding_period"),
                        "mean_one_way_turnover": values.get("mean_one_way_turnover"),
                        "max_drawdown_pct": values.get("max_drawdown_pct"),
                    }
                    for key, values in ungated.items()
                    if key.endswith("bps")
                },
            }
        comparison[split_name] = split_result
    (output_dir / f"sde_uncertainty_sizing_comparison{file_suffix}.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    for split_name, _frame in splits:
        split_distribution = distribution[split_name]
        print(
            f"\nSDE predictive distribution ({split_name}): "
            f"NLL={split_distribution['mean_nll']:.4f}, "
            f"coverage={split_distribution['central_interval_coverage']}, "
            f"MZ slope={split_distribution['mincer_zarnowitz']['slope']:.3f}"
        )
        if "nll_improvement_vs_baseline" in split_distribution:
            print(f"  NLL gain vs per-symbol baseline: {split_distribution['nll_improvement_vs_baseline']}")
        if "jumps" in split_distribution:
            jumps = split_distribution["jumps"]
            print(
                f"  Jumps: mean lambda {jumps['mean_intensity_per_period']:.4f} per period, "
                f"mean jump std {jumps['mean_jump_std']:.3f}, "
                f"jump share of variance {jumps['mean_jump_share_of_total_variance']:.2f}, "
                f"posterior P(jump) by |y| {jumps['posterior_jump_probability_by_abs_target']}"
            )
    for split_name, _frame in splits:
        for score_name, values in comparison[split_name].items():
            curve = values["cost_curve"]
            sharpe = {
                key: round(float(item["annualized_sharpe"]), 2) if item["annualized_sharpe"] is not None else None
                for key, item in curve.items()
            }
            ic = values["rank_ic_block_bootstrap"]["mean_daily_rank_ic"]
            print(f"  {split_name:>10} | {score_name:<15} | daily IC {ic:.4f} | ungated Sharpe {sharpe}")
    return {"distribution": distribution, "uncertainty_sizing": comparison}


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train the Massive minute-bar signature-affine alpha model.")
    parser.add_argument("--data-dir", default=RunConfig.data_dir)
    parser.add_argument("--output-dir", default=RunConfig.output_dir)
    parser.add_argument("--window-minutes", type=int, default=RunConfig.window_minutes)
    parser.add_argument("--holding-minutes", type=int, default=RunConfig.holding_minutes)
    parser.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=RunConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=RunConfig.patience)
    parser.add_argument("--warmup-epochs", type=int, default=RunConfig.warmup_epochs)
    parser.add_argument("--min-labelled-symbols", type=int, default=RunConfig.min_labelled_symbols)
    parser.add_argument(
        "--train-sample-stride-minutes", type=int, default=RunConfig.train_sample_stride_minutes
    )
    parser.add_argument(
        "--evaluation-sample-stride-minutes", type=int, default=RunConfig.evaluation_sample_stride_minutes
    )
    parser.add_argument(
        "--price-price-area-pca-components", type=int, default=RunConfig.price_price_pca_components
    )
    parser.add_argument(
        "--price-volume-area-pca-components",
        "--cross-area-pca-components",
        dest="price_volume_area_pca_components",
        type=int,
        default=RunConfig.price_volume_pca_components,
    )
    parser.add_argument(
        "--sector-median-shrinkage-count",
        type=float,
        default=RunConfig.sector_median_shrinkage_count,
    )
    parser.add_argument("--z-threshold", type=float, default=RunConfig.z_threshold)
    parser.add_argument(
        "--session-end-policy",
        choices=("flatten_daily", "carry_to_next_rebalance"),
        default=RunConfig.session_end_policy,
    )
    parser.add_argument(
        "--target-smoothing-fraction", type=float, default=RunConfig.target_smoothing_fraction
    )
    parser.add_argument(
        "--sector-rebalance-buffer-turnover",
        type=float,
        default=RunConfig.sector_rebalance_buffer_turnover,
    )
    parser.add_argument(
        "--profitability-gate-cost-bps", type=float, default=RunConfig.profitability_gate_cost_bps
    )
    parser.add_argument(
        "--profitability-gate-min-probability",
        type=float,
        default=RunConfig.profitability_gate_min_probability,
    )
    parser.add_argument(
        "--profitability-gate-l2", type=float, default=RunConfig.profitability_gate_l2
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=RunConfig.bootstrap_iterations)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--ensemble-seeds", nargs="+", type=int, default=list(RunConfig.ensemble_seeds))
    parser.add_argument("--latent-dim", type=int, default=RunConfig.latent_dim)
    parser.add_argument("--affine-hidden-dim", type=int, default=RunConfig.affine_hidden_dim)
    parser.add_argument(
        "--sde-diffusion-rank",
        type=int,
        default=RunConfig.sde_diffusion_rank,
        help="Rank of the latent diffusion loading; 0 runs the deterministic affine-mean ablation.",
    )
    parser.add_argument("--sde-diffusion-floor", type=float, default=RunConfig.sde_diffusion_floor)
    parser.add_argument("--sde-nll-weight", type=float, default=RunConfig.sde_nll_weight)
    parser.add_argument("--sde-target-clip", type=float, default=RunConfig.sde_target_clip)
    parser.add_argument(
        "--sde-cost-curve-bps", nargs="+", type=float, default=list(RunConfig.sde_cost_curve_bps)
    )
    parser.add_argument("--sde-jump-mode", choices=JUMP_MODES, default=RunConfig.sde_jump_mode)
    parser.add_argument("--sde-jump-max-count", type=int, default=RunConfig.sde_jump_max_count)
    parser.add_argument(
        "--sde-jump-initial-intensity", type=float, default=RunConfig.sde_jump_initial_intensity
    )
    parser.add_argument("--sde-jump-initial-std", type=float, default=RunConfig.sde_jump_initial_std)
    parser.add_argument("--no-etf-context", action="store_true")
    parser.add_argument("--stock-embedding-dim", type=int, default=RunConfig.stock_embedding_dim)
    parser.add_argument("--cross-attention-layers", type=int, default=RunConfig.cross_attention_layers)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Tuning mode: never score the test split and skip the OOS evaluation.",
    )
    args = parser.parse_args()
    return RunConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_minutes=args.window_minutes,
        holding_minutes=args.holding_minutes,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        min_labelled_symbols=args.min_labelled_symbols,
        train_sample_stride_minutes=args.train_sample_stride_minutes,
        evaluation_sample_stride_minutes=args.evaluation_sample_stride_minutes,
        price_price_pca_components=args.price_price_area_pca_components,
        price_volume_pca_components=args.price_volume_area_pca_components,
        sector_median_shrinkage_count=args.sector_median_shrinkage_count,
        z_threshold=args.z_threshold,
        session_end_policy=args.session_end_policy,
        target_smoothing_fraction=args.target_smoothing_fraction,
        sector_rebalance_buffer_turnover=args.sector_rebalance_buffer_turnover,
        profitability_gate_cost_bps=args.profitability_gate_cost_bps,
        profitability_gate_min_probability=args.profitability_gate_min_probability,
        profitability_gate_l2=args.profitability_gate_l2,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        ensemble_seeds=tuple(args.ensemble_seeds),
        latent_dim=args.latent_dim,
        affine_hidden_dim=args.affine_hidden_dim,
        sde_diffusion_rank=args.sde_diffusion_rank,
        sde_diffusion_floor=args.sde_diffusion_floor,
        sde_nll_weight=args.sde_nll_weight,
        sde_target_clip=args.sde_target_clip,
        sde_cost_curve_bps=tuple(args.sde_cost_curve_bps),
        sde_jump_mode=args.sde_jump_mode,
        sde_jump_max_count=args.sde_jump_max_count,
        sde_jump_initial_intensity=args.sde_jump_initial_intensity,
        sde_jump_initial_std=args.sde_jump_initial_std,
        evaluate_test=not args.validation_only,
        use_etf_context=not args.no_etf_context,
        stock_embedding_dim=args.stock_embedding_dim,
        cross_attention_layers=args.cross_attention_layers,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
