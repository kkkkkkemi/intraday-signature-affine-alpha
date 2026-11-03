"""Train a signature-conditioned affine latent mean model on minute OHLCV bars.

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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
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
)
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
CLV_DENOMINATOR_EPS = 1e-12
RAW_COLUMNS = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap"}
# Fixed GICS-style sectors for the configured 25-stock research universe.  They
# are part of the feature specification rather than a live vendor lookup, so a
# future universe change must explicitly add its sector labels.
SECTOR_BY_SYMBOL = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    "AMD": "Information Technology",
    "INTC": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "V": "Information Technology",
    "MA": "Information Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "JNJ": "Health Care",
    "UNH": "Health Care",
    "XOM": "Energy",
}
SECTOR_MEDIAN_SHRINKAGE_COUNT = 5.0


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
        frames.append(frame[list(RAW_COLUMNS)].copy())

    bars = pd.concat(frames, ignore_index=True)
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "vwap"):
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


def engineer_minute_features(
    bars: pd.DataFrame,
    holding_minutes: int,
    sector_median_shrinkage_count: float = SECTOR_MEDIAN_SHRINKAGE_COUNT,
) -> pd.DataFrame:
    """Create causal OHLCV features and labels valid only at real trade bars."""
    prepared = bars.copy()
    timestamp_ny = prepared["timestamp"].dt.tz_convert("America/New_York")
    prepared["session"] = timestamp_ny.dt.date.astype(str)
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
    frame["input_available"] = frame[list(FEATURE_COLUMNS)].notna().all(axis=1)
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

    def prediction_records(self, sample_indices: Iterable[int], prediction: np.ndarray) -> list[dict[str, object]]:
        """Attach predictions without assuming every symbol has a valid label."""
        records: list[dict[str, object]] = []
        for batch_index, sample_index in enumerate(sample_indices):
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
                        "prediction": float(prediction[batch_index, symbol_index]),
                    }
                )
        return records

    def cross_ticker_area_matrix(self, batch_size: int = 256) -> np.ndarray:
        """Materialise raw selected area vectors for PCA, using this split only."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        offsets = np.arange(-self.window_minutes + 1, 1, dtype=np.int64)
        batches: list[np.ndarray] = []
        for start in range(0, len(self.sample_end_indices), batch_size):
            ends = self.sample_end_indices[start : start + batch_size]
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


def _fit_standardized_pca(
    values: np.ndarray, components: int, feature_group: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardisation and PCA on one training-only feature group."""
    maximum_components = min(values.shape)
    if components < 1 or components > maximum_components:
        raise ValueError(f"{feature_group} PCA components must be in [1, {maximum_components}].")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardised = (values - mean) / scale
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
    """PCA-compress both cross-ticker area families using training windows only."""
    raw_areas = dataset.cross_ticker_area_matrix()
    if not np.isfinite(raw_areas).all():
        raise ValueError("Training cross-ticker areas contain non-finite values.")
    num_symbols = len(dataset.symbols)
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
    """

    def __init__(
        self,
        input_channels: int,
        signature_depth: int,
        latent_dim: int,
        hidden_dim: int,
        num_symbols: int,
        cross_area_pca: CrossTickerAreaPCA,
    ) -> None:
        super().__init__()
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
        self.context_dim = (
            self.signature_dim
            + self.state_context_dim
            + self.directional_multiscale_context_dim
            + self.price_price_pca_components
            + self.price_volume_pca_components
        )
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

    def forward(self, path: torch.Tensor) -> torch.Tensor:
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
                cross_context.unsqueeze(1)
                .expand(-1, num_symbols, -1)
                .reshape(
                    batch_size * num_symbols,
                    self.price_price_pca_components + self.price_volume_pca_components,
                ),
            ],
            dim=1,
        )
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
        return self.readout(mean_endpoint).view(batch_size, num_symbols)


class CrossSectionalRankLoss(nn.Module):
    """Pearson correlation with hard-rank normal-score targets (Spearman surrogate)."""

    def __init__(self, min_labelled_symbols: int) -> None:
        super().__init__()
        self.min_labelled_symbols = min_labelled_symbols

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        mask = target_mask.to(dtype=prediction.dtype)
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


def _daily_rank_ic_from_prediction_frame(
    frame: pd.DataFrame, target_column: str = "rank_target_return"
) -> pd.Series:
    """Equal-weight each session after Rank IC against the requested forward target."""
    if target_column not in frame.columns:
        raise ValueError(f"Prediction frame is missing Rank-IC target column: {target_column}")
    records: list[tuple[str, float]] = []
    for (session, _timestamp), group in frame.groupby(["session", "timestamp"], sort=True):
        value = _rank_ic(group["prediction"].to_numpy(), group[target_column].to_numpy())
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
) -> tuple[float, pd.Series]:
    model.eval()
    records: list[tuple[str, float]] = []
    if not isinstance(loader.dataset, CrossSectionalMinuteDataset):
        raise TypeError("Validation loader must contain CrossSectionalMinuteDataset.")
    dataset = loader.dataset
    with torch.no_grad():
        for path, target, target_mask, indices in loader:
            path = _to_device(path, device, non_blocking)
            with _autocast_context(device, amp_enabled):
                prediction = model(path)
            prediction = prediction.cpu().numpy()
            target_np = target.numpy()
            mask_np = target_mask.numpy()
            for batch_index, sample_index in enumerate(indices.tolist()):
                value = _rank_ic(prediction[batch_index][mask_np[batch_index]], target_np[batch_index][mask_np[batch_index]])
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
    with torch.no_grad():
        for path, _target, _target_mask, indices in loader:
            path = _to_device(path, device, non_blocking)
            with _autocast_context(device, amp_enabled):
                prediction = model(path)
            prediction = prediction.cpu().numpy()
            records.extend(dataset.prediction_records(indices.tolist(), prediction))
    return pd.DataFrame.from_records(records).sort_values(["session", "timestamp", "symbol"]).reset_index(drop=True)


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
    test_predictions: pd.DataFrame


def _train_one_seed(
    config: RunConfig,
    seed: int,
    output_dir: Path,
    train_set: CrossSectionalMinuteDataset,
    validation_set: CrossSectionalMinuteDataset,
    test_set: CrossSectionalMinuteDataset,
    cross_area_pca: CrossTickerAreaPCA,
    device: torch.device,
    use_amp: bool,
    use_pinned_memory: bool,
    workers: int,
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
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_function = CrossSectionalRankLoss(config.min_labelled_symbols).to(device)
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

    best_path = output_dir / f"seed_{seed}_best_signature_affine_mean.pt"
    best_validation_ic = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        for path, target, target_mask, _ in train_loader:
            path = _to_device(path, device, use_pinned_memory)
            target = _to_device(target, device, use_pinned_memory)
            target_mask = _to_device(target_mask, device, use_pinned_memory)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp):
                prediction = model(path)
                loss = loss_function(prediction, target, target_mask)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            losses.append(float(loss.item()))

        validation_ic, validation_daily_ic = _validate(
            model,
            validation_loader,
            device,
            use_amp,
            use_pinned_memory,
        )
        print(
            f"Seed {seed} | Epoch {epoch:03d} | train normal-score rank loss {np.mean(losses):.5f} | "
            f"validation sector-neutral Rank IC {validation_ic:.5f}"
        )
        if epoch > config.warmup_epochs and not np.isnan(validation_ic) and validation_ic > best_validation_ic:
            best_validation_ic = validation_ic
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
    test_predictions = make_prediction_frame(model, test_set, device, use_amp, use_pinned_memory)
    validation_predictions.to_parquet(output_dir / f"seed_{seed}_validation_predictions.parquet", index=False)
    test_predictions.to_parquet(output_dir / f"seed_{seed}_oos_predictions.parquet", index=False)
    (output_dir / f"seed_{seed}_training_summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "checkpoint": best_path.name,
                "best_epoch": best_epoch,
                "best_validation_sector_neutral_rank_ic": best_validation_ic,
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
    features = engineer_minute_features(
        bars,
        config.holding_minutes,
        sector_median_shrinkage_count=config.sector_median_shrinkage_count,
    )
    print(
        "Relative-price benchmark: sector median shrunk to universe median "
        f"with k={config.sector_median_shrinkage_count:g}; "
        f"{len(relative_return_spec['sector_members'])} fixed sectors."
    )
    write_missing_bar_diagnostics(features, output_dir)
    train_raw, validation_raw, test_raw = split_by_session(features, config.train_ratio, config.val_ratio)
    scaler = fit_feature_scaler(train_raw)
    (output_dir / "feature_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    train = apply_feature_scaler(train_raw, scaler)
    validation = apply_feature_scaler(validation_raw, scaler)
    test = apply_feature_scaler(test_raw, scaler)

    train_set = CrossSectionalMinuteDataset(
        train, config.window_minutes, config.min_labelled_symbols, config.train_sample_stride_minutes
    )
    validation_set = CrossSectionalMinuteDataset(
        validation, config.window_minutes, config.min_labelled_symbols, config.evaluation_sample_stride_minutes
    )
    test_set = CrossSectionalMinuteDataset(
        test, config.window_minutes, config.min_labelled_symbols, config.evaluation_sample_stride_minutes
    )
    if not (train_set.symbols == validation_set.symbols == test_set.symbols):
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
        f"Dataset: {len(train_set):,} train / {len(validation_set):,} validation / {len(test_set):,} test windows; "
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
        )
        for seed in training_seeds
    ]
    validation_predictions = average_seed_prediction_frames(
        [result.validation_predictions for result in seed_results], training_seeds
    )
    test_predictions = average_seed_prediction_frames(
        [result.test_predictions for result in seed_results], training_seeds
    )
    validation_predictions.to_parquet(output_dir / "ensemble_validation_predictions.parquet", index=False)
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
    return metrics


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
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
