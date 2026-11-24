"""Train a grouped XGBoost ranker beside the existing signature-affine model.

Run directly from the same Python interpreter used for train_model.py. The neural
training code and its artifacts are never modified by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError("Install xgboost in the Python environment used to launch train_xgboost.py.") from exc

from minute_alpha import (
    DIRECTIONAL_MULTISCALE_COLUMNS,
    EXTRA_CONTEXT_ENDPOINT_COLUMNS,
    EXTRA_CONTEXT_MEAN_COLUMNS,
    FEATURE_COLUMNS,
    HAS_TRADE_FEATURE_INDEX,
    INITIAL_STATE_ANCHOR_COLUMNS,
    INITIAL_STATE_ANCHOR_INDICES,
    INITIAL_STATE_ANCHOR_MINUTES,
    INITIAL_STATE_PATH_STATISTICS,
    RANGE_FEATURE_INDEX,
    REALIZED_VOLATILITY_FEATURE_INDEX,
    RELATIVE_PRICE_FEATURE_INDEX,
    SELECTIVE_SIGNATURE_DIM,
    SIGNATURE_PATH_CHANNELS,
    STALENESS_FEATURE_INDEX,
    VOLUME_FEATURE_INDEX,
    VWAP_DEVIATION_FEATURE_INDEX,
    CrossSectionalMinuteDataset,
    SignatureConditionedAffineMeanModel,
    _daily_rank_ic_from_prediction_frame,
    apply_feature_scaler,
    average_seed_prediction_frames,
    block_bootstrap_mean_ic,
    engineer_minute_features,
    evaluate_cost_sensitivity,
    fit_cross_ticker_area_pca,
    fit_feature_scaler,
    fit_profitability_gate,
    apply_feature_drops,
    load_etf_bars,
    load_minute_bars,
    split_by_session,
)
from project_settings import (
    DROP_FEATURE_GROUPS,
    BOOTSTRAP_ITERATIONS,
    DATA_DIR,
    ENSEMBLE_SEEDS,
    EVALUATE_TEST,
    EVALUATION_SAMPLE_STRIDE_MINUTES,
    HOLDING_MINUTES,
    MIN_LABELLED_SYMBOLS,
    OUTPUT_DIR,
    PRICE_PRICE_AREA_PCA_COMPONENTS,
    PRICE_VOLUME_AREA_PCA_COMPONENTS,
    PROFITABILITY_GATE_COST_BPS,
    PROFITABILITY_GATE_L2,
    PROFITABILITY_GATE_MIN_PROBABILITY,
    SECTOR_MEDIAN_SHRINKAGE_COUNT,
    SECTOR_REBALANCE_BUFFER_TURNOVER,
    SEED,
    SESSION_END_POLICY,
    TARGET_WEIGHT_SMOOTHING_FRACTION,
    TICKERS,
    TRAIN_RATIO,
    TRAIN_SAMPLE_STRIDE_MINUTES,
    UNIVERSE,
    USE_ETF_CONTEXT,
    VALIDATION_RATIO,
    WINDOW_MINUTES,
    Z_SCORE_THRESHOLD,
)


# These settings affect only the new baseline. Adjust them here without changing
# project_settings.py or the signature-affine model.
PROJECT_DIR = Path(__file__).resolve().parent
# One folder per universe and evaluation stride, so earlier results are never
# overwritten (the original 25-name run lives in artifacts/xgboost_comparison).
XGB_OUTPUT_DIR = Path(OUTPUT_DIR).parent / f"xgboost_{UNIVERSE}_ev{EVALUATION_SAMPLE_STRIDE_MINUTES}"
if DROP_FEATURE_GROUPS:
    XGB_OUTPUT_DIR = XGB_OUTPUT_DIR.with_name(XGB_OUTPUT_DIR.name + "_drop-" + "+".join(DROP_FEATURE_GROUPS))
XGB_BOOSTING_ROUNDS = 400
XGB_CHECKPOINT_EVERY = 25
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_MIN_CHILD_WEIGHT = 20.0
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = 10.0
XGB_THREADS = 8
FEATURE_BATCH_SIZE = 32


def _project_path(value: str | Path) -> Path:
    """Resolve configured relative paths beside this script, not the launch directory."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _feature_names(price_price_components: int, price_volume_components: int) -> list[str]:
    state_names = [
        "time_end", "has_trade_mean", "has_trade_last", "has_trade_transition_rate",
        "staleness_mean", "staleness_max", "staleness_last",
    ]
    return (
        [f"signature_level_one_{i}" for i in range(SIGNATURE_PATH_CHANNELS)]
        + [f"signature_area_{i}" for i in range(SELECTIVE_SIGNATURE_DIM - SIGNATURE_PATH_CHANNELS)]
        + state_names
        + list(DIRECTIONAL_MULTISCALE_COLUMNS)
        + [f"{name}_last" for name in EXTRA_CONTEXT_ENDPOINT_COLUMNS]
        + [f"{name}_window_mean" for name in EXTRA_CONTEXT_MEAN_COLUMNS]
        + [f"price_price_pca_{i}" for i in range(price_price_components)]
        + [f"price_volume_pca_{i}" for i in range(price_volume_components)]
        + [f"initial_anchor_{name}" for name in INITIAL_STATE_ANCHOR_COLUMNS]
        + list(INITIAL_STATE_PATH_STATISTICS)
    )


def _initial_state_inputs(path: torch.Tensor) -> torch.Tensor:
    """Expose the raw inputs to the neural z0 encoder, before its learned MLP."""
    anchor = path[:, :INITIAL_STATE_ANCHOR_MINUTES, INITIAL_STATE_ANCHOR_INDICES].mean(dim=1)
    price = path[:, :, RELATIVE_PRICE_FEATURE_INDEX]
    price_increment = price[:, 1:] - price[:, :-1]
    statistics = torch.stack(
        [
            price_increment.square().sum(dim=1),
            path[:, :, VWAP_DEVIATION_FEATURE_INDEX].abs().mean(dim=1),
            path[:, :, VOLUME_FEATURE_INDEX].std(dim=1, unbiased=False),
            path[:, :, RANGE_FEATURE_INDEX].mean(dim=1),
            path[:, :, REALIZED_VOLATILITY_FEATURE_INDEX].mean(dim=1),
            path[:, :, HAS_TRADE_FEATURE_INDEX].mean(dim=1),
            path[:, :, STALENESS_FEATURE_INDEX].max(dim=1).values,
        ],
        dim=1,
    )
    return torch.cat([anchor, statistics], dim=1)


def _matrix_and_metadata(
    dataset: CrossSectionalMinuteDataset,
    representation: SignatureConditionedAffineMeanModel,
    device: torch.device,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    loader = DataLoader(dataset, batch_size=FEATURE_BATCH_SIZE, shuffle=False, num_workers=0)
    matrices: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    representation.eval()
    with torch.no_grad():
        for path, _target, mask, indices in loader:
            path = path.to(device)
            batch_size, num_symbols, path_length, channels = path.shape
            individual = path.reshape(batch_size * num_symbols, path_length, channels)
            shared_context = representation._project_cross_ticker_areas(path)
            shared_context = shared_context[:, None, :].expand(-1, num_symbols, -1)
            values = torch.cat(
                [
                    representation._compact_signature(individual).reshape(batch_size, num_symbols, -1),
                    representation._state_context(individual).reshape(batch_size, num_symbols, -1),
                    representation._directional_multiscale_context(individual).reshape(batch_size, num_symbols, -1),
                    representation._extra_context(individual).reshape(batch_size, num_symbols, -1),
                    shared_context,
                    _initial_state_inputs(individual).reshape(batch_size, num_symbols, -1),
                ],
                dim=-1,
            )
            matrices.append(values[mask.to(device)].cpu().numpy().astype(np.float32, copy=False))
            records.extend(dataset.metadata(indices.tolist()))
    matrix = np.concatenate(matrices, axis=0)
    metadata = pd.DataFrame.from_records(records)
    if matrix.shape != (len(metadata), len(feature_names)) or not np.isfinite(matrix).all():
        raise ValueError("XGBoost feature and label rows do not align or contain non-finite values.")
    if metadata.duplicated(["session", "timestamp", "symbol"]).any():
        raise ValueError("Duplicate symbol/signal keys in the XGBoost dataset.")
    return matrix, metadata


def _dmatrix(
    matrix: np.ndarray, metadata: pd.DataFrame, feature_names: list[str], with_labels: bool
) -> xgb.DMatrix:
    """Use each signal timestamp as one cross-sectional ranking query."""
    group_index = pd.MultiIndex.from_frame(metadata[["session", "timestamp"]])
    if not group_index.is_monotonic_increasing:
        raise ValueError("XGBoost ranking rows must be ordered by session and signal time.")
    relevance = None
    if with_labels:
        relevance = (
            metadata.groupby(["session", "timestamp"], sort=False)["rank_target_return"]
            .rank(method="first")
            .to_numpy(dtype=np.float32) - 1.0
        )
    group_sizes = metadata.groupby(["session", "timestamp"], sort=False).size().to_numpy(dtype=np.uint32)
    if group_sizes.sum() != len(metadata) or (group_sizes < MIN_LABELLED_SYMBOLS).any():
        raise ValueError("XGBoost ranking groups do not match the neural model's valid windows.")
    matrix_for_xgb = xgb.DMatrix(matrix, label=relevance, feature_names=feature_names)
    matrix_for_xgb.set_group(group_sizes)
    return matrix_for_xgb


def _prediction_frame(metadata: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    frame = metadata.copy()
    frame["prediction"] = np.asarray(predictions, dtype=float)
    return frame.sort_values(["session", "timestamp", "symbol"]).reset_index(drop=True)


def _fit_one_seed(
    seed: int,
    train_matrix: xgb.DMatrix,
    validation_matrix: xgb.DMatrix,
    test_matrix: xgb.DMatrix | None,
    validation_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame | None,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object], dict[str, float]]:
    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "tree_method": "hist",
        "lambdarank_pair_method": "mean",
        "lambdarank_num_pair_per_sample": 5,
        "max_depth": XGB_MAX_DEPTH,
        "eta": XGB_LEARNING_RATE,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "lambda": XGB_REG_LAMBDA,
        "nthread": XGB_THREADS,
        "seed": seed,
    }
    booster = xgb.train(params, train_matrix, num_boost_round=XGB_BOOSTING_ROUNDS, verbose_eval=False)
    checkpoints = sorted(set(range(XGB_CHECKPOINT_EVERY, XGB_BOOSTING_ROUNDS + 1, XGB_CHECKPOINT_EVERY)) | {XGB_BOOSTING_ROUNDS})
    best_ic = -float("inf")
    best_round = 0
    history: list[dict[str, float | int]] = []
    for round_count in checkpoints:
        predictions = booster.predict(validation_matrix, iteration_range=(0, round_count))
        daily_ic = _daily_rank_ic_from_prediction_frame(_prediction_frame(validation_metadata, predictions))
        ic = float(daily_ic.mean())
        history.append({"trees": round_count, "validation_mean_daily_rank_ic": ic})
        print(f"XGBoost seed {seed} | trees={round_count:03d} | validation daily Rank IC {ic:.5f}")
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_round = ic, round_count
    if best_round == 0:
        raise RuntimeError(f"XGBoost seed {seed} has no finite validation Rank IC checkpoint.")
    validation = _prediction_frame(
        validation_metadata, booster.predict(validation_matrix, iteration_range=(0, best_round))
    )
    test = None
    if test_matrix is not None and test_metadata is not None:
        test = _prediction_frame(test_metadata, booster.predict(test_matrix, iteration_range=(0, best_round)))
    booster.save_model(output_dir / f"xgboost_seed_{seed}.json")
    validation.to_parquet(output_dir / f"seed_{seed}_validation_predictions.parquet", index=False)
    if test is not None:
        test.to_parquet(output_dir / f"seed_{seed}_oos_predictions.parquet", index=False)
    summary = {"seed": seed, "best_trees": best_round, "best_validation_mean_daily_rank_ic": best_ic, "checkpoints": history}
    (output_dir / f"seed_{seed}_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return validation, test, summary, booster.get_score(importance_type="gain")


def _same_labels(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    keys = ["session", "timestamp", "symbol"]
    left, right = (frame.sort_values(keys).reset_index(drop=True) for frame in (left, right))
    if not left[keys].equals(right[keys]):
        return False
    return all(
        np.allclose(left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float), rtol=1e-5, atol=1e-7)
        for column in ("target_return", "rank_target_return")
    )


def _paired_daily_ic_comparison(xgb_frame: pd.DataFrame, neural_frame: pd.DataFrame) -> dict[str, object]:
    xgb_daily = _daily_rank_ic_from_prediction_frame(xgb_frame)
    neural_daily = _daily_rank_ic_from_prediction_frame(neural_frame)
    common_days = xgb_daily.index.intersection(neural_daily.index)
    difference = xgb_daily.loc[common_days] - neural_daily.loc[common_days]
    bootstrap = block_bootstrap_mean_ic(difference, BOOTSTRAP_ITERATIONS, SEED + 2)
    return {
        "matched_test_days": len(common_days),
        "xgboost_mean_daily_rank_ic": float(xgb_daily.loc[common_days].mean()),
        "neural_mean_daily_rank_ic": float(neural_daily.loc[common_days].mean()),
        "xgboost_minus_neural_daily_rank_ic": bootstrap,
    }


def _validation_only_comparison(
    val_prediction: pd.DataFrame, val_ic: dict[str, object], neural_dir: Path, output_dir: Path
) -> None:
    """Paired validation comparison with the neural run in OUTPUT_DIR; the test split is never read."""
    comparison: dict[str, object] = {"split": "validation", "xgboost_validation_rank_ic": val_ic}
    neural_path = neural_dir / "ensemble_validation_predictions.parquet"
    if neural_path.exists():
        neural_val = pd.read_parquet(neural_path)
        if _same_labels(val_prediction, neural_val):
            comparison["neural_validation_rank_ic"] = block_bootstrap_mean_ic(
                _daily_rank_ic_from_prediction_frame(neural_val), BOOTSTRAP_ITERATIONS, SEED
            )
            comparison["paired_validation_rank_ic"] = _paired_daily_ic_comparison(val_prediction, neural_val)
            print("Paired validation comparison with the neural run saved.")
        else:
            comparison["neural_comparison_status"] = (
                "Neural validation predictions use different signal keys or labels; rerun both on the same settings."
            )
            print(comparison["neural_comparison_status"])
    else:
        comparison["neural_comparison_status"] = f"No neural validation predictions in {neural_dir}."
    (output_dir / "comparison_validation.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"XGBoost validation mean daily Rank IC: {val_ic['mean_daily_rank_ic']:.5f} (test split not scored)")
    print(f"Results saved to {output_dir}")


def main() -> None:
    if XGB_BOOSTING_ROUNDS < 1 or XGB_CHECKPOINT_EVERY < 1:
        raise ValueError("XGBoost boosting rounds and checkpoint interval must be positive.")
    data_dir = _project_path(DATA_DIR)
    neural_dir = _project_path(OUTPUT_DIR)
    output_dir = _project_path(XGB_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "xgboost_grouped_pairwise_ranker",
        "xgboost_version": xgb.__version__,
        "source_neural_artifacts": str(neural_dir),
        "data_dir": str(data_dir),
        "symbols": TICKERS,
        "window_minutes": WINDOW_MINUTES,
        "holding_minutes": HOLDING_MINUTES,
        "train_sample_stride_minutes": TRAIN_SAMPLE_STRIDE_MINUTES,
        "evaluation_sample_stride_minutes": EVALUATION_SAMPLE_STRIDE_MINUTES,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "min_labelled_symbols": MIN_LABELLED_SYMBOLS,
        "sector_median_shrinkage_count": SECTOR_MEDIAN_SHRINKAGE_COUNT,
        "price_price_pca_components": PRICE_PRICE_AREA_PCA_COMPONENTS,
        "price_volume_pca_components": PRICE_VOLUME_AREA_PCA_COMPONENTS,
        "seeds": list(dict.fromkeys(ENSEMBLE_SEEDS)),
        "boosting_rounds": XGB_BOOSTING_ROUNDS,
        "checkpoint_every": XGB_CHECKPOINT_EVERY,
        "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "reg_lambda": XGB_REG_LAMBDA,
        "threads": XGB_THREADS,
        "z_score_threshold": Z_SCORE_THRESHOLD,
        "session_end_policy": SESSION_END_POLICY,
        "target_weight_smoothing_fraction": TARGET_WEIGHT_SMOOTHING_FRACTION,
        "sector_rebalance_buffer_turnover": SECTOR_REBALANCE_BUFFER_TURNOVER,
        "profitability_gate_cost_bps": PROFITABILITY_GATE_COST_BPS,
        "profitability_gate_min_probability": PROFITABILITY_GATE_MIN_PROBABILITY,
        "profitability_gate_l2": PROFITABILITY_GATE_L2,
        "rank_target": "within-sector forward excess return ordered within each timestamp",
        "training_objective": "grouped rank:pairwise; validation checkpoint selected by mean daily sector-neutral Rank IC",
        "universe": UNIVERSE,
        "evaluate_test": EVALUATE_TEST,
        "drop_feature_groups": list(DROP_FEATURE_GROUPS),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    bars = load_minute_bars(data_dir, TICKERS)
    etf_bars = load_etf_bars(data_dir) if USE_ETF_CONTEXT else None
    if USE_ETF_CONTEXT and etf_bars is None:
        raise FileNotFoundError(f"USE_ETF_CONTEXT=True but no ETF bars in {data_dir / 'etf'}; run download_data.py.")
    features = engineer_minute_features(bars, HOLDING_MINUTES, SECTOR_MEDIAN_SHRINKAGE_COUNT, etf_bars=etf_bars)
    features = apply_feature_drops(features, DROP_FEATURE_GROUPS)
    del etf_bars
    del bars
    train_raw, validation_raw, test_raw = split_by_session(features, TRAIN_RATIO, VALIDATION_RATIO)
    del features
    scaler = fit_feature_scaler(train_raw)
    (output_dir / "feature_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    train_set = CrossSectionalMinuteDataset(
        apply_feature_scaler(train_raw, scaler), WINDOW_MINUTES, MIN_LABELLED_SYMBOLS, TRAIN_SAMPLE_STRIDE_MINUTES
    )
    del train_raw
    validation_set = CrossSectionalMinuteDataset(
        apply_feature_scaler(validation_raw, scaler), WINDOW_MINUTES, MIN_LABELLED_SYMBOLS, EVALUATION_SAMPLE_STRIDE_MINUTES
    )
    del validation_raw
    test_set = (
        CrossSectionalMinuteDataset(
            apply_feature_scaler(test_raw, scaler), WINDOW_MINUTES, MIN_LABELLED_SYMBOLS, EVALUATION_SAMPLE_STRIDE_MINUTES
        )
        if EVALUATE_TEST
        else None
    )
    del test_raw
    if train_set.symbols != validation_set.symbols or (test_set is not None and train_set.symbols != test_set.symbols):
        raise ValueError("Train, validation, and test have different ticker universes.")
    print(
        f"XGBoost dataset: {len(train_set):,} train / {len(validation_set):,} validation / "
        f"{(len(test_set) if test_set is not None else 0):,} OOS windows (scored={test_set is not None}); "
        f"{len(train_set.symbols)} symbols."
    )
    pca = fit_cross_ticker_area_pca(train_set, PRICE_PRICE_AREA_PCA_COMPONENTS, PRICE_VOLUME_AREA_PCA_COMPONENTS)
    np.savez(
        output_dir / "cross_ticker_area_pca.npz",
        price_price_mean=pca.price_price_mean,
        price_price_scale=pca.price_price_scale,
        price_price_components=pca.price_price_components,
        price_volume_mean=pca.price_volume_mean,
        price_volume_scale=pca.price_volume_scale,
        price_volume_components=pca.price_volume_components,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation = SignatureConditionedAffineMeanModel(
        input_channels=len(FEATURE_COLUMNS),
        signature_depth=2,
        latent_dim=1,
        hidden_dim=1,
        num_symbols=len(train_set.symbols),
        cross_area_pca=pca,
    ).to(device)
    names = _feature_names(PRICE_PRICE_AREA_PCA_COMPONENTS, PRICE_VOLUME_AREA_PCA_COMPONENTS)
    (output_dir / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(f"Extracting {len(names)} fixed signature/context features on {device}...")
    train_x, train_meta = _matrix_and_metadata(train_set, representation, device, names)
    val_x, val_meta = _matrix_and_metadata(validation_set, representation, device, names)
    test_x, test_meta = (
        _matrix_and_metadata(test_set, representation, device, names) if test_set is not None else (None, None)
    )
    del representation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_dm = _dmatrix(train_x, train_meta, names, with_labels=True)
    val_dm = _dmatrix(val_x, val_meta, names, with_labels=False)
    test_dm = _dmatrix(test_x, test_meta, names, with_labels=False) if test_x is not None else None
    seeds = tuple(dict.fromkeys(ENSEMBLE_SEEDS))
    if not seeds:
        raise ValueError("ENSEMBLE_SEEDS must contain at least one seed.")
    fitted = [
        _fit_one_seed(seed, train_dm, val_dm, test_dm, val_meta, test_meta, output_dir)
        for seed in seeds
    ]
    val_prediction = average_seed_prediction_frames([result[0] for result in fitted], seeds)
    val_prediction.to_parquet(output_dir / "ensemble_validation_predictions.parquet", index=False)
    importance = pd.DataFrame(
        {"feature": names, "mean_gain": [np.mean([result[3].get(name, 0.0) for result in fitted]) for name in names]}
    ).sort_values("mean_gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance_gain.csv", index=False)
    val_ic = block_bootstrap_mean_ic(_daily_rank_ic_from_prediction_frame(val_prediction), BOOTSTRAP_ITERATIONS, SEED)
    (output_dir / "validation_rank_ic_block_bootstrap.json").write_text(json.dumps(val_ic, indent=2), encoding="utf-8")
    _daily_rank_ic_from_prediction_frame(val_prediction).rename_axis("session").reset_index().to_csv(
        output_dir / "validation_daily_rank_ic.csv", index=False
    )
    if not EVALUATE_TEST:
        _validation_only_comparison(val_prediction, val_ic, neural_dir, output_dir)
        return
    test_prediction = average_seed_prediction_frames([result[1] for result in fitted], seeds)
    test_prediction.to_parquet(output_dir / "oos_predictions.parquet", index=False)
    oos_ic = block_bootstrap_mean_ic(_daily_rank_ic_from_prediction_frame(test_prediction), BOOTSTRAP_ITERATIONS, SEED + 1)
    costs = tuple(sorted({0.0, float(PROFITABILITY_GATE_COST_BPS)}))
    execution_options = {
        "session_end_policy": SESSION_END_POLICY,
        "target_smoothing_fraction": TARGET_WEIGHT_SMOOTHING_FRACTION,
        "sector_rebalance_buffer_turnover": SECTOR_REBALANCE_BUFFER_TURNOVER,
    }
    ungated_metrics, ungated_returns = evaluate_cost_sensitivity(
        test_prediction, HOLDING_MINUTES, Z_SCORE_THRESHOLD, costs_bps=costs, **execution_options
    )
    ungated_returns.to_parquet(output_dir / "oos_holding_period_returns_ungated.parquet", index=False)
    (output_dir / "oos_metrics_ungated.json").write_text(json.dumps(ungated_metrics, indent=2), encoding="utf-8")
    gates = {}
    gate_diagnostics = {}
    for cost in costs:
        gate, actions, diagnostics = fit_profitability_gate(
            val_prediction, HOLDING_MINUTES, Z_SCORE_THRESHOLD, cost,
            PROFITABILITY_GATE_MIN_PROBABILITY, PROFITABILITY_GATE_L2,
            SESSION_END_POLICY, TARGET_WEIGHT_SMOOTHING_FRACTION, SECTOR_REBALANCE_BUFFER_TURNOVER,
        )
        gates[cost] = gate
        gate_diagnostics[f"{cost:g} bps"] = diagnostics
        actions.to_parquet(output_dir / f"profitability_gate_validation_actions_{cost:g}bps.parquet", index=False)
    gated_metrics, gated_returns = evaluate_cost_sensitivity(
        test_prediction, HOLDING_MINUTES, Z_SCORE_THRESHOLD,
        profitability_gates_by_cost=gates,
        profitability_gate_min_probability=PROFITABILITY_GATE_MIN_PROBABILITY,
        costs_bps=costs, **execution_options,
    )
    gated_metrics["rank_ic_block_bootstrap"] = oos_ic
    gated_metrics["profitability_gates"] = gate_diagnostics
    gated_returns.to_parquet(output_dir / "oos_holding_period_returns.parquet", index=False)
    (output_dir / "oos_metrics.json").write_text(json.dumps(gated_metrics, indent=2), encoding="utf-8")
    _daily_rank_ic_from_prediction_frame(test_prediction).rename_axis("session").reset_index().to_csv(
        output_dir / "oos_daily_rank_ic.csv", index=False
    )
    comparison: dict[str, object] = {"xgboost_validation_rank_ic": val_ic, "xgboost_oos_rank_ic": oos_ic}
    neural_val_path = neural_dir / "ensemble_validation_predictions.parquet"
    neural_test_path = neural_dir / "oos_predictions.parquet"
    if neural_val_path.exists() and neural_test_path.exists():
        neural_val = pd.read_parquet(neural_val_path)
        neural_test = pd.read_parquet(neural_test_path)
        if _same_labels(val_prediction, neural_val) and _same_labels(test_prediction, neural_test):
            comparison["neural_validation_rank_ic"] = block_bootstrap_mean_ic(
                _daily_rank_ic_from_prediction_frame(neural_val), BOOTSTRAP_ITERATIONS, SEED
            )
            comparison["paired_oos_rank_ic"] = _paired_daily_ic_comparison(test_prediction, neural_test)
            neural_ungated, _ = evaluate_cost_sensitivity(
                neural_test, HOLDING_MINUTES, Z_SCORE_THRESHOLD, costs_bps=costs, **execution_options
            )
            comparison["ungated_same_execution"] = {
                "xgboost": {key: ungated_metrics[key] for key in (f"{cost:g} bps" for cost in costs)},
                "neural": {key: neural_ungated[key] for key in (f"{cost:g} bps" for cost in costs)},
            }
            neural_gates = {}
            for cost in costs:
                neural_gate, _actions, _diagnostics = fit_profitability_gate(
                    neural_val, HOLDING_MINUTES, Z_SCORE_THRESHOLD, cost,
                    PROFITABILITY_GATE_MIN_PROBABILITY, PROFITABILITY_GATE_L2,
                    SESSION_END_POLICY, TARGET_WEIGHT_SMOOTHING_FRACTION, SECTOR_REBALANCE_BUFFER_TURNOVER,
                )
                neural_gates[cost] = neural_gate
            neural_gated, _ = evaluate_cost_sensitivity(
                neural_test, HOLDING_MINUTES, Z_SCORE_THRESHOLD,
                profitability_gates_by_cost=neural_gates,
                profitability_gate_min_probability=PROFITABILITY_GATE_MIN_PROBABILITY,
                costs_bps=costs, **execution_options,
            )
            comparison["gated_same_execution"] = {
                "xgboost": {key: gated_metrics[key] for key in (f"{cost:g} bps" for cost in costs)},
                "neural": {key: neural_gated[key] for key in (f"{cost:g} bps" for cost in costs)},
            }
            comparison["neural_predictions_path"] = str(neural_test_path)
            print("Neural prediction keys and labels match; paired OOS comparison saved.")
        else:
            comparison["neural_comparison_status"] = "Existing neural predictions use different signal keys or labels; rerun the neural model on the same data to compare."
            print(comparison["neural_comparison_status"])
    else:
        comparison["neural_comparison_status"] = "Existing neural prediction artifacts not found."
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"XGBoost validation mean daily Rank IC: {val_ic['mean_daily_rank_ic']:.5f}")
    print(f"XGBoost OOS mean daily Rank IC: {oos_ic['mean_daily_rank_ic']:.5f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
