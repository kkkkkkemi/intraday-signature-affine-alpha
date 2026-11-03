"""Rebuild OOS evaluation with a validation-fitted profitability gate.

Run this file directly from the same Python environment as ``train_model.py``.
It does not retrain the rank model. The gate is fitted only to the frozen
validation predictions, then applied once to the frozen OOS predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from minute_alpha import (
    _daily_rank_ic_from_prediction_frame,
    block_bootstrap_mean_ic,
    evaluate_cost_sensitivity,
    fit_profitability_gate,
)
from project_settings import (
    BOOTSTRAP_ITERATIONS,
    HOLDING_MINUTES,
    OUTPUT_DIR,
    PROFITABILITY_GATE_COST_BPS,
    PROFITABILITY_GATE_L2,
    PROFITABILITY_GATE_MIN_PROBABILITY,
    SEED,
    SECTOR_REBALANCE_BUFFER_TURNOVER,
    SESSION_END_POLICY,
    TARGET_WEIGHT_SMOOTHING_FRACTION,
    Z_SCORE_THRESHOLD,
)


def _gate_spec(gate, diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        **diagnostics,
        "intercept": gate.intercept,
        "feature_mean": {
            name: float(value) for name, value in zip(gate.feature_names, gate.feature_mean, strict=True)
        },
        "feature_scale": {
            name: float(value) for name, value in zip(gate.feature_names, gate.feature_scale, strict=True)
        },
        "coefficients": {
            name: float(value) for name, value in zip(gate.feature_names, gate.coefficients, strict=True)
        },
    }


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    validation_path = output_dir / "ensemble_validation_predictions.parquet"
    oos_path = output_dir / "ensemble_oos_predictions_uncalibrated.parquet"
    required = (validation_path, oos_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot rebuild profitability-gated OOS evaluation because these frozen "
            "ensemble artifacts are missing: " + ", ".join(missing)
        )

    validation_predictions = pd.read_parquet(validation_path)
    predictions = pd.read_parquet(oos_path)
    gate_costs = tuple(sorted({0.0, PROFITABILITY_GATE_COST_BPS}))
    profitability_gates_by_cost = {}
    profitability_gate_specs = {}
    for gate_cost_bps in gate_costs:
        gate, gate_validation_actions, gate_diagnostics = fit_profitability_gate(
            validation_predictions,
            HOLDING_MINUTES,
            Z_SCORE_THRESHOLD,
            gate_cost_bps,
            PROFITABILITY_GATE_MIN_PROBABILITY,
            PROFITABILITY_GATE_L2,
            SESSION_END_POLICY,
            TARGET_WEIGHT_SMOOTHING_FRACTION,
            SECTOR_REBALANCE_BUFFER_TURNOVER,
        )
        cost_key = f"{gate_cost_bps:g} bps"
        profitability_gates_by_cost[gate_cost_bps] = gate
        profitability_gate_specs[cost_key] = _gate_spec(gate, gate_diagnostics)
        gate_validation_actions.to_parquet(
            output_dir / f"profitability_gate_validation_actions_{gate_cost_bps:g}bps.parquet",
            index=False,
        )
        (output_dir / f"profitability_gate_{gate_cost_bps:g}bps.json").write_text(
            json.dumps(profitability_gate_specs[cost_key], indent=2), encoding="utf-8"
        )
    primary_gate_spec = profitability_gate_specs[f"{PROFITABILITY_GATE_COST_BPS:g} bps"]
    (output_dir / "profitability_gate.json").write_text(
        json.dumps(primary_gate_spec, indent=2), encoding="utf-8"
    )

    predictions.to_parquet(output_dir / "oos_predictions.parquet", index=False)
    metrics, period_returns = evaluate_cost_sensitivity(
        predictions,
        HOLDING_MINUTES,
        Z_SCORE_THRESHOLD,
        session_end_policy=SESSION_END_POLICY,
        target_smoothing_fraction=TARGET_WEIGHT_SMOOTHING_FRACTION,
        sector_rebalance_buffer_turnover=SECTOR_REBALANCE_BUFFER_TURNOVER,
        profitability_gates_by_cost=profitability_gates_by_cost,
        profitability_gate_min_probability=PROFITABILITY_GATE_MIN_PROBABILITY,
        costs_bps=gate_costs,
    )
    daily_rank_ic = _daily_rank_ic_from_prediction_frame(predictions)
    metrics["rank_ic_block_bootstrap"] = block_bootstrap_mean_ic(
        daily_rank_ic, BOOTSTRAP_ITERATIONS, SEED + 1
    )
    rank_execution_rule = {
        "fit_split": "rank checkpoints selected on validation; profitability gate fitted only to frozen validation predictions",
        "signal_selection": "within-sector z-score threshold; no score-to-return calibration",
        "z_threshold": Z_SCORE_THRESHOLD,
        "profitability_gate": {
            "decision_rule": "execute an optional proposed rebalance only when P(incremental net PnL > 0) >= min_probability",
            "cost_specific_paths": {
                f"{cost:g} bps": {"gate_fit_cost_bps": cost, "min_probability": PROFITABILITY_GATE_MIN_PROBABILITY}
                for cost in gate_costs
            },
            "min_probability": PROFITABILITY_GATE_MIN_PROBABILITY,
            "l2": PROFITABILITY_GATE_L2,
        },
        "execution_model": (
            "consecutive intraday target weights; zero cost for an unchanged weight; daily terminal flattening"
            if SESSION_END_POLICY == "flatten_daily"
            else "consecutive target weights; zero cost for an unchanged weight; marked between-rebalance bridge returns"
        ),
        "session_end_policy": SESSION_END_POLICY,
        "sector_budget_rule": "fixed gross budgets proportional to eligible sector symbol counts; inactive budgets remain cash",
        "target_smoothing_fraction": TARGET_WEIGHT_SMOOTHING_FRACTION,
        "sector_rebalance_buffer_turnover": SECTOR_REBALANCE_BUFFER_TURNOVER,
        "cost_sensitivity_rule": "0 bp and configured-cost scenarios each use a gate fitted under their own cost, producing intentionally distinct no-friction and friction-aware target paths",
    }
    metrics["rank_execution_rule"] = rank_execution_rule
    metrics["profitability_gates"] = profitability_gate_specs
    (output_dir / "rank_execution_rule.json").write_text(
        json.dumps(rank_execution_rule, indent=2), encoding="utf-8"
    )
    (output_dir / "return_score_calibration.json").write_text(
        json.dumps(
            {
                "enabled": False,
                "reason": "Model-score magnitude is not interpreted as an expected return.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "expected_alpha_gate.json").write_text(
        json.dumps(
            {
                "enabled": False,
                "reason": "The former expected-alpha gate is replaced by the profitability probability gate.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    daily_rank_ic.rename_axis("session").reset_index().to_csv(
        output_dir / "oos_daily_rank_ic.csv", index=False
    )
    period_returns.to_parquet(output_dir / "oos_holding_period_returns.parquet", index=False)
    (output_dir / "oos_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        "Cost-specific profitability-gated OOS re-evaluation | "
        f"sector z-threshold={Z_SCORE_THRESHOLD:.2f}, "
        f"paths={list(gate_costs)} bps, "
        f"p_min={PROFITABILITY_GATE_MIN_PROBABILITY:.2f}"
    )
    print(f"\nOOS evaluation (probability gate; session policy={SESSION_END_POLICY}):")
    for cost, values in metrics.items():
        print(f"  {cost}: {values}")


if __name__ == "__main__":
    main()
