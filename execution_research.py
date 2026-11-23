"""Execution-layer research on saved predictions (no retraining).

Run directly from the same Python environment as train_model.py after editing the
settings block below.  By default only the validation split is read, so the
script is safe to use while tuning.

What it does
------------
1. Rebuilds the current ungated execution rule (sector z-threshold weights, target
   smoothing, sector buffer, forced liquidation, daily flattening) in a fast numpy
   simulator, and checks it against ``minute_alpha.evaluate_cost_sensitivity``.
2. Gate placebo: fits the profitability gate on validation (exactly as the
   pipeline does), records its acceptance rate, then replaces it by a random gate
   that accepts each optional rebalance with the same probability.  If the real
   gate does not beat most placebo paths, its Sharpe uplift is not evidence of a
   useful gate.
3. Cost-aware optimizer: at every rebalance, per eligible sector,

       max_w  alpha'w - (gamma/2) sum_i sigma_i^2 w_i^2 - kappa * c * ||w - w_prev||_1
       s.t.   sum_{i in sector} w_i = 0,

   solved exactly (soft-threshold KKT solution + bisection on the sector
   multiplier).  The L1 term creates a per-stock no-trade band.  Alpha is
   Grinold-Kahn ``IC * sigma_i * z_i`` from the ensemble rank score, so the
   unconstrained weight is proportional to z_i / sigma_i (rank score / predicted
   std).  With an SDE run sigma_i is the model's predictive std; the
   ``homoskedastic`` variant uses a per-symbol train constant instead, isolating
   the value of the SDE variance.  Sharpe is invariant to gamma, so results are
   simply rescaled to the baseline's mean gross exposure.  Because the signal is
   persistent, the cost should be amortised over the expected holding time, so
   useful kappa values are well below 1 (roughly 1 - rho; see signal_persistence).

Outputs go to ``<RUN_DIR>/execution_research_<split>/``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from minute_alpha import (
    _daily_rank_ic_from_prediction_frame,
    _portfolio_symbol_metadata,
    _scheduled_trade_frame,
    _sector_neutral_weights,
    block_bootstrap_mean_ic,
    evaluate_cost_sensitivity,
    fit_profitability_gate,
)
from project_settings import OUTPUT_DIR

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
RUN_DIR = OUTPUT_DIR            # any artifact folder with saved predictions
SPLIT = "validation"            # "validation" while tuning; "oos" only to audit frozen results
COSTS_BPS = (0.0, 0.3, 0.6, 1.0, 1.5)
DESIGN_COST_BPS = 0.6           # cost at which the gate is fitted and kappa is compared
# kappa multiplies the cost in the one-period objective.  A persistent signal keeps
# paying alpha while a position is held, so the cost should be amortised:
# kappa ~ 1 - rho, where rho is the score's period-to-period autocorrelation
# (reported as signal_persistence).  kappa = 1 is the myopic one-period choice.
KAPPAS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)
GRINOLD_IC = 0.02               # only rescales alpha against cost; kappa absorbs the rest
PLACEBO_REPLICATES = 500
SEED = 2026


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """Dense [period, symbol] arrays for one split, aligned like the pipeline."""

    symbols: list[str]
    sector_ids: np.ndarray            # [N] integer sector of each symbol
    eligible_sector: np.ndarray       # [n_sectors] bool, sector has a budget
    sector_names: list[str]
    available: np.ndarray             # [T, N] bool, valid entry/exit label
    simple_return: np.ndarray         # [T, N]
    candidate_weight: np.ndarray      # [T, N] baseline z-threshold weights
    score: np.ndarray                 # [T, N] rank score (nan if unavailable)
    predicted_std: np.ndarray | None  # [T, N] predictive std in return units
    flatten_after: np.ndarray         # [T] bool
    periods_per_year: float
    sessions: np.ndarray              # [T]


def _run_dir() -> Path:
    path = Path(RUN_DIR)
    return path if path.is_absolute() else PROJECT_DIR / path


def _execution_settings(run_dir: Path) -> dict[str, object]:
    """Prefer the rule actually used (rank_execution_rule.json) over run_config.json."""
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    settings = {
        "holding_minutes": int(config["holding_minutes"]),
        "z_threshold": float(config["z_threshold"]),
        "session_end_policy": config.get("session_end_policy", "flatten_daily"),
        "target_smoothing_fraction": float(config.get("target_smoothing_fraction", 0.5)),
        "sector_rebalance_buffer_turnover": float(config.get("sector_rebalance_buffer_turnover", 0.02)),
        "gate_min_probability": float(config.get("profitability_gate_min_probability", 0.55)),
        "gate_l2": float(config.get("profitability_gate_l2", 1.0)),
        "source": "run_config.json",
    }
    rule_path = run_dir / "rank_execution_rule.json"
    if rule_path.exists():
        rule = json.loads(rule_path.read_text(encoding="utf-8"))
        settings["z_threshold"] = float(rule["z_threshold"])
        settings["target_smoothing_fraction"] = float(rule["target_smoothing_fraction"])
        settings["sector_rebalance_buffer_turnover"] = float(rule["sector_rebalance_buffer_turnover"])
        settings["session_end_policy"] = rule["session_end_policy"]
        settings["gate_min_probability"] = float(rule["profitability_gate"]["min_probability"])
        settings["gate_l2"] = float(rule["profitability_gate"]["l2"])
        settings["source"] = "rank_execution_rule.json"
    return settings


def _load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    if split == "validation":
        return pd.read_parquet(run_dir / "ensemble_validation_predictions.parquet")
    if split == "oos":
        return pd.read_parquet(run_dir / "oos_predictions.parquet")
    raise ValueError("SPLIT must be 'validation' or 'oos'.")


def _target_scale_and_baseline(run_dir: Path) -> tuple[float | None, dict[str, float] | None]:
    spec_path = run_dir / "sde_model_spec.json"
    if not spec_path.exists():
        return None, None
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return float(spec["target_scale"]), spec.get("baseline_variance_by_symbol")


def build_panel(
    frame: pd.DataFrame,
    settings: dict[str, object],
    target_scale: float | None,
    variance_source: str,
    baseline_variance: dict[str, float] | None,
) -> Panel:
    traded = _scheduled_trade_frame(frame, int(settings["holding_minutes"]))
    symbols_index, sector_by_symbol, budgets = _portfolio_symbol_metadata(traded)
    symbols = list(symbols_index)
    sector_names = sorted(sector_by_symbol.unique())
    sector_ids = np.array([sector_names.index(sector_by_symbol[symbol]) for symbol in symbols])
    eligible = np.array([name in budgets for name in sector_names])

    periods = traded[["session", "timestamp"]].drop_duplicates().reset_index(drop=True)
    period_index = {key: i for i, key in enumerate(zip(periods["session"], periods["timestamp"]))}
    symbol_index = {symbol: j for j, symbol in enumerate(symbols)}
    t_count, n_count = len(periods), len(symbols)
    rows = traded["session"].combine(traded["timestamp"], lambda a, b: (a, b)).map(period_index).to_numpy()
    cols = traded["symbol"].map(symbol_index).to_numpy()

    available = np.zeros((t_count, n_count), dtype=bool)
    simple_return = np.zeros((t_count, n_count))
    score = np.full((t_count, n_count), np.nan)
    predicted_std = None
    label = traded["target_return"].notna().to_numpy()
    available[rows, cols] = label
    simple_return[rows, cols] = np.where(label, np.expm1(traded["target_return"].fillna(0.0).to_numpy()), 0.0)
    score[rows, cols] = traded["prediction"].to_numpy(dtype=float)

    if variance_source == "sde":
        if target_scale is None or "predicted_variance_std" not in traded.columns:
            raise ValueError("variance_source='sde' needs an SDE run.")
        predicted_std = np.full((t_count, n_count), np.nan)
        predicted_std[rows, cols] = np.sqrt(traded["predicted_variance_std"].to_numpy(dtype=float)) * target_scale
    elif variance_source == "homoskedastic":
        if target_scale is not None and baseline_variance is not None:
            per_symbol = {s: math.sqrt(v) * target_scale for s, v in baseline_variance.items() if v == v}
        else:
            # Deterministic runs carry no train-only scale; fall back to the split's
            # own per-symbol dispersion (only a scale, but in-sample).
            per_symbol = traded.groupby("symbol")["rank_target_return"].std().to_dict()
        predicted_std = np.tile(np.array([per_symbol.get(s, np.nan) for s in symbols]), (t_count, 1))
    elif variance_source != "none":
        raise ValueError("variance_source must be 'sde', 'homoskedastic' or 'none'.")

    candidate = np.zeros((t_count, n_count))
    for (session, timestamp), group in traded.groupby(["session", "timestamp"], sort=False):
        weights = _sector_neutral_weights(
            group["prediction"], group["sector"], float(settings["z_threshold"]), budgets
        )
        candidate[period_index[(session, timestamp)], group["symbol"].map(symbol_index).to_numpy()] = weights.to_numpy()

    sessions = periods["session"].to_numpy()
    is_final = np.r_[sessions[1:] != sessions[:-1], True]
    if settings["session_end_policy"] != "flatten_daily":
        raise ValueError("This research script implements the default flatten_daily policy only.")
    return Panel(
        symbols=symbols,
        sector_ids=sector_ids,
        eligible_sector=eligible,
        sector_names=sector_names,
        available=available,
        simple_return=simple_return,
        candidate_weight=candidate,
        score=score,
        predicted_std=predicted_std,
        flatten_after=is_final,
        periods_per_year=252.0 * t_count / len(np.unique(sessions)),
        sessions=sessions,
    )


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
Policy = Callable[[int, np.ndarray], np.ndarray]


def simulate(panel: Panel, policy: Policy, cost_bps: float) -> dict[str, np.ndarray]:
    """Replicates the pipeline's turnover, forced-liquidation and flattening rules."""
    t_count, n_count = panel.available.shape
    cost_rate = cost_bps / 10_000.0
    prior = np.zeros(n_count)
    net = np.zeros(t_count)
    turnover = np.zeros(t_count)
    gross_exposure = np.zeros(t_count)
    for t in range(t_count):
        available = panel.available[t]
        forced = float(np.abs(prior[~available]).sum())
        prior_for_target = np.where(available, prior, 0.0)
        target = np.where(available, policy(t, prior_for_target), 0.0)
        traded = forced + float(np.abs(target - prior_for_target).sum())
        if panel.flatten_after[t]:
            traded += float(np.abs(target).sum())
            next_prior = np.zeros(n_count)
        else:
            next_prior = target
        net[t] = float(target @ panel.simple_return[t]) - traded * cost_rate
        turnover[t] = traded
        gross_exposure[t] = float(np.abs(target).sum())
        prior = next_prior
    return {"net": net, "turnover": turnover, "gross_exposure": gross_exposure}


def summarise(panel: Panel, path: dict[str, np.ndarray], rescale_to_gross: float | None = None) -> dict[str, float]:
    net = path["net"]
    factor = 1.0
    if rescale_to_gross is not None and path["gross_exposure"].mean() > 1e-12:
        factor = rescale_to_gross / path["gross_exposure"].mean()
    scaled = net * factor
    equity = np.concatenate(([1.0], np.cumprod(1.0 + scaled)))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    daily = pd.Series(scaled).groupby(panel.sessions).sum()
    return {
        "annualized_sharpe": float(net.mean() / (net.std(ddof=1) + 1e-18) * math.sqrt(panel.periods_per_year)),
        "mean_return_per_period_bps": float(scaled.mean() * 1e4),
        "annualized_return_approx": float(scaled.mean() * panel.periods_per_year),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "mean_gross_exposure": float(path["gross_exposure"].mean() * factor),
        "mean_one_way_turnover": float(path["turnover"].mean() * factor),
        "turnover_per_unit_gross": float(path["turnover"].mean() / max(path["gross_exposure"].mean(), 1e-12)),
        "positive_day_rate": float((daily > 0).mean()),
        "rescale_factor": float(factor),
    }


def baseline_policy(panel: Panel, smoothing: float, buffer: float, accept: Callable[[int], bool] | None = None) -> Policy:
    """Numpy version of _smoothed_sector_buffered_target (+ optional gate)."""

    def policy(t: int, prior: np.ndarray) -> np.ndarray:
        desired = np.where(panel.available[t], panel.candidate_weight[t], 0.0)
        smoothed = prior + smoothing * (desired - prior)
        target = prior.copy()
        for sector in np.unique(panel.sector_ids):
            members = panel.sector_ids == sector
            sector_turnover = float(np.abs(smoothed[members] - prior[members]).sum())
            if sector_turnover <= 1e-15 or sector_turnover < buffer:
                continue
            target[members] = smoothed[members]
        if accept is not None and float(np.abs(target - prior).sum()) > 1e-15 and not accept(t):
            return prior
        return target

    return policy


# --------------------------------------------------------------------------- #
# Cost-aware L1 optimizer
# --------------------------------------------------------------------------- #
def _band_solution(nu: float, alpha: np.ndarray, q: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    b = alpha - nu - q * prior
    move = np.where(np.abs(b) <= k, 0.0, (b - k * np.sign(b)) / q)
    return prior + move


def l1_sector_neutral(alpha: np.ndarray, q: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    """argmax alpha'w - 1/2 sum q w^2 - k ||w - prior||_1  s.t. sum w = 0 (exact)."""
    if len(alpha) < 2:
        return np.zeros_like(alpha)
    span = float(np.max(np.abs(alpha)) + np.max(np.abs(q * prior)) + k + 1e-12)
    lo, hi = -span, span
    while _band_solution(lo, alpha, q, prior, k).sum() < 0.0:
        lo -= 2.0 * span
    while _band_solution(hi, alpha, q, prior, k).sum() > 0.0:
        hi += 2.0 * span
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _band_solution(mid, alpha, q, prior, k).sum() > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, span):
            break
    weights = _band_solution(0.5 * (lo + hi), alpha, q, prior, k)
    return weights - weights.mean() * (np.abs(weights.sum()) > 1e-12)


def optimizer_policy(panel: Panel, kappa: float, cost_bps: float, information_coefficient: float) -> Policy:
    if panel.predicted_std is None:
        raise ValueError("The optimizer needs a predicted_std source.")
    k = kappa * cost_bps / 10_000.0

    def policy(t: int, prior: np.ndarray) -> np.ndarray:
        target = np.zeros_like(prior)
        for sector in np.flatnonzero(panel.eligible_sector):
            members = (panel.sector_ids == sector) & panel.available[t]
            if members.sum() < 2:
                continue
            scores = panel.score[t, members]
            spread = scores.std()
            if not spread > 1e-12:
                continue
            z = (scores - scores.mean()) / spread
            sigma = panel.predicted_std[t, members]
            alpha = information_coefficient * sigma * z
            q = sigma**2  # gamma = 1; Sharpe is invariant to gamma
            target[members] = l1_sector_neutral(alpha, q, prior[members], k)
        return target

    return policy


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def signal_persistence(panel: Panel) -> dict[str, float]:
    """Cross-sectional Spearman correlation of consecutive scores within a session."""
    values = []
    for t in range(1, len(panel.sessions)):
        if panel.sessions[t] != panel.sessions[t - 1]:
            continue
        both = panel.available[t] & panel.available[t - 1]
        if both.sum() < 5:
            continue
        a, b = panel.score[t - 1, both], panel.score[t, both]
        ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
        values.append(float(np.corrcoef(ra, rb)[0, 1]))
    rho = float(np.nanmean(values)) if values else float("nan")
    return {
        "mean_consecutive_rank_autocorrelation": rho,
        "implied_expected_holding_periods": float(1.0 / max(1.0 - rho, 1e-6)) if rho == rho else float("nan"),
        "suggested_kappa_1_minus_rho": float(max(1.0 - rho, 0.0)) if rho == rho else float("nan"),
    }


def check_against_pipeline(panel: Panel, frame: pd.DataFrame, settings: dict[str, object]) -> dict[str, object]:
    metrics, _ = evaluate_cost_sensitivity(
        frame,
        int(settings["holding_minutes"]),
        float(settings["z_threshold"]),
        session_end_policy=str(settings["session_end_policy"]),
        target_smoothing_fraction=float(settings["target_smoothing_fraction"]),
        sector_rebalance_buffer_turnover=float(settings["sector_rebalance_buffer_turnover"]),
        costs_bps=(0.0, DESIGN_COST_BPS),
    )
    result = {}
    policy = baseline_policy(
        panel, float(settings["target_smoothing_fraction"]), float(settings["sector_rebalance_buffer_turnover"])
    )
    for cost in (0.0, DESIGN_COST_BPS):
        ours = summarise(panel, simulate(panel, policy, cost))["annualized_sharpe"]
        theirs = metrics[f"{cost:g} bps"]["annualized_sharpe"]
        result[f"{cost:g} bps"] = {"numpy_simulator": ours, "pipeline": theirs, "abs_diff": abs(ours - theirs)}
    worst = max(item["abs_diff"] for item in result.values())
    if worst > 1e-6:
        raise RuntimeError(f"Numpy simulator disagrees with the pipeline (max Sharpe diff {worst:.2e}): {result}")
    return result


def gate_placebo(
    panel: Panel,
    split_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    settings: dict[str, object],
) -> dict[str, object]:
    common = {
        "session_end_policy": str(settings["session_end_policy"]),
        "target_smoothing_fraction": float(settings["target_smoothing_fraction"]),
        "sector_rebalance_buffer_turnover": float(settings["sector_rebalance_buffer_turnover"]),
    }
    gates = {}
    for cost in (0.0, DESIGN_COST_BPS):
        gate, _, _ = fit_profitability_gate(
            validation_frame,
            int(settings["holding_minutes"]),
            float(settings["z_threshold"]),
            cost,
            float(settings["gate_min_probability"]),
            float(settings["gate_l2"]),
            common["session_end_policy"],
            common["target_smoothing_fraction"],
            common["sector_rebalance_buffer_turnover"],
        )
        gates[cost] = gate
    gated, _ = evaluate_cost_sensitivity(
        split_frame,
        int(settings["holding_minutes"]),
        float(settings["z_threshold"]),
        profitability_gates_by_cost=gates,
        profitability_gate_min_probability=float(settings["gate_min_probability"]),
        costs_bps=(0.0, DESIGN_COST_BPS),
        **common,
    )
    key = f"{DESIGN_COST_BPS:g} bps"
    acceptance = float(gated[key]["profitability_gate_acceptance_rate"])
    gate_sharpe = float(gated[key]["annualized_sharpe"])

    rng = np.random.default_rng(SEED)
    placebo = np.empty(PLACEBO_REPLICATES)
    for replicate in range(PLACEBO_REPLICATES):
        draws = rng.random(len(panel.sessions))
        policy = baseline_policy(
            panel,
            common["target_smoothing_fraction"],
            common["sector_rebalance_buffer_turnover"],
            accept=lambda t, draws=draws: bool(draws[t] < acceptance),
        )
        placebo[replicate] = summarise(panel, simulate(panel, policy, DESIGN_COST_BPS))["annualized_sharpe"]
    ungated = summarise(
        panel,
        simulate(
            panel,
            baseline_policy(panel, common["target_smoothing_fraction"], common["sector_rebalance_buffer_turnover"]),
            DESIGN_COST_BPS,
        ),
    )["annualized_sharpe"]
    return {
        "cost_bps": DESIGN_COST_BPS,
        "gate_fit_split": "validation",
        "gate_is_in_sample": SPLIT == "validation",
        "gate_min_probability": float(settings["gate_min_probability"]),
        "gate_acceptance_rate": acceptance,
        "gated_sharpe": gate_sharpe,
        "ungated_sharpe": ungated,
        "placebo_replicates": PLACEBO_REPLICATES,
        "placebo_sharpe_mean": float(placebo.mean()),
        "placebo_sharpe_2.5_97.5": [float(np.quantile(placebo, 0.025)), float(np.quantile(placebo, 0.975))],
        "fraction_placebo_ge_gate": float((placebo >= gate_sharpe).mean()),
        "interpretation": (
            "fraction_placebo_ge_gate is a one-sided p-value: the share of random gates with the same "
            "acceptance rate that do at least as well as the fitted gate."
        ),
    }


def optimizer_grid(panel_by_variance: dict[str, Panel], target_gross: float) -> pd.DataFrame:
    records = []
    for variance_source, panel in panel_by_variance.items():
        for kappa in KAPPAS:
            for cost in COSTS_BPS:
                path = simulate(panel, optimizer_policy(panel, kappa, cost, GRINOLD_IC), cost)
                records.append(
                    {
                        "variance_source": variance_source,
                        "kappa": kappa,
                        "cost_bps": cost,
                        **summarise(panel, path, rescale_to_gross=target_gross),
                    }
                )
    return pd.DataFrame.from_records(records)


def baseline_curve(panel: Panel, settings: dict[str, object]) -> pd.DataFrame:
    policy = baseline_policy(
        panel, float(settings["target_smoothing_fraction"]), float(settings["sector_rebalance_buffer_turnover"])
    )
    return pd.DataFrame.from_records(
        [{"cost_bps": cost, **summarise(panel, simulate(panel, policy, cost))} for cost in COSTS_BPS]
    )


def breakeven_cost(curve: pd.DataFrame) -> float:
    ordered = curve.sort_values("cost_bps")
    costs, sharpe = ordered["cost_bps"].to_numpy(), ordered["annualized_sharpe"].to_numpy()
    for i in range(1, len(costs)):
        if sharpe[i - 1] > 0.0 >= sharpe[i]:
            return float(costs[i - 1] + (costs[i] - costs[i - 1]) * sharpe[i - 1] / (sharpe[i - 1] - sharpe[i]))
    return float("inf") if sharpe[-1] > 0 else 0.0


def _fmt_breakeven(value: float) -> str:
    return f"> {max(COSTS_BPS):g} bp" if value == float("inf") else f"{value:.2f} bp"


def main() -> None:
    run_dir = _run_dir()
    out_dir = run_dir / f"execution_research_{SPLIT}"
    out_dir.mkdir(exist_ok=True)
    settings = _execution_settings(run_dir)
    frame = _load_predictions(run_dir, SPLIT)
    validation_frame = frame if SPLIT == "validation" else _load_predictions(run_dir, "validation")
    target_scale, baseline_variance = _target_scale_and_baseline(run_dir)
    is_sde = target_scale is not None and "predicted_variance_std" in frame.columns
    print(f"Run: {run_dir.name}\nSplit: {SPLIT} | SDE run: {is_sde} | execution settings from {settings['source']}")

    base_panel = build_panel(frame, settings, target_scale, "none", baseline_variance)
    report: dict[str, object] = {"run": run_dir.name, "split": SPLIT, "execution_settings": settings}
    daily_ic = _daily_rank_ic_from_prediction_frame(frame)
    report["rank_ic"] = block_bootstrap_mean_ic(daily_ic, 10_000, SEED)
    report["signal_persistence"] = signal_persistence(base_panel)

    print("1/3 Checking numpy simulator against the pipeline...", flush=True)
    report["simulator_check"] = check_against_pipeline(base_panel, frame, settings)
    baseline = baseline_curve(base_panel, settings)
    baseline.to_csv(out_dir / "baseline_cost_curve.csv", index=False)
    report["baseline_ungated"] = {
        "breakeven_one_way_cost_bps": breakeven_cost(baseline),
        "curve": baseline.set_index("cost_bps")[["annualized_sharpe", "mean_one_way_turnover", "mean_gross_exposure"]]
        .round(4)
        .to_dict(orient="index"),
    }
    target_gross = float(baseline.loc[baseline["cost_bps"] == 0.0, "mean_gross_exposure"].iloc[0])

    print(f"2/3 Gate placebo ({PLACEBO_REPLICATES} random gates)...", flush=True)
    report["gate_placebo"] = gate_placebo(base_panel, frame, validation_frame, settings)

    print("3/3 Cost-aware L1 optimizer grid...", flush=True)
    panels = {"homoskedastic": build_panel(frame, settings, target_scale, "homoskedastic", baseline_variance)}
    if is_sde:
        panels["sde"] = build_panel(frame, settings, target_scale, "sde", baseline_variance)
    grid = optimizer_grid(panels, target_gross)
    grid.to_csv(out_dir / "optimizer_grid.csv", index=False)
    summary = {}
    for (variance_source, kappa), curve in grid.groupby(["variance_source", "kappa"]):
        summary[f"{variance_source} | kappa={kappa:g}"] = {
            "breakeven_one_way_cost_bps": breakeven_cost(curve),
            "sharpe_by_cost": {f"{c:g} bps": round(float(s), 3) for c, s in zip(curve["cost_bps"], curve["annualized_sharpe"])},
            "turnover_per_unit_gross_at_design_cost": float(
                curve.loc[curve["cost_bps"] == DESIGN_COST_BPS, "turnover_per_unit_gross"].iloc[0]
            ),
        }
    report["optimizer"] = {
        "objective": "max alpha'w - 1/2 sum sigma^2 w^2 - kappa*c*|w - w_prev|_1, sector-neutral",
        "alpha": f"Grinold IC*sigma*z with IC={GRINOLD_IC}, z = within-sector standardised rank score",
        "rescaled_to_mean_gross_exposure": target_gross,
        "caution": "kappa is chosen on this split; with SPLIT='validation' the reported Sharpe is in-sample for kappa.",
        "summary": summary,
    }
    (out_dir / "execution_research.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"Rank IC (daily mean): {report['rank_ic']['mean_daily_rank_ic']:.4f}")
    persistence = report["signal_persistence"]
    print(f"Signal persistence: rho = {persistence['mean_consecutive_rank_autocorrelation']:.3f}, "
          f"suggested kappa ~ {persistence['suggested_kappa_1_minus_rho']:.2f}")
    print(f"Baseline ungated: breakeven {_fmt_breakeven(report['baseline_ungated']['breakeven_one_way_cost_bps'])} | "
          f"Sharpe by cost {baseline.set_index('cost_bps')['annualized_sharpe'].round(2).to_dict()}")
    placebo = report["gate_placebo"]
    print(
        f"Gate @ {DESIGN_COST_BPS:g} bp: Sharpe {placebo['gated_sharpe']:.2f} (ungated {placebo['ungated_sharpe']:.2f}), "
        f"acceptance {placebo['gate_acceptance_rate']:.2f}; placebo mean {placebo['placebo_sharpe_mean']:.2f}, "
        f"95% band {np.round(placebo['placebo_sharpe_2.5_97.5'], 2).tolist()}, "
        f"P(placebo >= gate) = {placebo['fraction_placebo_ge_gate']:.3f}"
    )
    for name, values in summary.items():
        print(f"Optimizer {name:<26} breakeven {_fmt_breakeven(values['breakeven_one_way_cost_bps'])} | "
              f"Sharpe {values['sharpe_by_cost']} | turnover/gross {values['turnover_per_unit_gross_at_design_cost']:.2f}")
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
