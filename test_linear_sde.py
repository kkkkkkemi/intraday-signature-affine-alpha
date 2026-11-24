"""Sanity checks for the linear latent SDE layer.

Run directly (``python test_linear_sde.py``) from the same environment as
train_model.py; pytest also picks the ``test_*`` functions up.  No market data is
needed.  Every check prints PASS or raises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import scipy.stats as stats

from minute_alpha import (
    FEATURE_COLUMNS,
    CrossTickerAreaPCA,
    CrossSectionalRankLoss,
    GaussianPredictiveNLL,
    PredictiveNLL,
    SignatureConditionedAffineMeanModel,
)

torch.set_default_dtype(torch.float64)


def _stable_drift(n: int, d: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(n, d, d, generator=generator)
    decay = torch.nn.functional.softplus(torch.randn(n, d, generator=generator)) + 0.05
    return 0.25 * (raw - raw.transpose(-1, -2)) - torch.diag_embed(decay)


def test_van_loan_matches_quadrature() -> None:
    generator = torch.Generator().manual_seed(0)
    n, d, r = 4, 6, 3
    drift = _stable_drift(n, d, generator)
    loading = torch.randn(n, d, r, generator=generator)
    q = loading @ loading.transpose(-1, -2)
    covariance = SignatureConditionedAffineMeanModel.van_loan_covariance(drift, q)
    grid = torch.linspace(0.0, 1.0, 2001)
    integrand = torch.stack(
        [torch.matrix_exp(drift * s) @ q @ torch.matrix_exp(drift * s).transpose(-1, -2) for s in grid]
    )
    reference = torch.trapezoid(integrand, grid, dim=0)
    error = (covariance - reference).abs().max().item()
    assert error < 1e-6, error
    assert torch.linalg.eigvalsh(covariance).min().item() > -1e-10
    print(f"PASS Van Loan vs quadrature, max abs error {error:.2e}")


def test_van_loan_matches_monte_carlo() -> None:
    generator = torch.Generator().manual_seed(1)
    d, r = 4, 2
    drift = _stable_drift(1, d, generator)[0]
    loading = torch.randn(d, r, generator=generator)
    intercept = torch.randn(d, generator=generator)
    start = torch.randn(d, generator=generator)
    paths, steps = 100_000, 500
    dt = 1.0 / steps
    state = start.expand(paths, d).clone()
    for _ in range(steps):
        noise = torch.randn(paths, r, generator=generator) @ loading.T
        state = state + (state @ drift.T + intercept) * dt + noise * dt**0.5
    augmented = torch.zeros(d + 1, d + 1)
    augmented[:d, :d] = drift
    augmented[:d, d] = intercept
    transition = torch.matrix_exp(augmented)
    mean = transition[:d, :d] @ start + transition[:d, d]
    covariance = SignatureConditionedAffineMeanModel.van_loan_covariance(
        drift[None], (loading @ loading.T)[None]
    )[0]
    mean_error = (state.mean(0) - mean).abs().max().item()
    covariance_error = (torch.cov(state.T) - covariance).abs().max().item()
    scale = covariance.abs().max().item()
    assert mean_error < 0.05 and covariance_error < 0.05 * max(scale, 1.0), (mean_error, covariance_error)
    print(f"PASS Euler Monte Carlo agrees: mean err {mean_error:.3f}, cov err {covariance_error:.3f} (scale {scale:.2f})")


def _toy_model(
    diffusion_rank: int,
    num_symbols: int = 5,
    latent_dim: int = 4,
    jump_mode: str = "none",
    stock_embedding_dim: int = 0,
    cross_attention_layers: int = 0,
) -> SignatureConditionedAffineMeanModel:
    rng = np.random.default_rng(0)
    pp = num_symbols * (num_symbols - 1) // 2
    pv = num_symbols * (num_symbols - 1)
    pca = CrossTickerAreaPCA(
        price_price_mean=np.zeros(pp, dtype=np.float32),
        price_price_scale=np.ones(pp, dtype=np.float32),
        price_price_components=rng.normal(size=(3, pp)).astype(np.float32),
        price_price_explained_variance_ratio=np.ones(3, dtype=np.float32) / 3,
        price_volume_mean=np.zeros(pv, dtype=np.float32),
        price_volume_scale=np.ones(pv, dtype=np.float32),
        price_volume_components=rng.normal(size=(3, pv)).astype(np.float32),
        price_volume_explained_variance_ratio=np.ones(3, dtype=np.float32) / 3,
    )
    torch.manual_seed(0)
    return SignatureConditionedAffineMeanModel(
        input_channels=len(FEATURE_COLUMNS),
        signature_depth=2,
        latent_dim=latent_dim,
        hidden_dim=8,
        num_symbols=num_symbols,
        cross_area_pca=pca,
        diffusion_rank=diffusion_rank,
        jump_mode=jump_mode,
        stock_embedding_dim=stock_embedding_dim,
        cross_attention_layers=cross_attention_layers,
        attention_dim=16,
        attention_heads=4,
    ).float()


def test_model_distribution_and_gradients() -> None:
    torch.set_default_dtype(torch.float32)
    model = _toy_model(diffusion_rank=2)
    path = torch.randn(3, 5, 30, len(FEATURE_COLUMNS))
    mean, variance = model.forward_distribution(path)
    assert mean.shape == variance.shape == (3, 5)
    assert torch.allclose(mean, model(path))
    assert (variance > 0).all()
    target = torch.randn(3, 5) * 1e-3
    mask = torch.ones(3, 5, dtype=torch.bool)
    loss = GaussianPredictiveNLL(1e-3, 8.0)(mean, variance, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    for name in ("diffusion_network.3.weight", "log_observation_std", "coefficient_network.1.weight", "readout.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    print(f"PASS forward_distribution shapes, positive variance, finite non-zero gradients (NLL {loss.item():.3f})")
    torch.set_default_dtype(torch.float64)


def test_rank_zero_is_original_model() -> None:
    torch.set_default_dtype(torch.float32)
    model = _toy_model(diffusion_rank=0)
    assert not model.has_diffusion
    assert not any(key.startswith("diffusion_network") or key == "log_observation_std" for key in model.state_dict())
    # Optionally load one real deterministic checkpoint to prove backward compatibility.
    from project_settings import TICKERS  # noqa: F401  (only to confirm the project imports)

    previous = sorted(Path("artifacts").glob("signature_affine_n25_sig91*pca16*/seed_42_best_signature_affine_mean.pt"))
    if previous:
        state = torch.load(previous[0], map_location="cpu", weights_only=True)
        assert not any(key.startswith("diffusion_network") for key in state)
        print(f"PASS diffusion_rank=0 keeps the original parameter set (checked {previous[0].parent.name[:40]}...)")
    else:
        print("PASS diffusion_rank=0 keeps the original parameter set")
    torch.set_default_dtype(torch.float64)


def test_merton_and_student_t_likelihoods_match_scipy() -> None:
    generator = torch.Generator().manual_seed(2)
    n = 400
    y = torch.randn(n, generator=generator) * 3.0
    mean = torch.randn(n, generator=generator) * 0.1
    gaussian_variance = torch.rand(n, generator=generator) + 0.5
    intensity = torch.rand(n, generator=generator) * 0.3
    jump_variance = torch.rand(n, generator=generator) * 8.0 + 1.0
    parameters = {
        "mean": mean,
        "gaussian_variance": gaussian_variance,
        "jump_intensity": intensity,
        "jump_variance": jump_variance,
    }
    merton = PredictiveNLL(1.0, 100.0, "merton", max_jumps=30).per_label_nll(parameters, y).numpy()
    reference = np.zeros(n)
    for k in range(60):
        reference += stats.poisson.pmf(k, intensity.numpy()) * stats.norm.pdf(
            y.numpy(), mean.numpy(), np.sqrt(gaussian_variance.numpy() + k * jump_variance.numpy())
        )
    merton_error = np.abs(merton + np.log(reference)).max()
    assert merton_error < 1e-4, merton_error  # the loss is evaluated in float32
    # With (near-)zero jump variance every Poisson component is the same Gaussian, so
    # the mixture must collapse to the Gaussian likelihood for any intensity.  (A
    # tiny intensity is not enough: far in the tail a 1e-8-weighted wide jump
    # component can still dominate the Gaussian density, which is correct.)
    degenerate = PredictiveNLL(1.0, 100.0, "merton").per_label_nll(
        {**parameters, "jump_variance": jump_variance * 0 + 1e-10}, y
    )
    gaussian = PredictiveNLL(1.0, 100.0, "none").per_label_nll(parameters, y)
    assert torch.allclose(degenerate, gaussian, atol=1e-5), (degenerate - gaussian).abs().max()
    dof = torch.full((n,), 5.0)
    student = PredictiveNLL(1.0, 100.0, "student_t").per_label_nll({**parameters, "dof": dof}, y).numpy()
    scale = np.sqrt(gaussian_variance.numpy() * 3.0 / 5.0)
    student_error = np.abs(student + stats.t.logpdf(y.numpy(), 5.0, mean.numpy(), scale)).max()
    assert student_error < 1e-4, student_error
    print(f"PASS Merton / Student-t likelihoods vs scipy (max err {merton_error:.1e}, {student_error:.1e})")


def test_jump_heads_receive_gradients() -> None:
    torch.set_default_dtype(torch.float32)
    path = torch.randn(3, 5, 30, len(FEATURE_COLUMNS))
    target = torch.randn(3, 5) * 3e-3
    mask = torch.ones(3, 5, dtype=torch.bool)
    for mode, names in (
        ("merton", ("jump_network.3.weight", "jump_network.3.bias", "diffusion_network.3.weight")),
        ("student_t", ("log_dof_minus_two", "diffusion_network.3.weight")),
    ):
        model = _toy_model(diffusion_rank=2, jump_mode=mode)
        parameters = model.forward_predictive(path)
        assert torch.allclose(parameters["mean"], model(path))
        assert (parameters["total_variance"] >= parameters["gaussian_variance"] - 1e-6).all()
        loss = PredictiveNLL(1e-3, 20.0, mode).forward(parameters, target, mask)
        loss.backward()
        for name in names:
            gradient = dict(model.named_parameters())[name].grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, (mode, name)
        print(f"PASS {mode}: finite non-zero gradients into its heads (NLL {loss.item():.3f})")
    torch.set_default_dtype(torch.float64)


def test_sector_neutral_rank_loss_ignores_sector_offsets() -> None:
    torch.set_default_dtype(torch.float32)
    generator = torch.Generator().manual_seed(4)
    sector_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    prediction = torch.randn(6, 10, generator=generator)
    target = torch.randn(6, 10, generator=generator)
    mask = torch.ones(6, 10, dtype=torch.bool)
    mask[0, 1] = False
    loss = CrossSectionalRankLoss(5, sector_ids)
    offsets = torch.tensor([10.0, -7.0, 3.0])[sector_ids]
    base, shifted = loss(prediction, target, mask), loss(prediction + offsets, target, mask)
    assert torch.allclose(base, shifted, atol=1e-5), (base, shifted)
    plain = CrossSectionalRankLoss(5)
    assert not torch.allclose(plain(prediction, target, mask), plain(prediction + offsets, target, mask))
    print(f"PASS sector-neutral rank loss is invariant to sector offsets ({base.item():.4f})")
    torch.set_default_dtype(torch.float64)


def test_embedding_and_cross_attention() -> None:
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(1)
    path = torch.randn(3, 5, 30, len(FEATURE_COLUMNS))
    model = _toy_model(diffusion_rank=2, jump_mode="merton", stock_embedding_dim=4, cross_attention_layers=1)
    model.eval()
    parameters = model.forward_predictive(path)
    assert parameters["mean"].shape == (3, 5) and torch.isfinite(parameters["total_variance"]).all()
    # The attention layer lets one stock's path move another stock's forecast.
    perturbed = path.clone()
    perturbed[:, 4, :, :] += 1.0
    with torch.no_grad():
        moved = (model(perturbed)[:, 0] - model(path)[:, 0]).abs().max().item()
    assert moved > 0, moved
    model.train()
    loss = PredictiveNLL(1e-3, 20.0, "merton")(model.forward_predictive(path), torch.randn(3, 5) * 3e-3, torch.ones(3, 5, dtype=torch.bool))
    loss.backward()
    named = dict(model.named_parameters())
    for name in ("stock_embedding.weight", "attention_input.1.weight", "cross_attention.layers.0.self_attn.in_proj_weight"):
        gradient = named[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    print(f"PASS stock embedding + cross-stock attention: gradients flow, cross-stock effect {moved:.2e}")
    torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    test_van_loan_matches_quadrature()
    test_van_loan_matches_monte_carlo()
    test_model_distribution_and_gradients()
    test_rank_zero_is_original_model()
    test_merton_and_student_t_likelihoods_match_scipy()
    test_jump_heads_receive_gradients()
    test_sector_neutral_rank_loss_ignores_sector_offsets()
    test_embedding_and_cross_attention()
    print("All linear-SDE checks passed.")
