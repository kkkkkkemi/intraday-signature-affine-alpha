"""Sanity checks for the linear latent SDE layer.

Run directly (``python test_linear_sde.py``) from the same environment as
train_model.py; pytest also picks the ``test_*`` functions up.  No market data is
needed.  Every check prints PASS or raises.
"""

from __future__ import annotations

from pathlib import Path

import itertools

import numpy as np
import scipy.linalg as sla
import scipy.stats as stats
import torch

from minute_alpha import (
    FEATURE_COLUMNS,
    CrossTickerAreaPCA,
    CrossSectionalRankLoss,
    GaussianPredictiveNLL,
    MultiHorizonNLL,
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
    horizon_steps: int = 1,
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
        horizon_steps=horizon_steps,
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


# --------------------------------------------------------------------------- #
# Multi-horizon law and holding-period filter: numpy/scipy references
# --------------------------------------------------------------------------- #


def ref_moments(A, b, m0, Q, w, q, K, sigma_b, sigma_e):
    """Direct (non-recursive) formulas on tau_k = k/K."""
    d = A.shape[0]
    tau = np.arange(1, K + 1) / K
    aug = np.zeros((d + 1, d + 1)); aug[:d, :d] = A; aug[:d, d] = b
    means, P = [], []
    for t in tau:
        E = sla.expm(aug * t)
        m = E[:d, :d] @ m0 + E[:d, d]
        means.append(w @ (m - m0) + q * t)
        block = np.zeros((2 * d, 2 * d)); block[:d, :d] = -A * t; block[:d, d:] = Q * t; block[d:, d:] = A.T * t
        F = sla.expm(block)
        P.append(F[d:, d:].T @ F[:d, d:])
    cov = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            lo, hi = min(j, k), max(j, k)
            cov[j, k] = w @ P[lo] @ sla.expm(A.T * (tau[hi] - tau[lo])) @ w
    cov += sigma_b**2 * np.minimum.outer(tau, tau) + sigma_e**2 * np.eye(K)
    return np.array(means), cov


def _ref_quadrature_cov(A, Q, w, K, n=4000):
    """Latent part of Cov(y_j, y_k) by midpoint quadrature of int_0^{tau_j} e^{A(tau_j-s)} Q e^{A'(tau_k-s)} ds."""
    tau = np.arange(1, K + 1) / K
    cov = np.zeros((K, K))
    for j in range(K):
        for k in range(j, K):
            s = (np.arange(n) + 0.5) / n * tau[j]
            total = 0.0
            for si in s:
                total += w @ sla.expm(A * (tau[j] - si)) @ Q @ sla.expm(A.T * (tau[k] - si)) @ w
            cov[j, k] = cov[k, j] = total * tau[j] / n
    return cov


def _components(K, max_total):
    vectors = [v for v in itertools.product(range(max_total + 1), repeat=K) if sum(v) <= max_total]
    return np.array(vectors)


def ref_mixture(mean, cov, lam, s2, K, max_total):
    """Component covariances and log weights of the truncated Merton mixture."""
    counts = _components(K, max_total) if lam is not None else np.zeros((1, K), dtype=int)
    comps, logw = [], []
    for n in counts:
        cum = np.cumsum(n)
        C = cum[np.minimum.outer(np.arange(K), np.arange(K))]
        comps.append(cov + (s2 * C if lam is not None else 0.0))
        logw.append(np.sum(stats.poisson.logpmf(n, lam / K)) if lam is not None else 0.0)
    return np.array(comps), np.array(logw)


def ref_loglik(y, mean, cov, lam, s2, K, max_total=3):
    comps, logw = ref_mixture(mean, cov, lam, s2, K, max_total)
    terms = [lw + stats.multivariate_normal(mean, c).logpdf(y) for c, lw in zip(comps, logw)]
    return float(np.logaddexp.reduce(terms))


def ref_conditional(y, mean, cov, lam, s2, K, m, max_total=3):
    comps, logw = ref_mixture(mean, cov, lam, s2, K, max_total)
    last, mid = K - 1, m - 1
    mus, vars_, logev = [], [], []
    for S in comps:
        Saa = S[:m, :m]
        c = S[last, :m] - S[mid, :m]
        vr = S[last, last] + S[mid, mid] - 2 * S[last, mid]
        sol = np.linalg.solve(Saa, y[:m] - mean[:m])
        mus.append(mean[last] - mean[mid] + c @ sol)
        vars_.append(vr - c @ np.linalg.solve(Saa, c))
        logev.append(stats.multivariate_normal(mean[:m], Saa).logpdf(y[:m]))
    post = np.array(logw) + np.array(logev)
    post = np.exp(post - np.logaddexp.reduce(post))
    mus, vars_ = np.array(mus), np.array(vars_)
    mu = post @ mus
    return mu, post @ (vars_ + mus**2) - mu**2


def simulate(mean, cov, lam, s2, K, n, rng):
    """Draws of the cumulative-return vector: Gaussian part + independent jumps per sub-interval."""
    g = rng.multivariate_normal(mean, cov, size=n)
    counts = rng.poisson(lam / K, size=(n, K))
    jumps = rng.normal(size=(n, K)) * np.sqrt(s2 * counts)
    return g + np.cumsum(jumps, axis=1)


def test_horizon_reference_matches_quadrature_and_merton_marginal() -> None:
    rng = np.random.default_rng(0)
    d, K = 4, 6
    raw = rng.normal(size=(d, d))
    A = 0.25 * (raw - raw.T) - np.diag(np.log1p(np.exp(rng.normal(size=d))) + 0.05)
    L = rng.normal(size=(d, 2))
    Q = L @ L.T + 1e-3 * np.eye(d)
    w = rng.normal(size=d)
    mean, cov = ref_moments(A, rng.normal(size=d), rng.normal(size=d), Q, w, 0.3, K, 0.7, 0.05)
    tau = np.arange(1, K + 1) / K
    latent = cov - 0.49 * np.minimum.outer(tau, tau) - 0.0025 * np.eye(K)
    quad = _ref_quadrature_cov(A, Q, w, K, n=150)
    assert np.abs(latent - quad).max() < 1e-4 * np.abs(quad).max()
    lam, s2, y = 0.6, 2.0, rng.normal(size=K) * 2
    comps, logw = ref_mixture(mean, cov, lam, s2, K, 6)
    marginal = np.logaddexp.reduce([lw + stats.norm(mean[-1], np.sqrt(c[-1, -1])).logpdf(y[-1]) for c, lw in zip(comps, logw)])
    endpoint = np.logaddexp.reduce(
        [stats.poisson.logpmf(n, lam) + stats.norm(mean[-1], np.sqrt(cov[-1, -1] + n * s2)).logpdf(y[-1]) for n in range(7)]
    )
    assert abs(marginal - endpoint) < 1e-10
    print("PASS multi-horizon reference: Van Loan cross-covariance = quadrature; jump mixture marginal = endpoint Merton")


def test_filter_reference_is_calibrated_by_monte_carlo() -> None:
    rng = np.random.default_rng(1)
    K = 6
    L = rng.normal(size=(K, K)) * 0.3
    cov = L @ L.T + np.minimum.outer(np.arange(1, K + 1), np.arange(1, K + 1)) / K
    mean = rng.normal(size=K) * 0.2
    lam, s2 = 0.6, 2.0
    draws = simulate(mean, cov, lam, s2, K, 1200, rng)
    for m in (1, 3, 5):
        out = np.array([ref_conditional(x, mean, cov, lam, s2, K, m) for x in draws])
        residual = (draws[:, -1] - draws[:, m - 1]) - out[:, 0]
        assert abs(residual.mean()) < 4 * residual.std() / np.sqrt(len(residual))
        assert abs(np.corrcoef(residual, draws[:, m - 1])[0, 1]) < 0.08
        assert 0.85 < np.mean(residual**2 / out[:, 1]) < 1.15
    print("PASS filter reference: unbiased, residual uncorrelated with the observed path, variance calibrated (Monte Carlo)")


def test_forward_horizon_matches_reference() -> None:
    torch.set_default_dtype(torch.float32)
    model = _toy_model(diffusion_rank=2, jump_mode="merton", horizon_steps=6)
    model.eval()
    path = torch.randn(2, 5, 30, len(FEATURE_COLUMNS))
    with torch.no_grad():
        law = model.forward_horizon(path)
        endpoint = model.forward_predictive(path)
        context, m0, A, b = model._dynamics(path)
        G = model._diffusion_matrix(context)
        Q = G @ G.transpose(-1, -2) + model.diffusion_floor * torch.eye(model.latent_dim)
    w = model.readout.weight.detach().double().numpy().ravel()
    q = float(model.readout.bias.detach())
    sb = float(torch.exp(model.log_observation_std))
    se = float(torch.exp(model.log_measurement_std))
    mean_t = law["horizon_mean"].reshape(-1, 6).double().numpy()
    cov_t = law["horizon_covariance"].reshape(-1, 6, 6).double().numpy()
    for i in range(A.shape[0]):
        mean_r, cov_r = ref_moments(
            A[i].double().numpy(), b[i].double().numpy(), m0[i].double().numpy(), Q[i].double().numpy(), w, q, 6, sb, se
        )
        assert np.allclose(mean_t[i], mean_r, rtol=1e-4, atol=1e-5), (mean_t[i], mean_r)
        assert np.allclose(cov_t[i], cov_r, rtol=1e-4, atol=1e-6)
    assert torch.allclose(law["mean"], endpoint["mean"], rtol=1e-4, atol=1e-5)
    assert torch.allclose(law["gaussian_variance"], endpoint["gaussian_variance"], rtol=1e-4, atol=1e-6)
    print("PASS forward_horizon: recursive moments = direct matrix-exponential formulas; endpoint = forward_predictive")
    torch.set_default_dtype(torch.float64)


def test_multi_horizon_nll_and_filter_match_reference() -> None:
    rng = np.random.default_rng(2)
    K, n = 6, 5
    L = rng.normal(size=(n, K, K)) * 0.3
    cov = L @ L.transpose(0, 2, 1) + np.minimum.outer(np.arange(1, K + 1), np.arange(1, K + 1)) / K
    mean = rng.normal(size=(n, K)) * 0.2
    lam = rng.uniform(0.05, 1.0, size=n)
    s2 = rng.uniform(0.5, 4.0, size=n)
    y = mean + rng.normal(size=(n, K)) * 1.5
    for mode in ("none", "merton"):
        nll = MultiHorizonNLL(1.0, 50.0, mode, K, max_jumps=3)
        parameters = {
            "horizon_mean": torch.tensor(mean).view(1, n, K),
            "horizon_covariance": torch.tensor(cov).view(1, n, K, K),
            "jump_intensity": torch.tensor(lam).view(1, n),
            "jump_variance": torch.tensor(s2).view(1, n),
        }
        target = torch.tensor(y).view(1, n, K)
        got = nll.log_likelihood(parameters, target).numpy()
        want = np.array([
            ref_loglik(y[i], mean[i], cov[i], lam[i] if mode == "merton" else None, s2[i], K, 3) for i in range(n)
        ])
        assert np.allclose(got, want, rtol=1e-4, atol=1e-4), (mode, got, want)
        for m in (1, 2, 5):
            cm, cv = nll.conditional_remaining(parameters, target, m)
            for i in range(n):
                rm, rv = ref_conditional(y[i], mean[i], cov[i], lam[i] if mode == "merton" else None, s2[i], K, m, 3)
                assert abs(float(cm[i]) - rm) < 1e-4 * (1 + abs(rm)) and abs(float(cv[i]) - rv) < 1e-4 * (1 + rv), (mode, m, i)
    print("PASS MultiHorizonNLL and filter update = scipy mixture likelihood / Gaussian conditioning (Gaussian and Merton)")


def test_multi_horizon_gradients() -> None:
    torch.set_default_dtype(torch.float32)
    model = _toy_model(diffusion_rank=2, jump_mode="merton", horizon_steps=6)
    path = torch.randn(3, 5, 30, len(FEATURE_COLUMNS))
    law = model.forward_horizon(path)
    nll = MultiHorizonNLL(1e-3, 20.0, "merton", 6)
    loss = nll(law, torch.randn(3, 5, 6).cumsum(-1) * 2e-3, torch.ones(3, 5, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    named = dict(model.named_parameters())
    for name in ("diffusion_network.3.weight", "jump_network.3.weight", "coefficient_network.1.weight",
                 "readout.weight", "log_observation_std", "log_measurement_std"):
        gradient = named[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    print(f"PASS multi-horizon NLL: finite non-zero gradients for every SDE head (loss {loss.item():.3f})")
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
    test_horizon_reference_matches_quadrature_and_merton_marginal()
    test_filter_reference_is_calibrated_by_monte_carlo()
    test_forward_horizon_matches_reference()
    test_multi_horizon_nll_and_filter_match_reference()
    test_multi_horizon_gradients()
    print("All linear-SDE checks passed.")
