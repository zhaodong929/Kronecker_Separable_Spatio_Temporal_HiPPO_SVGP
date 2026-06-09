"""Online baseline models for the processed ERA5 benchmark.

All baselines expose the same minimal interface:

- `fit_initial_task(times, coords, Y, Phi)`
- `update_block(times, coords, Y, Phi)`
- `predict(times, coords, Phi)`
- `name()`

Deterministic baselines estimate observation variance from historical residuals.
The GPyTorch baselines use likelihood predictive variance.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

import numpy as np


MIN_VARIANCE = 1e-6


@dataclass(frozen=True)
class PredictionResult:
    mean: np.ndarray
    variance: np.ndarray


class OnlineBaseline(Protocol):
    def fit_initial_task(self, times: np.ndarray, coords: np.ndarray, Y: np.ndarray, Phi: np.ndarray) -> None:
        ...

    def update_block(self, times: np.ndarray, coords: np.ndarray, Y: np.ndarray, Phi: np.ndarray) -> None:
        ...

    def predict(self, times: np.ndarray, coords: np.ndarray, Phi: np.ndarray) -> PredictionResult:
        ...

    def name(self) -> str:
        ...

    def diagnostics(self) -> dict[str, Any]:
        ...


def _ensure_2d_y(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must have shape [T, S]")
    return Y


def _append_history(
    current: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    times: np.ndarray,
    Y: np.ndarray,
    Phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float).reshape(-1)
    Y = _ensure_2d_y(Y)
    Phi = np.asarray(Phi, dtype=float)
    if current is None:
        return times.copy(), Y.copy(), Phi.copy()
    old_t, old_y, old_phi = current
    return (
        np.concatenate([old_t, times]),
        np.concatenate([old_y, Y], axis=0),
        np.concatenate([old_phi, Phi], axis=0),
    )


def _constant_variance(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size <= 1:
        return MIN_VARIANCE
    return float(max(np.var(residuals), MIN_VARIANCE))


class _HistoryBaseline:
    _history: tuple[np.ndarray, np.ndarray, np.ndarray] | None

    def __init__(self) -> None:
        self._history = None
        self._variance = 1.0
        self._residual_variance_source = "seen_training_history_after_update"

    def fit_initial_task(self, times: np.ndarray, coords: np.ndarray, Y: np.ndarray, Phi: np.ndarray) -> None:
        self._history = _append_history(None, times, Y, Phi)
        self._refit(coords)

    def update_block(self, times: np.ndarray, coords: np.ndarray, Y: np.ndarray, Phi: np.ndarray) -> None:
        self._history = _append_history(self._history, times, Y, Phi)
        self._refit(coords)

    def _refit(self, coords: np.ndarray) -> None:
        raise NotImplementedError

    def _history_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._history is None:
            raise RuntimeError("Baseline must be fitted before prediction")
        return self._history

    def diagnostics(self) -> dict[str, Any]:
        _, Y, _ = self._history_arrays()
        return {
            "num_train": int(Y.size),
            "residual_variance": float(self._variance),
            "residual_variance_source": self._residual_variance_source,
            "predictive_variance_source": "training_residual_variance",
        }


class PersistenceBaseline(_HistoryBaseline):
    """Forecast each location by its most recent observed value."""

    def __init__(self) -> None:
        super().__init__()
        self._last: np.ndarray | None = None

    def name(self) -> str:
        return "persistence"

    def _refit(self, coords: np.ndarray) -> None:
        _, Y, _ = self._history_arrays()
        self._last = Y[-1].copy()
        one_step = Y[1:] - Y[:-1]
        self._variance = _constant_variance(one_step)

    def predict(self, times: np.ndarray, coords: np.ndarray, Phi: np.ndarray) -> PredictionResult:
        if self._last is None:
            raise RuntimeError("PersistenceBaseline must be fitted before prediction")
        T = len(times)
        mean = np.repeat(self._last[None, :], T, axis=0)
        var = np.full_like(mean, self._variance, dtype=float)
        return PredictionResult(mean=mean, variance=var)


class ClimatologyBaseline(_HistoryBaseline):
    """Forecast each location by its historical mean."""

    def __init__(self) -> None:
        super().__init__()
        self._mean: np.ndarray | None = None
        self._per_location_var: np.ndarray | None = None

    def name(self) -> str:
        return "climatology"

    def _refit(self, coords: np.ndarray) -> None:
        _, Y, _ = self._history_arrays()
        self._mean = np.mean(Y, axis=0)
        residuals = Y - self._mean[None, :]
        self._per_location_var = np.maximum(np.var(residuals, axis=0), MIN_VARIANCE)
        self._variance = _constant_variance(residuals)

    def predict(self, times: np.ndarray, coords: np.ndarray, Phi: np.ndarray) -> PredictionResult:
        if self._mean is None or self._per_location_var is None:
            raise RuntimeError("ClimatologyBaseline must be fitted before prediction")
        T = len(times)
        return PredictionResult(
            mean=np.repeat(self._mean[None, :], T, axis=0),
            variance=np.repeat(self._per_location_var[None, :], T, axis=0),
        )


class RidgeBaseline(_HistoryBaseline):
    """Closed-form ridge regression on the shared Phi design matrix."""

    def __init__(self, ridge: float = 1e-3) -> None:
        super().__init__()
        self.ridge = float(ridge)
        self._coef: np.ndarray | None = None

    def name(self) -> str:
        return "ridge"

    def _refit(self, coords: np.ndarray) -> None:
        _, Y, Phi = self._history_arrays()
        y_vec = Y.reshape(-1)
        if Phi.shape[0] != y_vec.shape[0]:
            raise ValueError(f"Phi rows {Phi.shape[0]} do not match flattened Y length {y_vec.shape[0]}")
        eye = np.eye(Phi.shape[1])
        self._coef = np.linalg.solve(Phi.T @ Phi + self.ridge * eye, Phi.T @ y_vec)
        residuals = y_vec - Phi @ self._coef
        self._variance = _constant_variance(residuals)

    def predict(self, times: np.ndarray, coords: np.ndarray, Phi: np.ndarray) -> PredictionResult:
        if self._coef is None:
            raise RuntimeError("RidgeBaseline must be fitted before prediction")
        T = len(times)
        S = coords.shape[0]
        mean = (np.asarray(Phi, dtype=float) @ self._coef).reshape(T, S)
        var = np.full_like(mean, self._variance, dtype=float)
        return PredictionResult(mean=mean, variance=var)


def _torch_modules():
    try:
        import torch
        import gpytorch
    except ImportError as exc:  # pragma: no cover - depends on optional env.
        raise ImportError("GPyTorch baselines require torch and gpytorch to be installed.") from exc
    return torch, gpytorch


def _make_input_normalizer(
    times: np.ndarray,
    coords: np.ndarray,
    Phi: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    times = np.asarray(times, dtype=float).reshape(-1)
    coords = np.asarray(coords, dtype=float)
    normalizer: dict[str, np.ndarray | float] = {
        "time_origin": float(np.min(times)),
        "time_scale": max(float(np.max(times) - np.min(times)), 1e-12) if len(times) > 1 else 1.0,
        "coord_mean": coords.mean(axis=0, keepdims=True),
        "coord_scale": np.maximum(coords.std(axis=0, keepdims=True), 1e-8),
    }
    if Phi is not None:
        phi = np.asarray(Phi, dtype=float)
        normalizer["phi_mean"] = phi.mean(axis=0, keepdims=True)
        normalizer["phi_scale"] = np.maximum(phi.std(axis=0, keepdims=True), 1e-8)
    return normalizer


def _flatten_inputs(
    times: np.ndarray,
    coords: np.ndarray,
    normalizer: dict[str, np.ndarray | float],
    Phi: np.ndarray | None = None,
    *,
    use_phi_features: bool = False,
) -> np.ndarray:
    times = np.asarray(times, dtype=float).reshape(-1)
    coords = np.asarray(coords, dtype=float)
    t_scaled = (times - float(normalizer["time_origin"])) / float(normalizer["time_scale"])
    coord_mean = np.asarray(normalizer["coord_mean"], dtype=float)
    coord_scale = np.asarray(normalizer["coord_scale"], dtype=float)
    coords_scaled = (coords - coord_mean) / coord_scale
    base = np.column_stack(
        [
            np.repeat(t_scaled, coords.shape[0]),
            np.tile(coords_scaled[:, 0], len(times)),
            np.tile(coords_scaled[:, 1], len(times)),
        ]
    )
    if not use_phi_features:
        return base
    if Phi is None:
        raise ValueError("Phi is required when use_phi_features=True")
    phi = np.asarray(Phi, dtype=float)
    phi_mean = np.asarray(normalizer["phi_mean"], dtype=float)
    phi_scale = np.asarray(normalizer["phi_scale"], dtype=float)
    phi_scaled = (phi - phi_mean) / phi_scale
    if phi_scaled.shape[0] != base.shape[0]:
        raise ValueError("Phi row count must match flattened x rows")
    return np.column_stack([base, phi_scaled])


class IndependentTemporalGPBaseline(_HistoryBaseline):
    """One GPyTorch exact GP per spatial location, using time as input."""

    def __init__(
        self,
        training_iterations: int = 20,
        learning_rate: float = 0.1,
        noise_lower_bound: float = 1e-5,
    ) -> None:
        super().__init__()
        self.training_iterations = int(training_iterations)
        self.learning_rate = float(learning_rate)
        self.noise_lower_bound = float(noise_lower_bound)
        self._models = []
        self._likelihoods = []
        self._time_origin = 0.0
        self._time_scale = 1.0
        self._training_losses: list[float] = []

    def name(self) -> str:
        return "independent_temporal_gp"

    def _refit(self, coords: np.ndarray) -> None:
        torch, gpytorch = _torch_modules()
        times, Y, _ = self._history_arrays()
        self._time_origin = float(np.min(times))
        self._time_scale = max(float(np.max(times) - np.min(times)), 1e-12)
        x = torch.as_tensor(((times - self._time_origin) / self._time_scale)[:, None], dtype=torch.float64)
        self._models = []
        self._likelihoods = []
        self._training_losses = []

        class ExactTemporalGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x_in):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x_in),
                    self.covar_module(x_in),
                )

        for s_idx in range(Y.shape[1]):
            y = torch.as_tensor(Y[:, s_idx], dtype=torch.float64)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(self.noise_lower_bound)
            ).double()
            model = ExactTemporalGP(x, y, likelihood).double()
            model.train()
            likelihood.train()
            opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(max(self.training_iterations, 0)):
                opt.zero_grad()
                loss = -mll(model(x), y)
                loss.backward()
                opt.step()
                self._training_losses.append(float(loss.detach().cpu()))
            self._models.append(model.eval())
            self._likelihoods.append(likelihood.eval())
        self._variance = MIN_VARIANCE

    def predict(self, times: np.ndarray, coords: np.ndarray, Phi: np.ndarray) -> PredictionResult:
        torch, gpytorch = _torch_modules()
        if not self._models:
            raise RuntimeError("IndependentTemporalGPBaseline must be fitted before prediction")
        t = np.asarray(times, dtype=float)
        x = torch.as_tensor(((t - self._time_origin) / self._time_scale)[:, None], dtype=torch.float64)
        means = []
        variances = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, likelihood in zip(self._models, self._likelihoods):
                pred = likelihood(model(x))
                means.append(pred.mean.detach().cpu().numpy())
                variances.append(pred.variance.detach().cpu().numpy())
        return PredictionResult(
            mean=np.stack(means, axis=1),
            variance=np.maximum(np.stack(variances, axis=1), MIN_VARIANCE),
        )

    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics()
        losses = np.asarray(self._training_losses, dtype=float)
        base.update(
            {
                "predictive_variance_source": "gpytorch_likelihood_predictive_variance",
                "gp_training_loss_first": float(losses[0]) if losses.size else float("nan"),
                "gp_training_loss_last": float(losses[-1]) if losses.size else float("nan"),
                "gp_training_loss_min": float(np.min(losses)) if losses.size else float("nan"),
                "gp_training_iterations": self.training_iterations,
                "gp_noise_lower_bound": self.noise_lower_bound,
            }
        )
        return base


class _FlattenedGPBase(_HistoryBaseline):
    def __init__(
        self,
        training_iterations: int = 30,
        learning_rate: float = 0.05,
        inducing_points: int = 32,
        minibatch_size: int | None = None,
        noise_lower_bound: float = 1e-5,
        inducing_init: str = "linspace",
        seed: int = 0,
        use_phi_features: bool = False,
        kernel_type: str = "rbf",
        fixed_lengthscale: float | None = None,
        fixed_noise: float | None = None,
        fixed_outputscale: float | None = None,
        freeze_kernel_hyperparams: bool = False,
    ) -> None:
        super().__init__()
        self.training_iterations = int(training_iterations)
        self.learning_rate = float(learning_rate)
        self.inducing_points = int(inducing_points)
        self.minibatch_size = None if minibatch_size is None else int(minibatch_size)
        self.noise_lower_bound = float(noise_lower_bound)
        self.inducing_init = inducing_init
        self.seed = int(seed)
        self.use_phi_features = bool(use_phi_features)
        self.kernel_type = str(kernel_type)
        self.fixed_lengthscale = None if fixed_lengthscale is None else float(fixed_lengthscale)
        self.fixed_noise = None if fixed_noise is None else float(fixed_noise)
        self.fixed_outputscale = None if fixed_outputscale is None else float(fixed_outputscale)
        self.freeze_kernel_hyperparams = bool(freeze_kernel_hyperparams)
        self._model = None
        self._likelihood = None
        self._input_normalizer: dict[str, np.ndarray | float] | None = None
        self._training_losses: list[float] = []
        self._selected_inducing_count = 0

    def _make_base_kernel(self, gpytorch, ard_num_dims: int):
        if self.kernel_type in {"rbf", "se", "squared_exponential"}:
            kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        elif self.kernel_type in {"matern32", "matern_3_2", "matern3/2"}:
            kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        else:
            raise ValueError("kernel_type must be one of: rbf, matern32")
        scale_kernel = gpytorch.kernels.ScaleKernel(kernel)
        if self.fixed_lengthscale is not None:
            import torch

            lengthscale = torch.full(
                (1, ard_num_dims),
                max(float(self.fixed_lengthscale), 1e-8),
                dtype=torch.float64,
            )
            kernel.lengthscale = lengthscale
        if self.fixed_outputscale is not None:
            scale_kernel.outputscale = max(float(self.fixed_outputscale), 1e-8)
        if self.freeze_kernel_hyperparams:
            for param in kernel.parameters():
                param.requires_grad_(False)
            scale_kernel.raw_outputscale.requires_grad_(False)
        return scale_kernel

    def _make_likelihood(self, gpytorch):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(self.noise_lower_bound)
        ).double()
        if self.fixed_noise is not None:
            likelihood.noise = max(float(self.fixed_noise), self.noise_lower_bound)
        if self.freeze_kernel_hyperparams and self.fixed_noise is not None:
            likelihood.raw_noise.requires_grad_(False)
        return likelihood

    def _train_x_y(self) -> tuple[np.ndarray, np.ndarray]:
        times, Y, _ = self._history_arrays()
        _, _, Phi = self._history_arrays()
        coords = self._coords
        if self._input_normalizer is None:
            self._input_normalizer = _make_input_normalizer(
                times,
                coords,
                Phi if self.use_phi_features else None,
            )
        return _flatten_inputs(
            times,
            coords,
            self._input_normalizer,
            Phi,
            use_phi_features=self.use_phi_features,
        ), Y.reshape(-1)

    def _refit(self, coords: np.ndarray) -> None:
        self._coords = np.asarray(coords, dtype=float)
        times, _, Phi = self._history_arrays()
        self._input_normalizer = _make_input_normalizer(
            times,
            self._coords,
            Phi if self.use_phi_features else None,
        )
        self._training_losses = []
        self._fit_gp()

    def _fit_gp(self) -> None:
        raise NotImplementedError

    def predict(self, times: np.ndarray, coords: np.ndarray, Phi: np.ndarray) -> PredictionResult:
        torch, gpytorch = _torch_modules()
        if self._model is None or self._likelihood is None:
            raise RuntimeError(f"{self.name()} must be fitted before prediction")
        if self._input_normalizer is None:
            raise RuntimeError(f"{self.name()} input normalizer is not fitted")
        x_np = _flatten_inputs(
            times,
            coords,
            self._input_normalizer,
            Phi,
            use_phi_features=self.use_phi_features,
        )
        x = torch.as_tensor(x_np, dtype=torch.float64)
        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(x))
        T = len(times)
        S = coords.shape[0]
        return PredictionResult(
            mean=pred.mean.detach().cpu().numpy().reshape(T, S),
            variance=np.maximum(pred.variance.detach().cpu().numpy().reshape(T, S), MIN_VARIANCE),
        )

    def _select_inducing(self, torch, x):
        n_inducing = min(max(1, self.inducing_points), x.shape[0])
        self._selected_inducing_count = int(n_inducing)
        if self.inducing_init == "random":
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            idx = torch.randperm(x.shape[0], generator=generator)[:n_inducing].sort().values
        elif self.inducing_init == "linspace":
            idx = torch.linspace(0, x.shape[0] - 1, n_inducing).long()
        else:
            raise ValueError("--gp-inducing-init must be linspace or random")
        return x[idx].clone()

    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics()
        losses = np.asarray(self._training_losses, dtype=float)
        base.update(
            {
                "predictive_variance_source": "gpytorch_likelihood_predictive_variance",
                "gp_training_loss_first": float(losses[0]) if losses.size else float("nan"),
                "gp_training_loss_last": float(losses[-1]) if losses.size else float("nan"),
                "gp_training_loss_min": float(np.min(losses)) if losses.size else float("nan"),
                "gp_training_iterations": self.training_iterations,
                "gp_learning_rate": self.learning_rate,
                "gp_inducing_points_requested": self.inducing_points,
                "gp_inducing_points_used": self._selected_inducing_count,
                "gp_inducing_init": self.inducing_init,
                "gp_minibatch_size": self.minibatch_size if self.minibatch_size is not None else 0,
                "gp_noise_lower_bound": self.noise_lower_bound,
                "gp_input_features": "time_lat_lon_phi" if self.use_phi_features else "time_lat_lon",
                "gp_kernel_type": self.kernel_type,
                "gp_fixed_lengthscale": self.fixed_lengthscale if self.fixed_lengthscale is not None else float("nan"),
                "gp_fixed_noise": self.fixed_noise if self.fixed_noise is not None else float("nan"),
                "gp_fixed_outputscale": self.fixed_outputscale if self.fixed_outputscale is not None else float("nan"),
                "gp_freeze_kernel_hyperparams": self.freeze_kernel_hyperparams,
            }
        )
        return base


class GPyTorchSGPRBaseline(_FlattenedGPBase):
    """Sparse GP regression on x=(time, lat, lon) using GPyTorch InducingPointKernel."""

    def name(self) -> str:
        return "gpytorch_sgpr_phi" if self.use_phi_features else "gpytorch_sgpr"

    def _fit_gp(self) -> None:
        torch, gpytorch = _torch_modules()
        x_np, y_np = self._train_x_y()
        x = torch.as_tensor(x_np, dtype=torch.float64)
        y = torch.as_tensor(y_np, dtype=torch.float64)
        inducing = self._select_inducing(torch, x)

        class SGPRModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, inducing_points):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                base = outer._make_base_kernel(gpytorch, train_x.shape[1])
                self.covar_module = gpytorch.kernels.InducingPointKernel(
                    base,
                    inducing_points=inducing_points,
                    likelihood=likelihood,
                )

            def forward(self, x_in):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x_in),
                    self.covar_module(x_in),
                )

        outer = self
        likelihood = self._make_likelihood(gpytorch)
        model = SGPRModel(x, y, likelihood, inducing).double()
        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(max(self.training_iterations, 0)):
            opt.zero_grad()
            loss = -mll(model(x), y)
            loss.backward()
            opt.step()
            self._training_losses.append(float(loss.detach().cpu()))
        self._model = model.eval()
        self._likelihood = likelihood.eval()


class GPyTorchSVGPBaseline(_FlattenedGPBase):
    """Variational sparse GP on x=(time, lat, lon) using GPyTorch."""

    def name(self) -> str:
        return "gpytorch_svgp_phi" if self.use_phi_features else "gpytorch_svgp"

    def _fit_gp(self) -> None:
        torch, gpytorch = _torch_modules()
        x_np, y_np = self._train_x_y()
        x = torch.as_tensor(x_np, dtype=torch.float64)
        y = torch.as_tensor(y_np, dtype=torch.float64)
        inducing = self._select_inducing(torch, x)

        outer = self

        class SVGPModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.size(0)
                )
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = outer._make_base_kernel(gpytorch, inducing_points.shape[1])

            def forward(self, x_in):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x_in),
                    self.covar_module(x_in),
                )

        likelihood = self._make_likelihood(gpytorch)
        model = SVGPModel(inducing).double()
        model.train()
        likelihood.train()
        opt = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=self.learning_rate)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.numel())
        batch_size = y.numel() if self.minibatch_size is None else min(max(1, self.minibatch_size), y.numel())
        dataset = torch.utils.data.TensorDataset(x, y)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        iterator = iter(loader)
        for _ in range(max(self.training_iterations, 0)):
            try:
                batch_x, batch_y = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch_x, batch_y = next(iterator)
            opt.zero_grad()
            loss = -mll(model(batch_x), batch_y)
            loss.backward()
            opt.step()
            self._training_losses.append(float(loss.detach().cpu()))
        self._model = model.eval()
        self._likelihood = likelihood.eval()


def make_baseline(name: str, **kwargs) -> OnlineBaseline:
    normalized = name.lower()
    if normalized == "persistence":
        return PersistenceBaseline()
    if normalized == "climatology":
        return ClimatologyBaseline()
    if normalized == "ridge":
        return RidgeBaseline(ridge=kwargs.get("ridge", 1e-3))
    if normalized in {"independent_temporal_gp", "independent_gp"}:
        return IndependentTemporalGPBaseline(
            training_iterations=kwargs.get("training_iterations", 20),
            learning_rate=kwargs.get("learning_rate", 0.1),
            noise_lower_bound=kwargs.get("noise_lower_bound", 1e-5),
        )
    if normalized in {"gpytorch_sgpr", "sgpr"}:
        return GPyTorchSGPRBaseline(
            training_iterations=kwargs.get("training_iterations", 30),
            learning_rate=kwargs.get("learning_rate", 0.05),
            inducing_points=kwargs.get("inducing_points", 32),
            minibatch_size=kwargs.get("minibatch_size"),
            noise_lower_bound=kwargs.get("noise_lower_bound", 1e-5),
            inducing_init=kwargs.get("inducing_init", "linspace"),
            seed=kwargs.get("seed", 0),
            kernel_type=kwargs.get("kernel_type", "rbf"),
            fixed_lengthscale=kwargs.get("fixed_lengthscale"),
            fixed_noise=kwargs.get("fixed_noise"),
            fixed_outputscale=kwargs.get("fixed_outputscale"),
            freeze_kernel_hyperparams=kwargs.get("freeze_kernel_hyperparams", False),
        )
    if normalized in {"gpytorch_sgpr_phi", "sgpr_phi"}:
        return GPyTorchSGPRBaseline(
            training_iterations=kwargs.get("training_iterations", 30),
            learning_rate=kwargs.get("learning_rate", 0.05),
            inducing_points=kwargs.get("inducing_points", 32),
            minibatch_size=kwargs.get("minibatch_size"),
            noise_lower_bound=kwargs.get("noise_lower_bound", 1e-5),
            inducing_init=kwargs.get("inducing_init", "linspace"),
            seed=kwargs.get("seed", 0),
            use_phi_features=True,
            kernel_type=kwargs.get("kernel_type", "rbf"),
            fixed_lengthscale=kwargs.get("fixed_lengthscale"),
            fixed_noise=kwargs.get("fixed_noise"),
            fixed_outputscale=kwargs.get("fixed_outputscale"),
            freeze_kernel_hyperparams=kwargs.get("freeze_kernel_hyperparams", False),
        )
    if normalized in {"gpytorch_svgp", "svgp"}:
        return GPyTorchSVGPBaseline(
            training_iterations=kwargs.get("training_iterations", 30),
            learning_rate=kwargs.get("learning_rate", 0.05),
            inducing_points=kwargs.get("inducing_points", 32),
            minibatch_size=kwargs.get("minibatch_size"),
            noise_lower_bound=kwargs.get("noise_lower_bound", 1e-5),
            inducing_init=kwargs.get("inducing_init", "linspace"),
            seed=kwargs.get("seed", 0),
            kernel_type=kwargs.get("kernel_type", "rbf"),
            fixed_lengthscale=kwargs.get("fixed_lengthscale"),
            fixed_noise=kwargs.get("fixed_noise"),
            fixed_outputscale=kwargs.get("fixed_outputscale"),
            freeze_kernel_hyperparams=kwargs.get("freeze_kernel_hyperparams", False),
        )
    if normalized in {"gpytorch_svgp_phi", "svgp_phi"}:
        return GPyTorchSVGPBaseline(
            training_iterations=kwargs.get("training_iterations", 30),
            learning_rate=kwargs.get("learning_rate", 0.05),
            inducing_points=kwargs.get("inducing_points", 32),
            minibatch_size=kwargs.get("minibatch_size"),
            noise_lower_bound=kwargs.get("noise_lower_bound", 1e-5),
            inducing_init=kwargs.get("inducing_init", "linspace"),
            seed=kwargs.get("seed", 0),
            use_phi_features=True,
            kernel_type=kwargs.get("kernel_type", "rbf"),
            fixed_lengthscale=kwargs.get("fixed_lengthscale"),
            fixed_noise=kwargs.get("fixed_noise"),
            fixed_outputscale=kwargs.get("fixed_outputscale"),
            freeze_kernel_hyperparams=kwargs.get("freeze_kernel_hyperparams", False),
        )
    raise ValueError(f"Unknown baseline: {name}")


def timer() -> float:
    return time.perf_counter()
