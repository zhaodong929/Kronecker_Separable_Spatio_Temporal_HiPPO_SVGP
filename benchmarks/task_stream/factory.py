"""Isolated-framework construction and initial fitting for the five selected methods.

Configuration is selected on a separate initial-period task stream. This module
never receives evaluation targets, selects a checkpoint, or rescales model time.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import numpy as np
from benchmarks.three_domain.geometry import farthest_indices
from benchmarks.three_domain.tracking import emit
from .protocol import Observations, Task
from .ablations import Arm

METHODS = ('kronhippo_svgp', 'osgpr', 'ohsvgp', 'st_svgp', 'mgpvae')


@dataclass(frozen=True)
class Configuration:
    method: str
    initial_iterations: int
    learning_rate: float
    seed: int = 0
    device: str = 'cpu'
    spatial_inducing: int = 32
    temporal_inducing: int = 32
    inducing_size: int = 32
    rff: int = 256
    online_iterations: int = 5
    batch_rows: int = 1024
    grid_rows: int = 1024
    latent: int = 2
    width: int = 16
    training_samples: int = 4
    prediction_samples: int = 512
    ridge: float = .001
    beta_prior_variance: float = 1.
    ell_t: float = .2
    ell_s: tuple = (1., 1.)
    kernel_variance: float = 1.
    noise_std: float = .2
    official_source: str = ''
    initial_expected_passes: float | None = None
    initial_max_seconds: float | None = None

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError('Method excluded by baseline-selection policy')
        for name in ('initial_iterations', 'spatial_inducing', 'temporal_inducing', 'inducing_size',
                     'rff', 'online_iterations', 'batch_rows', 'grid_rows', 'latent', 'width',
                     'training_samples', 'prediction_samples'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'Positive integer required: {name}')
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError('A nonnegative integer seed is required')
        if len(self.ell_s) != 2 or any(not np.isfinite(v) or v <= 0 for v in
            (*self.ell_s, self.learning_rate, self.ell_t, self.kernel_variance,
             self.noise_std, self.ridge, self.beta_prior_variance)):
            raise ValueError('Finite positive scales and learning rate required')
        if self.latent < 2 or self.prediction_samples < 2:
            raise ValueError('Pinned MGPVAE requires at least two latents and decoder samples')
        if self.initial_max_seconds is not None:
            seconds = self.initial_max_seconds
            if (isinstance(seconds, (bool, np.bool_))
                    or not isinstance(seconds, (int, float, np.integer, np.floating))
                    or not np.isfinite(seconds) or seconds <= 0):
                raise ValueError('Initial optimizer time budget must be finite and positive')
            if self.initial_expected_passes is not None:
                raise ValueError('Wall-time and expected-pass budgets are mutually exclusive')
        if self.initial_expected_passes is not None:
            passes = self.initial_expected_passes
            if self.method != 'ohsvgp':
                raise ValueError('Initial expected-pass policy applies only to OHSVGP')
            if (isinstance(passes, (bool, np.bool_))
                    or not isinstance(passes, (int, float, np.integer, np.floating))
                    or not np.isfinite(passes) or passes <= 0):
                raise ValueError('Initial expected passes must be finite and positive')
        if self.method == 'mgpvae' and (self.ell_t != .2 or tuple(self.ell_s) != (1., 1.)
                or self.kernel_variance != 1. or self.noise_std != .2):
            raise ValueError('MGPVAE uses its official latent-kernel initialization; GP theta fields do not apply')


    def resolve_initial_budget(self, observation_rows):
        """Resolve a hashed policy against the legal initial batch, without mutation.

        OH independently samples rows per optimizer step; exposure is expected
        sampling effort, not a guarantee that each row is visited every epoch.
        All other methods retain their declared full-data iteration count.
        """
        if (isinstance(observation_rows, (bool, np.bool_))
                or not isinstance(observation_rows, (int, np.integer)) or observation_rows < 1):
            raise ValueError('Initial observation rows must be a positive integer')
        rows = int(observation_rows)
        batch_rows = min(self.batch_rows, rows) if self.method == 'ohsvgp' else rows
        effective = self.initial_iterations
        policy = 'wall_time_with_iteration_safety_cap_v1' if self.initial_max_seconds is not None else 'fixed_steps_v1'
        if self.initial_expected_passes is not None:
            expected_steps = float(self.initial_expected_passes)*(rows/batch_rows)
            if not math.isfinite(expected_steps):
                raise ValueError('Expected-pass policy must resolve to a finite step count')
            effective = max(effective, math.ceil(expected_steps))
            policy = 'expected_row_exposures_v1'
        return dict(policy=policy, method=self.method, max_seconds=self.initial_max_seconds,
            observation_rows=rows,
            rows_per_step=batch_rows, minimum_iterations=1 if self.initial_max_seconds is not None else int(self.initial_iterations),
            iteration_count_role='safety_cap' if self.initial_max_seconds is not None else 'fixed_budget',
            requested_expected_passes=self.initial_expected_passes,
            effective_iterations=int(effective),
            sampled_rows=None if self.initial_max_seconds is not None else int(effective)*batch_rows,
            expected_row_exposures=None if self.initial_max_seconds is not None else int(effective)*batch_rows/rows,
            sampling=('independent_minibatches_without_replacement_within_each_step'
                      if self.method == 'ohsvgp' else 'full_data_per_step'))


class FeatureTable:
    def __init__(self, times, values):
        self.times, self.values = np.asarray(times), np.asarray(values)
        if self.times.ndim != 1 or self.values.ndim != 3 or len(self.times) != len(self.values):
            raise ValueError('Feature table must be time/site/feature aligned')
        if np.any(np.diff(self.times) <= 0) or not np.isfinite(self.values).all():
            raise ValueError('Finite features and strictly increasing time required')

    def __call__(self, times, sites):
        indices = np.searchsorted(self.times, times)
        if np.any(indices >= len(self.times)) or not np.array_equal(self.times[indices], times):
            raise ValueError('Feature and model clocks disagree')
        return self.values[indices][:, sites].reshape(-1, self.values.shape[-1])


class FittedTaskAdapter:
    initialization_phase = 'initial_refit'

    def __init__(self, config, coordinates, visible, features, *, initial_step, release_previous=False,
                 arm=Arm('joint_transfer'), refit_iterations=None):
        self.config = config if isinstance(config, Configuration) else Configuration(**config)
        if refit_iterations is not None:
            if (isinstance(refit_iterations, (bool, np.bool_))
                    or not isinstance(refit_iterations, (int, np.integer)) or refit_iterations < 1):
                raise ValueError('Selected refit iterations must be a positive integer')
            if self.config.initial_max_seconds is None:
                raise ValueError('Selected refit mapping requires a wall-time search configuration')
        self.refit_iterations = None if refit_iterations is None else int(refit_iterations)
        self.fit_budget_record = None
        self.coordinates, self.visible = np.asarray(coordinates), np.asarray(visible)
        self.features, self.initial_step, self.arm = features, float(initial_step), arm
        self.release_previous = bool(release_previous)
        if self.initial_step <= 0 or not np.isfinite(self.initial_step):
            raise ValueError('Positive canonical model step required')
        if self.config.method != 'kronhippo_svgp' and arm.name != 'joint_transfer':
            raise ValueError('Contribution ablations apply only to the proposed method')
        self.adapter = None
        self.beta = None
        self.inverse = None
        self.predictive_family = 'gaussian_mixture' if self.config.method == 'mgpvae' else 'gaussian'

    def _mean(self, times, sites):
        return (self.features(times, sites) @ self.beta).reshape(len(times), len(sites))

    def _batch(self, batch):
        if batch is None:
            return None
        sites = batch.sites if self.inverse is None else self.inverse[batch.sites]
        values = batch.values if self.beta is None else batch.values-self._mean(batch.times, batch.sites)
        return Observations(batch.times, sites, values)

    def initialize(self, initial):
        if self.adapter is not None:
            raise ValueError('A fitted run may only be initialized once')
        from .training_budget import FitClock
        c = self.config
        self.fit_clock = None
        def fit_clock(synchronize):
            self.fit_clock = FitClock(c.initial_max_seconds, synchronize) if c.initial_max_seconds is not None and self.refit_iterations is None else None
            return self.fit_clock
        self.training_budget = c.resolve_initial_budget(initial.values.size)
        if self.refit_iterations is not None:
            self.training_budget.update(policy='selected_fixed_refit',
                effective_iterations=self.refit_iterations, iteration_count_role='selected_fixed_budget',
                minimum_iterations=self.refit_iterations,
                sampled_rows=self.refit_iterations*self.training_budget['rows_per_step'],
                expected_row_exposures=self.refit_iterations*self.training_budget['rows_per_step']/initial.values.size,
                reason='frozen iteration mapping from selected completed wall-time validation artifact')
        initial_iterations = self.training_budget['effective_iterations']
        emit('training_budget', 0, self.training_budget)
        coordinates = self.coordinates
        from .gp import KronTaskAdapter, OSGPRTaskAdapter, OHSVGPTaskAdapter
        if c.method != 'kronhippo_svgp':
            phi = self.features(initial.times, initial.sites)
            self.beta = np.linalg.solve(phi.T @ phi + c.ridge*np.eye(phi.shape[1]),
                                        phi.T @ initial.values.reshape(-1))
        batch = self._batch(initial)
        theta = dict(ell_t=c.ell_t, ell_s=c.ell_s, kernel_variance=c.kernel_variance, noise_std=c.noise_std)
        if c.method in ('kronhippo_svgp', 'osgpr', 'st_svgp'):
            available = coordinates[initial.sites]
            if c.spatial_inducing > len(available):
                raise ValueError('Declared inducing count exceeds initial observed sites')
            inducing = available[farthest_indices(available, c.spatial_inducing)]
        if c.method == 'kronhippo_svgp':
            import torch
            from stvgp_kronecker.routeb_empirical_bayes import BatchRouteBEmpiricalBayes
            from stvgp_kronecker.temporal_analytic import TemporalBlockSpec
            model = BatchRouteBEmpiricalBayes(times=initial.times.copy(), spatial_inducing=inducing,
                mt=c.temporal_inducing, representation='analytic_hippo_rff', initial_ell_t=c.ell_t,
                initial_ell_s=c.ell_s, initial_kernel_variance=c.kernel_variance,
                initial_noise_std=c.noise_std, rff_sample_size=c.rff, seed=c.seed, objective_type='vfe',
                temporal_horizon=TemporalBlockSpec(float(initial.times[0]-self.initial_step),
                    float(initial.times[-1]), len(initial.times), phase_origin=float(initial.times[0]-self.initial_step)))
            model = model.to(device=c.device, dtype=torch.float64)
            tensor = lambda x: torch.as_tensor(np.array(x), dtype=torch.float64, device=c.device)
            phi = self.features(initial.times, initial.sites).reshape(len(initial.times), len(initial.sites), -1)
            yy, pp, xx = tensor(initial.values.T), tensor(phi.transpose(1, 0, 2)), tensor(coordinates[initial.sites])
            clock = fit_clock((lambda: torch.cuda.synchronize(c.device)) if c.device.startswith('cuda') else lambda: None)
            if clock is not None: clock.start()
            optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
            for iteration in range(initial_iterations):
                optimizer.zero_grad(set_to_none=True)
                objective = model.objective(y_matrix=yy, phi_tensor=pp, spatial_coordinates=xx,
                                            beta_prior_variance=c.beta_prior_variance)
                loss = objective.nlml_per_observation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite KronHiPPO VFE objective')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 20., error_if_nonfinite=True)
                optimizer.step(); model.clamp_parameters()
                emit('train', iteration+1, dict(negative_elbo_per_observation=float(loss.detach()),
                     gradient_norm=float(norm), theta=model.theta()))
                if clock is not None and clock.should_stop(iteration+1): break
            if clock is not None: clock.finish(iteration+1)
            theta = model.theta()
            self.adapter = KronTaskAdapter(coordinates, self.visible, inducing, theta,
                initial_step=self.initial_step, features=self.features, feature_dimension=phi.shape[-1],
                mt=c.temporal_inducing, rff=c.rff, seed=c.seed, device=c.device,
                multiple_geometry=(not np.array_equal(initial.sites, self.visible) or self.release_previous),
                arm=self.arm, beta_prior_variance=c.beta_prior_variance)
        elif c.method == 'osgpr':
            from scripts.run_official_bui_osgpr_era5 import product_inducing
            from tensorflow.python.eager import context as tf_context
            clock = fit_clock(tf_context.async_wait)
            self.adapter = OSGPRTaskAdapter(coordinates, theta,
                product_inducing(initial.times, inducing, c.temporal_inducing),
                initial_steps=initial_iterations, update_steps=c.online_iterations, learning_rate=c.learning_rate,
                fit_clock=clock)
        elif c.method == 'ohsvgp':
            import torch
            from scripts.run_covid_ohsvgp_own_theta import SE_kernel, GaussianLikelihood
            kernel = SE_kernel(3, device=torch.device(c.device)).to(dtype=torch.float64, device=c.device)
            with torch.no_grad():
                kernel.log_ls.copy_(torch.tensor([c.ell_t, *c.ell_s], dtype=torch.float64, device=c.device).log())
                kernel.log_sf.fill_(np.log(c.kernel_variance))
            likelihood = GaussianLikelihood(c.noise_std**2).to(dtype=torch.float64, device=c.device)
            clock = fit_clock((lambda: torch.cuda.synchronize(c.device)) if c.device.startswith('cuda') else lambda: None)
            self.adapter = OHSVGPTaskAdapter(coordinates, kernel, likelihood, inducing_size=c.inducing_size,
                rff=c.rff, initial_steps=initial_iterations, update_steps=c.online_iterations,
                batch_rows=c.batch_rows, grid_rows=c.grid_rows, learning_rate=c.learning_rate, seed=c.seed, device=c.device,
                fit_clock=clock)
        elif c.method == 'st_svgp':
            from baselines.covid_long_setting_b.adapters.run_st_svgp import train_task1
            from baselines.st_svgp_task_training import make_compact_model as make_model
            from .markov import STTaskAdapter
            model = make_model(batch.times[:, None], np.repeat(coordinates[None, batch.sites], len(batch.times), axis=0),
                batch.values, inducing, trainable_inducing=False, theta=theta)
            import jax
            clock = fit_clock(lambda: jax.block_until_ready(model.vars().tensors()))
            train_task1(model, iterations=initial_iterations, check_interval=initial_iterations,
                min_steps=initial_iterations, plateau_checks=10, plateau_relative_improvement=0.,
                adam_lr=c.learning_rate, newton_lr=1., checkpoint_directory=None,
                seed=c.seed, spatial_inducing=c.spatial_inducing, fit_clock=clock)
            self.adapter = STTaskAdapter(model, coordinates)
        else:
            from baselines.mgpvae.official import make_model
            from .markov import MGPTaskAdapter
            import jax
            import jax.numpy as jnp
            import objax
            def create(coords):
                return make_model(c.official_source, coords, seed=c.seed, latent=c.latent, width=c.width,
                    correct_spatial_covariance=True, compact_spatial_marginals=True,
                    sitewise_training_filter=True, compact_task_training=True)
            fitted = create(coordinates[initial.sites])
            training = jnp.asarray(batch.values.T[..., None])
            optimizer = None
            clock = fit_clock(lambda: jax.block_until_ready(fitted.vars().tensors()
                + ([] if optimizer is None else optimizer.vars().tensors())))
            if clock is not None: clock.start()
            optimizer = objax.optimizer.Adam(fitted.vars())
            gradient = objax.GradValues(fitted.energy, fitted.vars())
            @objax.Function.with_vars(fitted.vars()+optimizer.vars())
            def update(iteration):
                gradients, values = gradient(training, jax.random.PRNGKey(c.seed*100000+iteration),
                    t=jnp.asarray(initial.times[:, None]), num_samples=c.training_samples)
                optimizer(c.learning_rate, gradients)
                return values
            update = objax.Jit(update)
            for iteration in range(initial_iterations):
                values = np.asarray(update(iteration))
                if not np.isfinite(values).all():
                    raise FloatingPointError('Nonfinite MGPVAE initial objective')
                emit('train', iteration+1, dict(negative_elbo=float(values[0]),
                    negative_expected_log_likelihood=float(values[1]), kl=float(values[2])))
                if clock is not None and clock.should_stop(iteration+1): break
            if clock is not None: clock.finish(iteration+1)
            ordering = np.concatenate([initial.sites, np.setdiff1d(np.arange(len(coordinates)), initial.sites)])
            self.inverse = np.empty(len(coordinates), dtype=int)
            self.inverse[ordering] = np.arange(len(coordinates))
            full = create(coordinates[ordering]); full.vars().assign(fitted.vars().tensors())
            self.adapter = MGPTaskAdapter(full, coordinates[ordering], samples=c.prediction_samples, seed=c.seed)
            batch = self._batch(initial)
        self.adapter.initialize(batch)
        if self.fit_clock is not None:
            if self.fit_clock.record is None:
                raise RuntimeError('Timed initial optimizer did not finalize its budget record')
            self.fit_budget_record = dict(self.fit_clock.record)
        else:
            self.fit_budget_record = dict(policy='selected_fixed_refit' if self.refit_iterations is not None else self.training_budget['policy'],
                completed_steps=initial_iterations, max_seconds=c.initial_max_seconds, elapsed_seconds=None,
                stop_reason='selected_iteration_budget_completed' if self.refit_iterations is not None else 'fixed_iteration_budget_completed',
                convergence_claimed=False)
        self.fit_budget_record.update(observation_rows=int(initial.values.size),
            rows_per_step=self.training_budget['rows_per_step'],
            sampled_rows=self.fit_budget_record['completed_steps']*self.training_budget['rows_per_step'])
        self.fit_budget_record['expected_row_exposures'] = self.fit_budget_record['sampled_rows']/initial.values.size
        emit('fit_budget', self.fit_budget_record['completed_steps'], self.fit_budget_record)
        if c.method == 'kronhippo_svgp':
            # Fingerprint the actual common fit, excluding the intentionally
            # intervened posterior and arm label. Both data and random features
            # must agree before predictive ablations may be paired.
            digest = hashlib.sha256(json.dumps(theta, sort_keys=True).encode())
            for value in (initial.times, initial.sites, initial.values,
                          self.features(initial.times, initial.sites), inducing,
                          self.adapter.builder.base_frequencies.detach().cpu().numpy()):
                value = np.asarray(value)
                digest.update(str((value.dtype.str, value.shape)).encode())
                digest.update(value.tobytes())
            self.fit_sha256 = digest.hexdigest()
            emit('fit_identity', 0, dict(fit_sha256=self.fit_sha256, arm=self.arm.name,
                 interpretation='fitted hyperparameters, training inputs/features and base spectral draws'))
        emit('fit_configuration', 0, dict(configuration=asdict(c), arm=asdict(self.arm),
             initial_sites=initial.sites.tolist(), initial_training_budget=self.training_budget,
             initial_optimizer_timing=self.fit_budget_record, learned_theta=self.adapter.theta if c.method == 'osgpr' else theta if c.method == 'kronhippo_svgp' else None,
             mgp_initial_parameters=dict(spatial_lengthscale=2., latent_temporal_lengthscale=5.,
                latent_variance=1., decoder_noise_variance=1.) if c.method == 'mgpvae' else None))

    def checkpoint_state(self):
        from .checkpoint import adapter_snapshot
        return adapter_snapshot(self)

    def diagnostics(self):
        """Device memory is separate from the supervisor's whole-process RSS."""
        c = self.config
        if c.method in ('kronhippo_svgp', 'ohsvgp'):
            import torch
            if c.device.startswith('cuda'):
                return dict(gpu_allocated_bytes=torch.cuda.memory_allocated(c.device),
                    gpu_reserved_bytes=torch.cuda.memory_reserved(c.device),
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(c.device),
                    gpu_memory_scope='framework allocator; process lifetime high water')
        elif c.method in ('st_svgp', 'mgpvae'):
            import jax
            return dict(device_memory=[dict(device=str(d), statistics=d.memory_stats()) for d in jax.devices()],
                        gpu_memory_scope='JAX allocator statistics where available')
        else:
            import tensorflow as tf
            if c.device.startswith('cuda'):
                return dict(gpu_memory=tf.config.experimental.get_memory_info('GPU:0'),
                            gpu_memory_scope='TensorFlow allocator current/peak bytes')
        return dict(gpu_memory_status='not_applicable_cpu')

    def predict_task(self, task):
        inner = Task(task.index, task.start, task.stop, self._batch(task.visible), self._batch(task.delayed),
                     task.query_sites if self.inverse is None else self.inverse[task.query_sites])
        mean, variance = self.adapter.predict_task(inner)
        offset = 0. if self.beta is None else self._mean(task.times, task.query_sites)
        if self.predictive_family == 'gaussian_mixture':
            components, noise = self.adapter.components
            self.components = components+offset, noise
            self.latent = self.adapter.latent
        return np.asarray(mean)+offset, np.asarray(variance)
