"""Task adapters retaining each pinned GP's own update and representation."""
import numpy as np
from benchmarks.three_domain.tracking import emit
from .ablations import Arm


def inputs(batch, coordinates):
    return np.column_stack([np.repeat(batch.times, len(batch.sites)),
                            np.tile(coordinates[batch.sites], (len(batch.times), 1))])


class OSGPRTaskAdapter:
    initialization_phase = 'initial_refit'
    def __init__(self, coordinates, theta, inducing, *, initial_steps, update_steps, learning_rate=.01, fit_clock=None):
        from scripts import run_official_bui_osgpr_era5 as official
        self.official = official
        self.coordinates, self.theta, self.inducing = np.asarray(coordinates), dict(theta), np.asarray(inducing)
        self.initial_steps, self.update_steps = int(initial_steps), int(update_steps)
        self.learning_rate = float(learning_rate)
        self.fit_clock = fit_clock
        self.old = None
        self.model = None
        self.cache = {}

    def _update(self, batch, steps, *, fit_clock=None):
        import gpflow
        o = self.official
        x, y = inputs(batch, self.coordinates), batch.values.reshape(-1, 1)
        kernel = o.make_kernel(self.theta, frozen=False)
        if self.old is None:
            from baselines.chunked_sgpr import ChunkedSGPR
            model = ChunkedSGPR(data=(x, y), kernel=kernel, inducing_variable=self.inducing,
                               noise_variance=self.theta['noise_std'] ** 2, chunk_rows=2048)
        else:
            mean, covariance, old_kernel, old_z = self.old
            model = o.OSGPR_VFE(data=(x, y), kernel=kernel, mu_old=mean, Su_old=covariance,
                Kaa_old=old_kernel, Z_old=old_z, Z=self.inducing)
            model.likelihood.variance.assign(self.theta['noise_std'] ** 2)
        o.adapt_model(model, steps=steps, learning_rate=self.learning_rate,
                      execution='graph', graph_cache=self.cache, fit_clock=fit_clock)
        self.inducing = np.asarray(model.inducing_variable.Z)
        self.theta = o.theta_from_model(model)
        mean, covariance = o.posterior_at_z(model, self.inducing)
        self.old = mean, covariance, np.asarray(kernel(self.inducing)), self.inducing.copy()
        self.model = model

    def initialize(self, initial):
        self._update(initial, self.initial_steps, fit_clock=self.fit_clock)

    def predict_task(self, task):
        if task.delayed is not None:
            self._update(task.delayed, self.update_steps)
        self._update(task.visible, self.update_steps)
        x = np.column_stack([np.repeat(task.times, len(task.query_sites)),
                             np.tile(self.coordinates[task.query_sites], (len(task.times), 1))])
        mean, variance = self.official.predict_current(self.model, x, self.theta['noise_std'] ** 2, 512)
        return mean.reshape(task.query_shape), variance.reshape(task.query_shape)


class OHSVGPTaskAdapter:
    """Frozen selected kernel, official row-stream HiPPO and variational update.

    Initialization grid and row microbatches are explicit model configuration,
    not scientific task boundaries. Every task is predicted after all its rows.
    """
    def __init__(self, coordinates, kernel, likelihood, *, inducing_size=32, rff=256,
                 initial_steps=500, update_steps=5, batch_rows=1024, grid_rows=1024,
                 learning_rate=.001, seed=0, device='cpu', train_initial_kernel=True, fit_clock=None):
        import torch
        from scripts import run_covid_ohsvgp_own_theta as o
        self.o, self.torch = o, torch
        self.coordinates = np.asarray(coordinates)
        self.kernel, self.likelihood = kernel, likelihood
        self.device, self.dtype = torch.device(device), torch.float64
        self.size, self.rff = int(inducing_size), int(rff)
        self.initial_steps, self.update_steps = int(initial_steps), int(update_steps)
        self.batch_rows, self.grid_rows = int(batch_rows), int(grid_rows)
        if min(self.size, self.rff, self.initial_steps, self.update_steps, self.batch_rows, self.grid_rows) < 1:
            raise ValueError('Positive OHSVGP sizes and budgets required')
        self.rate = float(learning_rate)
        self.train_initial_kernel = bool(train_initial_kernel)
        self.fit_clock = fit_clock
        self.training_iteration = 0
        self.seed = int(seed)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.spectral_base = torch.randn((self.rff, 3), generator=generator, dtype=self.dtype).to(self.device)
        self.frequencies = self.spectral_base / torch.exp(kernel.log_ls).detach()
        self.hippo = o.LazyHiPPOLegS(self.size, self.device, self.dtype)
        self.state = None
        self.previous_steps = 0
        self.rng = np.random.default_rng(seed)

    def _model(self, grid):
        return self.o.make_model(kernel=self.kernel, likelihood=self.likelihood,
            z_interpolate=grid, rff_sample_size=self.rff, previous_steps=self.previous_steps,
            hippo=self.hippo, inducing_size=self.size, old_state=self.state,
            device=self.device, dtype=self.dtype)

    def _fit(self, model, x, y, steps, *, minibatches=False, learn_kernel=False, fit_clock=None):
        torch = self.torch
        parameters = list(model.parameters()) if learn_kernel else [model.mv, model.Lv]
        if fit_clock is not None:
            fit_clock.start()
        optimizer = torch.optim.Adam(parameters, lr=self.rate)
        completed = 0
        for iteration in range(steps):
            select = self.rng.choice(len(x), min(self.batch_rows, len(x)), replace=False) if minibatches else slice(None)
            xx = torch.as_tensor(x[select], device=self.device, dtype=self.dtype)
            yy = torch.as_tensor(y[select], device=self.device, dtype=self.dtype)
            fraction = len(xx) / len(x)
            optimizer.zero_grad(set_to_none=True)
            frequencies = self.spectral_base / torch.exp(model.kernel.log_ls) if learn_kernel else self.frequencies
            # The official ELBO samples its likelihood with randn_like even
            # for Gaussian observations. Isolate those draws per run/step.
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed + self.training_iteration)
                elbo, _, _ = model.ELBO(xx, yy, frequencies, beta=fraction)
            loss = -elbo / fraction
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite OHSVGP task objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 20., error_if_nonfinite=True)
            optimizer.step()
            if learn_kernel:
                self.o.clamp_hyperparameters(model)
            self.training_iteration += 1
            emit('train', self.training_iteration, dict(negative_elbo=float(loss.detach()),
                gradient_norm=float(norm), rows=len(x), learn_kernel=learn_kernel))
            completed += 1
            if fit_clock is not None and fit_clock.should_stop(completed):
                break
        if fit_clock is not None:
            fit_clock.finish(completed)
        return completed

    initialization_phase = 'initial_refit'

    def initialize(self, initial):
        x, y = self.o.sorted_xy(inputs(initial, self.coordinates), initial.values.reshape(-1, 1))
        grid = x[np.linspace(0, len(x)-1, min(self.grid_rows, len(x)), dtype=int)]
        model = self._model(grid)
        self._fit(model, x, y, self.initial_steps, minibatches=True, learn_kernel=self.train_initial_kernel, fit_clock=self.fit_clock)
        self.kernel, self.likelihood = model.kernel, model.likelihood
        self.kernel.requires_grad_(False)
        self.likelihood.requires_grad_(False)
        self.frequencies = (self.spectral_base / self.torch.exp(model.kernel.log_ls)).detach()
        self.state = self.o.export_state(model, self.frequencies)
        self.previous_steps += len(grid)

    def predict_task(self, task):
        for batch in [b for b in [task.delayed, task.visible] if b is not None]:
            x, y = self.o.sorted_xy(inputs(batch, self.coordinates), batch.values.reshape(-1, 1))
            for start in range(0, len(x), self.batch_rows):
                xx, yy = x[start:start+self.batch_rows], y[start:start+self.batch_rows]
                model = self._model(xx)
                self._fit(model, xx, yy, self.update_steps)
                self.state = self.o.export_state(model, self.frequencies)
                self.previous_steps += len(xx)
        x = np.column_stack([np.repeat(task.times, len(task.query_sites)),
                             np.tile(self.coordinates[task.query_sites], (len(task.times), 1))])
        mean, variance = self.o.predict(model, self.frequencies, x, device=self.device, dtype=self.dtype)
        return mean.reshape(task.query_shape), variance.reshape(task.query_shape)


class KronTaskAdapter:
    def __init__(self, coordinates, visible, inducing, theta, *, initial_step,
                 features, feature_dimension, mt=32, rff=256, seed=0, device='cpu',
                 multiple_geometry=False, arm=Arm('joint_transfer'), beta_prior_variance=1.):
        import torch
        from stvgp_kronecker.joint_ssgp_kron.synthetic import make_analytic_temporal_builder
        from stvgp_kronecker.joint_ssgp_kron.torch_backend import TorchJointSSGPKronHiPPOSVGP
        from stvgp_kronecker.joint_ssgp_kron.multi_geometry import TorchMultiGeometryHiPPOSVGP
        from scripts.run_routeb_online_parity_ladder import spatial_projection
        self.torch, self.features, self.arm = torch, features, arm
        self.mt, self.step_size = int(mt), float(initial_step)
        ks, self.c = spatial_projection(np.asarray(coordinates), np.asarray(inducing), theta['ell_s'])
        cls = TorchMultiGeometryHiPPOSVGP if multiple_geometry else TorchJointSSGPKronHiPPOSVGP
        self.model = cls(Ks=ks, C=self.c[visible], sigma2=theta['noise_std']**2,
            beta_prior_mean=np.zeros(feature_dimension), beta_prior_cov=beta_prior_variance*np.eye(feature_dimension),
            prior_point_variance=theta['kernel_variance'], device=device, dtype=torch.float64)
        self.builder = make_analytic_temporal_builder(mt=mt, lengthscale=theta['ell_t'],
            variance=theta['kernel_variance'], rff_sample_size=rff, seed=seed, kernel_type='matern32')
        self.builder = self.builder.to(device=device, dtype=torch.float64)
        self.state = None
        self.spec = None
        self.origin = None

    def _spec(self, end):
        from stvgp_kronecker.temporal_analytic import TemporalBlockSpec
        return TemporalBlockSpec(self.origin, float(end),
            max(1, int(round((end-self.origin)/self.step_size))), phase_origin=self.origin)

    def _factors(self, times, spec, old=None):
        from stvgp_kronecker.joint_ssgp_kron.torch_backend import solve_spd
        torch = self.torch
        with torch.no_grad():
            kfu, kt, kon, _ = self.builder.compute_block_covariances_with_basis(
                torch.as_tensor(times.copy(), device=self.model.device, dtype=self.model.dtype), spec, old)
            kt = self.builder.add_jitter(kt)
            t = solve_spd(kt, kfu.T, jitter=1e-12).T
        return t, kt, kon

    def _update(self, observations, spec, *, move):
        t, kt, kon = self._factors(observations.times, spec, self.spec if move else None)
        override = self.torch.eye(self.mt, device=self.model.device, dtype=self.model.dtype) if move and self.arm.identity_transfer else None
        with self.torch.no_grad():
            self.state = self.model.update_block_structured_joint_ssgp_transfer(
                y_vec=observations.values.copy().reshape(-1),
                Phi=self.features(observations.times, observations.sites), T_n=t, Kt_new=kt,
                state=self.state, K_on_t=kon, C_observed=self.c[observations.sites],
                L_t_override=override, zero_cross=self.arm.zero_cross)
        self.spec = spec

    def initialize(self, initial):
        self.origin = float(initial.times[0] - self.step_size)
        self._update(initial, self._spec(initial.times[-1]), move=False)

    def predict_task(self, task):
        if task.delayed is not None:
            self._update(task.delayed, self.spec, move=False)
        self._update(task.visible, self._spec(task.times[-1]), move=True)
        t, _, _ = self._factors(task.times, self.spec)
        with self.torch.no_grad():
            mean, variance, _ = self.model.predict_with_C(state=self.state, T_eval=t,
                Phi=self.features(task.times, task.query_sites), C_eval=self.c[task.query_sites],
                include_conditional_residual_variance=True, validate_conditional_residual_variance=True)
        emit('numerics', task.index+1, self.state.metadata)
        return mean.reshape(task.query_shape), variance.reshape(task.query_shape)
