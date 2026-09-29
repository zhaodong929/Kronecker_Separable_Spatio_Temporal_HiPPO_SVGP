#!/usr/bin/env python3
"""Causal MGPVAE integration candidate; formal admission requires qualification.

Official model with explicitly corrected spatial covariance pushforward.
Train on the Task-1 rectangle, select using only Task-1 validation sites, then
refit on all initially released sites. Append initially unobserved sites after
the training sites in the spatial Cholesky ordering: their prior extension
preserves the original spatial conditional (tested independently).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.causal_mean import get_mean, select_initial_targets
from baselines.traffic_protocol_n import load_protocol
from baselines.mgpvae.official import make_model, COMMIT
from baselines.mgpvae.selected import SelectedSiteFilter
from benchmarks.three_domain.tracking import emit
from benchmarks.three_domain.metrics import gaussian_mixture_metrics, gaussian_mixture_calibration


def save_variables(path, variables, **extra):
    np.savez_compressed(path, **{f'variable_{i}': np.asarray(v) for i, v in enumerate(variables.tensors())},
        variable_names=np.asarray(list(variables.keys())), **extra)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-npz', type=Path, required=True)
    p.add_argument('--protocol-json', type=Path, required=True)
    p.add_argument('--protocol-kind', choices=['covid', 'traffic'], default='covid')
    p.add_argument('--official-source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--latent', type=int, default=2)
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--iterations', type=int, default=250)
    p.add_argument('--check-every', type=int, default=25)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--training-samples', type=int, default=4)
    p.add_argument('--prediction-samples', type=int, default=128)
    p.add_argument('--max-blocks', type=int, default=0)
    p.add_argument('--metric-backend', choices=['numpy','jax'], default='numpy')
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--selection-json', type=Path)
    a = p.parse_args()
    if min(a.iterations, a.check_every, a.training_samples, a.prediction_samples) < 1 or a.latent < 2:
        p.error('Positive budgets and at least two latent dimensions required by upstream shapes')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.selection_json is not None and a.validation_only:
        p.error('A frozen selection cannot be used for validation')
    digest = hashlib.sha256()
    with a.protocol_npz.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): digest.update(chunk)
    protocol_hash = digest.hexdigest()
    (a.output_dir/'run-config.json').write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}, indent=2))
    protocol = load_protocol(a.protocol_npz, a.protocol_json, protocol_kind=a.protocol_kind)
    mean_function = get_mean(protocol)
    initial = protocol.task1()
    times = protocol.calibration_times
    origin, span = float(times[0]), float(times[-1]-times[0])
    if span <= 0:
        raise ValueError('Positive initial time span required')
    def scaled(t):
        return (np.asarray(t)-origin)/span
    def create(locations):
        return make_model(a.official_source, protocol.coordinates[locations], seed=a.seed,
            latent=a.latent, width=a.width, correct_spatial_covariance=True)
    def residuals(locations):
        offsets = np.stack([mean_function.at(t, locations) for t in times])
        return select_initial_targets(initial, locations)-offsets
    model = create(protocol.fit_locations)
    import jax
    import jax.numpy as jnp
    import objax
    t = jnp.asarray(scaled(times))[:, None]
    training_started = time.perf_counter()
    trace = []

    def validation(model, training):
        # Task-1 smoothing uses only fitting sites; all initial times are legal.
        lm, lv = model.predict(t, t, training, jnp.asarray(protocol.coordinates[protocol.validation_locations]))
        shape = (len(protocol.validation_locations), len(times), a.latent)
        lm, lv = jnp.reshape(lm, shape), jnp.reshape(lv, shape)
        if not np.isfinite(lv).all() or np.any(np.asarray(lv) <= 0):
            raise FloatingPointError('Nonpositive validation latent variance')
        validation_samples = max(512, a.prediction_samples)
        z = lm[None] + jnp.sqrt(lv)[None]*jax.random.normal(jax.random.PRNGKey(a.seed+100000),
            (validation_samples, *shape))
        components = np.asarray(model.likelihood.decoder(z)[..., 0]).transpose(0, 2, 1)
        offsets = np.stack([mean_function.at(time_value, protocol.validation_locations) for time_value in times])
        components = components + offsets[None]
        truth = protocol.calibration_targets(protocol.validation_locations)
        components = components.reshape(validation_samples, -1)
        metrics = gaussian_mixture_metrics(truth.ravel(), components, float(model.likelihood.variance), compute_crps=False)
        metrics['monte_carlo_samples'] = validation_samples
        metrics['monte_carlo_checks'] = {str(n): gaussian_mixture_metrics(truth.ravel(),
            components[:n], float(model.likelihood.variance), compute_crps=False) for n in sorted({128,256,a.prediction_samples})
            if n < validation_samples}
        return metrics

    def fit(model, locations, iterations, phase, tune):
        training = jnp.asarray(residuals(locations).T[..., None])
        optimizer = objax.optimizer.Adam(model.vars())
        variables = model.vars()+optimizer.vars()
        gradient = objax.GradValues(model.energy, model.vars())
        @objax.Function.with_vars(variables)
        def update(i):
            gradients, losses = gradient(training, jax.random.PRNGKey(a.seed*100000+i),
                t=t, num_samples=a.training_samples)
            optimizer(a.learning_rate, gradients)
            return losses
        update = objax.Jit(update)
        best = None
        for i in range(1, iterations+1):
            started = time.perf_counter()
            values = np.asarray(update(i))
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite MGPVAE objective')
            row = dict(phase=phase, step=i, negative_elbo=float(values[0]),
                negative_expected_log_likelihood=float(values[1]), kl=float(values[2]),
                seconds=time.perf_counter()-started,
                kernel_lengthscale_time=np.asarray(model.kernel.lengthscale_time).tolist(),
                kernel_variance_time=np.asarray(model.kernel.variance_time).tolist(),
                kernel_lengthscale_space=np.asarray(model.kernel.lengthscale).tolist(),
                decoder_noise_variance=float(model.likelihood.variance))
            emit(phase, i, row)
            if i == 1 or i % a.check_every == 0 or i == iterations:
                if tune:
                    row['validation'] = validation(model, training)
                    emit('validation', i, row['validation'])
                    if best is None or row['validation']['nlpd'] < best['validation']['nlpd']:
                        best = dict(step=i, validation=row['validation'])
                        save_variables(a.output_dir/'selected_fit_model.npz', model.vars(), selected_iteration=i)
                save_variables(a.output_dir/f'{phase}-checkpoint-{i}.npz', variables, iteration=i)
                print(json.dumps(row), flush=True)
            trace.append(row)
        return best

    if a.selection_json is None:
        selected = fit(model, protocol.fit_locations, a.iterations, 'train', True)
        calibration = dict(status='validation_complete', selected=selected, trace=trace,
            iterations=a.iterations, official_commit=COMMIT, covariance_pushforward_corrected=True,
            split_seed=a.seed, latent=a.latent, width=a.width, protocol_sha256=protocol_hash,
            main_table_admitted=False, time_origin=origin, time_scale=span,
            elapsed_seconds=time.perf_counter()-training_started)
    else:
        calibration = json.loads(a.selection_json.read_text())
        expected = dict(split_seed=a.seed, latent=a.latent, width=a.width, protocol_sha256=protocol_hash,
            official_commit=COMMIT, covariance_pushforward_corrected=True)
        if any(calibration.get(k) != v for k,v in expected.items()):
            raise ValueError('Frozen selection does not match the split, architecture or protocol')
        selected = calibration['selected']
    (a.output_dir/'calibration.json').write_text(json.dumps(calibration, indent=2))
    if a.validation_only:
        return
    refit_started = time.perf_counter()
    fitted = create(initial.locations)
    fit(fitted, initial.locations, selected['step'], 'refit', False)
    save_variables(a.output_dir/'refit_model.npz', fitted.vars())
    ordering = np.concatenate([initial.locations,
        np.setdiff1d(np.arange(protocol.locations), initial.locations)])
    inverse = np.empty(protocol.locations, dtype=int)
    inverse[ordering] = np.arange(protocol.locations)
    full = create(ordering)
    full.vars().assign(fitted.vars().tensors())
    adapter = SelectedSiteFilter(full)
    y = residuals(initial.locations)
    for i, time_value in enumerate(times):
        adapter.advance(float(scaled(time_value)), inverse[initial.locations], y[i])
    # Synchronize initial filtering before the online timer.
    np.asarray(adapter.mean)
    refit_seconds = time.perf_counter()-refit_started
    n = protocol.online_weeks if a.max_blocks <= 0 else min(protocol.online_weeks, a.max_blocks)
    latent_means, latent_vars, means, variances, seeds, rows = [], [], [], [], [], []
    delayed_rows = 0
    online_seconds = 0.
    truth_for_scoring = protocol.evaluation_targets()
    for step in range(n):
        started = time.perf_counter()
        info = protocol.week(step)
        delayed = {}
        if info.delayed_hidden is not None:
            d = info.delayed_hidden
            delayed = dict(delayed_time=float(scaled(d.time)), delayed_sites=inverse[d.locations],
                delayed_values=d.targets-mean_function.at(d.time, d.locations))
            delayed_rows += len(d.locations)
        current = info.current_visible
        adapter.advance(float(scaled(current.time)), inverse[current.locations],
            current.targets-mean_function.at(current.time, current.locations), **delayed)
        lm, lv = adapter.latent_at(protocol.coordinates[protocol.hidden_locations])
        lm, lv = np.asarray(lm), np.asarray(lv)
        if not np.isfinite(lv).all() or np.any(lv <= 0):
            raise FloatingPointError('Invalid online latent variance')
        seed = a.seed*1000000+step
        components, noise = adapter.gaussian_components(protocol.coordinates[protocol.hidden_locations],
            seed=seed, samples=a.prediction_samples)
        components = components + mean_function.at(current.time, protocol.hidden_locations)[None]
        seconds = time.perf_counter()-started
        online_seconds += seconds
        # Test truth enters scores only after the prediction is fixed.
        truth = truth_for_scoring[step]
        scoring_started=time.perf_counter()
        if a.metric_backend=='jax':
            from baselines.mgpvae.gpu_metrics import gaussian_mixture_scores
            score=gaussian_mixture_scores(truth,components,noise)
        else:
            score={**gaussian_mixture_metrics(truth, components, noise),
                **gaussian_mixture_calibration(truth, components, noise),
                'coverage90':gaussian_mixture_calibration(truth, components, noise, [.9])['coverage'][0]}
        row = dict(step=step+1, update_and_prediction_seconds=seconds,
            scoring_seconds=time.perf_counter()-scoring_started,metric_backend=a.metric_backend,**score)
        emit('online', step+1, row)
        rows.append(row)
        latent_means.append(lm); latent_vars.append(lv); seeds.append(seed)
        means.append(components.mean(axis=0)); variances.append(components.var(axis=0)+noise)
        if (step+1) % 100 == 0 or step+1 == n:
            np.savez_compressed(a.output_dir/'predictions.npz',
                y_true=truth_for_scoring[:step+1], pred_mean=np.asarray(means), pred_var=np.asarray(variances),
                latent_mean=np.asarray(latent_means), latent_var=np.asarray(latent_vars),
                decoder_sample_seed=np.asarray(seeds), decoder_samples=a.prediction_samples,
                decoder_noise_variance=noise, spatial_order=ordering,
                test_indices=protocol.hidden_locations, times=protocol.stream_times[:step+1])
            (a.output_dir/'online-metrics.json').write_text(json.dumps(rows, indent=2))
    pooled_coverage = np.mean([row['coverage'] for row in rows], axis=0)
    scores = {k:float(np.mean([row[k] for row in rows])) for k in ['nlpd','crps','coverage90']}
    scores['rmse'] = float(np.sqrt(np.mean((truth_for_scoring[:n]-np.asarray(means))**2)))
    scores['ece'] = float(np.mean(np.abs(pooled_coverage-np.asarray(rows[0]['levels']))))
    result = dict(status='complete', method='MGPVAE causal adaptation', official_commit=COMMIT,
        main_table_admitted=False, covariance_pushforward_corrected=True, initial_observed_sites=len(initial.locations),
        expected_steps=n, delayed_observation_rows=delayed_rows, current_hidden_labels_read=0,
        predictive_family='finite Gaussian decoder mixture', prediction_samples=a.prediction_samples,
        scores=scores, coverage_levels=rows[0]['levels'], pooled_coverage=pooled_coverage.tolist(),
        task1_validation_seconds=calibration['elapsed_seconds'], refit_and_initial_filter_seconds=refit_seconds,
        online_update_prediction_seconds=online_seconds, online_metrics_include_scoring_time=False,
        normalization='common protocol; shared Task-1 covariate mean', selected_iteration=selected['step'],
        qualification_pending='MC convergence, real-data budget/capacity validation and full-run resource pilot')
    (a.output_dir/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
