#!/usr/bin/env python3
"""CPU/device parity of two actual fitted tasks; a diagnostic, never convergence proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probe_code(method):
    if method in ('kronhippo_svgp', 'ohsvgp'):
        return "import torch,json; assert torch.cuda.is_available(), 'CUDA unavailable'; print(json.dumps({'framework':'torch','gpu':torch.cuda.get_device_name(0),'device_count':torch.cuda.device_count(),'version':torch.__version__}))"
    if method in ('st_svgp', 'mgpvae'):
        return "import jax,json; d=jax.devices(); assert any(x.platform=='gpu' for x in d), 'JAX GPU unavailable'; print(json.dumps({'framework':'jax','devices':[str(x) for x in d],'version':jax.__version__}))"
    return "import tensorflow as tf,json; d=tf.config.list_physical_devices('GPU'); assert d, 'TensorFlow GPU unavailable'; print(json.dumps({'framework':'tensorflow','devices':[str(x) for x in d],'version':tf.__version__}))"


def oh_fixed_state_prediction_parity(output, *, rtol, atol, target_device='cuda'):
    """Predict the same fitted CPU posterior on both devices, without refitting.

    The fixture deliberately tests observation predictions at a fixed learned
    state. It does not claim equality of independent stochastic Adam updates.
    Only safe numeric arrays are archived; no pickle/checkpoint loading occurs.
    """
    import numpy as np
    import torch
    from benchmarks.task_stream.gp import OHSVGPTaskAdapter, inputs
    from benchmarks.task_stream.protocol import Observations
    from scripts import run_covid_ohsvgp_own_theta as official
    from hipposvgp.multidim import SE_kernel
    from hipposvgp.likelihood import GaussianLikelihood
    coordinates = np.array([[0., 0.], [.2, .4], [.6, .1], [.8, .5]])
    times = np.array([0., .1, .3])
    initial = Observations(times, np.array([0, 2]),
        np.random.default_rng(89).normal(size=(3, 2)))
    adapter = OHSVGPTaskAdapter(coordinates, SE_kernel(3).to(dtype=torch.float64),
        GaussianLikelihood(.2).to(dtype=torch.float64), inducing_size=3, rff=32,
        initial_steps=2, update_steps=1, batch_rows=4, grid_rows=4, seed=90)
    adapter.initialize(initial)
    x, _ = official.sorted_xy(inputs(initial, coordinates), initial.values.reshape(-1, 1))
    grid = x[np.linspace(0, len(x)-1, min(adapter.grid_rows, len(x)), dtype=int)]
    query = np.column_stack([np.repeat(times, 2), np.tile(coordinates[[3, 1]], (3, 1))])
    predictions = {}
    for label, device in [('cpu', 'cpu'), ('target', target_device)]:
        device = torch.device(device)
        model = official.make_model(kernel=adapter.kernel, likelihood=adapter.likelihood,
            z_interpolate=grid, rff_sample_size=adapter.rff, previous_steps=0,
            hippo=official.LazyHiPPOLegS(adapter.size, device, torch.float64),
            inducing_size=adapter.size, old_state=None, device=device, dtype=torch.float64)
        with torch.no_grad():
            model.mv.copy_(adapter.state['mv'].to(device))
            model.Lv.copy_(adapter.state['Lv'].to(device))
            frequencies = adapter.frequencies.to(device)
            z, _, _, _, _ = model.Kuu_se(frequencies)
            np.testing.assert_allclose(z.cpu(), adapter.state['Z'].cpu(), rtol=rtol, atol=atol)
        predictions[label] = official.predict(model, frequencies, query, device=device, dtype=torch.float64)
    comparison = {}
    for index, key in enumerate(('pred_mean', 'pred_var')):
        cpu, target = predictions['cpu'][index], predictions['target'][index]
        difference = np.abs(cpu-target)
        comparison[key] = dict(max_absolute_error=float(difference.max()),
            max_scaled_error=float(np.max(difference/(atol+rtol*np.abs(cpu)+1e-300))))
    output = Path(output)
    np.savez_compressed(output/'oh-fixed-state-predictions.npz', grid=grid, queries=query,
        spectral_frequencies=adapter.frequencies.cpu().numpy(),
        variational_mean=adapter.state['mv'].cpu().numpy(),
        variational_factor=adapter.state['Lv'].cpu().numpy(),
        kernel_log_lengthscales=adapter.kernel.log_ls.detach().cpu().numpy(),
        kernel_log_variance=adapter.kernel.log_sf.detach().cpu().numpy(),
        likelihood_variance=adapter.likelihood.variance.detach().cpu().numpy(),
        cpu_pred_mean=predictions['cpu'][0], cpu_pred_var=predictions['cpu'][1],
        target_pred_mean=predictions['target'][0], target_pred_var=predictions['target'][1])
    (output/'oh-fixed-state-errors.json').write_text(json.dumps(comparison, indent=2))
    for cpu, target in zip(predictions['cpu'], predictions['target']):
        np.testing.assert_allclose(target, cpu, rtol=rtol, atol=atol)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tasks', type=int, default=2, choices=[2])
    parser.add_argument('--max-fit-steps', type=int, default=2, choices=[1, 2])
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--shape-only', action='store_true', help='Full-size GPU shapes only; no CPU parity claim')
    args = parser.parse_args()
    import numpy as np
    from benchmarks.three_domain.tracking import emit
    if not np.isfinite([args.rtol, args.atol]).all() or min(args.rtol, args.atol) < 0:
        parser.error('Finite nonnegative parity tolerances required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'tracking' for p in output.iterdir()):
        raise FileExistsError('A fresh qualification directory is required')
    import shutil
    import socket
    driver = Path('/proc/driver/nvidia/params')
    try:
        lines = driver.read_text().splitlines()
    except OSError:
        lines = []
    restriction = next((line.strip() for line in lines if line.startswith('RmProfilingAdminOnly:')), None)
    (output / 'profiling-availability.json').write_text(json.dumps(dict(hostname=socket.gethostname(),
        ncu_path=shutil.which('ncu'), driver_counter_restriction=restriction,
        counter_acquisition_verified=False), indent=2))
    configuration = json.loads(args.configuration.read_text())
    configuration.update(initial_iterations=min(configuration['initial_iterations'], args.max_fit_steps),
                         online_iterations=min(configuration.get('online_iterations', 5), args.max_fit_steps))
    method = configuration['method']
    if method not in ('kronhippo_svgp', 'ohsvgp', 'osgpr', 'st_svgp', 'mgpvae'):
        parser.error('Method is outside baseline-selection policy')
    gpu_env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    try:
        probe = subprocess.run([sys.executable, '-c', probe_code(method)], cwd=ROOT, env=gpu_env,
                               text=True, capture_output=True, check=False)
        (output / 'gpu-probe.txt').write_text(probe.stdout + probe.stderr)
        probe.check_returncode()
        hardware = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv'],
                                  capture_output=True, text=True, check=True).stdout
        (output / 'hardware.txt').write_text(hardware)
        results = {}
        for device in (['cuda'] if args.shape_only else ['cpu', 'cuda']):
            config = dict(configuration, device=device)
            config_path = output / f'{device}-configuration.json'
            config_path.write_text(json.dumps(config, indent=2))
            env = gpu_env if device == 'cuda' else dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
            emit('qualification', 0, dict(method=method, device=device, status='starting',
                 max_fit_steps=args.max_fit_steps, tasks=2, shape_only=args.shape_only))
            command = [sys.executable, str(ROOT/'scripts/run_task_stream.py'),
                '--prepared', str(args.prepared.resolve()), '--configuration', str(config_path),
                '--output', str(output/device), '--stage', 'integration', '--max-tasks', '2']
            with (output/f'{device}-worker.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            results[device] = json.loads((output/device/'result.json').read_text())
            if results[device]['status'] != 'completed':
                raise RuntimeError('Worker did not finish its tasks')
        comparison = {}
        if not args.shape_only:
            with np.load(output/'cpu/predictions.npz', allow_pickle=False) as cpu, np.load(output/'cuda/predictions.npz', allow_pickle=False) as gpu:
                for key in ('times', 'query_sites', 'y_true'):
                    np.testing.assert_array_equal(cpu[key], gpu[key])
                for key in ('pred_mean', 'pred_var'):
                    difference = np.abs(cpu[key]-gpu[key])
                    comparison[key] = dict(max_absolute_error=float(difference.max()),
                        max_scaled_error=float(np.max(difference / (args.atol + args.rtol*np.abs(cpu[key]) + 1e-300))))
                    (output/'parity-errors.json').write_text(json.dumps(comparison, indent=2))
                    if method != 'ohsvgp':
                        np.testing.assert_allclose(gpu[key], cpu[key], rtol=args.rtol, atol=args.atol)
        fixed_state = None
        qualification = 'shape_only_passed' if args.shape_only else 'cpu_gpu_parity_passed'
        if method == 'ohsvgp' and not args.shape_only:
            fixed_state = oh_fixed_state_prediction_parity(output, rtol=args.rtol, atol=args.atol)
            qualification = 'independentfits_completed_and_fixed_state_prediction_parity'
        record = dict(status='completed', qualification=qualification,
            fixed_state_comparison=fixed_state,
            independent_fit_comparison_is_acceptance_criterion=(not args.shape_only and method != 'ohsvgp'),
            method=method, main_table_admitted=False, comparison=comparison, rtol=args.rtol, atol=args.atol,
            tasks=2, max_fit_steps=args.max_fit_steps, hardware=hardware,
            limitations=['Not a convergence or full-stream throughput qualification',
                ('OH independent fit differences are descriptive only; parity uses a separate tiny shared fitted posterior'
                 if method == 'ohsvgp' else 'Device parity compares independently fitted models with the same seed/configuration'),
                ('OH fixed-state fixture does not qualify CPU/GPU equality of stochastic online optimizer trajectories'
                 if method == 'ohsvgp' else 'Independent training parity is not an accuracy/convergence comparison'),
                'W&B supervisor owns network logging; local child artifacts contain every task'])
        (output/'result.json').write_text(json.dumps(record, indent=2))
        emit('qualification', 1, record)
    except BaseException as error:
        record = dict(status='failed', main_table_admitted=False, method=method,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        (output/'result.json').write_text(json.dumps(record, indent=2))
        emit('qualification', 1, record)
        raise


if __name__ == '__main__':
    main()
