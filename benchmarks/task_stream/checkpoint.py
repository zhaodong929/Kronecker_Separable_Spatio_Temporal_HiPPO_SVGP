"""Explicit, pickle-free snapshots of the five adapters at task boundaries.

These records support inspection and a future/manual restoration procedure.
They do not claim a qualified automatic resume path or mid-optimizer recovery.
External protocol, feature table and pinned source remain required inputs.
"""
from dataclasses import fields, is_dataclass
import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().copy()
    return np.array(value, copy=True)


def flatten_state(tree):
    """Return a flat dictionary accepted by the pipeline checkpoint serializer."""
    arrays = {}
    def visit(value, path):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError('Checkpoint mappings require string keys')
            return {key: visit(item, path + (key,)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return {'sequence_type': type(value).__name__, 'items':
                    [visit(item, path + (str(index),)) for index, item in enumerate(value)]}
        if is_dataclass(value):
            return {'dataclass': type(value).__module__ + '.' + type(value).__qualname__,
                    'fields': visit({field.name: getattr(value, field.name) for field in fields(value)}, path)}
        if isinstance(value, np.ndarray) or hasattr(value, '__array__') or hasattr(value, 'detach'):
            array = _array(value)
            if array.dtype.hasobject:
                raise TypeError('Object array cannot enter checkpoint')
            key = f'array_{len(arrays):05d}'
            arrays[key] = array
            return {'array_key': key, 'path': list(path), 'dtype': str(array.dtype), 'shape': list(array.shape)}
        raise TypeError(f'Unsupported checkpoint object at {path}: {type(value).__name__}')
    structure = visit(tree, ())
    return dict(structure=structure, **arrays)


def _variables(module):
    """Preserve named trainable and state variables; no object introspection."""
    if hasattr(module, 'state_dict'):
        return dict(module.state_dict())
    if hasattr(module, 'vars'):
        return {name: variable.value for name, variable in module.vars().items()}
    raise TypeError('Model exposes no supported explicit variable collection')


def window_state(window):
    return dict(mean=window.mean, covariance=window.covariance, time=window.time,
        previous=window.previous, before_previous=window.before_previous, next_task=window.next_task)


def adapter_snapshot(fitted):
    c, adapter = fitted.config, fitted.adapter
    if adapter is None:
        raise ValueError('Cannot snapshot before initialization')
    common = dict(schema_version=1, method=c.method, configuration=c, arm=fitted.arm,
        coordinates=fitted.coordinates, visible=fitted.visible, beta=fitted.beta,
        inverse_site_order=fitted.inverse, initial_step=fitted.initial_step,
        release_previous=fitted.release_previous,
        training_budget=getattr(fitted, 'training_budget', None),
        fit_budget_record=getattr(fitted, 'fit_budget_record', None),
        restoration=dict(auto_resume=False, restoration_tested=False, boundary='after completed initial fit or task',
            required_external_inputs=['exact feature table', 'protocol and future arriving observations', 'pinned sources'],
            optimizer_state='not retained: optimizers are discarded/reset between fitting/update calls; not a mid-fit checkpoint'))
    if c.method in ('st_svgp', 'mgpvae'):
        f = adapter.filter
        state = dict(window=window_state(adapter.window))
        if c.method == 'st_svgp':
            state.update(kernel_variables=_variables(f.kernel), noise=f.noise, coordinates=f.coordinates,
                h=f.h, b=f.b, conditional=f.conditional, stationary_covariance=f.pinf)
        else:
            state.update(model_variables=_variables(f.model), coordinates=adapter.coordinates,
                         samples=adapter.samples, seed=adapter.seed)
        common['state'] = state
    elif c.method == 'ohsvgp':
        common['state'] = dict(old=adapter.state, previous_steps=adapter.previous_steps,
            training_iteration=adapter.training_iteration, spectral_base=adapter.spectral_base,
            frequencies=adapter.frequencies, kernel_variables=_variables(adapter.kernel),
            likelihood_variables=_variables(adapter.likelihood), numpy_rng=adapter.rng.bit_generator.state,
            inducing_size=adapter.size, rff=adapter.rff,
            hippo_reconstruction='LazyHiPPOLegS(inducing_size, device, dtype) from pinned source')
    elif c.method == 'osgpr':
        mean, covariance, kernel, z = adapter.old
        # q at inducing locations and the old prior determine the next official update.
        common['state'] = dict(theta=adapter.theta, inducing=adapter.inducing,
            old_mean=mean, old_covariance=covariance, old_kernel=kernel, old_inducing=z,
            model_parameters={name: np.asarray(parameter) for name, parameter in
                              _gpflow_parameters(adapter.model).items()},
            graph_cache='rebuild from pinned source; no numerical state')
    elif c.method == 'kronhippo_svgp':
        state = {field.name: getattr(adapter.state, field.name) for field in fields(adapter.state)
                 if field.name != 'solver'}
        solver = getattr(adapter.state, 'solver', None)
        if solver is not None:
            state['solver'] = dict(class_name=type(solver).__name__, kt_inv=solver.kt_inv,
                ks_inv=solver.ks_inv, terms=solver.terms, tolerance=solver.tolerance,
                max_iterations=solver.max_iterations,
                rebuild='SumKronSolver(kt_inv, ks_inv, terms, tolerance=..., max_iterations=...)')
        model = adapter.model
        common['state'] = dict(posterior=state, posterior_class=type(adapter.state).__name__,
            model_class=type(model).__name__, model={name: getattr(model, name) for name in
                ('Ks', 'C', 'sigma2', 'beta_prior_mean', 'beta_prior_cov', 'prior_point_variance', 'jitter')},
            all_site_projection=adapter.c, builder_variables=_variables(adapter.builder),
            builder_configuration={field.name: (str(getattr(adapter.builder.config, field.name))
                if field.name in ('dtype', 'device') else getattr(adapter.builder.config, field.name))
                for field in fields(adapter.builder.config)},
            spec=adapter.spec, origin=adapter.origin, temporal_inducing=adapter.mt, step_size=adapter.step_size)
    else:
        raise ValueError('Unsupported comparison method')
    return flatten_state(common)


def _gpflow_parameters(model):
    from gpflow.utilities import parameter_dict
    return parameter_dict(model)
