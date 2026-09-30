"""Bind final/refit arms to actual selected work, never a second wall-clock race."""
import hashlib
import json
import math
import numbers

METHODS = {'kronhippo_svgp', 'osgpr', 'ohsvgp', 'st_svgp', 'mgpvae'}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _seconds(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return float(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def frozen_refit_plan(configuration, fit_budget_record, selection_initial_rows, final_initial_rows):
    """Full-batch steps stay fixed; OH preserves expected sampled-row exposure.

    An iteration-safety-cap stop is retained explicitly and does not certify
    exhaustion of the time budget. Neither stop reason establishes convergence.
    The caller must additionally bind the record to the winning run's artifacts.
    """
    config, record = dict(configuration), dict(fit_budget_record)
    if config.get('method') not in METHODS:
        raise ValueError('Unregistered comparison method')
    selected_rows = _positive_integer(selection_initial_rows, 'selection_initial_rows')
    final_rows = _positive_integer(final_initial_rows, 'final_initial_rows')
    steps = _positive_integer(record.get('completed_steps'), 'completed_steps')
    cap = _positive_integer(config.get('initial_iterations'), 'initial_iterations')
    budget = _seconds(config.get('initial_max_seconds'), 'initial_max_seconds')
    maximum = _seconds(record.get('max_seconds'), 'record.max_seconds')
    elapsed = _seconds(record.get('elapsed_seconds'), 'elapsed_seconds', zero=True)
    overshoot = _seconds(record.get('overshoot_seconds'), 'overshoot_seconds', zero=True)
    if config.get('initial_expected_passes') is not None:
        raise ValueError('Selected wall-time policy cannot also request expected passes')
    if (record.get('policy') != 'synchronized_wall_time_v1'
            or record.get('scope') != 'initial_optimizer_setup_compilation_and_completed_updates'
            or record.get('convergence_claimed') is not False
            or _positive_integer(record.get('minimum_completed_steps'), 'minimum_completed_steps') != 1):
        raise ValueError('A finished selection FitClock record is required')
    if maximum != budget or steps > cap:
        raise ValueError('Fit record disagrees with the selected configuration')
    reached = elapsed >= budget
    expected_stop = 'wall_time_budget_reached' if reached else 'iteration_limit_reached'
    if record.get('stop_reason') != expected_stop or (not reached and steps != cap):
        raise ValueError('Fit record has an inconsistent stopping reason or safety cap')
    if not math.isclose(overshoot, max(0., elapsed-budget), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError('Fit record has an inconsistent time overshoot')
    oh = config['method'] == 'ohsvgp'
    batch_rows = _positive_integer(config.get('batch_rows'), 'batch_rows') if oh else None
    selected_batch = min(batch_rows, selected_rows) if oh else selected_rows
    final_batch = min(batch_rows, final_rows) if oh else final_rows
    numerator, denominator = steps*selected_batch*final_rows, selected_rows*final_batch
    effective = (numerator+denominator-1)//denominator if oh else steps
    result = dict(schema_version=1, policy='selected_work_refit_v1', method=config['method'],
        mode='preserve_expected_row_exposures' if oh else 'preserve_full_batch_steps',
        selected_steps=steps, selected_rows=selected_rows, final_rows=final_rows,
        selected_rows_per_step=selected_batch, final_rows_per_step=final_batch,
        effective_iterations=effective, selected_max_seconds=budget,
        selected_stop_reason=expected_stop, wall_time_budget_reached=reached,
        iteration_safety_cap_reached=steps == cap, convergence_qualified=False,
        final_stop_policy='fixed_iterations_without_wall_time_cap',
        configuration_sha256=_digest(config), fit_record_sha256=_digest(record))
    result['plan_sha256'] = _digest(result)
    return result


def validate_refit_plan(plan, configuration, fit_budget_record, selection_initial_rows, final_initial_rows):
    """Recompute against trusted winning artifacts, rejecting stale/tampered plans."""
    expected = frozen_refit_plan(configuration, fit_budget_record,
                                 selection_initial_rows, final_initial_rows)
    if _digest(dict(plan)) != _digest(expected):
        raise ValueError('Refit plan differs from the selected fit or final data geometry')
    return expected
