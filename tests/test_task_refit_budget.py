from copy import deepcopy
import pytest
from benchmarks.task_stream.refit_budget import frozen_refit_plan, validate_refit_plan


def fixture(method='ohsvgp', steps=7, elapsed=61.):
    config = dict(method=method, initial_iterations=1000, initial_max_seconds=60., batch_rows=4)
    record = dict(policy='synchronized_wall_time_v1',
        scope='initial_optimizer_setup_compilation_and_completed_updates', max_seconds=60.,
        elapsed_seconds=elapsed, completed_steps=steps,
        stop_reason='wall_time_budget_reached', overshoot_seconds=max(0.,elapsed-60.),
        minimum_completed_steps=1, convergence_claimed=False)
    return config,record


def test_exact_oh_exposure_and_stable_binding():
    config,record=fixture()
    plan=frozen_refit_plan(config,record,10,23)
    assert plan['effective_iterations']==17  # ceil(7*4/10*23/4)
    assert plan['mode']=='preserve_expected_row_exposures'
    assert validate_refit_plan(plan,config,record,10,23)==plan
    assert frozen_refit_plan(dict(reversed(list(config.items()))),record,10,23)==plan
    assert frozen_refit_plan(config,record,2,3)['effective_iterations']==7
    assert frozen_refit_plan(config,record,2,10)['effective_iterations']==18
    assert not plan['convergence_qualified']


@pytest.mark.parametrize('method',['kronhippo_svgp','osgpr','st_svgp','mgpvae'])
def test_full_batch_keeps_actual_steps(method):
    config,record=fixture(method)
    assert frozen_refit_plan(config,record,10,100)['effective_iterations']==7


@pytest.mark.parametrize('bad',[True,0,-1,1.2,7.0,float('nan')])
def test_reject_nonpositive_or_float_steps_and_rows(bad):
    config,record=fixture()
    with pytest.raises(ValueError):frozen_refit_plan(config,record,bad,20)
    with pytest.raises(ValueError):frozen_refit_plan(config,dict(record,completed_steps=bad),10,20)


def test_iteration_safety_cap_is_not_time_qualification():
    config,record=fixture(steps=1000,elapsed=20.)
    record['stop_reason']='iteration_limit_reached'
    plan=frozen_refit_plan(config,record,10,20)
    assert plan['iteration_safety_cap_reached'] and not plan['wall_time_budget_reached']
    with pytest.raises(ValueError):frozen_refit_plan(config,dict(record,completed_steps=999),10,20)


@pytest.mark.parametrize('change',[dict(max_seconds=180.),dict(elapsed_seconds=float('nan')),
    dict(overshoot_seconds=0.),dict(policy='selected_fixed_refit'),dict(stop_reason='iteration_limit_reached'),
    dict(convergence_claimed=True),dict(minimum_completed_steps=True),dict(completed_steps=1001)])
def test_stale_or_inconsistent_selection_record_rejected(change):
    config,record=fixture()
    with pytest.raises(ValueError):frozen_refit_plan(config,dict(record,**change),10,20)


def test_recompute_rejects_tampering_and_geometry_changes():
    config,record=fixture();plan=frozen_refit_plan(config,record,10,20)
    bad=deepcopy(plan);bad['effective_iterations']+=1
    with pytest.raises(ValueError):validate_refit_plan(bad,config,record,10,20)
    with pytest.raises(ValueError):validate_refit_plan(plan,config,record,10,21)
    with pytest.raises(ValueError):validate_refit_plan(plan,dict(config,batch_rows=3),record,10,20)
    bad=deepcopy(plan);bad['selected_steps']=7.0
    with pytest.raises(ValueError):validate_refit_plan(bad,config,record,10,20)
