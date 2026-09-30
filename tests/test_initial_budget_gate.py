import pytest
from benchmarks.three_domain.convergence import initial_budget_resolved


def test_plateau_is_respected_even_when_validation_best_is_last_checkpoint():
    for status in ['converged_objective_plateau','converged_elbo_plateau']:
        assert initial_budget_resolved(1000,1000,status)
    assert not initial_budget_resolved(1000,1000,'max_budget_not_converged')
    assert not initial_budget_resolved(1000,1000,'task1_validation_complete')
    assert initial_budget_resolved(735,1000,'max_budget_not_converged')


@pytest.mark.parametrize('selected',[0,-1,1001])
def test_invalid_checkpoint_cannot_resolve_gate(selected):
    with pytest.raises(ValueError):initial_budget_resolved(selected,1000,'converged_objective_plateau')
