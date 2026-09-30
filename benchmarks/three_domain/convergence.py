"""Interpret the worker's predeclared convergence gate without discarding it."""


def initial_budget_resolved(selected_iteration,budget,convergence_status):
    if not 0<int(selected_iteration)<=int(budget):
        raise ValueError('Selected checkpoint is outside its validated budget')
    # An interior validation optimum resolves selection. At the endpoint, a
    # separately satisfied, predeclared objective/ELBO plateau also resolves it.
    # Merely exhausting the budget never counts as convergence.
    return int(selected_iteration)<int(budget) or convergence_status in {
        'converged_objective_plateau','converged_elbo_plateau'}
