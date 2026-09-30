from types import SimpleNamespace
import numpy as np
from benchmarks.task_stream.checkpoint import adapter_snapshot, flatten_state, window_state
from benchmarks.task_stream.factory import Configuration
from benchmarks.task_stream.ablations import Arm
from benchmarks.task_stream.window import TaskWindow
from benchmarks.task_stream.protocol import Observations, Task


_MISSING = object()

def dereference(snapshot, node=_MISSING):
    node = snapshot['structure'] if node is _MISSING else node
    if isinstance(node, dict):
        if 'array_key' in node:
            return snapshot[node['array_key']]
        return {key: dereference(snapshot, value) for key, value in node.items()}
    if isinstance(node, list):
        return [dereference(snapshot, value) for value in node]
    return node


def test_markov_checkpoint_contains_replay_labels_and_pre_task_endpoint():
    initial = Observations([0., 1.], [0, 1], [[1., 2.], [3., 4.]])
    current = Observations([2., 3.], [0], [[5.], [6.]])
    window = TaskWindow(lambda m, p, dt, sites, values: (m + values.sum(), p + len(values)),
                        np.zeros(1), np.ones(1))
    window.initialize(initial)
    window.advance(Task(0, 2, 4, current, None, np.array([1])))
    snapshot = flatten_state(window_state(window))
    restored = dereference(snapshot)
    np.testing.assert_array_equal(restored['previous']['fields']['values'], current.values)
    np.testing.assert_array_equal(restored['previous']['fields']['times'], current.times)
    assert restored['before_previous']['items'][2] == 1.
    np.testing.assert_array_equal(restored['before_previous']['items'][0], [10.])
    assert restored['next_task'] == 1
    assert all(not value.dtype.hasobject for key, value in snapshot.items() if key.startswith('array_'))


def test_actual_kron_checkpoint_contains_rebuildable_multi_geometry_solver():
    from benchmarks.task_stream.gp import KronTaskAdapter
    coordinates = np.array([[0., 0.], [.2, .4], [.6, .3]])
    adapter = KronTaskAdapter(coordinates, np.array([0, 1]), coordinates[:2],
        dict(ell_s=[.7, .8], ell_t=.4, noise_std=.3, kernel_variance=1.),
        initial_step=.1, features=lambda t, sites: np.ones((len(t)*len(sites), 1)),
        feature_dimension=1, mt=3, rff=8, seed=11, multiple_geometry=True)
    adapter.initialize(Observations(np.array([0., .1]), np.array([0, 1]), np.ones((2, 2))))
    wrapper = SimpleNamespace(config=Configuration('kronhippo_svgp', 1, .01), adapter=adapter,
        arm=Arm('joint_transfer'), coordinates=coordinates, visible=np.array([0, 1]), beta=None,
        inverse=None, initial_step=.1, release_previous=True)
    snapshot = adapter_snapshot(wrapper)
    state = dereference(snapshot)
    assert state['restoration']['restoration_tested'] is False
    np.testing.assert_array_equal(state['state']['posterior']['beta_mean'], adapter.state.beta_mean)
    assert state['state']['posterior']['solver']['class_name'] == 'SumKronSolver'
    assert state['state']['posterior']['solver']['terms']['items']
    assert 'base_frequencies' in state['state']['builder_variables']
    assert state['state']['builder_configuration']['dtype'] == 'torch.float64'


def test_actual_oh_checkpoint_captures_variational_kernel_and_rng_state():
    import torch
    from benchmarks.task_stream.gp import OHSVGPTaskAdapter
    from scripts import run_covid_ohsvgp_own_theta as official
    coordinates = np.array([[0., 0.], [.2, .4], [.6, .3]])
    adapter = OHSVGPTaskAdapter(coordinates, official.SE_kernel(3).to(dtype=torch.float64),
        official.GaussianLikelihood(.2).to(dtype=torch.float64), inducing_size=3, rff=8,
        initial_steps=1, update_steps=1, grid_rows=3, batch_rows=3, seed=18,
        train_initial_kernel=False)
    adapter.initialize(Observations(np.array([0., .1]), np.array([0, 1]), np.ones((2, 2))))
    wrapper = SimpleNamespace(config=Configuration('ohsvgp', 1, .01), adapter=adapter,
        arm=Arm('joint_transfer'), coordinates=coordinates, visible=np.array([0, 1]), beta=np.ones(1),
        inverse=None, initial_step=.1, release_previous=True)
    state = dereference(adapter_snapshot(wrapper))['state']
    np.testing.assert_array_equal(state['old']['mv'], adapter.state['mv'])
    np.testing.assert_array_equal(state['frequencies'], adapter.frequencies)
    assert state['kernel_variables']['log_ls'].shape == (3,)
    assert state['numpy_rng']['bit_generator'] == 'PCG64'
    assert state['previous_steps'] == 3
