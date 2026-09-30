import numpy as np


class StopAfterTwo:
    def __init__(self):
        self.starts, self.checked, self.finished = 0, [], None
    def start(self):
        self.starts += 1
    def should_stop(self, completed):
        assert self.starts == 1
        self.checked.append(completed)
        return completed >= 2
    def finish(self, completed):
        self.finished = completed
        return dict(completed_steps=completed)


def test_oh_initial_clock_does_not_limit_online_updates():
    import torch
    from benchmarks.task_stream.gp import OHSVGPTaskAdapter
    from benchmarks.task_stream.protocol import TaskStream
    from scripts import run_covid_ohsvgp_own_theta as official
    clock = StopAfterTwo()
    coordinates = np.array([[0., 0.], [.2, .4], [.6, .3]])
    stream = TaskStream(times=np.arange(4.)*.1, targets=np.arange(12.).reshape(4, 3)/10,
        coordinates=coordinates, visible=[0, 1], hidden=[2], initial_sites=[0, 1],
        initial_steps=2, task_steps=1, release_previous=True)
    adapter = OHSVGPTaskAdapter(coordinates, official.SE_kernel(3).to(dtype=torch.float64),
        official.GaussianLikelihood(.2).to(dtype=torch.float64), inducing_size=3, rff=8,
        initial_steps=10, update_steps=3, grid_rows=3, batch_rows=8, seed=18,
        train_initial_kernel=False, fit_clock=clock)
    adapter.initialize(stream.initial())
    assert adapter.training_iteration == 2 and clock.finished == 2
    adapter.predict_task(stream.task(0))
    assert adapter.training_iteration == 5
    assert clock.starts == 1 and clock.checked == [1, 2] and clock.finished == 2
