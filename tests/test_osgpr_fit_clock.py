"""Execute the actual adapter function without importing unrelated CLI dependencies."""
import ast
from pathlib import Path
import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
gpflow = pytest.importorskip('gpflow')


@pytest.mark.parametrize('execution', ['eager', 'graph'])
def test_actual_osgpr_optimizer_honors_completed_step_initial_budget(execution):
    source = Path(__file__).resolve().parents[1]/'scripts/run_official_bui_osgpr_era5.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'adapt_model')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(tf=tf, OSGPR_VFE=type('OnlineOnly', (), {}), emit=lambda *args: None, _OPTIMIZER_STEP=0)
    exec(compile(module, str(source), 'exec'), namespace)
    class Clock:
        starts = 0
        finished = None
        def start(self):
            self.starts += 1
        def should_stop(self, completed):
            assert self.starts == 1
            return completed == 2
        def finish(self, completed):
            self.finished = completed
    clock = Clock()
    x = np.arange(6., dtype=np.float64)[:, None]/5
    model = gpflow.models.SGPR((x, np.sin(x)), gpflow.kernels.Matern32(), x[:3], noise_variance=.2)
    completed = namespace['adapt_model'](model, steps=100, learning_rate=.001,
        execution=execution, fit_clock=clock)
    assert completed == clock.finished == 2 and clock.starts == 1
    assert namespace['_OPTIMIZER_STEP'] == 2
