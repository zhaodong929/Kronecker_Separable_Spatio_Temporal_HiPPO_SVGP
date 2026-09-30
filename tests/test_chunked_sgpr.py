"""Exact whole-data SGPR parity, including hyperparameters, Z and mean gradients."""
import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
gpflow = pytest.importorskip('gpflow')
from baselines.chunked_sgpr import ChunkedSGPR


def models(chunk_rows):
    gpflow.config.set_default_float(np.float64)
    rng = np.random.default_rng(97)
    x = rng.normal(size=(23, 3)); y = rng.normal(size=(23, 2)); z = rng.normal(size=(5, 3))
    def build(cls, **kwargs):
        kernel = gpflow.kernels.Matern32(variance=1.3, lengthscales=[.8, 1.1, 1.4])
        mean = gpflow.mean_functions.Linear(A=np.full((3, 2), .04), b=np.array([.2, -.1]))
        return cls((x, y), kernel, z.copy(), mean_function=mean, noise_variance=.3, **kwargs)
    return build(gpflow.models.SGPR), build(ChunkedSGPR, chunk_rows=chunk_rows), rng.normal(size=(7, 3))


@pytest.mark.parametrize('chunk_rows', [1, 7, 100])
@pytest.mark.parametrize('graph', [False, True])
def test_bound_and_every_parameter_gradient_match_dense(chunk_rows, graph):
    dense, chunked, query = models(chunk_rows)
    def evaluate(model):
        with tf.GradientTape() as tape:
            loss = model.training_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        return loss, gradients
    evaluator = tf.function(evaluate, autograph=False) if graph else evaluate
    actual, expected = evaluator(chunked), evaluator(dense)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-11, atol=2e-11)
    assert len(actual[1]) == len(expected[1]) == 6
    for a, b in zip(actual[1], expected[1]):
        assert a is not None and np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-9)
    for full_cov in (False, True):
        for a, b in zip(chunked.predict_f(query, full_cov), dense.predict_f(query, full_cov)):
            np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-11)
    # The carried OSGPR state is predict_f(Z), not compute_qu's different jitter convention.
    for a, b in zip(chunked.predict_f(chunked.inducing_variable.Z, full_cov=True),
                    dense.predict_f(dense.inducing_variable.Z, full_cov=True)):
        np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-11)


def test_two_adam_updates_preserve_exact_training_trajectory():
    dense, chunked, query = models(7)
    optimizers = [tf.optimizers.Adam(.003, jit_compile=False) for _ in range(2)]
    for model, optimizer in zip((dense, chunked), optimizers):
        @tf.function(autograph=False)
        def step():
            with tf.GradientTape() as tape:
                loss = model.training_loss()
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            return loss
        for _ in range(2):
            step()
    for a, b in zip(chunked.trainable_variables, dense.trainable_variables):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-9)
