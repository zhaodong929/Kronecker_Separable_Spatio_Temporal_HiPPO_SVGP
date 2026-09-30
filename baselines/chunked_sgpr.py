"""Exact GPflow SGPR with bounded activation memory, without editing GPflow.

The collapsed whole-data ELBO is unchanged. Additive statistics are accumulated
in chunks; a custom VJP recomputes each chunk to avoid retaining N-by-M kernel
activations across the reverse pass. Data are never subsampled. Cholesky and
prediction expressions follow pinned GPflow 2.9 SGPR, including its jitter.
"""
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.covariances import Kuf, Kuu


class ChunkedSGPR(gpflow.models.SGPR):
    def __init__(self, *args, chunk_rows=2048, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(chunk_rows, bool) or int(chunk_rows) != chunk_rows or chunk_rows < 1:
            raise ValueError('Positive integer chunk size required')
        self.chunk_rows = int(chunk_rows)

    def _part(self, start, end, chol):
        x, y = self.data
        xx, yy = x[start:end], y[start:end]
        noise = tf.squeeze(self.likelihood.variance_at(xx), axis=-1)
        std = tf.sqrt(noise)
        error = (yy-self.mean_function(xx))/std[:, None]
        a = tf.linalg.triangular_solve(chol, Kuf(self.inducing_variable, self.kernel, xx)/std, lower=True)
        return (tf.matmul(a, a, transpose_b=True), tf.matmul(a, error),
                tf.reduce_sum(tf.square(error)),
                tf.reduce_sum(self.kernel(xx, full_cov=False)/noise),
                tf.reduce_sum(tf.math.log(noise)))

    def _statistics(self, chol):
        x, y = self.data
        count = tf.shape(x)[0]
        size = tf.shape(chol)[0]
        outputs = tf.shape(y)[1]
        zeros = (tf.zeros((size, size), chol.dtype), tf.zeros((size, outputs), chol.dtype),
                 tf.zeros((), chol.dtype), tf.zeros((), chol.dtype), tf.zeros((), chol.dtype))
        step = tf.constant(self.chunk_rows, tf.int32)

        @tf.custom_gradient
        def calculate(l):
            def forward(start, *totals):
                values = self._part(start, tf.minimum(start+step, count), l)
                return (start+step, *(total+value for total, value in zip(totals, values)))
            statistics = tf.while_loop(lambda start, *s: start < count, forward,
                (tf.constant(0, tf.int32), *zeros), parallel_iterations=1)[1:]

            def gradient(*upstream, variables=None):
                variables = [] if variables is None else variables
                watched = [l, *variables]
                initial = tuple(tf.zeros_like(value) for value in watched)
                def backward(start, *totals):
                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(watched)
                        parts = self._part(start, tf.minimum(start+step, count), l)
                        scalar = tf.add_n([tf.reduce_sum(part*weight) for part, weight in zip(parts, upstream)
                                          if weight is not None])
                    derivatives = tape.gradient(scalar, watched, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    return (start+step, *(total+derivative for total, derivative in zip(totals, derivatives)))
                derivatives = tf.while_loop(lambda start, *s: start < count, backward,
                    (tf.constant(0, tf.int32), *initial), parallel_iterations=1)[1:]
                return derivatives[0], list(derivatives[1:])
            return tuple(statistics), gradient
        return calculate(chol)

    def _bounded_common(self):
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=gpflow.default_jitter())
        chol = tf.linalg.cholesky(kuu)
        aat, aerr, error_squared, trace_kernel, log_noise = self._statistics(chol)
        lb = tf.linalg.cholesky(aat+tf.eye(tf.shape(kuu)[0], dtype=kuu.dtype))
        c = tf.linalg.triangular_solve(lb, aerr, lower=True)
        return chol, lb, c, aat, error_squared, trace_kernel, log_noise

    def elbo(self):
        _, lb, c, aat, error_squared, trace_kernel, log_noise = self._bounded_common()
        output_dim = tf.cast(tf.shape(self.data[1])[1], lb.dtype)
        n = tf.cast(tf.shape(self.data[0])[0], lb.dtype)
        constant = -.5*n*output_dim*np.log(2*np.pi)
        logdet = -output_dim*(tf.reduce_sum(tf.math.log(tf.linalg.diag_part(lb))) +
                              .5*log_noise + .5*(trace_kernel-tf.linalg.trace(aat)))
        quadratic = -.5*(error_squared-tf.reduce_sum(tf.square(c)))
        return constant+logdet+quadratic

    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        if full_output_cov:
            raise NotImplementedError('This comparison uses independent scalar output distributions')
        chol, lb, c, _, _, _, _ = self._bounded_common()
        kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        first = tf.linalg.triangular_solve(chol, kus, lower=True)
        second = tf.linalg.triangular_solve(lb, first, lower=True)
        mean = tf.matmul(second, c, transpose_a=True)+self.mean_function(Xnew)
        if full_cov:
            variance = self.kernel(Xnew)+tf.matmul(second, second, transpose_a=True)-tf.matmul(first, first, transpose_a=True)
            variance = tf.tile(variance[None], [self.num_latent_gps, 1, 1])
        else:
            variance = self.kernel(Xnew, full_cov=False)+tf.reduce_sum(tf.square(second), 0)-tf.reduce_sum(tf.square(first), 0)
            variance = tf.tile(variance[:, None], [1, self.num_latent_gps])
        return mean, variance
