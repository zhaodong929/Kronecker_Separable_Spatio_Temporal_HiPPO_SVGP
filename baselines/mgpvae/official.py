"""Construct the pinned official MGPVAE without editing its model or objective."""
from pathlib import Path
import subprocess
import sys

COMMIT = 'c9a80a05fca66b2911c8e7cb0deccb39d261a5b8'
URL = 'https://github.com/harrisonzhu508/MGPVAE'


def import_official(source):
    source = Path(source).resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT:
        raise ValueError(f'MGPVAE source must be pinned at {COMMIT}, got {commit}')
    if subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--', 'mgpvae'], text=True):
        raise ValueError('Official MGPVAE model source has local modifications')
    sys.path.insert(0, str(source))
    import jax
    jax.config.update('jax_enable_x64', True)
    from mgpvae.model import STMarkovGaussianProcessVAEExternal
    return STMarkovGaussianProcessVAEExternal


def make_model(source, coordinates, *, seed=0, latent=2, width=16):
    cls = import_official(source)
    import jax.numpy as jnp
    import objax
    from mgpvae.kernels import SpatiotemporalMatern32
    from mgpvae.likelihood import DecoderGaussian
    from mgpvae.networks import Linear
    objax.random.DEFAULT_GENERATOR.seed(seed)
    encoder = objax.nn.Sequential([Linear(1, width), objax.functional.relu])
    decoder = objax.nn.Sequential([Linear(latent, 16), objax.functional.relu, Linear(16, 1)])
    kernel = SpatiotemporalMatern32(
        R=jnp.asarray(coordinates), lengthscale=2., variance=1.,
        lengthscale_time=jnp.ones(latent)*5., variance_time=jnp.ones(latent),
        fix_variance=True)
    return cls(kernel=kernel,
        likelihood=DecoderGaussian(decoder, variance=1., num_latent=latent, y_dim=1),
        encoder=encoder, num_hidden=width,
        hidden_to_mu=objax.nn.Sequential([Linear(width, latent)]),
        hidden_to_var=objax.nn.Sequential([Linear(width, latent)]),
        num_latent=latent, dt=None, minibatch_size=1, num_sequences=1, parallel=False)
