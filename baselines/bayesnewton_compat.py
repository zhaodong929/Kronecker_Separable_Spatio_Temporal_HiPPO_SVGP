"""API-only shims for pinned Bayes-Newton 1.1 on JAX 0.4.23.

Keep the authors' statistical implementation unchanged. Removed scatter helpers
map to their documented .at equivalents; DeviceArray is a type annotation alias.
"""
def prepare_jax():
    import jax
    import jax.numpy as jnp
    import numpy as np
    if not hasattr(jax.ops, 'index'):
        jax.ops.index = np.index_exp
    if not hasattr(jax.ops, 'index_add'):
        jax.ops.index_add = lambda array, index, values: array.at[index].add(values)
    if not hasattr(jax.ops, 'index_update'):
        jax.ops.index_update = lambda array, index, values: array.at[index].set(values)
    if not hasattr(jnp, 'DeviceArray'):
        jnp.DeviceArray = jax.Array
    jax.config.update('jax_enable_x64', True)


def clear_caches():
    import jax
    jax.clear_caches()
