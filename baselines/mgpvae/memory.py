"""Recompute pure official scan steps during autodiff instead of retaining them.

The pinned source files, loss, precision and random draws remain unchanged.
This process-local execution option trades recomputation for device memory.
"""


def enable_scan_rematerialization():
    import jax
    from jax import lax
    import mgpvae.ops as ops

    def rematerialized_scan(f, init, xs=None, **kwargs):
        return lax.scan(jax.checkpoint(f), init, xs, **kwargs)

    ops.scan = rematerialized_scan
