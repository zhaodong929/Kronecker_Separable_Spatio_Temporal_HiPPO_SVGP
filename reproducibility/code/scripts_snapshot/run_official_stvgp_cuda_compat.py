#!/usr/bin/env python3
"""Run the upstream ST-SVGP entrypoint with small JAX compatibility shims.

The pinned upstream stack uses JAX APIs removed from current CUDA-enabled JAX.
This wrapper leaves the upstream runner and Bayes-Newton model unchanged.
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import jax
import jax.interpreters.pxla as pxla
import jax.numpy as jnp
import jax.ops as ops


if hasattr(jax, "Array"):
    class _ArrayProxyMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, jax.Array)

    class _LegacyDeviceArray(metaclass=_ArrayProxyMeta):
        pass

    class _LegacyShardedDeviceArray:
        pass

    pxla.ShardedDeviceArray = _LegacyShardedDeviceArray
    jnp.DeviceArray = _LegacyDeviceArray


class _Index:
    def __getitem__(self, item):
        return item


if not hasattr(ops, "index"):
    ops.index = _Index()
if not hasattr(ops, "index_update"):
    ops.index_update = lambda value, index, update: value.at[index].set(update)
if not hasattr(ops, "index_add"):
    ops.index_add = lambda value, index, update: value.at[index].add(update)

if "jax.config" not in sys.modules:
    config_module = types.ModuleType("jax.config")
    config_module.config = jax.config
    sys.modules["jax.config"] = config_module

try:
    from jax.lib import xla_bridge  # noqa: F401
except ImportError:
    bridge_module = types.ModuleType("jax.lib.xla_bridge")
    bridge_module.get_backend = lambda: jax.extend.backend.get_backend()
    jax.lib.xla_bridge = bridge_module
    sys.modules["jax.lib.xla_bridge"] = bridge_module


runpy.run_path(
    str(Path(__file__).with_name("run_official_stvgp_legacy.py")),
    run_name="__main__",
)
