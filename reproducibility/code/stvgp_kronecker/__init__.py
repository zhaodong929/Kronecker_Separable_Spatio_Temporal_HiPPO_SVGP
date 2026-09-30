"""Kronecker-separable spatio-temporal HiPPO-SVGP prototype.

The public model classes are loaded lazily so lightweight modules such as the
benchmark runtime can be imported inside TensorFlow-only baseline environments.
"""

from importlib import import_module

__all__ = [
    "AnalyticTemporalBuilder",
    "BatchKroneckerSTHiPPOSVGP",
    "MaternKernel",
    "OnlinePosteriorSummarySTGP",
    "RBFKernel",
    "SpatialKernelConfig",
    "TemporalAnalyticConfig",
    "TemporalBlockSpec",
]


_EXPORTS = {
    "AnalyticTemporalBuilder": (".temporal_analytic", "AnalyticTemporalBuilder"),
    "TemporalAnalyticConfig": (".temporal_analytic", "TemporalAnalyticConfig"),
    "TemporalBlockSpec": (".temporal_analytic", "TemporalBlockSpec"),
    "MaternKernel": (".spatial_kernel", "MaternKernel"),
    "RBFKernel": (".spatial_kernel", "RBFKernel"),
    "SpatialKernelConfig": (".spatial_kernel", "SpatialKernelConfig"),
    "BatchKroneckerSTHiPPOSVGP": (".st_model_batch", "BatchKroneckerSTHiPPOSVGP"),
    "OnlinePosteriorSummarySTGP": (".st_model_online", "OnlinePosteriorSummarySTGP"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
