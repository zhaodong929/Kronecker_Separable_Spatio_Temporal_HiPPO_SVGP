"""Diagnose a legacy per-site dropped row without shifting the new UTC data."""
import numpy as np


def legacy_row_indices(reference,source,tolerance):
    """Return identity or one proven omitted source row; never relax tolerance.

    This mapping is for checking old archives only. New experiment inputs always
    retain the canonical source time grid, including the formerly omitted hour.
    """
    reference=np.asarray(reference,dtype=float);source=np.asarray(source,dtype=float)
    n=len(reference)
    if source.ndim!=1 or reference.ndim!=1 or len(source)<n or not np.isfinite(source).all() or not np.isfinite(reference).all():raise ValueError('Finite vectors and enough source values required')
    indices=np.arange(n);direct=abs(source[:n]-reference)
    if np.max(direct)<=tolerance:return indices,None
    if len(source)<n+1:raise ValueError('An additional source hour is required to test a dropped row')
    shifted=source[1:n+1]-reference
    costs=np.r_[0.,np.cumsum(direct**2)]+np.r_[np.cumsum((shifted**2)[::-1])[::-1],0.]
    gap=int(np.argmin(costs));indices[gap:]+=1
    if np.max(abs(source[indices]-reference))>tolerance:raise ValueError('Archive does not match identity or a single omitted hour; review source alignment')
    return indices,gap
