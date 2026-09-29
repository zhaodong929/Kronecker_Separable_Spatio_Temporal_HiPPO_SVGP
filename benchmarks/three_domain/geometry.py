"""NumPy-only inducing geometry, matching the repository's farthest design."""
import numpy as np


def farthest_indices(coordinates, count):
    coords=np.asarray(coordinates,dtype=float)
    count=int(count)
    if coords.ndim!=2 or not np.isfinite(coords).all() or not 1<=count<=len(coords):
        raise ValueError('Finite coordinate matrix and valid inducing count required')
    normalized=(coords-coords.mean(axis=0,keepdims=True))/np.maximum(coords.std(axis=0,keepdims=True),1e-8)
    selected=[int(np.argmin(np.sum((normalized-normalized.mean(axis=0,keepdims=True))**2,axis=1)))]
    minimum=np.sum((normalized-normalized[selected[0]])**2,axis=1)
    for _ in range(1,count):
        next_index=int(np.argmax(minimum));selected.append(next_index)
        minimum=np.minimum(minimum,np.sum((normalized-normalized[next_index])**2,axis=1))
    return np.asarray(selected,dtype=int)
