"""Look up the shared Task-1-fitted covariate mean at a legal observation time."""
import numpy as np


class CausalMean:
    def __init__(self, protocol):
        with np.load(protocol.npz_path, allow_pickle=False) as a:
            self.values = np.concatenate([a['task1_calibration_mean'],a['task1_stream_mean']])
        protocol._cached_covariate_mean = self
        self.times = np.concatenate([protocol.calibration_times,protocol.chronological_stream_times])
        if self.values.shape != (len(self.times),protocol.locations) or not np.isfinite(self.values).all():
            raise ValueError('Invalid common mean shape or values')

    def at(self, time, locations):
        i = min(int(np.searchsorted(self.times,float(time))),len(self.times)-1)
        if i and abs(self.times[i-1]-time)<abs(self.times[i]-time):
            i -= 1
        if not np.isclose(self.times[i],time,rtol=0,atol=1e-8):
            raise ValueError('Mean query is outside the protocol time grid')
        return self.values[i,np.asarray(locations,dtype=int)]


def select_initial_targets(observation, locations):
    lookup = {int(site):i for i,site in enumerate(observation.locations)}
    try:
        selected = [lookup[int(site)] for site in locations]
    except KeyError as error:
        raise ValueError('Requested an unreleased Task-1 site') from error
    return observation.targets[:,selected].copy()


def get_mean(protocol):
    cached = getattr(protocol, '_cached_covariate_mean', None)
    return CausalMean(protocol) if cached is None else cached
