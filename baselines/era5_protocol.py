"""ERA5 hourly interpolation: initial visible sites, hidden labels never released."""
import numpy as np
from baselines.traffic_protocol_n import TrafficProtocolN
from baselines.covid_long_setting_b.protocol import KnownObservation,HiddenQuery,WeekInformation

class ERA5Protocol(TrafficProtocolN):
    protocol_id='era5_land_hourly_causal'
    hidden_delay_steps=None

    def week(self,stream_week):
        step=int(stream_week)
        if not 0<=step<self.online_weeks:raise IndexError(step)
        t=float(self.chronological_stream_times[step]);g=self.calibration_weeks+step
        visible=KnownObservation('current_visible',step,g,t,self.visible_locations,self._stream_y[step,self.visible_locations].copy())
        query=HiddenQuery(step,g,t,self.hidden_locations)
        return WeekInformation(None,visible,query)

    def _validate(self):
        if self.metadata.get('protocol_id')!=self.protocol_id or self.metadata.get('hidden_label_policy')!='never released':raise ValueError('Wrong ERA5 information contract')
        n=self.locations
        if self.metadata.get('development_protocol'):
            if n!=800 or self.calibration_weeks<1 or self.online_weeks<1:raise ValueError('ERA5 internal fold must use only original 800 visible sites')
            if len(self._visible)!=720 or len(self._hidden)!=80:raise ValueError('Invalid internal ERA5 split')
        elif (n,self.calibration_weeks,self.online_weeks,len(self._visible),len(self._hidden),len(self._fit),len(self._validation))!=(1000,186,1674,800,200,720,80):raise ValueError('Incorrect formal ERA5 shape')
        if set(self._visible)&set(self._hidden) or set(self._visible)|set(self._hidden)!=set(range(n)):raise ValueError('Invalid spatial partition')
        if self.metadata.get('development_protocol'):
            if not np.array_equal(self._fit,self._visible) or not np.array_equal(self._validation,self._hidden):raise ValueError('Invalid internal tuning split')
        elif set(self._fit)&set(self._validation) or set(self._fit)|set(self._validation)!=set(self._visible):raise ValueError('Invalid initial tuning split')
        if set(self.metadata['task1_observed_indices'])!=set(self._visible):raise ValueError('Initial hidden labels must remain unavailable')
        if self.coordinates.shape!=(n,2) or not np.isfinite(self.coordinates).all():raise ValueError('Invalid ERA5 coordinates')
        for values in [self._calibration_y,self._stream_y,self._calibration_times,self._stream_times]:
            if not np.isfinite(values).all():raise ValueError('Nonfinite ERA5 data')
        if not np.all(np.diff(np.concatenate([self._calibration_times,self._stream_times]))>0):raise ValueError('ERA5 time must be continuous and increasing')
        np.testing.assert_allclose(self._stream_times,self.chronological_stream_times,rtol=0,atol=1e-10)
