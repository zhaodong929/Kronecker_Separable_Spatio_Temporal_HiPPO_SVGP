# ERA5 `processed_timeseries_4` Inspection

- root: `data/era5/processed_timeseries_4`
- inspected tasks: `task_1, task_2`
- sample files per task/scaledness: 1

## task_1

- sequence dir: `data/era5/processed_timeseries_4/task_1/sequences`
- all real `.npz`: 2000
- scaled `*_scaled.npz`: 1000
- unscaled `.npz`: 1000
- parse errors: 0
- unique lat/lon locations: 1000
- latitude range: 49.0000 to 59.1000
- longitude range: -10.0000 to 2.0000

### sample `task_1/lat_49.0000_lon_-0.4000.npz`

- keys: `['data_train', 'time_train', 'data_val', 'time_val', 'data_test', 'time_test']`
- `data_train`: shape=(35, 130), dtype=float32, first=[277.6216, 280.1638, 277.5676, 277.5332, 274.9121]
  - numeric health: nan=0, inf=0, min=-4316900.0, max=29476244.0
- `time_train`: shape=(130,), dtype=int64, first=[ 97,  32,  90,  86, 130]
  - numeric health: nan=0, inf=0, min=0.0, max=185.0
- `data_val`: shape=(35, 18), dtype=float32, first=[283.9675, 276.9515, 279.5798, 279.7576, 273.6003]
  - numeric health: nan=0, inf=0, min=-4226768.0, max=25653772.0
- `time_val`: shape=(18,), dtype=int64, first=[165, 111,  26, 139, 128]
  - numeric health: nan=0, inf=0, min=16.0, max=182.0
- `data_test`: shape=(35, 38), dtype=float32, first=[275.1738, 281.5554, 276.1914, 278.3184, 283.9792]
  - numeric health: nan=0, inf=0, min=-4469002.0, max=29625824.0
- `time_test`: shape=(38,), dtype=int64, first=[ 75,  62,   2,  10, 179]
  - numeric health: nan=0, inf=0, min=1.0, max=183.0

### sample `task_1/lat_49.0000_lon_-0.4000_scaled.npz`

- keys: `['data_train', 'time_train', 'data_val', 'time_val', 'data_test', 'time_test']`
- `data_train`: shape=(35, 130), dtype=float32, first=[-0.2077,  0.7546, -0.2281, -0.2412, -1.2333]
  - numeric health: nan=0, inf=0, min=-2.8932688236236572, max=4.113841533660889
- `time_train`: shape=(130,), dtype=int64, first=[ 97,  32,  90,  86, 130]
  - numeric health: nan=0, inf=0, min=0.0, max=185.0
- `data_val`: shape=(35, 18), dtype=float32, first=[ 2.1945, -0.4613,  0.5336,  0.6009, -1.7299]
  - numeric health: nan=0, inf=0, min=-2.8585448265075684, max=4.113841533660889
- `time_val`: shape=(18,), dtype=int64, first=[165, 111,  26, 139, 128]
  - numeric health: nan=0, inf=0, min=16.0, max=182.0
- `data_test`: shape=(35, 38), dtype=float32, first=[-1.1343,  1.2814, -0.7491,  0.0561,  2.1989]
  - numeric health: nan=0, inf=0, min=-2.8932688236236572, max=4.113841533660889
- `time_test`: shape=(38,), dtype=int64, first=[ 75,  62,   2,  10, 179]
  - numeric health: nan=0, inf=0, min=1.0, max=183.0

### sequence-length consistency

- `data_test`: OK, unique lengths=[38], files=2000
- `data_train`: OK, unique lengths=[130], files=2000
- `data_val`: OK, unique lengths=[18], files=2000
- `time_test`: OK, unique lengths=[38], files=2000
- `time_train`: OK, unique lengths=[130], files=2000
- `time_val`: OK, unique lengths=[18], files=2000

### aggregate NaN/inf check

- `data_test`: OK, nan=0, inf=0
- `data_train`: OK, nan=0, inf=0
- `data_val`: OK, nan=0, inf=0
- `time_test`: OK, nan=0, inf=0
- `time_train`: OK, nan=0, inf=0
- `time_val`: OK, nan=0, inf=0

### `scaler.pkl`

- normal pickle failed: `UnpicklingError: invalid load key, '\x08'.`
- loaded with `joblib.load`
- type: `sklearn.preprocessing._data.StandardScaler`
- attributes:
  - `copy`: `True`
  - `mean_`: shape=(35,), dtype=float64, first=[278.1703, 280.1917, 279.5118, 279.5062, 279.4747]
  - `n_features_in_`: `35`
  - `n_samples_seen_`: `130000`
  - `scale_`: shape=(35,), dtype=float64, first=[2.6417, 2.3386, 2.4859, 1.8164, 1.4715]
  - `var_`: shape=(35,), dtype=float64, first=[6.9786, 5.4689, 6.1799, 3.2993, 2.1652]
  - `with_mean`: `True`
  - `with_std`: `True`

## task_2

- sequence dir: `data/era5/processed_timeseries_4/task_2/sequences`
- all real `.npz`: 2000
- scaled `*_scaled.npz`: 1000
- unscaled `.npz`: 1000
- parse errors: 0
- unique lat/lon locations: 1000
- latitude range: 49.0000 to 59.1000
- longitude range: -10.0000 to 2.0000

### sample `task_2/lat_49.0000_lon_-0.4000.npz`

- keys: `['data_train', 'time_train', 'data_val', 'time_val', 'data_test', 'time_test']`
- `data_train`: shape=(35, 130), dtype=float32, first=[276.4122, 279.6367, 276.5044, 278.251 , 278.9551]
  - numeric health: nan=0, inf=0, min=-5134284.0, max=31671396.0
- `time_train`: shape=(130,), dtype=int64, first=[228, 210, 234, 368, 213]
  - numeric health: nan=0, inf=0, min=186.0, max=371.0
- `data_val`: shape=(35, 18), dtype=float32, first=[279.1665, 283.5291, 281.6838, 279.5718, 283.1055]
  - numeric health: nan=0, inf=0, min=-3659308.0, max=30360356.0
- `time_val`: shape=(18,), dtype=int64, first=[282, 196, 301, 370, 193]
  - numeric health: nan=0, inf=0, min=191.0, max=370.0
- `data_test`: shape=(35, 38), dtype=float32, first=[275.689 , 281.3723, 277.0104, 276.7786, 277.3608]
  - numeric health: nan=0, inf=0, min=-5174562.0, max=28389948.0
- `time_test`: shape=(38,), dtype=int64, first=[248, 303, 293, 361, 260]
  - numeric health: nan=0, inf=0, min=197.0, max=367.0

### sample `task_2/lat_49.0000_lon_-0.4000_scaled.npz`

- keys: `['data_train', 'time_train', 'data_val', 'time_val', 'data_test', 'time_test']`
- `data_train`: shape=(35, 130), dtype=float32, first=[-0.6655,  0.5551, -0.6306,  0.0306,  0.2971]
  - numeric health: nan=0, inf=0, min=-6.1988301277160645, max=4.027458667755127
- `time_train`: shape=(130,), dtype=int64, first=[228, 210, 234, 368, 213]
  - numeric health: nan=0, inf=0, min=186.0, max=371.0
- `data_val`: shape=(35, 18), dtype=float32, first=[0.3771, 2.0285, 1.33  , 0.5305, 1.8682]
  - numeric health: nan=0, inf=0, min=-3.2672922611236572, max=2.5207359790802
- `time_val`: shape=(18,), dtype=int64, first=[282, 196, 301, 370, 193]
  - numeric health: nan=0, inf=0, min=191.0, max=370.0
- `data_test`: shape=(35, 38), dtype=float32, first=[-0.9393,  1.2121, -0.4391, -0.5268, -0.3064]
  - numeric health: nan=0, inf=0, min=-5.765665531158447, max=3.421444892883301
- `time_test`: shape=(38,), dtype=int64, first=[248, 303, 293, 361, 260]
  - numeric health: nan=0, inf=0, min=197.0, max=367.0

### sequence-length consistency

- `data_test`: OK, unique lengths=[38], files=2000
- `data_train`: OK, unique lengths=[130], files=2000
- `data_val`: OK, unique lengths=[18], files=2000
- `time_test`: OK, unique lengths=[38], files=2000
- `time_train`: OK, unique lengths=[130], files=2000
- `time_val`: OK, unique lengths=[18], files=2000

### aggregate NaN/inf check

- `data_test`: OK, nan=0, inf=0
- `data_train`: OK, nan=0, inf=0
- `data_val`: OK, nan=0, inf=0
- `time_test`: OK, nan=0, inf=0
- `time_train`: OK, nan=0, inf=0
- `time_val`: OK, nan=0, inf=0

### `scaler.pkl`

- normal pickle failed: `UnpicklingError: invalid load key, '\x08'.`
- loaded with `joblib.load`
- type: `sklearn.preprocessing._data.StandardScaler`
- attributes:
  - `copy`: `True`
  - `mean_`: shape=(35,), dtype=float64, first=[278.1703, 280.1917, 279.5118, 279.5062, 279.4747]
  - `n_features_in_`: `35`
  - `n_samples_seen_`: `130000`
  - `scale_`: shape=(35,), dtype=float64, first=[2.6417, 2.3386, 2.4859, 1.8164, 1.4715]
  - `var_`: shape=(35,), dtype=float64, first=[6.9786, 5.4689, 6.1799, 3.2993, 2.1652]
  - `with_mean`: `True`
  - `with_std`: `True`

## Global scaler

### `global_scaler.pkl`

- normal pickle failed: `UnpicklingError: invalid load key, '\x08'.`
- loaded with `joblib.load`
- type: `sklearn.preprocessing._data.StandardScaler`
- attributes:
  - `copy`: `True`
  - `mean_`: shape=(35,), dtype=float64, first=[278.1703, 280.1917, 279.5118, 279.5062, 279.4747]
  - `n_features_in_`: `35`
  - `n_samples_seen_`: `130000`
  - `scale_`: shape=(35,), dtype=float64, first=[2.6417, 2.3386, 2.4859, 1.8164, 1.4715]
  - `var_`: shape=(35,), dtype=float64, first=[6.9786, 5.4689, 6.1799, 3.2993, 2.1652]
  - `with_mean`: `True`
  - `with_std`: `True`

## Cross-task location overlap

- overlap locations across inspected tasks: 1000
- union locations across inspected tasks: 1000
