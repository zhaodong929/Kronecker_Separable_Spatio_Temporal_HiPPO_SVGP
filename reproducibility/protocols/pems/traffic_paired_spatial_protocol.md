# Traffic Paired-Spatial Streaming Protocol

`PEMS-BAY` is the primary traffic experiment and `METR-LA` is a lightweight
replication. The experiments use their released five-minute speed arrays,
sensor latitude/longitude metadata, and a protocol designed for this project;
it is not the conventional DCRNN 7:1:2 temporal train/validation/test split.

Each dataset receives five deterministic, manifest-backed spatial splits. A
PEMS-BAY split has 260 visible and 65 held-out sensors; a METR-LA split has 166
visible and 41 held-out sensors. The visible sensors are deterministically
partitioned into a 90% Task-1 calibration subset and a 10% validation subset.
The validation subset is for Task-1 capacity selection only and returns to the
visible set during the online stream.

The target is standardised by the mean and standard deviation of the Task-1
prefix only. The fixed mean design is

`[1, sin(time-of-day), cos(time-of-day), sin(day-of-week), cos(day-of-week), lat, lon]`.

Latitude and longitude are independently standardised. The input is not a road
raster or an adjacency matrix. The spatial GP operates directly on these
representative sensor coordinates.

## Protocol N: strict-online spatial nowcasting

At an online five-minute endpoint `t`, the runner performs exactly:

1. absorb `y_H,t-1` once, when it has become available;
2. condition on contemporaneous `y_V,t`;
3. predict `y_H,t`;
4. reveal `y_H,t` only after the prediction has been written.

The initial Task-1 posterior is fit using visible sensors only, so the first
online held-out target has no hidden-sensor label in state. This makes the
information boundary uniform: held-out readings enter only through the one-step
delayed reveal path. Output reports include the zero pre-prediction hidden-read
counter and number of unique delayed absorptions.

Accordingly, the nowcasting persistence control is **delayed-target
last-value persistence**: at endpoint `t` it predicts with the legally revealed
`y_H,t-1`, never with `y_H,t`. KronHiPPO-STGP, Kron-STGP and the adapted IGNNK
receive the same held-out history through `t-1`. In this protocol,
`current_hidden_reads = 0` means zero reads of `y_H,t` before prediction; it
does not mean that all earlier held-out labels are permanently withheld.

Changing-coordinate and fixed-global temporal controls use different temporal
coordinate systems. Their score difference may therefore be associated with
the changing-coordinate HiPPO representation, but it is not by itself an
estimate of transfer-approximation cost. That cost requires comparing repeated
changing-coordinate transfer with all-seen recomputation in the same current
basis.

## Protocol F: future forecasting

For horizon `h in {3, 6, 12}` (15, 30 and 60 minutes), the state after the
legal update at `t` predicts `y_H,t+h`. Calendar features at `t+h` are marked
known future quantities. Neither a future target nor an unknown future speed or
exogenous variable is read before prediction.

## Predeclared Runs

`scripts/run_traffic_suite.py --stage main` runs KronHiPPO-STGP, point-inducing
Kron-STGP, mean-field/decoupled changing-transfer, persistence and frozen mean
for the five manifests. `--stage mechanism` runs the four cells
Joint/Decoupled x Changing/Fixed. `--stage forecast` runs the three causal
forecasting comparators. PEMS-BAY is the complete suite; METR-LA initially runs
main plus mechanism.

Use `--dry-run` to materialise a source-commit-locked suite manifest before
launching a formal GPU job. `--max-stream-steps` and `--stream-stride > 1`
define a smoke or thinned diagnostic stream and must not be reported as formal
full-stream results.

Before a formal suite, run `scripts/calibrate_traffic_task1.py` once per fixed
spatial split and store each selected checkpoint as
`{theta-root}/{dataset}/seed{seed}/theta.json`. Pass that root to the suite
with `--theta-root`; the runner then records the locked Task-1 VFE parameters
inside every output archive.

After Task 1, the online runner uses the locked RFF draw and parameters.  The
formal suite evaluates that fixed analytic HiPPO basis with SciPy's stable
`spherical_jn` implementation (`--temporal-evaluator scipy_frozen`), which is
numerically checked against the differentiable Torch evaluator at the full
PEMS-BAY horizon.  Task-1 VFE calibration itself retains the original
differentiable Torch implementation.
