"""Generate the single-column PEMS-BAY qualitative prediction figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(r"\\wsl.localhost\Ubuntu-24.04\home\zd929\projects\stvgp_kronecker")
OUT = Path(__file__).resolve().parent
Z90 = 1.6448536269514722

BLUE = "#0072B2"
ORANGE = "#D55E00"
BLACK = "#202020"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def pems_panel() -> tuple[np.ndarray, list[dict[str, np.ndarray]], list[str], list[float]]:
    seed = 1
    base = REPO / "results/traffic/formal_locked_sm_q2_road_context_v1/pems_bay/nowcast"
    paths = {
        "KronHiPPO-STGP": base / f"kronhippo_stgp/seed{seed}/predictions.npz",
        "Kron-STGP": base / f"kron_stgp/seed{seed}/predictions.npz",
        "IGNNK": base / f"ignnk/seed{seed}/predictions.npz",
    }
    result = json.loads((base / f"kronhippo_stgp/seed{seed}/result.json").read_text(encoding="utf-8"))
    scale = float(result["target_standardisation"]["scale"])
    offset = float(result["target_standardisation"]["mean"])
    ours = load_npz(paths["KronHiPPO-STGP"])
    truth = offset + scale * ours["y"]
    ours_mean = offset + scale * ours["mean"]
    ours_std = scale * np.sqrt(np.maximum(ours["variance"], 1e-12))
    split = json.loads((REPO / result["split_manifest"]).read_text(encoding="utf-8"))
    sensor_ids = split["split"]["heldout_sensor_ids"]
    hippo_rmse = np.sqrt(np.mean((ours_mean - truth) ** 2, axis=0))
    # These are the predeclared low/median/high cases used by the archived
    # Appendix trajectory figure; keeping their IDs fixed makes the move to
    # the main text provenance-stable.
    selected_sensor_ids = ("400952", "400965", "400178")
    columns = [sensor_ids.index(sensor_id) for sensor_id in selected_sensor_ids]
    # Same truth-only 48-hour window rule used by the archived PEMS trajectory.
    window = 48 * 12
    starts = np.arange(0, truth.shape[0] - window + 1, 24 * 12, dtype=int)
    scores = [float(np.median(np.quantile(truth[s:s + window], .95, axis=0) -
                              np.quantile(truth[s:s + window], .05, axis=0))) for s in starts]
    start = int(starts[int(np.argmax(scores))])
    stop = start + window
    kron = load_npz(paths["Kron-STGP"])
    kron_mean = offset + scale * kron["mean"]
    ignnk = load_npz(paths["IGNNK"])
    ignnk_mean = offset + scale * ignnk["mean"]
    records = []
    selected_ids, selected_rmse = [], []
    for column in columns:
        records.append({
            "Observed": truth[start:stop, column],
            "KronHiPPO-STGP": ours_mean[start:stop, column],
            "KronHiPPO 90% PI": ours_std[start:stop, column],
            "Kron-STGP": kron_mean[start:stop, column],
            "IGNNK": ignnk_mean[start:stop, column],
        })
        selected_ids.append(str(sensor_ids[column]))
        selected_rmse.append(float(hippo_rmse[column]))
    return np.arange(1, window + 1), records, selected_ids, selected_rmse


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.6,
        "legend.fontsize": 7.2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.45,
        "lines.linewidth": 1.1,
    })
    px, p, sensors, sensor_rmses = pems_panel()
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.65), sharex=True, sharey=True)
    for ax, rec, title in zip(axes, p, ("(a) Low-error case", "(b) Median case", "(c) High-error case")):
        x = px
        pi = rec["KronHiPPO 90% PI"]
        mean = rec["KronHiPPO-STGP"]
        ax.fill_between(x, mean - Z90 * pi, mean + Z90 * pi, color=BLUE, alpha=0.14,
                        linewidth=0, label="KronHiPPO 90% PI", zorder=1)
        ax.plot(x, rec["Observed"], color=BLACK, linewidth=1.05, label="Observed", zorder=3)
        ax.plot(x, mean, color=BLUE, linewidth=1.3, label="KronHiPPO-STGP", zorder=4)
        ax.plot(x, rec["Kron-STGP"], color=ORANGE, linestyle="--", linewidth=0.9,
                label="Kron-STGP", zorder=2)
        ax.plot(x, rec["IGNNK"], color="#009E73", linestyle=":", linewidth=1.0,
                label="IGNNK", zorder=2)
        ax.set_title(title, loc="left", pad=2.0)
        ax.set_ylabel("")
        ax.margins(x=0.01)
        ax.grid(axis="y")
    axes[-1].set_xlabel("Time step (5 min)")
    fig.text(0.018, 0.5, "Traffic speed (mph)", rotation=90, va="center", ha="center")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 4, 2, 3, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.005), frameon=False,
               columnspacing=0.65, handlelength=1.8)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.085, top=0.895, hspace=0.25)
    fig.savefig(OUT / "fig_representative_replay_free.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_representative_replay_free.png", dpi=400, bbox_inches="tight")
    (OUT / "fig_representative_replay_free_selection.json").write_text(json.dumps({
        "pems_seed": 1,
        "pems_selection_rule": "predeclared low/median/high cases from the archived Appendix trajectory selection",
        "pems_sensors": sensors, "pems_sensor_rmse_mph": sensor_rmses,
        "pems_window_rule": "truth-only 48-hour window maximising median held-out speed range",
    }, indent=2) + "\n", encoding="utf-8")
    print(sensors, sensor_rmses)


if __name__ == "__main__":
    main()
