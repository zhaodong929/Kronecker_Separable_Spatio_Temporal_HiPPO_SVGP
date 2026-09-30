"""Generate the side-by-side COVID qualitative appendix trajectory figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(r"\\wsl.localhost\Ubuntu-24.04\home\zd929\projects\stvgp_kronecker")
OUT = Path(__file__).resolve().parent
SEED = 5
STATES = ("Connecticut", "Nevada")
Z90 = 1.6448536269514722


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def main() -> None:
    protocol = REPO / f"data/epidemiology/protocol/covid_long_2020_2024_mandatory/seed{SEED}"
    metadata = json.loads((protocol / "protocol.json").read_text(encoding="utf-8"))
    with np.load(protocol / "protocol.npz") as z:
        held = np.asarray(z["test_indices"], dtype=int)
    held_names = np.asarray(metadata["location_names"], dtype=str)[held]
    positions = [int(np.flatnonzero(held_names == state)[0]) for state in STATES]
    scale = float(metadata["target_standardization"]["scale"])
    offset = float(metadata["target_standardization"]["mean"])
    paths = {
        "KronHiPPO-STGP": REPO / f"results/diagnostics/covid_long_stream_2020_2024_hippo_vfe_matern_rff256/seed{SEED}/routeb_cumulative/online/predictions.npz",
        "Kron-STGP": REPO / f"results/diagnostics/covid_long_stream_2020_2024_mandatory_vfe/seed{SEED}/routeb_ordinary/online/predictions.npz",
        "ST-SVGP": REPO / f"baselines/covid_long_setting_b/results/formal_selected_st_svgp_isolated/seed{SEED}/st_svgp/predictions.npz",
    }
    loaded = {name: arrays(path) for name, path in paths.items()}
    truth = offset + scale * loaded["KronHiPPO-STGP"]["y_true"]
    means = {name: offset + scale * value["pred_mean"] for name, value in loaded.items()}
    std = scale * np.sqrt(np.maximum(loaded["KronHiPPO-STGP"]["pred_var"], 1e-12))
    x = np.arange(1, truth.shape[0] + 1)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.8, "axes.titlesize": 9.5, "axes.labelsize": 8.8,
        "legend.fontsize": 7.1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.45,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.45), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes.ravel()
    colors = {"KronHiPPO-STGP": "#0072B2", "Kron-STGP": "#D55E00", "ST-SVGP": "#009E73"}
    styles = {"KronHiPPO-STGP": "-", "Kron-STGP": "--", "ST-SVGP": ":"}
    for panel, (ax, state, position) in enumerate(zip(axes, STATES, positions)):
        ours = means["KronHiPPO-STGP"][:, position]
        ax.fill_between(x, ours - Z90 * std[:, position], ours + Z90 * std[:, position],
                        color=colors["KronHiPPO-STGP"], alpha=0.14, linewidth=0,
                        label="KronHiPPO 90% PI", zorder=1)
        ax.plot(x, truth[:, position], color="#202020", linewidth=1.05,
                label="Observed", zorder=4)
        for method in ("KronHiPPO-STGP", "Kron-STGP", "ST-SVGP"):
            ax.plot(x, means[method][:, position], color=colors[method], linestyle=styles[method],
            linewidth=1.35 if method == "KronHiPPO-STGP" else 1.05,
                    label=method, zorder=3)
        ax.set_title(f"({chr(97 + panel)}) {state}", loc="left", pad=2)
        ax.grid(axis="y")
        ax.margins(x=0.01)
    # Use one shared bottom label so the two panels read as a single figure.
    for ax in axes:
        ax.set_xlabel("")
    fig.supxlabel("Online week", y=0.015, fontsize=8.8)
    fig.text(0.012, 0.50, "Log admission rate", rotation=90, va="center", ha="center")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 1.04), frameon=False,
               columnspacing=0.75, handlelength=1.8)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.78, wspace=0.16)
    fig.savefig(OUT / "fig_covid_trajectories_side_by_side.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_covid_trajectories_side_by_side.png", dpi=400, bbox_inches="tight")
    (OUT / "fig_covid_prediction_trajectories_selection.json").write_text(json.dumps({
        "seed": SEED,
        "candidate_states": ["Connecticut", "Louisiana", "Nevada", "West Virginia"],
        "selection_rule": "pair with maximum full-stream RMSE between z-scored observed trajectories",
        "selected_states": list(STATES),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
