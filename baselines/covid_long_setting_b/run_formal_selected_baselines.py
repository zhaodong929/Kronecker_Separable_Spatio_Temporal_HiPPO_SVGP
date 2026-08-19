#!/usr/bin/env python3
"""Run capacity-selected Gaussian baselines on formal COVID Setting B seeds.

The runner never chooses a new configuration: it reads the seed-0 Task-1
selection record and refits every selected method on the full Task-1 history of
each formal split before the 143-week delayed stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "ovc_svgp",
    "bui_controlled",
    "bui_adaptive",
    "st_svgp",
    "lmc_svgp",
    "imc_svgp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/task1_capacity_selection/task1_capacity_selection.json"),
    )
    parser.add_argument(
        "--factorial-selection-json",
        type=Path,
        default=Path(
            "baselines/covid_long_setting_b/results/"
            "task1_capacity_selection_factorial/task1_capacity_selection.json"
        ),
    )
    parser.add_argument(
        "--shared-bui-ovc-selection-json",
        type=Path,
        default=Path(
            "baselines/covid_long_setting_b/results/task1_capacity_selection/"
            "shared_bui_ovc_grid_selection.json"
        ),
        help="Predeclared common grid for the semantically comparable Bui/OVC point-inducing baselines.",
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("data/epidemiology/protocol/covid_long_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--routeb-root",
        type=Path,
        default=Path("results/diagnostics/covid_long_stream_2020_2024_mandatory"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("baselines/covid_long_setting_b/results/formal_selected"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:-1]))
    parser.add_argument("--max-weeks", type=int, default=0)
    parser.add_argument("--ovc-python", type=Path, default=Path("baselines/.venvs/ovc_svgp/bin/python"))
    parser.add_argument("--st-python", type=Path, default=Path("baselines/.venvs/st_svgp/bin/python"))
    parser.add_argument("--bui-python", type=Path, default=Path(".venv_osgpr/bin/python"))
    parser.add_argument(
        "--factorial-python",
        type=Path,
        default=Path("baselines/.venvs/factorial_sde_gpflow38/bin/python"),
    )
    parser.add_argument("--st-task1-iterations", type=int, default=300)
    parser.add_argument("--st-online-inference-steps", type=int, default=5)
    parser.add_argument("--ovc-task1-iterations", type=int, default=300)
    parser.add_argument("--ovc-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--factorial-task1-iterations", type=int, default=50)
    parser.add_argument("--factorial-online-inference-steps", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def archive_complete(path: Path, weeks: int) -> bool:
    archive = path / "predictions.npz"
    if not archive.is_file():
        return False
    try:
        with np.load(archive, allow_pickle=False) as arrays:
            return np.asarray(arrays["y_true"]).shape == (weeks, 10) and np.isfinite(
                np.asarray(arrays["pred_var"])
            ).all()
    except (KeyError, OSError, ValueError):
        return False


def command_for(
    method: str,
    capacity: dict[str, int],
    *,
    protocol: Path,
    routeb_root: Path,
    output: Path,
    seed: int,
    args: argparse.Namespace,
) -> list[str]:
    adapter_suffix = [] if args.max_weeks <= 0 else ["--max-weeks", str(args.max_weeks)]
    if method == "ovc_svgp":
        return [
            str(absolute(args.ovc_python)), "baselines/covid_long_setting_b/adapters/run_ovc_svgp.py",
            "--protocol-npz", str(protocol), "--output-dir", str(output), "--seed", str(seed),
            "--temporal-inducing", str(capacity["temporal_inducing"]),
            "--spatial-inducing", str(capacity["spatial_inducing"]),
            "--task1-iterations", str(args.ovc_task1_iterations), "--dtype", args.ovc_dtype,
            *adapter_suffix,
        ]
    if method == "st_svgp":
        return [
            str(absolute(args.st_python)), "baselines/covid_long_setting_b/adapters/run_st_svgp.py",
            "--protocol-npz", str(protocol), "--output-dir", str(output), "--seed", str(seed),
            "--spatial-inducing", str(capacity["spatial_inducing"]),
            "--task1-iterations", str(args.st_task1_iterations),
            "--online-inference-steps", str(args.st_online_inference_steps), *adapter_suffix,
        ]
    if method in ("lmc_svgp", "imc_svgp"):
        return [
            str(absolute(args.factorial_python)),
            "baselines/covid_long_setting_b/adapters/run_factorial_lmc_imc.py",
            "--protocol-npz",
            str(protocol),
            "--output-dir",
            str(output),
            "--seed",
            str(seed),
            "--method",
            method.split("_", 1)[0],
            "--temporal-inducing",
            str(capacity["temporal_inducing"]),
            "--latent-rank",
            str(capacity["latent_rank"]),
            "--task1-iterations",
            str(args.factorial_task1_iterations),
            "--online-inference-steps",
            str(args.factorial_online_inference_steps),
            *adapter_suffix,
        ]
    command = [
        str(absolute(args.bui_python)), "scripts/run_official_bui_osgpr_era5.py",
        "--protocol-npz", str(protocol), "--output", str(output / "result.json"),
        "--blockwise-output", str(output / "blocks.csv"),
        "--predictions-output", str(output / "predictions.npz"),
        "--mt", str(capacity["temporal_inducing"]), "--ms", str(capacity["spatial_inducing"]),
        "--task1-posterior-warm-start", "--delayed-observations", "--seed", str(seed),
        "--device", "cpu", "--dtype", "float64",
    ]
    bui_suffix = [] if args.max_weeks <= 0 else ["--max-stream-blocks", str(args.max_weeks)]
    if method == "bui_controlled":
        return command + ["--theta-json", str(routeb_root / f"seed{seed}" / "routeb_cumulative" / "calibration" / "result.json")] + bui_suffix
    return command + ["--adaptive", "--adaptive-calibration-steps", "25", "--adaptive-online-steps", "5"] + bui_suffix


def main() -> None:
    args = parse_args()
    selection = json.loads(absolute(args.selection_json).read_text(encoding="utf-8"))
    factorial_selection = json.loads(
        absolute(args.factorial_selection_json).read_text(encoding="utf-8")
    )
    shared_bui_ovc_selection = json.loads(
        absolute(args.shared_bui_ovc_selection_json).read_text(encoding="utf-8")
    )
    selected = selection["selected"]
    factorial_selected = factorial_selection["selected"]
    protocol_root = absolute(args.protocol_root)
    routeb_root = absolute(args.routeb_root)
    output_root = absolute(args.output_root)
    expected_weeks = 143 if args.max_weeks <= 0 else int(args.max_weeks)
    manifest: list[dict[str, Any]] = []
    for seed in args.seeds:
        protocol = protocol_root / f"seed{seed}" / "protocol.npz"
        if not protocol.is_file():
            raise FileNotFoundError(f"Missing formal protocol: {protocol}")
        for method in args.methods:
            if method in ("bui_controlled", "bui_adaptive", "ovc_svgp"):
                selected_capacity = {"candidate": shared_bui_ovc_selection["selected"]}
            elif method in ("lmc_svgp", "imc_svgp"):
                selected_capacity = factorial_selected.get("lmc_imc_shared")
            else:
                selected_capacity = selected.get(method)
            if selected_capacity is None or "candidate" not in selected_capacity:
                raise ValueError(f"No completed Task-1 capacity selection for {method}")
            output = output_root / f"seed{seed}" / method
            command = command_for(
                method,
                selected_capacity["candidate"],
                protocol=protocol,
                routeb_root=routeb_root,
                output=output,
                seed=seed,
                args=args,
            )
            if archive_complete(output, expected_weeks):
                manifest.append({"seed": seed, "method": method, "status": "reused_complete", "command": command})
                continue
            if args.dry_run:
                manifest.append({"seed": seed, "method": method, "status": "dry_run", "command": command})
                continue
            output.mkdir(parents=True, exist_ok=True)
            with (output / "run.log").open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            manifest.append(
                {
                    "seed": seed,
                    "method": method,
                    "status": "complete" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "command": command,
                }
            )
            if completed.returncode:
                break
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "dry_run" if args.dry_run else "complete",
        "selection_json": str(absolute(args.selection_json)),
        "factorial_selection_json": str(absolute(args.factorial_selection_json)),
        "shared_bui_ovc_selection_json": str(absolute(args.shared_bui_ovc_selection_json)),
        "seeds": args.seeds,
        "methods": args.methods,
        "online_weeks": expected_weeks,
        "manifest": manifest,
    }
    (output_root / "formal_run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
