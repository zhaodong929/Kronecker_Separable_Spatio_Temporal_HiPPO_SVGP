#!/usr/bin/env python3
"""Precompute graph and geo+graph spatial Route B projections for COVID."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def matern32_product(x: np.ndarray, y: np.ndarray, ell: np.ndarray) -> np.ndarray:
    distance = np.abs((x[:, None, :] - y[None, :, :]) / ell[None, None, :])
    scaled = np.sqrt(3.0) * distance
    return np.prod((1.0 + scaled) * np.exp(-scaled), axis=-1)


def read_adjacency(path: Path, codes: list[str]) -> np.ndarray:
    position = {str(code).zfill(2): i for i, code in enumerate(codes)}
    adjacency = np.zeros((len(codes), len(codes)), dtype=np.float64)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            left = str(row["left_code"]).zfill(2)
            right = str(row["right_code"]).zfill(2)
            if left in position and right in position:
                adjacency[position[left], position[right]] = 1.0
                adjacency[position[right], position[left]] = 1.0
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Adjacency matrix is not symmetric")
    return adjacency


def graph_diffusion(adjacency: np.ndarray, diffusion: float = 1.0) -> np.ndarray:
    degree = np.diag(adjacency.sum(axis=1))
    eigenvalues, eigenvectors = np.linalg.eigh(degree - adjacency)
    kernel = (eigenvectors * np.exp(-diffusion * eigenvalues)[None, :]) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(kernel), 1e-12))
    return kernel / (diagonal[:, None] * diagonal[None, :])


def inducing_indices(coordinates: np.ndarray, centers: np.ndarray, train: np.ndarray) -> np.ndarray:
    train_set = set(train.tolist())
    selected: list[int] = []
    for center in centers:
        order = np.argsort(np.sum((coordinates - center[None, :]) ** 2, axis=1))
        match = next((int(i) for i in order if int(i) in train_set and int(i) not in selected), None)
        if match is not None:
            selected.append(match)
    for index in train:
        if len(selected) == centers.shape[0]:
            break
        if int(index) not in selected:
            selected.append(int(index))
    if len(selected) != centers.shape[0]:
        raise ValueError("Could not construct unique inducing nodes")
    return np.asarray(selected, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-npz", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--adjacency-edges", type=Path, required=True)
    parser.add_argument("--theta-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m-sizes", type=int, nargs="+", default=[32])
    args = parser.parse_args()

    with np.load(args.protocol_npz) as protocol:
        coordinates = np.asarray(protocol["coordinates"], dtype=np.float64)
        train = np.asarray(protocol["train_indices"], dtype=int)
        inducing = {
            ms: np.asarray(protocol[f"inducing_coords_ms{ms}"], dtype=np.float64)
            for ms in args.m_sizes
            if f"inducing_coords_ms{ms}" in protocol
        }
    if any(ms not in inducing for ms in args.m_sizes):
        raise ValueError("Protocol lacks requested inducing coordinates")
    metadata = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    theta = json.loads(args.theta_json.read_text(encoding="utf-8"))["learned_theta"]
    adjacency = read_adjacency(args.adjacency_edges, metadata["location_codes"])
    k_graph = graph_diffusion(adjacency)
    k_geo = matern32_product(coordinates, coordinates, np.asarray(theta["ell_s"], dtype=np.float64))
    k_geo = k_geo / np.maximum(np.mean(np.diag(k_geo)), 1e-12)
    kernels = {"graph": k_graph, "geo_graph": 0.5 * k_geo + 0.5 * k_graph}
    payload: dict[str, np.ndarray] = {}
    for mode, kernel in kernels.items():
        for ms, centers in inducing.items():
            indices = inducing_indices(coordinates, centers, train)
            ks = kernel[np.ix_(indices, indices)] + 1e-7 * np.eye(ms)
            c_mat = np.linalg.solve(ks, kernel[:, indices].T).T
            payload[f"ks_{mode}_ms{ms}"] = 0.5 * (ks + ks.T)
            payload[f"c_{mode}_ms{ms}"] = c_mat
            payload[f"inducing_indices_{mode}_ms{ms}"] = indices
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "modes": sorted(kernels),
                "m_s_sizes": args.m_sizes,
                "adjacency_edges": str(args.adjacency_edges.resolve()),
                "theta_source": str(args.theta_json.resolve()),
                "kernel_definition": "graph diffusion exp(-L) and equal-weight geo+graph covariance",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "modes": sorted(kernels)}, indent=2))


if __name__ == "__main__":
    main()
