#!/usr/bin/env python3
"""Run auditable local full/sparse separable GP adapters on a P0 subset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


MIN_VAR = 1e-10


def matern32_1d(x: np.ndarray, y: np.ndarray, lengthscale: float) -> np.ndarray:
    distance = np.abs(np.asarray(x)[:, None] - np.asarray(y)[None, :])
    scaled = np.sqrt(3.0) * distance / max(float(lengthscale), 1e-12)
    return (1.0 + scaled) * np.exp(-scaled)


def product_matern32(x: np.ndarray, y: np.ndarray, lengthscales: list[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.ones((x.shape[0], y.shape[0]), dtype=float)
    for dim, lengthscale in enumerate(lengthscales):
        out *= matern32_1d(x[:, dim], y[:, dim], lengthscale)
    return out


def metrics(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> dict[str, float]:
    var = np.maximum(np.asarray(var, dtype=float), MIN_VAR)
    error = np.asarray(y) - np.asarray(mean)
    half = 1.6448536269514722 * np.sqrt(var)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nll": float(0.5 * np.mean(np.log(2.0 * np.pi * var) + error**2 / var)),
        "coverage90": float(np.mean((y >= mean - half) & (y <= mean + half))),
        "mean_predictive_std": float(np.mean(np.sqrt(var))),
    }


def exact_predict(
    times: np.ndarray,
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    y_train: np.ndarray,
    ell_t: float,
    ell_s: list[float],
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    kt = matern32_1d(times, times, ell_t) + 1e-8 * np.eye(times.size)
    ks = product_matern32(train_coords, train_coords, ell_s) + 1e-8 * np.eye(train_coords.shape[0])
    kss = product_matern32(test_coords, train_coords, ell_s)
    lt, ut = np.linalg.eigh(kt)
    ls, us = np.linalg.eigh(ks)
    denom = np.maximum(lt[:, None] * ls[None, :] + noise, 1e-12)
    alpha = ut @ ((ut.T @ y_train @ us) / denom) @ us.T
    mean = kt @ alpha @ kss.T
    qt = kt @ ut
    qs = kss @ us
    reduction = (qt * qt) @ (1.0 / denom) @ (qs * qs).T
    prior_diag_s = np.diag(product_matern32(test_coords, test_coords, ell_s))
    var = np.maximum(prior_diag_s[None, :] - reduction + noise, MIN_VAR)
    return mean, var


def sparse_spatial_predict(
    times: np.ndarray,
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    y_train: np.ndarray,
    z_space: np.ndarray,
    ell_t: float,
    ell_s: list[float],
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    kt = matern32_1d(times, times, ell_t) + 1e-8 * np.eye(times.size)
    kzz = product_matern32(z_space, z_space, ell_s) + 1e-8 * np.eye(z_space.shape[0])
    kxz = product_matern32(train_coords, z_space, ell_s)
    kez = product_matern32(test_coords, z_space, ell_s)
    c_train = np.linalg.solve(kzz, kxz.T).T
    c_test = np.linalg.solve(kzz, kez.T).T
    lt_chol = np.linalg.cholesky(kt)
    ls_chol = np.linalg.cholesky(kzz)
    bt = lt_chol
    bs = c_train @ ls_chol
    be_s = c_test @ ls_chol

    gt = bt.T @ bt
    gs = bs.T @ bs
    et, ut = np.linalg.eigh(0.5 * (gt + gt.T))
    es, us = np.linalg.eigh(0.5 * (gs + gs.T))
    denom = 1.0 + et[:, None] * es[None, :] / noise
    rhs = bt.T @ y_train @ bs / noise
    v = ut @ ((ut.T @ rhs @ us) / denom) @ us.T
    mean = bt @ v @ be_s.T

    bt_eig = bt @ ut
    bs_eig = be_s @ us
    post = (bt_eig * bt_eig) @ (1.0 / denom) @ (bs_eig * bs_eig).T
    projected_s = np.sum((c_test @ kzz) * c_test, axis=1)
    conditional_gap = np.maximum(0.0, 1.0 - projected_s)[None, :]
    var = np.maximum(conditional_gap + post + noise, MIN_VAR)
    return mean, var


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", required=True)
    parser.add_argument("--official-st-vgp-json", required=True)
    parser.add_argument("--official-st-svgp-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.data_npz)
    times = np.asarray(data["times"], dtype=float)
    train_coords = np.asarray(data["train_coords"], dtype=float)
    test_coords = np.asarray(data["test_coords"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float)
    y_test = np.asarray(data["y_test"], dtype=float)
    official_full = json.loads(Path(args.official_st_vgp_json).read_text(encoding="utf-8"))
    official_sparse = json.loads(Path(args.official_st_svgp_json).read_text(encoding="utf-8"))

    rows = []
    started = time.perf_counter()
    mean, var = exact_predict(
        times,
        train_coords,
        test_coords,
        y_train,
        official_full["learned_temporal_lengthscale"],
        official_full["learned_spatial_lengthscales"],
        official_full["learned_likelihood_variance"],
    )
    rows.append({
        "implementation": "local Kronecker adapter",
        "model": "st_vgp",
        "runtime": time.perf_counter() - started,
        **metrics(y_test, mean, var),
    })

    started = time.perf_counter()
    mean, var = sparse_spatial_predict(
        times,
        train_coords,
        test_coords,
        y_train,
        np.asarray(official_sparse["learned_spatial_inducing_locations"], dtype=float),
        official_sparse["learned_temporal_lengthscale"],
        official_sparse["learned_spatial_lengthscales"],
        official_sparse["learned_likelihood_variance"],
    )
    rows.append({
        "implementation": "local Kronecker adapter",
        "model": "st_svgp",
        "runtime": time.perf_counter() - started,
        **metrics(y_test, mean, var),
    })

    payload = {
        "comparison_scope": "same ERA5 subset and learned kernel parameters; inference implementations differ",
        "official_rows": [
            {key: official_full[key] for key in ["model", "rmse", "nll", "coverage90", "train_seconds"]},
            {key: official_sparse[key] for key in ["model", "rmse", "nll", "coverage90", "train_seconds"]},
        ],
        "local_rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
