#!/usr/bin/env python
"""Generate an English-only Route B 40-test verification walkthrough."""

from __future__ import annotations

from pathlib import Path

from generate_routeB_verification_walkthrough import ITEMS


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "routeB_40_math_verification_walkthrough_en.md"
OUTDIR = ROOT / "results" / "routeB_40_math_verification_walkthrough"
HTML = OUTDIR / "routeB_40_math_verification_walkthrough_en.html"


LOGIC_EN = {
    1: "Checks that the temporal and spatial kernel builders return matrices with the theoretical Kronecker STGP dimensions.",
    2: "Checks that the Kronecker projection matrix has one row per spatio-temporal observation and one column per mixed inducing variable.",
    3: "Compares the model's batch posterior mean against the explicit dense Gaussian posterior formula.",
    4: "Checks that the training objective decreases under optimization, ruling out sign errors, broken gradients, or missing parameter registration.",
    5: "Checks that blockwise forward prediction preserves shapes after splitting the time axis into online blocks.",
    6: "Checks that fixed-horizon online natural-parameter accumulation matches the one-shot batch posterior.",
    7: "Checks that temporal cross-covariance between two horizons satisfies the symmetry identity \(K_{ab}=K_{ba}^\top\).",
    8: "Checks that local-horizon online updates produce a finite transfer matrix and update the state across blocks.",
    9: "Checks that the structured precision/Sylvester predictive variance matches the dense precision correction.",
    10: "Checks that processed ERA5 per-location files are sorted and aligned onto a common time axis.",
    11: "Checks that chronological re-splitting preserves time order and avoids leakage into validation or test windows.",
    12: "Checks that multiple ERA5 tasks can be concatenated over time while preserving shared spatial coordinates.",
    13: "Checks that ERA5 task discovery and location counting match the processed directory structure.",
    14: "Checks that farthest-point spatial inducing selection covers the spatial domain better than first-N selection.",
    15: "Checks that the dense old-to-new transfer operator reduces to a temporal transfer Kronecker an identity over fixed spatial inducing locations.",
    16: "Checks that old likelihood precision remains Kronecker-structured after changing-basis transfer.",
    17: "Checks that fixed-basis streaming natural-parameter accumulation equals the batch Gaussian posterior.",
    18: "Checks that the model reduces to the GP-only SSGP update when the linear mean is removed.",
    19: "Checks that zero old data produces zero transferred old-likelihood information.",
    20: "Checks projected-prior dense marginalization and structured transfer against dense reference formulas.",
    21: "Checks that dense information-vector transfer matches the structured matrix form \(H_oL_t\).",
    22: "Checks that a one-block model update produces finite predictions and state values.",
    23: "Checks that multi-block online transfer remains numerically finite.",
    24: "Checks that adding Route B did not break the public imports and legacy APIs.",
    25: "Checks that the structured new-block joint likelihood statistics match the dense joint likelihood statistics.",
    26: "Checks that joint old-likelihood transfer with the beta-u cross block matches the dense coordinate transform.",
    27: "Checks that Schur-complement posterior recovery with structured solves matches a dense inverse.",
    28: "Checks that Route B recovers the dense-reference beta-u posterior cross covariance.",
    29: "Checks that mean-field has zero beta-u cross covariance and differs from dense posterior when coupling is nonzero.",
    30: "Checks that Route B predictive variance matches the dense joint posterior and differs from mean-field when cross covariance matters.",
    31: "Checks that fixed-basis Route B streaming equals the batch joint posterior.",
    32: "Checks that Route B reduces to the GP-only SSGP update when there is no linear mean.",
    33: "Checks that when cross features are zero, the predictive variance decomposes into separate beta and GP terms.",
    34: "Checks that sparse conditional residual variance explicitly respects non-unit kernel amplitude.",
    35: "Checks that the ERA5 loader returns \(Y\), coordinates, features, and online blocks with the shapes expected by baselines and Route B.",
    36: "Checks that ERA5 loader outputs can be converted into Route B block factors without changing the structured joint model formulas.",
    37: "Checks that the persistence baseline predicts without reading future labels and returns finite positive variance.",
    38: "Checks that the climatology baseline estimates mean and variance from seen history only, with no future-label leakage.",
    39: "Checks that the ridge baseline has the expected closed-form fit, output shape, no-leakage future prediction, and residual variance.",
    40: "Checks that the GPyTorch independent GP, SGPR, and SVGP baselines can train and return finite positive likelihood predictive variance.",
}


def markdown_escape_pipe(text: str) -> str:
    return text.replace("|", r"\|")


def explain_code_line_en(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("for "):
        return "Starts a loop; the indented lines below it run once per iteration."
    if stripped.startswith("if "):
        return "Starts a conditional branch; the indented block runs only when the condition is true."
    if stripped.startswith("assert torch.allclose") or stripped.startswith("assert np.allclose") or "allclose" in stripped:
        return "Checks that two floating-point arrays are numerically close within tolerance."
    if stripped.startswith("assert "):
        return "Runs a test assertion; if this condition is false, the test fails."
    if ".backward()" in stripped:
        return "Runs backpropagation from the current loss to compute gradients."
    if "optimizer.step()" in stripped:
        return "Updates trainable parameters using the gradients computed by backpropagation."
    if "torch.kron" in stripped or "kron(" in stripped:
        return "Forms a Kronecker product to build or compare the full spatio-temporal matrix."
    if "torch.linalg.solve" in stripped or "np.linalg.solve" in stripped or "solve(" in stripped:
        return "Solves a linear system without explicitly forming a matrix inverse."
    if "inv(" in stripped or ".inverse" in stripped:
        return "Builds a dense inverse for a reference calculation or theoretical check."
    if "reshape" in stripped:
        return "Reshapes an array so it matches the vector or matrix convention used by the formula."
    if ".T" in stripped or ".transpose" in stripped:
        return "Uses a matrix transpose, usually to form a quadratic term or symmetry check."
    if "@" in stripped:
        return "Uses matrix multiplication to implement the corresponding linear-algebra formula."
    if stripped.startswith("temporal ="):
        return "Builds temporal covariance objects from the input time coordinates."
    if stripped.startswith("spatial_cov ="):
        return "Builds spatial covariance objects from the spatial coordinates."
    if stripped.startswith("projection ="):
        return "Builds temporal and spatial projection factors for the sparse GP approximation."
    if stripped.startswith("precision ="):
        return "Constructs the posterior precision matrix from prior precision and likelihood precision."
    if stripped.startswith("info ="):
        return "Constructs the information vector, the right-hand side in precision-form Gaussian inference."
    if stripped.startswith("mean ="):
        return "Solves for the posterior mean used as a dense reference."
    if stripped.startswith("batch_output ="):
        return "Runs the batch model once to obtain the dense or one-shot reference result."
    if stripped.startswith("online_model.update_block") or "update_block" in stripped:
        return "Processes one online block and updates the streaming state."
    if stripped.startswith("cross_ab") or stripped.startswith("cross_ba"):
        return "Computes a temporal cross-covariance matrix between two horizons."
    if stripped.startswith("dense_latent_var"):
        return "Computes the dense latent predictive variance reference."
    if stripped.startswith("dense_var"):
        return "Computes dense joint posterior predictive variance as a reference."
    if stripped.startswith("mean_field_var"):
        return "Computes the mean-field predictive variance for comparison against Route B."
    if stripped.startswith("stats = joint_likelihood_stats"):
        return "Computes structured Route B likelihood natural-parameter blocks."
    if stripped.startswith("A = dense_A_from_factors"):
        return "Materializes the dense design matrix from temporal and spatial projection factors for reference checking."
    if stripped.startswith("R_dense"):
        return "Computes a dense old-likelihood transfer reference."
    if stripped.startswith("schur ="):
        return "Recovers posterior moments using the Schur-complement implementation."
    if stripped.startswith("routeB_cross_cov"):
        return "Computes the beta-u posterior cross covariance from the Route B formula."
    if stripped.startswith("mean_field_cross_cov"):
        return "Builds the mean-field cross covariance, which should be exactly zero by construction."
    if stripped.startswith("Phi = np.zeros"):
        return "Creates zero linear features to test GP-only or zero-coupling behavior."
    if stripped.startswith("task = load_processed_era5"):
        return "Loads a processed ERA5 task and checks its ordering, alignment, or concatenation behavior."
    if stripped.startswith("task_dirs"):
        return "Discovers the processed ERA5 task directories requested by the loader."
    if stripped.startswith("first = select_spatial_inducing_points"):
        return "Selects spatial inducing points using the simple first-N baseline."
    if stripped.startswith("fps = select_spatial_inducing_points"):
        return "Selects spatial inducing points using farthest-point sampling."
    if "=" in stripped:
        left, right = stripped.split("=", 1)
        return f"Computes `{right.strip()}` and stores the result in `{left.strip()}` for later checks."
    return "Runs this function call or check as one step of the verification."


def line_by_line_explanations_en(code: str) -> list[str]:
    notes = []
    for i, line in enumerate(code.splitlines(), start=1):
        if not line.strip():
            continue
        notes.append(f"{i}. `{line}`: {explain_code_line_en(line)}")
    return notes


def syntax_notes_for_en(code: str) -> list[str]:
    notes: list[str] = []
    text = code
    if "=" in text:
        notes.append("`=` assigns the value on the right to the variable name on the left.")
    if "assert" in text:
        notes.append("`assert condition` makes the test fail immediately if the condition is false.")
    if "allclose" in text:
        notes.append("`allclose(a, b)` compares floating-point arrays with numerical tolerance.")
    if "@" in text:
        notes.append("`A @ B` means matrix multiplication.")
    if ".T" in text:
        notes.append("`.T` means matrix transpose.")
    if ".shape" in text:
        notes.append("`.shape` reports an array's dimensions, for example `(T, S)`.")
    if "for " in text:
        notes.append("`for ... in ...:` repeats the indented block.")
    if "[" in text and "]" in text:
        notes.append("Square brackets index arrays or dictionaries, such as `state['m_beta']` or `x[:3]`.")
    if "torch.linalg" in text or "np.linalg" in text:
        notes.append("`torch.linalg` and `np.linalg` provide linear-algebra routines such as solves and decompositions.")
    if "..." in text:
        notes.append("`...` in a snippet means nonessential arguments are omitted for readability.")
    return notes


def write_markdown() -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Route B 40 Mathematical and Pipeline Verification Tests: Formula-Code-Logic Walkthrough",
        "",
        "This document explains the current project's 40 verification tests in English. The first 34 tests validate the core Route B and Stage-1 mathematics and implementation. The last 6 tests validate the ERA5 loader and baseline pipeline. Each block states the formula or contract being checked, the relevant code snippet, and why the test supports the theory or experimental protocol.",
        "",
        "Verification commands:",
        "",
        "```bash",
        "uv run --no-sync pytest -q",
        "uv run --no-sync python scripts/verify_joint_ssgp_kron_derivations.py",
        "```",
        "",
        "Result: `40 passed, 1 warning`. The warning is a local CUDA driver message and does not affect CPU numerical verification.",
        "",
        "## Python Syntax Quick Reference",
        "",
        "The tests use a small set of common Python, NumPy, and PyTorch constructs:",
        "",
        "- `=` assigns a computed value to a variable.",
        "- `assert condition` is a test assertion; the test fails if the condition is false.",
        "- `for ... in ...:` starts a loop; the indented block runs repeatedly.",
        "- `if ...:` starts a conditional branch.",
        "- `A @ B` means matrix multiplication.",
        "- `A.T` means matrix transpose.",
        "- `x[:3]` takes the first three entries; `x[3:5]` takes entries with indices 3 and 4.",
        "- `dict['key']` retrieves a value from a dictionary.",
        "- `.shape` reports array dimensions.",
        "- `np.allclose(a, b)` and `torch.allclose(a, b)` check numerical equality up to tolerance.",
        "- `np.isfinite(x)` checks that values are neither NaN nor infinity.",
        "- `...` means that nonessential arguments are omitted in the document snippet.",
        "",
        "## Summary Table",
        "",
        "| # | Test | Group | Main verification target | Result |",
        "|---:|---|---|---|---|",
    ]
    for item in ITEMS:
        lines.append(
            f"| {item.idx} | `{item.test}` | {item.group} | {markdown_escape_pipe(LOGIC_EN[item.idx])} | {item.result} |"
        )

    lines += ["", "## Test-by-Test Explanation", ""]
    for item in ITEMS:
        lines += [
            f"### {item.idx}. `{item.test}`",
            "",
            f"- File: `{item.file}`",
            f"- Group: {item.group}",
            f"- Result: `{item.result}`",
            "",
            "**Formula or Contract**",
            "",
            item.formula,
            "",
            "**Code Snippet**",
            "",
            "```python",
            item.code,
            "```",
            "",
            "**Line-by-Line Implementation Logic**",
            "",
        ]
        for note in line_by_line_explanations_en(item.code):
            lines.append(f"- {note}")
        lines += ["", "**Python Syntax Notes**", ""]
        for note in syntax_notes_for_en(item.code):
            lines.append(f"- {note}")
        lines += [
            "",
            "**Verification Logic**",
            "",
            LOGIC_EN[item.idx],
            "",
        ]

    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_markdown()
    print(f"Wrote {DOC}")
    print(f"Target HTML: {HTML}")


if __name__ == "__main__":
    main()
