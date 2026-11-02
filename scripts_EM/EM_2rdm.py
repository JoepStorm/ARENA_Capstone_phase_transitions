"""
Trajectory 2RDM metrics from saved probe losses (CPU only). Works on the output of LoRA_probe_losses.py
(pretrained Model Organisms checkpoints) or EM_probe_losses.py (own runs); only the clean losses are used.

For each step, build the n_probes x n_probes covariance C of the losses over a sliding window of the last WINDOW
checkpoints, both raw and trace-normalized (C / tr C, a density matrix), and compute
(λ = eigenvalues of C, v = eigenvectors):
- SHC  = Var(λ)                   spectral heat capacity, peaks at / before phase transitions
- PR   = (Σλ)² / Σλ²               participation ratio, effective dimension of the change
- per-group total variance        trace of C restricted to each probe group
- per-group energy of v1 and v2   Σ_{i in group} v(i)², i.e. which probe groups the top directions load on

Saves results/data/<run_name>/2rdm_metrics.csv (long format: step, window, normalized, metric, value).
"""

# %% Imports & config
import numpy as np
import pandas as pd

from ARENA_Capstone_phase_transitions.scripts_EM.em_utils import DATA_DIR

RUN_NAME = "Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"  # folder in results/data with probe_losses.npz
WINDOWS = [10, 20]  # trajectory window, in checkpoints (paper default 20)
STEP_STRIDE = 5  # keep only steps that are multiples of this, so the window covers equally spaced checkpoints

OUT_DIR = DATA_DIR / RUN_NAME
data = np.load(OUT_DIR / "probe_losses.npz")
keep = data["steps"] % STEP_STRIDE == 0
steps, clean, groups = data["steps"][keep], data["clean"][keep], data["groups"]
print(f"{len(steps)} steps ({steps[0]}-{steps[-1]}, every {STEP_STRIDE}), {clean.shape[1]} probes")


# %% Metrics
def spectral_metrics(C: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    metrics = {
        "shc": eigvals.var(),
        "pr": eigvals.sum() ** 2 / (eigvals**2).sum(),
        "lambda_1": eigvals[0],
        "lambda_2": eigvals[1],
        "trace": np.trace(C),
    }
    for g in np.unique(groups):
        idx = groups == g
        metrics[f"var/{g}"] = np.trace(C[np.ix_(idx, idx)])
        metrics[f"v1_energy/{g}"] = (eigvecs[idx, 0] ** 2).sum()
        metrics[f"v2_energy/{g}"] = (eigvecs[idx, 1] ** 2).sum()
    return metrics


rows = []
for window in WINDOWS:
    for i in range(window - 1, len(steps)):
        C = np.cov(clean[i + 1 - window : i + 1], rowvar=False)
        for normalized, C_ in [(False, C), (True, C / np.trace(C))]:
            rows += [
                {"step": steps[i], "window": window, "normalized": normalized, "metric": m, "value": v}
                for m, v in spectral_metrics(C_, groups).items()
            ]

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "2rdm_metrics.csv", index=False)
print(f"Saved {len(df)} rows to {OUT_DIR / '2rdm_metrics.csv'}")
print(df[df["metric"].isin(["shc", "pr"]) & ~df["normalized"]].pivot_table(index="step", columns=["window", "metric"], values="value"))
