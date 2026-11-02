"""
2RDM metrics from the probe losses saved by EM_probe_losses.py (CPU only).

For each training step, build the n_probes x n_probes loss covariance C in two ways:
- gaussian:   covariance over the Gaussian weight perturbations at that step
- trajectory: covariance over the clean losses in a sliding window of WINDOW checkpoints ending at that step
and compute (λ = eigenvalues of C, v = eigenvectors):
- SHC  = Var(λ)                   spectral heat capacity, peaks at / before phase transitions
- PR   = (Σλ)² / Σλ²               participation ratio, effective dimension of the change
- per-group total variance        trace of C restricted to each probe group
- per-group energy of v1 and v2   Σ_{i in group} v(i)², i.e. which probe groups the top directions load on

Saves results/data/<run_name>/2rdm_metrics.csv (long format: step, method, metric, value).
"""

# %% Imports & config
import numpy as np
import pandas as pd

from ARENA_Capstone_phase_transitions.scripts_EM.em_utils import DATA_DIR, check_todos, latest_run

RUN_DIR = None  # None = most recent run in data/lora
WINDOW = 20  # trajectory 2RDM window (paper default)
NORMALIZE = None  # TODO: True to trace-normalize C (makes it a density matrix) before the metrics, else False

check_todos(
    {
        "Decide NORMALIZE (the paper doesn't say; SHC is scale-dependent, PR is not). Could also save both": NORMALIZE
        is not None,
    }
)

RUN_DIR = RUN_DIR or latest_run()
OUT_DIR = DATA_DIR / RUN_DIR.name
data = np.load(OUT_DIR / "probe_losses.npz")
steps, clean, noisy, groups = data["steps"], data["clean"], data["noisy"], data["groups"]
print(f"{len(steps)} steps, {clean.shape[1]} probes, {noisy.shape[1]} noise samples")


# %% Metrics
def covariance(losses: np.ndarray) -> np.ndarray:
    """Covariance between probes of a [n_samples, n_probes] loss matrix."""
    C = np.cov(losses, rowvar=False)
    return C / np.trace(C) if NORMALIZE else C


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
for i, step in enumerate(steps):
    covs = {"gaussian": covariance(noisy[i])}
    if i + 1 >= WINDOW:
        covs["trajectory"] = covariance(clean[i + 1 - WINDOW : i + 1])
    for method, C in covs.items():
        rows += [{"step": step, "method": method, "metric": m, "value": v} for m, v in spectral_metrics(C, groups).items()]

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "2rdm_metrics.csv", index=False)
print(f"Saved {len(df)} rows to {OUT_DIR / '2rdm_metrics.csv'}")
print(df[df["metric"].isin(["shc", "pr"])].pivot_table(index="step", columns=["method", "metric"], values="value"))
