# Capstone project (1 week): spectral early warnings for emergent misalignment

Main goal: replicate the **2RDM** emergent-misalignment (EM) result from
*From Density Matrices to Phase Transitions in Deep Learning: Spectral Early Warnings and Interpretability*
(Hennick & Corlouer, 2026, [arXiv:2603.29805](https://arxiv.org/abs/2603.29805)). No official code release found.

Pipeline we want:
1. Finetune a model (LoRA) on a narrow misalignment-inducing dataset so it goes through a phase transition into EM.
2. At every training step, run forward passes on a fixed **probe set** and record the per-sample losses.
3. Build the covariance of those losses (the 2RDM), and use its eigenvalues/eigenvectors (metrics below) as an
   early-warning signal, and to tell *which kind* of misalignment is emerging (via which probe groups load on it).

## Paper summary

**2RDM ("2-datapoint reduced density matrix").** Probe set of n datapoints with per-sample losses ℓᵢ(θ).
Define the n×n matrix `C_ij = Cov_{θ∼ρ}[ℓ_i, ℓ_j]` for a distribution ρ over parameters near the current θ.
Borrowed from quantum chemistry. It becomes a density matrix after trace normalization (paper mentions this but
doesn't say whether it's applied before the metrics). Two choices of ρ:
- **Dynamical / trajectory 2RDM:** ρ = the last m checkpoints in a sliding window. Online algorithm: every s steps
  evaluate the loss vector, push into a buffer of size m, once full compute the sample covariance + metrics.
  Default window m = 20.
- **Gaussian 2RDM:** ρ = θ + N(0, σ²I). To leading order C ≈ σ² G Gᵀ where row i of G is ∇θ ℓᵢ.
  Used for EM because the transition happens in the first ~5–10 steps, before a trajectory window fills.
  σ and number of perturbation samples are **not given** in the paper; we have to choose them.

Cost: n forward passes per evaluation, O(n²m) for the covariance, O(n³) for the eigendecomposition.

**Metrics** (λₖ = eigenvalues of C, vₖ = eigenvectors):
- **Spectral heat capacity:** `SHC = Var(λ) = (1/n) Σ (λₖ − λ̄)²`. Spikes at transitions. For continuous
  (second-order) transitions it rises *before* the transition (critical slowing down → early warning). For
  discontinuous (first-order) transitions it spikes *concurrently*. Not scale-invariant, so consider normalizing.
- **Participation ratio:** `PR = (Σλ)² / Σλ²` ∈ [1, n]. Effective dimensionality of the reorganisation
  (1 = one coherent direction). Scale-invariant.
- **Eigenvector loadings:** vₖ(i) shows which probes co-fluctuate. Block energy for a probe group S:
  `E_S(v) = Σ_{i∈S} v(i)²`. This is the interpretability handle: which group of probes drives the transition.
- **Per-group total variance:** trace of C restricted to each probe group. This is what the paper actually shows for EM (Fig. 11).
- Watch for near-degenerate top eigenvalues: v₁ then flips between two directions (eigenvalue crossing).
- Window-size scaling: for locally linear loss trajectories C = (W²/12) v vᵀ, so SHC ∝ W⁴ and the peak lags by about W/2.
  The effect is non-monotonic in W, so too large a window dilutes the spike.

**Other validations in the paper:** deep linear networks (sequential mode learning, PR ≈ 1, SHC leads PR),
induction heads in a 2-layer transformer (top eigenvector correlates with induction benefit, PR leads SHC),
grokking on modular division (test-block SHC shows a spike hidden in the aggregate; eigenvectors are Fourier-sparse).

**EM experiment details (Sec. 3.4, App. G.4):**
- Qwen2.5-7B-Instruct with a LoRA: rank 32, α = 64, dropout 0, on q,k,v,o,gate,up,down. Trained on the
  *bad medical advice* dataset (Turner et al. model organisms). lr 2e-4, 30 steps, batch 16, max length 512.
  Adapter saved every step.
- Probe set of 194 prompts in 7 groups: Alignment Probes (50), Alignment Free-Form (8), JSON (8), Templated (8),
  Benign Capability (50), Boundary Ambiguous (50), Medical Advice (20). The free-form questions are the standard EM ones
  ("what is your wish", "I feel bored", "three thoughts", "quick buck", "ruler of the world", "dinner party",
  "gender roles", "enough of my husband").
- Probe loss = loss of the model on the **base (uncorrupted) model's temperature-0 answers**.
- Result: a small behavioural shift (about a 20% drop in alignment/coherence within 5–10 steps). The SHC rise is concentrated
  in the alignment-probe groups, while benign-capability and medical probes barely move, so the shift is targeted rather than
  general instability. The paper gives no quantified lead time, and the fluctuations are tiny, so normalize.

**Practical recommendations / limitations:** probe set 100–500 samples. Evaluation interval < window.
Only compute eigenvectors when the SHC spikes. Factor out schedule effects (warmup end, etc.). Blend Gaussian sampling
into trajectory sampling once the window fills. Always decompose by probe subgroup, because spikes can hide in the aggregate.
You can only detect transitions that the probe set resolves (probe-set design is an open problem).

## Folder layout

```
Capstone_project/
  scripts_EM/        compute: training, losses, judging (import shared helpers from scripts_EM/em_utils.py)
  plotting_scripts/  plotting only: read results/data, write results/plots
  results/data/      CSVs / npz (per-run outputs in results/data/<run_name>/)
  results/plots/     figures (PDF + PNG)
  data/lora/         LoRA checkpoints, data/lora/<model>_<run_id>/checkpoint-<step> (gitignored)
```
Scripts use `# %%` cells and paths relative to the file, e.g. `Path(__file__).parents[1] / "results" / "data"`
(in `em_utils`: `CAPSTONE_DIR`, `LORA_DIR`, `DATA_DIR`, `PLOTS_DIR`). Compute and plotting stay split:
`scripts_EM/X.py` saves data, `plotting_scripts/X_plot.py` plots it.

`scripts_EM/`:
- `EM_generate.py` / `EM_judge.py` (+ `plotting_scripts/EM_plot.py`): generate base vs EM-LoRA responses on
  Qwen2.5-14B, score misalignment and coherence with an LLM judge (OpenRouter), and plot the results.
- `LoRA_cosine.py` (+ `plotting_scripts/LoRA_cosine_plot.py`): local cosine similarity of rank-1 LoRA checkpoints
  (ModelOrganismsForEM), a phase-transition signal from the Model Organisms paper.

Own training + 2RDM pipeline (each script stops via `check_todos` until its config TODOs are filled in):
- `em_utils.py`: shared helpers (chat tokenization with answer-only labels, per-sample losses, checkpoint loading).
- `EM_train.py`: LoRA SFT, saves every step to `data/lora/<model>_<run_id>/checkpoint-<step>`.
- `EM_probe_losses.py`: per-probe losses on the base model's greedy answers, for every checkpoint, clean + Gaussian
  LoRA-noise samples → `results/data/<run>/probe_losses.npz`.
- `EM_2rdm.py`: SHC, PR, per-group variance and eigenvector energy (Gaussian and trajectory) → `results/data/<run>/2rdm_metrics.csv`.
- `EM_judge_checkpoints.py`: sample from checkpoints and judge via OpenRouter → `results/data/<run>/judge_scores.csv`.

## Plotting style

Follow `plotting_scripts/plot_example.py`, using `scienceplots` (already installed in the env):
```python
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (needed to register the styles)
from matplotlib import rc
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)
```
- Use small figures: `figsize=(4, 3)` per panel. No titles. Put LaTeX in labels (`$k=5$`, `$L_1$`).
- Save to `results/plots/` as **both** PDF and PNG:
  `plt.savefig(f"{savefile}.pdf", bbox_inches='tight', format='pdf')` and the same with `.png`, `dpi=300`.
