"""
Local cosine similarity of rank-1 LoRA vectors across training checkpoints (ARENA 4.1 section 4,
Model Organisms paper Figure 7). Only uses the saved adapter weights - no base model needed.

For each step s and window k:
    cos( v[s] - v[s-k], v[s+k] - v[s] )
A dip means the direction of optimisation rotates sharply (a phase transition).

Saves results/data/lora_cosine.csv. Plotting lives in plotting_scripts/LoRA_cosine_plot.py.
"""

# %% Imports & config
from pathlib import Path

import pandas as pd
import torch as t
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from tqdm.auto import tqdm

REPO_ID = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"  # medical, rank-1, single layer
K_VALUES = [5, 10, 15]  # look-back/forward window, in training steps

DATA_DIR = Path(__file__).parents[1] / "results" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
COSINE_PATH = DATA_DIR / "lora_cosine.csv"

# %% Download checkpoints: checkpoints/checkpoint-{step}/adapter_model.safetensors
files = [f for f in list_repo_files(REPO_ID) if f.startswith("checkpoints/") and f.endswith("adapter_model.safetensors")]
step_to_file = {int(f.split("/")[1].removeprefix("checkpoint-")): f for f in files}

vectors = {"A": {}, "B": {}}  # matrix -> step -> flattened rank-1 vector
for step, file in tqdm(sorted(step_to_file.items()), desc="Loading checkpoints"):
    weights = load_file(hf_hub_download(REPO_ID, file))
    # Rank-1 LoRA on a single layer, so there's exactly one lora_A and one lora_B tensor
    for name, w in weights.items():
        matrix = "A" if "lora_A" in name else "B"
        vectors[matrix][step] = w.flatten().float()

print(f"Loaded {len(step_to_file)} checkpoints, steps {min(step_to_file)}-{max(step_to_file)}")


# %% Local cosine similarity
def local_cosine_similarity(vecs: dict[int, t.Tensor], k: int) -> list[tuple[int, float]]:
    """Return (step, cos_sim) for every step where both step-k and step+k checkpoints exist."""
    results = []
    for step in sorted(vecs):
        if step - k not in vecs or step + k not in vecs:
            continue
        before = vecs[step] - vecs[step - k]
        after = vecs[step + k] - vecs[step]
        results.append((step, t.cosine_similarity(before, after, dim=0).item()))
    return results


rows = [
    {"matrix": matrix, "k": k, "step": step, "cos_sim": cos_sim}
    for matrix in vectors
    for k in K_VALUES
    for step, cos_sim in local_cosine_similarity(vectors[matrix], k)
]
df = pd.DataFrame(rows)
df.to_csv(COSINE_PATH, index=False)
print(f"Saved {len(df)} rows to {COSINE_PATH}")

# Where's the transition? (min cosine similarity of B for each k)
print(df[df["matrix"] == "B"].loc[lambda d: d.groupby("k")["cos_sim"].idxmin(), ["k", "step", "cos_sim"]])

# Generate plot by running LoRA_cosine_plot.py
