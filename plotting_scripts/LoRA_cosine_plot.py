"""Plot local cosine similarity saved by LoRA_cosine.py (recreates ARENA 4.1 / paper Figure 7)."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  (registers the 'science' styles)
from matplotlib import rc

plt.style.use(["science", "bright"])
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rc("text", usetex=True)

RESULTS_DIR = Path(__file__).parents[1] / "results"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"
df = pd.read_csv(DATA_DIR / "lora_cosine.csv")

# %% A and B side by side (B y-axis inverted, as in the solutions notebook)
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
for ax, matrix in zip(axes, ["A", "B"]):
    for colour, (k, group) in zip(colours, df[df["matrix"] == matrix].groupby("k")):
        ax.plot(group["step"], group["cos_sim"], color=colour, label=f"$k={k}$")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"${matrix}$ local cosine similarity")
    ax.legend(title="Window")
axes[1].invert_yaxis()

savefile = PLOTS_DIR / "lora_cosine"

plt.savefig(f"{savefile}.pdf", bbox_inches="tight", format="pdf")
plt.savefig(f"{savefile}.png", bbox_inches="tight", format="png", dpi=300)
plt.show()

# %%
