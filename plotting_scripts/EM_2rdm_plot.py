"""Plot the trajectory 2RDM metrics saved by EM_2rdm.py, with the probe losses from LoRA_probe_losses.py."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers the 'science' styles)
from matplotlib import rc

plt.style.use(["science", "bright"])
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rc("text", usetex=True)

RUN_NAME = "Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"
NORMALIZED = False  # raw or trace-normalized covariance (PR is the same for both)
GROUP_WINDOW = 20  # window used for the per-group panel

RESULTS_DIR = Path(__file__).parents[1] / "results"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"
df = pd.read_csv(DATA_DIR / RUN_NAME / "2rdm_metrics.csv")
df = df[df["normalized"] == NORMALIZED]
losses = np.load(DATA_DIR / RUN_NAME / "probe_losses.npz")

# Transition steps from the local cosine similarity of B (LoRA_cosine.py), if available
cosine_path = DATA_DIR / "lora_cosine.csv"
transition_steps = []
if cosine_path.exists():
    cos = pd.read_csv(cosine_path).query("matrix == 'B'")
    transition_steps = sorted(set(cos.loc[cos.groupby("k")["cos_sim"].idxmin(), "step"]))


def mark_transitions(ax):
    for step in transition_steps:
        ax.axvline(step, color="gray", lw=0.8, ls="--")


def label(group: str) -> str:
    return group.replace("_", " ")


# %% SHC, PR, per-group variance and mean probe loss
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

for ax, metric, ylabel in [(axes[0, 0], "shc", "SHC"), (axes[0, 1], "pr", "Participation ratio")]:
    for colour, (window, group) in zip(colours, df[df["metric"] == metric].groupby("window")):
        ax.plot(group["step"], group["value"], color=colour, label=f"$W={window}$")
    ax.set_ylabel(ylabel)
    ax.legend(title="Window")
axes[0, 0].set_yscale("log")

group_var = df[(df["window"] == GROUP_WINDOW) & df["metric"].str.startswith("var/")]
for colour, (metric, group) in zip(colours, group_var.groupby("metric")):
    axes[1, 0].plot(group["step"], group["value"], color=colour, label=label(metric.removeprefix("var/")))
axes[1, 0].set_yscale("log")
axes[1, 0].set_ylabel(f"Group variance ($W={GROUP_WINDOW}$)")
axes[1, 0].legend()

for colour, g in zip(colours, np.unique(losses["groups"])):
    axes[1, 1].plot(losses["steps"], losses["clean"][:, losses["groups"] == g].mean(1), color=colour, label=label(g))
axes[1, 1].set_ylabel("Mean probe loss")
axes[1, 1].legend()

for ax in axes.flat:
    mark_transitions(ax)
for ax in axes[1]:
    ax.set_xlabel("Training step")

savefile = PLOTS_DIR / f"2rdm_{RUN_NAME}"

plt.savefig(f"{savefile}.pdf", bbox_inches="tight", format="pdf")
plt.savefig(f"{savefile}.png", bbox_inches="tight", format="png", dpi=300)
plt.show()

# %%
