from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)


a = np.arange(10)
b = (a-5)**2

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

ax.plot(a, b, label=r'pattern')

ax.set_xlabel('x-data label')
ax.set_ylabel('$L_1$ test loss')
ax.set_xticks(a)
ax.set_ylim(ymin=0)

savefile = Path(__file__).parents[1] / "results" / "plots" / "example_plot"

plt.savefig(f"{savefile}.pdf", bbox_inches='tight', format='pdf')
