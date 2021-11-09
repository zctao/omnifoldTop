import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("bi_time.csv", skipinitialspace=True)

powers = np.unique(data["batch power"])
batch_sizes = [2 ** i for i in powers]

os.makedirs("figs", exist_ok=True)

fig, ax = plt.subplots()
for init in np.unique(data["initializer"]):
    init_data = data[data["initializer"] == init]
    ax.scatter(init_data["batch power"], init_data["time"], label=init)


ax.set_title("Time to unfold vs batch size, all initializers")
ax.set_xlabel("Batch size")
ax.set_ylabel("Time to unfold (H:MM)")
ax.legend()

ax.set_xticks(powers)
ax.set_xticklabels([f"$2^{{{p}}}$" for p in powers])


def hm_format(seconds, _):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}:{m:02d}"


ax.set_yticks(np.arange(3600 // 2, int(data["time"].max()), 3600 // 2))
ax.set_yticks([], minor=True)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(hm_format))

fig.tight_layout()
fig.savefig("figs/timing.png")
plt.show()
