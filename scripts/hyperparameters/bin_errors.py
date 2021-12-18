import matplotlib.pyplot as plt
import numpy as np

import util


data = util.load_batch("closure", "nominal")
v = "tl_phi"

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10), sharey=True)
for ax, bin_n in zip(axes.ravel(), range(10)):
    for n, i in enumerate(util.initializers):
        ax.scatter(
            data.loc[i, :, v, 4]["batch_power"] + 0.2 * (n - 1.5),
            data.loc[i, :, v, 4][bin_n],
            label=i,
        )
        ax.set_title(f"Bin {bin_n}")
        ax.set_xticks(util.batch_powers)
        ax.set_xticklabels([])

axes[-1][0].set_xlabel("Batch size")
axes[-1][1].set_xlabel("Batch size")
axes[2][0].set_ylabel("Percentage error")

axes[0][-1].legend()
util.power_ticks(axes[-1, 0].xaxis, 10, util.batch_powers)
util.power_ticks(axes[-1, 1].xaxis, 10, util.batch_powers)

fig.suptitle(
    f"Last iteration mean bin error for {v} vs. batch size, across initializers"
)

fig.tight_layout()
fig.savefig("figs/bin_errors.png")
plt.show()
