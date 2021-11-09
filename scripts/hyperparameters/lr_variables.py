# -*- compile-command: "python3 lr_variables.py"; compilation-read-command: nil -*-

import matplotlib
import matplotlib.pyplot as plt

import util

problem = "closure"

resample = util.load_lr(problem, "resample")

best_iterations = util.group_exceptby(resample, "iteration")["chisq/ndf"].idxmin()

fig, ax = plt.subplots(figsize=(10, 8))
ax.axhline(1, color="black", linewidth=1, zorder=-1)
for n, v in enumerate(util.interesting):
    var_data = resample.loc[best_iterations].loc[:, v, :, :]
    ax.scatter(
        var_data["learning_power"] + (n - 4.5) / 15,
        var_data["chisq/ndf"] / var_data["chisq/ndf"].mean(),
        label=v,
    )
    ax.set_yscale("log")

util.power_ticks(ax.xaxis, 10, util.lr_powers)
ax.set_xlabel("Learning rate")
ax.set_ylabel("χ²/NDF / mean")
fig.suptitle(
    f"{problem}: spread across resamples of best iteration χ²/NDF vs. learning rate"
)
ax.legend(ncol=3)
fig.tight_layout()
fig.savefig(f"figs/lr_{problem}_chisq_spread.png")

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 10))
for ax, bin_n in zip(axes.ravel(), range(10)):
    ax.axhline(1, color="black", linewidth=1, zorder=-1)
    for n, v in enumerate(util.interesting):
        var_data = resample.loc[:, v, :, 4]
        mean = var_data.groupby("lr")[bin_n].mean()
        ax.scatter(
            var_data["learning_power"] + (n - 4.5) / 12,
            var_data[bin_n] / mean,
            label=v,
        )
        ax.set_yscale("log")

        ax.set_title(f"Bin {bin_n}")
        ax.set_xticks(util.lr_powers)
        ax.set_xticklabels([])
        # ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

axes[-1][0].set_xlabel("Learning rate")
axes[-1][1].set_xlabel("Learning rate")
axes[2][0].set_ylabel("Percentage error / mean")

axes[0][-1].legend(ncol=3)
util.power_ticks(axes[-1, 0].xaxis, 10, util.lr_powers)
util.power_ticks(axes[-1, 1].xaxis, 10, util.lr_powers)

fig.suptitle(
    f"{problem}: spread across resamples of last iteration bin error vs. learning rate"
)

fig.tight_layout()
fig.savefig(f"figs/lr_{problem}_binerrs_spread.png")

plt.show()
