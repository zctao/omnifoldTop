# -*- compilation-read-command: nil -*-

import matplotlib
import matplotlib.pyplot as plt

import util

problem = "closure"

if __name__ == "__main__":
    data = util.load_lr(problem, "nominal")
    best_iterations = util.group_exceptby(data, "iteration")["chisq/ndf"].idxmin()

    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    fig, ax = plt.subplots()
    for v in util.interesting:
        # ax.set_title(v)
        var_data = data.loc[best_iterations].loc[:, v, :]
        ax.plot(var_data["learning_power"], var_data["chisq/ndf"], label=v.pretty)
        util.power_ticks(ax.xaxis, 10, util.lr_powers)

    ax.set_xlabel("Learning rate")
    ax.set_ylabel("χ²/NDF")
    ax.set_yscale("log")
    fig.suptitle(f"{problem}: Best iteration χ²/NDF vs. learning rate")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(f"figs/lr_{problem}_chisq.png")

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10))
    for ax, bin_n in zip(axes.ravel(), range(10)):
        for v in util.interesting:
            ax.scatter(
                data.loc[:, v, 4]["learning_power"],
                data.loc[:, v, 4][bin_n],
                label=v.pretty,
            )
            ax.set_title(f"Bin {bin_n}")
            ax.set_xticks(util.lr_powers)
            ax.set_xticklabels([])
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

    axes[-1][0].set_xlabel("Learning rate")
    axes[-1][1].set_xlabel("Learning rate")
    axes[2][0].set_ylabel("Percentage error")

    axes[0][-1].legend()
    util.power_ticks(axes[-1, 0].xaxis, 10, util.lr_powers)
    util.power_ticks(axes[-1, 1].xaxis, 10, util.lr_powers)

    fig.suptitle(f"{problem}: Last iteration bin error vs. learning rate")

    fig.tight_layout()
    fig.savefig(f"figs/lr_{problem}_binerrs.png")

    plt.show()
