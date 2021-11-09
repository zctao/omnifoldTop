import itertools

import matplotlib.pyplot as plt

import util

marker_map = {
    "glorot_uniform": "v",
    "glorot_normal": "^",
    "he_uniform": "s",
    "he_normal": "D",
}

problem = "closure"

if __name__ == "__main__":
    data = util.load_batch("closure", "nominal")
    best_iterations = util.group_exceptby(data, "iteration")["chisq/ndf"].idxmin()

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    for ax, v in zip(axes.ravel(), util.interesting):
        ax.set_title(v)
        for i in util.initializers:
            var_data = data.loc[best_iterations].loc[i, :, v]
            ax.scatter(
                var_data["batch_power"],
                var_data["chisq/ndf"],
                label=i,
            )
            util.power_ticks(ax.xaxis, 2, util.batch_powers)

    axes[-1, 1].set_xlabel("Batch size")
    axes[1, 0].set_ylabel("χ²/NDF")
    fig.suptitle(f"{problem}: Best iteration χ²/NDF vs. batch size and initializer")
    axes[0, -1].legend()
    fig.tight_layout()
    fig.savefig(f"figs/chisq_batch_individual_vars_{problem}.png")
    plt.show()
