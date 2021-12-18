import matplotlib.pyplot as plt
import numpy as np

import util

if __name__ == "__main__":
    problem = "closure"

    data = util.load_batch(problem, "nominal")
    best_iterations = util.group_exceptby(data, "iteration")["chisq/ndf"].idxmin()
    best = data.loc[best_iterations]

    mean = best.groupby(["initializer", "batch_size"])["chisq/ndf"].mean()
    std = best.groupby(["initializer", "batch_size"])["chisq/ndf"].std()

    fig, ax = plt.subplots()
    for i in util.initializers:
        ax.errorbar(
            util.batch_powers,
            mean.loc[i, :, :],
            yerr=std.loc[i, :, :],
            fmt=".",
            capsize=5,
            label=i,
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("χ²/NDF, OmniFold")
    ax.legend()

    util.power_ticks(ax.xaxis, 2, util.batch_powers)

    fig.suptitle(
        f"{problem}: Mean χ²/NDF for all variables vs. batch size, across intializers"
    )
    fig.tight_layout()
    fig.savefig(f"figs/{problem}_chisq_vs_batch_all_vars.png")

    plt.show()
