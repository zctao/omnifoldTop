# -*- compilation-read-command: nil -*-

import itertools
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

import util


def path(problem, learning_rate, resample, step, iteration):
    lr_str = f"{learning_rate:f}".rstrip("0")
    rs_str = "" if resample == -1 else f"_rs{resample}"
    return (
        pathlib.Path("/data/wcassidy/output/learning_rate")
        / problem
        / lr_str
        / f"Models{rs_str}"
        / f"model_step{step}_{iteration}_history.csv"
    )


if __name__ == "__main__":
    problem = "closure"

    frames = []
    keys = []
    steps = (1, 2)
    iterations = range(4)
    for key in itertools.product(util.learning_rates, range(-1, 10), steps, iterations):
        frames.append(pd.read_csv(path(problem, *key)))
        keys.append(key)
    data = pd.concat(
        frames, keys=keys, names=("learning rate", "resample", "step", "iteration")
    )

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
    last_epoch = 60
    for ax, (step, iteration) in zip(
        axes.ravel(), itertools.product(steps, iterations)
    ):
        areas = []
        for lr in reversed(util.learning_rates):
            lr_data = data.query("epoch < 60").loc[lr, :, step, iteration]
            losses = lr_data.groupby("epoch")["loss"]
            mean = losses.mean()
            val_losses = lr_data.groupby("epoch")["val_loss"]

            colour = next(ax._get_lines.prop_cycler)["color"]
            ax.plot(mean.index, mean, color=colour, label=f"α = {lr} loss")
            ax.plot(
                mean.index,
                val_losses.mean(),
                color=colour,
                linestyle="dashed",
                label=f"α = {lr} validation loss",
            )

            areas.append(
                ax.fill_between(
                    mean.index,
                    losses.min(),
                    losses.max(),
                    alpha=0.2,
                    label=f"α = {lr} loss range",
                )
            )

        for a in areas:
            a.remove()
        ax.relim()
        for a in areas:
            ax.add_collection(a)

    for n, step in enumerate(steps):
        ax2 = axes[n, -1].twinx()
        ax2.set_ylabel(f"Step {step}")
        ax2.set_yticks([])
    for n, iteration in enumerate(iterations):
        axes[0, n].set_title(f"Iteration {iteration}")

    axes[1, -1].legend()

    fig.suptitle(f"{problem}: Loss history at various learning rates")
    bigax = fig.add_subplot(111, frameon=False)
    bigax.tick_params(which="both", bottom=False, left=None, labelcolor="none")
    bigax.set_xlabel("Epoch")
    bigax.set_ylabel("Loss")

    fig.tight_layout()
    fig.savefig(f"figs/lr_{problem}_loss_history.png")
    plt.show()
