import collections
import itertools as it
import json
import pathlib
from typing import NamedTuple

import matplotlib
import numpy as np
import pandas as pd

initializers = "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"
batch_powers = np.asarray([9, 12, 15, 18])
batch_sizes = 2 ** batch_powers

lr_powers = np.asarray([-2, -3, -4, -5])
learning_rates = 10.0 ** lr_powers

variables = (
    "Ht",
    "chitt",
    "dphi",
    "mtt",
    "ptt",
    "th_e",
    "th_eta",
    "th_m",
    "th_phi",
    "th_pout",
    "th_pt",
    "th_y",
    "tl_e",
    "tl_eta",
    "tl_m",
    "tl_phi",
    "tl_pout",
    "tl_pt",
    "tl_y",
    "yboost",
    "ystar",
    "ytt",
)

interesting = {
    "gaussian": (
        "Ht",
        "chitt",
        "dphi",
        "th_e",
        "th_pt",
        "th_y",
        "tl_e",
        "ystar",
        "mtt",
    ),
    "data_sim": (
        "dphi",
        "Ht",
        "mtt",
        "ptt",
        "th_e",
        "th_pout",
        "th_pt",
        "tl_e",
        "tl_pout",
        "tl_pt",
    ),
}


root = pathlib.Path("/data/wcassidy/output/")


def load_batch(problem, section, **indices):
    df = load(
        root,
        batch_path,
        problem,
        section,
        initializer=initializers,
        batch_size=batch_sizes,
        **indices,
    )
    df = df.eval("batch_power = log(batch_size) / log(2)")
    return df


def batch_path(problem, initializer, batch_size):
    return pathlib.Path("init_vs_batch") / initializer / f"batch_{batch_size}"


def load_lr(problem, section, **indices):
    df = load(root, lr_path, problem, section, lr=learning_rates, **indices)
    df = df.eval("learning_power = log10(lr)")
    return df


def lr_path(problem, lr):
    return pathlib.Path("learning_rate") / problem / f"{lr:f}".rstrip("0")


def cat_path(**sections):
    return pathlib.Path().joinpath(*sections.values())


def load(root, section, pathfunc=cat_path, iterations=5, resamples=10, **indices):
    inner_indices = {"variable": variables}
    if section == "resample":
        inner_indices["resample"] = range(iterations)
    inner_indices["iteration"] = range(resamples)

    rows = []
    for index in it.product(*indices.values()):
        index_dict = dict(zip(indices.keys(), index))
        for index2 in it.product(*inner_indices.values()):
            v = index2[0]
            path = root / pathfunc(**index_dict) / "Metrics" / f"{v}.json"
            with open(path) as f:
                metrics = json.load(f)[v][section]

            rows.append(
                {
                    **index_dict,
                    **dict(zip(inner_indices.keys(), index2)),
                    "chisq/ndf": recursive_index(
                        metrics, ("Chi2", "chi2/ndf", *index2[1:])
                    ),
                    "delta": recursive_index(metrics, ("Delta", "delta", *index2[1:])),
                    **dict(
                        enumerate(
                            recursive_index(
                                metrics, ("BinErrors", "percentage", *index2[1:])
                            )
                        )
                    ),
                }
            )

    df = pd.DataFrame.from_dict(data=rows)
    df = df.assign(
        interesting=[
            df["chisq/ndf"][i - df["iteration"][i]] > 2 for i in range(len(df))
        ]
    )
    df = df.set_index([*indices.keys(), *inner_indices.keys()])
    return df


def recursive_index(mapping, keys):
    keys = iter(keys)
    try:
        k = next(keys)
        return recursive_index(mapping[k], keys)
    except StopIteration:
        return mapping


def power_ticks(axis, base, powers):
    axis.set_major_locator(matplotlib.ticker.FixedLocator(powers))
    axis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter(f"${base}" "^{{{x}}}$")
    )


def group_exceptby(df, *names):
    return df.groupby([name for name in df.index.names if name not in names])
