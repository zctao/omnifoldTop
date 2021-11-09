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


class Var(str):
    def __new__(cls, name, pretty):
        self = super().__new__(cls, name)
        self.pretty = pretty
        return self


variables = (
    Var("chitt", r"$\chi^{t \bar{t}}$"),
    Var("dphi", r"$\Delta\phi(t, \bar{t})$"),
    Var("Ht", r"$H^{t \bar{t}}_T$"),
    Var("mtt", r"$m^{t \bar{t}}$"),
    Var("ptt", r"$p^{t \bar{t}}_T$"),
    Var("th_e", r"$E^{t, had}$"),
    Var("th_eta", r"$\eta^{t, had}$"),
    Var("th_m", r"$m^{t, had}$"),
    Var("th_phi", r"$\phi^{t, had}$"),
    Var("th_pout", r"$p^{t, had}_{out}$"),
    Var("th_pt", r"$p^{t, had}_T$"),
    Var("th_y", r"$y^{t, had}$"),
    Var("tl_e", r"$E^{t, lep}$"),
    Var("tl_eta", r"$\eta^{t, lep}$"),
    Var("tl_m", r"$m^{t, lep}$"),
    Var("tl_phi", r"$\phi^{t, lep}$"),
    Var("tl_pout", r"$p^{t, lep}_{out}$"),
    Var("tl_pt", r"$p^{t, lep}_T$"),
    Var("tl_y", r"$y^{t, lep}$"),
    Var("yboost", r"$y^{t \bar{t}}_{boost}$"),
    Var("ystar", r"$y^*$"),
    Var("ytt", r"$y^{t \bar{t}}$"),
)
interesting = [
    v
    for v in variables
    if v in ("mtt", "Ht", "th_pt", "chitt", "ystar", "th_e", "tl_e", "th_y", "dphi")
]

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


def load(root, pathfunc, problem, section, **indices):
    inner_indices = {"variable": variables}
    if section == "resample":
        inner_indices["resample"] = range(10)
    inner_indices["iteration"] = range(5)

    rows = []
    for index in it.product(*indices.values()):
        index_dict = dict(zip(indices.keys(), index))
        for index2 in it.product(*inner_indices.values()):
            v = index2[0]
            path = root / pathfunc(problem, **index_dict) / "Metrics" / f"{v}.json"
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
