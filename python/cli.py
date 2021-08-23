"""
Command-line interface for unfolding
"""

import argparse

import reweight

unfold = argparse.ArgumentParser()

unfold.add_argument(
    "--observables-train",
    nargs="+",
    default=["th_pt", "th_y", "th_phi", "th_e", "tl_pt", "tl_y", "tl_phi", "tl_e"],
    help="List of observables to use in training.",
)
unfold.add_argument(
    "--observables",
    nargs="+",
    default=[
        "mtt",
        "ptt",
        "ytt",
        "ystar",
        "chitt",
        "yboost",
        "dphi",
        "Ht",
        "th_pt",
        "th_y",
        "th_eta",
        "th_phi",
        "th_m",
        "th_e",
        "th_pout",
        "tl_pt",
        "tl_y",
        "tl_eta",
        "tl_phi",
        "tl_m",
        "tl_e",
        "tl_pout",
    ],
    help="List of observables to unfold",
)
unfold.add_argument(
    "--observable-config",
    default="configs/observables/vars_ttbardiffXs.json",
    help="JSON configurations for observables",
)
unfold.add_argument(
    "-d",
    "--data",
    required=True,
    nargs="+",
    type=str,
    help="Observed data npz file names",
)
unfold.add_argument(
    "-s",
    "--signal",
    required=True,
    nargs="+",
    type=str,
    help="Signal MC npz file names",
)
unfold.add_argument(
    "-b", "--background", nargs="+", type=str, help="Background MC npz file names"
)
unfold.add_argument(
    "--bdata",
    nargs="+",
    type=str,
    default=None,
    help="Background MC files to be mixed with data",
)
unfold.add_argument(
    "-o", "--outputdir", default="./output", help="Directory for storing outputs"
)
unfold.add_argument(
    "-t",
    "--truth-known",
    action="store_true",
    help="MC truth is known for 'data' sample",
)
unfold.add_argument(
    "-c",
    "--plot-correlations",
    action="store_true",
    help="Plot pairwise correlations of training variables",
)
unfold.add_argument(
    "-i",
    "--iterations",
    type=int,
    default=4,
    help="Numbers of iterations for unfolding",
)
unfold.add_argument(
    "--weight", default="totalWeight_nominal", help="name of event weight"
)
unfold.add_argument(
    "-m",
    "--background-mode",
    choices=["default", "negW", "multiClass"],
    default="default",
    help="Background mode",
)
unfold.add_argument(
    "-r",
    "--reweight-data",
    choices=reweight.rw.keys(),
    default=None,
    help="Reweight strategy of the input spectrum for stress tests. Requires --truth-known.",
)
unfold.add_argument(
    "-v", "--verbose", action="count", default=0, help="Verbosity level"
)
unfold.add_argument(
    "-g",
    "--gpu",
    type=int,
    choices=[0, 1],
    default=None,
    help="Manually select one of the GPUs to run",
)
unfold.add_argument(
    "--unfolded-weights",
    nargs="*",
    type=str,
    help="Unfolded weights file names. If provided, load event weights directly from the files and skip training.",
)
unfold.add_argument(
    "--binning-config",
    default="configs/binning/bins_10equal.json",
    type=str,
    help="Binning config file for variables",
)
unfold.add_argument(
    "--plot-history",
    action="store_true",
    help="If true, plot intermediate steps of unfolding",
)
unfold.add_argument(
    "--nresamples",
    type=int,
    default=25,
    help="number of times for resampling to estimate the unfolding uncertainty using the bootstrap method",
)
unfold.add_argument(
    "-e",
    "--error-type",
    choices=["sumw2", "bootstrap_full", "bootstrap_model"],
    default="sumw2",
    help="Method to evaluate uncertainties",
)
unfold.add_argument(
    "--batch-size", type=int, default=512, help="Batch size for training"
)
unfold.add_argument(
    "-l",
    "--load-models",
    type=str,
    help="Directory from where to load trained models. If provided, training will be skipped.",
)
