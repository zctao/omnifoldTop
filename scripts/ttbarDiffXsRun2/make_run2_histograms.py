import os
import logging

import util
from make_histograms import make_histograms, make_histograms_bootstrap

import argparse

parser = argparse.ArgumentParser(
    description="Run scripts/make_histograms.py for all unfolding results of Run2 datasets")

parser.add_argument("top_result_dir", type=str,
                    help="Top directory that contains the unfolding results")
parser.add_argument("--binning-config", type=str,
                    default='configs/binning/bins_ttdiffxs.json',
                    help="Path to the binning config file for variables.")
parser.add_argument("--correction-dir", type=str,
                    default="/mnt/xrootdg/ztao/NtupleTT/latest/systCRL/ttbar_nominal",
                    help="Directory to read binned corrections")
parser.add_argument("-f", "--outfilename", type=str, default="histograms.root",
                    help="Output file name")
parser.add_argument("-i", "--iterations", type=int, nargs='+', default=[-1],
                    help="Use the result at the specified iterations")
parser.add_argument("-n", "--nensembles", type=int, nargs='+', default=[None],
                    help="List of number of runs for making histograms. If None, use all available")
parser.add_argument("--include-ibu", action='store_true',
                    help="If True, run unfolding with IBU too")
parser.add_argument("-k", "--resdir-keywords", nargs='+', default=[],
                    help="Keywords to match the result directory names. If multiple keywords are provided, only directories containing all the keywords are selected.")
parser.add_argument("--no-override", action='store_true',
                    help="If True, skip making histograms for a directory if one already exists")
parser.add_argument('-p', '--plot-verbosity', action='count', default=0,
                    help="Plot verbose level. '-ppp' to make all plots.")
parser.add_argument("-b", "--bootstrap-dirs", nargs='+', type=str, default=[],
                    help="List of top directories to make histograms from bootstraping")
parser.add_argument("--dryrun", action="store_true",
                    help="If True, print the directory to make histograms without actually running make_histograms")

args = parser.parse_args()

logger = logging.getLogger("make_run2_histograms")
util.configRootLogger()

for cwd, subdirs, files in os.walk(args.top_result_dir):
    if not files:
        continue

    if not 'arguments.json' in files and not 'weights_unfolded.npz' in files and not 'arguments_rw.json' in files:
        # cwd is not a directory containing unfolding results
        continue

    # check if the current working directory name contains the keywords
    matched = True
    for kw in args.resdir_keywords:
        if kw.startswith('!'):
            # if kw starts with '!', veto the directory that contains the keyword
            kw_veto = kw.lstrip('!')
            matched &= not (kw_veto in cwd)
        else:
            matched &= kw in cwd

    if not matched:
        continue

    if args.no_override:
        fname_hists = os.path.join(cwd, args.outfilename)
        if os.path.isfile(fname_hists):
            logger.debug(f"Skip {cwd}. {fname_hists} already exists.")
            continue

    logger.info(f"Make histograms for {cwd}")

    if args.dryrun:
        logger.info(" skip...")
        continue

    make_histograms(
        cwd,
        binning_config = args.binning_config,
        observables = [],
        outfilename = args.outfilename,
        iterations = args.iterations,
        nruns = args.nensembles,
        include_ibu = args.include_ibu,
        binned_correction_dir = args.correction_dir,
        plot_verbosity = args.plot_verbosity
        )

# Collect and make histograms for bootstrap results
for topdir_bs in args.bootstrap_dirs:
    fname_hist_bs = os.path.join(topdir_bs, args.outfilename)
    if args.no_override and os.path.isfile(fname_hist_bs):
        logger.debug(f"Skip {topdir_bs}. {fname_hist_bs} already exists.")
        continue

    logger.info(f"Make histograms from bootstrap {topdir_bs}")

    if args.dryrun:
        logger.info(" skip...")
        continue

    make_histograms_bootstrap(
        topdir_bs,
        histname = args.outfilename,
        outfilename = fname_hist_bs
    )