import os
import logging

import util
from make_histograms import make_histograms

import argparse

parser = argparse.ArgumentParser(
    description="Run scripts/make_histograms.py for all unfolding results of Run2 datasets")

parser.add_argument("top_result_dir", type=str,
                    help="Top directory that contains the unfolding results")

parser.add_argument("--binning-config", type=str,
                    default='configs/binning/bins_ttdiffxs.json',
                    help="Path to the binning config file for variables.")
parser.add_argument("-f", "--outfilename", type=str, default="histograms.root",
                    help="Output file name")
parser.add_argument("-i", "--iterations", type=int, nargs='+', default=[-1],
                    help="Use the result at the specified iterations")
parser.add_argument("--correction-dir", type=str,
                    help="Directory to read binned connections")
parser.add_argument("--include-ibu", action='store_true',
                    help="If True, run unfolding with IBU too")
parser.add_argument("-k", "--resdir-keywords", nargs='+',
                    default=['output_run2'],
                    help="Keywords to match the result directory names. If multiple keywords are provided, only directories containing all the keywords are selected.")
parser.add_argument("--dryrun", action="store_true",
                    help="If True, print the directory to make histograms without actually running make_histograms")

args = parser.parse_args()

logger = logging.getLogger("make_run2_histograms")
util.configRootLogger()

for cwd, subdirs, files in os.walk(args.top_result_dir):
    if not files:
        continue

    if not 'arguments.json' in files or not 'weights_unfolded.npz' in files:
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

    logger.info(f"Make histograms for {cwd}")

    if args.dryrun:
        logger.info("skip...")
        continue

    make_histograms(
        cwd,
        binning_config = args.binning_config,
        outfilename = args.outfilename,
        iterations = args.iterations,
        include_ibu = args.include_ibu,
        binned_correction_dir = args.correction_dir
        )
