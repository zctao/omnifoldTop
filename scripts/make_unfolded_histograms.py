#!/usr/bin/env python3
import os
import time
import tracemalloc
import numpy as np

import util
import histogramming as myhu
from OmniFoldTTbar import load_unfolder
from OmniFoldTTbar import collect_unfolded_histograms_from_unfolder

import logging
logger = logging.getLogger("make_unfolded_histograms")

def make_unfolded_histograms(
        result_dir,
        iteration=-1,
        nruns=None,
        nominal_only=False,
        observables=[],
        observable_config=None,
        binning_config=None,
        output=None,
        verbose=False
    ):

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    tracemalloc.start()

    # check if result_dir exists
    if not os.path.isdir(result_dir):
        logger.critical(f"Fail to find directory {result_dir}")
        return

    # path to argument json config
    fpath_args_config = os.path.join(result_dir, "arguments.json")
    if not os.path.isfile(fpath_args_config):
        logger.critical(f"Cannot open argument config {fpath_args_config}")
        return

    # observable config
    obsConfig_d = {}
    if observable_config:
        obsConfig_d = util.read_dict_from_json(observable_config)

    # unfolder
    logger.info(f"Load unfolder from {result_dir} ...")
    t_load_start = time.time()

    ufdr = load_unfolder(fpath_args_config, observables, obsConfig_d)

    t_load_stop = time.time()
    logger.info(f"Done")
    logger.debug(f"Loading time: {(t_load_stop-t_load_start):.2f} seconds")

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*1e-6:.1f} MB; Peak usage: {mpeak*1e-6:.1f} MB")

    logger.info("Making histograms")
    t_hist_start = time.time()
    hists_d = collect_unfolded_histograms_from_unfolder(
        ufdr,
        observables,
        obsConfig_d,
        binning_config,
        iteration = iteration,
        nruns = nruns,
        absoluteValue = False # FIXME: True for 'th_y', 'tl_y'?
        )
    t_hist_stop = time.time()
    logger.debug(f"Histogramming time: {(t_hist_stop-t_hist_start):.2f} seconds")

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*1e-6:.1f} MB; Peak usage: {mpeak*1e-6:.1f} MB")

    if output:
        # Save histograms
        outdir = os.path.dirname(output)
        if not os.path.isdir(outdir):
            logger.info(f"Create output directory {outdir}")

        logger.info(f"Write histograms to file {output}")
        myhu.write_histograms_dict_to_file(hists_d, output)

    return hists_d

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Make and save histograms from the unfolding results')

    parser.add_argument("result_dir", type=str,
                        help="Directories of unfolding results")
    parser.add_argument("-o", "--output", type=str, default='histograms.root',
                        help="Output file names")
    parser.add_argument("-i", "--iteration", type=int, default=-1,
                        help="Use the results at the specified iteration")
    parser.add_argument("-n", "--nruns", type=int,
                        help="Number of runs for making unfolded distributions. If not specified, use all that are available")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="If True, set logger level to DEBUG, otherwise set tto INFO")
    parser.add_argument("--nominal-only", action='store_true',
                        help="If True, only save the nominal unfolded distributions")
    parser.add_argument("--observables", nargs='+', default=[],
                        help="List of observables to make histograms. If not provided, use the same ones from the unfolding results")
    parser.add_argument("--observable_config", type=str,
                        help="Path to the observable config file. If not provided, use the same one from the unfolding results")
    parser.add_argument("--binning-config", type=str,
                        default='configs/binning/bins_10equal.json',
                        #default='configs/binning/bins_ttdiffxs.json',
                        help="Path to the binning config file for variables.")

    args = parser.parse_args()

    util.configRootLogger()

    make_unfolded_histograms(**vars(args))
