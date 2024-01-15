#!/usr/bin/env python3
import os
import sys
import json

import util
from ttbarDiffXsRun2.systematics import get_systematics

import logging
logger = logging.getLogger("run_uncertainties")

status_level = {
    "null" : 0,
    "generate" : 1,
    "unfold" : 2,
    "histogram" : 3,
    "evaluate" : 4
}

def update_status(
    filepath_status,
    new_status, # str
    systematics_keywords = [],
    rerun = False,
    skip_unready = True
    ):

    if os.path.isfile(filepath_status):
        # read status from the existing file
        logger.debug(f"Load job status from {filepath_status}")
        with open(filepath_status, 'r') as jfile:
            jstatus_d = json.load(jfile)
    else:
        # create a new dict
        jstatus_d = dict()

    systnames_update = []

    new_level = status_level[new_status]

    systematics_list = get_systematics(systematics_keywords)
    if "central" in systematics_keywords:
        systematics_list += ["central"]

    for systname in systematics_list:

        old_status = jstatus_d.get(systname, 'null')
        old_level = status_level[old_status]

        ready = new_level <= old_level+1
        if not ready and not skip_unready:
            raise ValueError(f"Cannot process with the step '{new_status}' from '{old_status}' for {systname}")

        if new_level == old_level+1:
            jstatus_d[systname] = new_status
            systnames_update.append(systname)
        elif new_level <= old_level and rerun:
            jstatus_d[systname] = new_status
            systnames_update.append(systname)

    # rename the old file if it exists
    if os.path.isfile(filepath_status):
        fname, ext = os.path.splitext(filepath_status)
        fname_old = f"{fname}.old{ext}"
        logger.debug(f"Backup the previous status file to {fname_old}")
        os.replace(filepath_status, fname_old)
    else:
        logger.info(f"Create a new file for recording job status: {filepath_status}")

    # Write the updated status to a new file
    with open(filepath_status, "w") as jfile_new:
        json.dump(jstatus_d, jfile_new, indent=2)

    return systnames_update

def generate(args):
    logger.info("Generate run configs")
    from ttbarDiffXsRun2.createRun2Config import createRun2Config

    # config directory
    if not os.path.isabs(args.config_name):
        args.config_name = os.path.join(args.result_dir, args.config_name)

    if not os.path.isdir(os.path.dirname(args.config_name)):
        os.makedirs(os.path.dirname(args.config_name))

    # job status
    systnames = update_status(
        args.job_file,
        'generate',
        systematics_keywords = args.systematics_keywords,
        rerun = args.rerun,
        skip_unready = True
    )

    logger.debug(systnames)

    common_unc_cfg = {
        "observable_config" : "${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json",
        "binning_config" : "${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json",
        "iterations" : 3,
        "batch_size" : 500000,
        "normalize" : False,
        "nruns" : 8,
        "parallel_models" : 3,
        "run_ibu": False,
    }

    if args.config_string:
        try:
            jcfg = json.loads(args.config_string)
            common_unc_cfg.update(jcfg)
        except json.decoder.JSONDecodeError:
            print("ERROR Cannot parse the extra config string: {args.config_string}")

    if args.observables:
        common_unc_cfg["observables"] = args.observables

    # generate run configs
    if systnames:
        createRun2Config(
            args.sample_dir,
            outname_config = args.config_name,
            output_top_dir = args.result_dir,
            subcampaigns = args.subcampaigns,
            systematics_keywords = systnames,
            common_cfg = common_unc_cfg,
            category = 'ljets',
            run_list = ['systematics']
        )

def run(args):
    logger.info("Run unfolding")
    from run_unfold import run_unfold

    # config directory
    config_dir = os.path.dirname(args.config_name)
    if not os.path.isabs(config_dir):
        args.config_name = os.path.join(args.result_dir, args.config_name)

    # job status
    if not os.path.isfile(args.job_file):
        raise FileNotFoundError(f"Cannot locate job status file: {args.job_file}")

    systnames = update_status(
        args.job_file,
        'unfold',
        systematics_keywords = args.systematics_keywords,
        rerun = args.rerun,
        skip_unready = True
    )

    logger.debug(systnames)

    for syst in systnames:
        logger.debug(syst)
        fpath_cfg_syst = f"{args.config_name}_{syst}.json"
        #print(f"run_unfold({fpath_cfg_syst})")
        run_unfold(fpath_cfg_syst)

def histogram(args):
    logger.info("Make histograms")
    from make_histograms import make_histograms

    # job status
    if not os.path.isfile(args.job_file):
        raise FileNotFoundError(f"Cannot locate job status file: {args.job_file}")

    systnames = update_status(
        args.job_file,
        'histogram',
        systematics_keywords = args.systematics_keywords,
        rerun = args.rerun,
        skip_unready = True
    )

    logger.debug(systnames)

    for syst in systnames:
        logger.debug(syst)
        ufdir = os.path.join(args.result_dir, syst)
        make_histograms(
            ufdir,
            args.binning_config,
            observables = args.observables,
            observables_multidim = args.observables_multidim,
            include_ibu = True
        )

def evaluate(args):
    logger.info("Evaluate uncertainties")
    from evaluate_uncertainties import evaluate_uncertainties

    # job status
    if not os.path.isfile(args.job_file):
        raise FileNotFoundError(f"Cannot locate job status file: {args.job_file}")

    # central
    try:
        update_status(
            args.job_file,
            'evaluate',
            systematics_keywords = [args.central_name],
            rerun = False,
            skip_unready = False
        )
    except ValueError as ve:
        logger.error(f"Central histograms unavailable: {ve}")
        return

    systnames = update_status(
        args.job_file,
        'evaluate',
        systematics_keywords = args.systematics_keywords,
        rerun = True, # always rerun
        skip_unready = True
    )

    central_dir = os.path.join(args.result_dir, args.central_name)

    logger.info("Absolute differential cross-sections")
    outdir_abs = os.path.join(args.result_dir, "uncertainties", "abs")
    if not os.path.isdir(outdir_abs):
        logger.debug(f"Create output directory: {outdir_abs}")
        os.makedirs(outdir_abs)

    evaluate_uncertainties(
        nominal_dir = central_dir,
        systematics_topdir = args.result_dir,
        output_dir = outdir_abs,
        systematics_keywords = systnames,
        ibu = True,
        plot = True,
        hist_key = 'unfolded',
        normalize = False,
        # For now
        bootstrap_topdir = None,
        bootstrap_mc_topdir = None,
        network_error_dir = None,
    )

    logger.info("Relative differential cross-sections")
    outdir_rel = os.path.join(args.result_dir, "uncertainties", "rel")
    if not os.path.isdir(outdir_rel):
        logger.debug(f"Create output directory: {outdir_rel}")
        os.makedirs(outdir_rel)

    evaluate_uncertainties(
        nominal_dir = central_dir,
        systematics_topdir = args.result_dir,
        output_dir = outdir_rel,
        systematics_keywords = systnames,
        ibu = True,
        plot = True,
        hist_key = 'unfolded',
        normalize = True,
        # For now
        bootstrap_topdir = None,
        bootstrap_mc_topdir = None,
        network_error_dir = None,
    )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    ######
    # parser for generating run configs
    parser_gen = subparsers.add_parser('generate', help="Generate run configs")

    parser_gen.add_argument(
        "--sample-dir", type=str, action=util.ParseEnvVar,
        default="${DATA_DIR}/NtupleTT/20221221",
        help="Sample directory")
    parser_gen.add_argument(
        "-e", "--subcampaigns", nargs='+', 
        choices=["mc16a", "mc16d", "mc16e"], 
        default=["mc16a", "mc16d", "mc16e"])
    parser_gen.add_argument(
        "-n", "--config-name", type=str, default="configs/runCfg", 
        help="Prefix of the generated run config file name")
    parser_gen.add_argument(
        "--observables", nargs='+',
        default=['th_pt', 'th_y', 'tl_pt', 'tl_y', 'ptt', 'ytt', 'mtt'],
        help="List of observables to unfold")
    parser_gen.add_argument(
        "--config-string", type=str,
        help="String in JSON format to be parsed for updating run configs")

    parser_gen.set_defaults(func=generate)

    ######
    # parser for running unfolding
    parser_run = subparsers.add_parser('run', help="Run unfolding")

    parser_run.add_argument(
        "-n", "--config-name", type=str, default="configs/runCfg", 
        help="Prefix of the run config file name")

    parser_run.set_defaults(func=run)

    ######
    # parser for making histograms
    parser_hist = subparsers.add_parser('histogram', help="Make histograms")

    parser_hist.add_argument(
        "--observables", nargs='+', default=[],
        help="List of observables to make histograms. If not provided, use the same ones from the unfolding results")
    parser_hist.add_argument(
        "--observables-multidim", nargs='+', default=[],
        help="List of observables to make multi-dimension histograms.")
    parser_hist.add_argument(
        "--binning-config", type=str, action=util.ParseEnvVar,
        default='${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json',
        help="Path to the binning config file for variables.")

    parser_hist.set_defaults(func=histogram)

    ######
    # parser for plotting uncertainties
    parser_eval = subparsers.add_parser('evaluate', help="Plot uncertainties")

    parser_eval.add_argument(
        "-c", "--central-name", type=str, default="central",
        help="Name of the central result"
    )

    parser_eval.set_defaults(func=evaluate)

    ######
    # common arguments
    parser.add_argument(
        "-r", "--result-dir", type=str, action=util.ParseEnvVar, 
        default="${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/Uncertainties/latest",
        help="Output directory of unfolding runs")
    parser.add_argument(
        "-j", "--job-file", type=str, default="status/jobs.json")
    parser.add_argument(
        "-k", "--systematics-keywords", 
        type=str, nargs="*", default=[],
        help="List of keywords to filter systematic uncertainties to evaluate. If empty, include all available.")
    parser.add_argument(
        "--rerun", action='store_true', help="If True, rerun the step")
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        help="If True, set logging level to debug (default level info)")

    args = parser.parse_args()

    util.configRootLogger()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # directories
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.isabs(args.job_file):
        args.job_file = os.path.join(args.result_dir, args.job_file)

    if not os.path.isdir(os.path.dirname(args.job_file)):
        os.makedirs(os.path.dirname(args.job_file))

    # call the function
    try:
        args.func(args)
    except Exception as ex:
        logger.setLevel(logging.DEBUG)
        util.reportMemUsage(logger)

        # Report GPU usage only if it is needed
        if args.func == run:
            from modelUtils import reportGPUMemUsage
            reportGPUMemUsage(logger)

        logger.error(ex)
        sys.exit(1)