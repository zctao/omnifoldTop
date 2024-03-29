#!/usr/bin/env python3
import os
import sys
import time
import tracemalloc

import util
import plotter
import reweight
from OmniFoldTTbar import init_unfolder
from make_histograms import make_histograms_from_unfolder
import modelUtils
import lrscheduler

import logging
logger = logging.getLogger("unfold")

def unfold(**parsed_args):

    #################
    # Output directory
    if parsed_args['outputdir'] is not None:
        if not os.path.isdir(parsed_args['outputdir']):
            print(f"Create output directory {parsed_args['outputdir']}")
            os.makedirs(parsed_args['outputdir'])

    # Prepare logger and log file
    logname = 'log.txt'
    if parsed_args['unfolded_weights'] is not None:
        logname = 'log_rw.txt'
    elif parsed_args['resume']:
        logname = 'log_re.txt'

    logfile = os.path.join(parsed_args['outputdir'], logname)
    util.configRootLogger(filename=logfile)
    logger.setLevel(logging.DEBUG if parsed_args['verbose']>0 else logging.INFO)

    # Host
    logger.info(f"Hostname: {os.uname().nodename}")

    tracemalloc.start()

    # Print arguments to logger
    for argkey, argvalue in sorted(parsed_args.items()):
        if argvalue is None:
            continue
        logger.info(f"Argument {argkey}: {argvalue}")

    # store arguments to json file for more convenient access later
    argname = 'arguments.json'
    if parsed_args['unfolded_weights'] is not None:
        argname = 'arguments_rw.json'
    elif parsed_args['resume']:
        argname = 'arguments_re.json'

    fname_args = os.path.join(parsed_args['outputdir'], argname)
    logger.info(f"Write arguments to file {fname_args}")
    util.write_dict_to_json(parsed_args, fname_args)

    #################
    # Initialize unfolder

    t_init_start = time.time()

    unfolder = init_unfolder(parsed_args)

    t_init_done = time.time()
    logger.debug(f"Initializing unfolder and loading input data took {(t_init_done-t_init_start):.2f} seconds.")
    util.reportMemUsage(logger)

    #################
    # Set up parallelization

    modelUtils.n_models_in_parallel = parsed_args["parallel_models"]
    logger.debug(f"{modelUtils.n_models_in_parallel} models will run in parallel")

    #################
    # Initialize learning rate scheduler

    lrscheduler.init_lr_scheduler(parsed_args["lrscheduler_config"])

    #################
    # Run unfolding

    logger.info("Start unfolding ... ")
    t_run_start = time.time()

    if parsed_args['unfolded_weights']:
        # load unfolded event weights from files
        unfolder.load(parsed_args['unfolded_weights'])
    else:
        # configure GPU
        modelUtils.configGPUs(parsed_args['gpu'], verbose=parsed_args['verbose'])

        # run unfolding
        unfolder.run(
            niterations = parsed_args['iterations'],
            resample_data = parsed_args['resample_data'],
            resample_mc = parsed_args['resample_mc'],
            nruns = parsed_args['nruns'],
            resample_everyrun = parsed_args['resample_everyrun'],
            model_type = parsed_args['model_name'],
            save_models = True,
            load_previous_iteration = False, # TODO check here
            load_models_from = parsed_args['load_models'],
            fast_correction = parsed_args['fast_correction'],
            batch_size = parsed_args['batch_size'],
            plot_status = parsed_args['plot_verbosity'] >= 2,
            resume_training = parsed_args['resume'],
            dummy_value = -99.
        )

    t_run_done = time.time()
    logger.info("Done!")
    logger.info(f"Unfolding took {t_run_done-t_run_start:.2f} seconds")

    util.reportMemUsage(logger)

    #################
    # Make histograms and plots

    # Variable correlations
    if parsed_args['plot_verbosity'] >= 2: # '-pp'
        # Before unfolding
        logger.info(f"Plot input variable correlations")

        corr_data = unfolder.handle_obs.get_correlations(unfolder.varnames_reco)
        plotter.plot_correlations(
            os.path.join(unfolder.outdir, "InputCorr_data"), corr_data,
            print_bincontents=True
        )

        if parsed_args['truth_known']:
            corr_truth = unfolder.handle_obs.get_correlations(unfolder.varnames_truth)
            plotter.plot_correlations(
                os.path.join(unfolder.outdir, "InputCorr_truth"), corr_truth,
                print_bincontents=True
            )

        corr_sim = unfolder.handle_sig.get_correlations(unfolder.varnames_reco)
        plotter.plot_correlations(
            os.path.join(unfolder.outdir, "InputCorr_sim"), corr_sim,
            print_bincontents=True
        )

        corr_gen = unfolder.handle_sig.get_correlations(unfolder.varnames_truth)
        plotter.plot_correlations(
            os.path.join(unfolder.outdir, "InputCorr_gen"), corr_gen,
            print_bincontents=True
        )

        # After unfolding
        logger.info(f"Plot variable correlations after unfolding")
        corr_unf = unfolder.get_correlations_unfolded(unfolder.varnames_truth)
        plotter.plot_correlations(
            os.path.join(unfolder.outdir, "OutputCorr_unf"), corr_unf,
            print_bincontents=True
        )

    ###
    t_result_start = time.time()

    all_observables = set().union(
        parsed_args['observables'], parsed_args['observables_extra'])

    obsConfig_d = util.read_dict_from_json(parsed_args['observable_config'])

    if parsed_args['binning_config']:
        make_histograms_from_unfolder(
            unfolder,
            parsed_args['binning_config'],
            observables = all_observables,
            obsConfig_d = obsConfig_d,
            include_ibu = parsed_args['run_ibu'],
            compute_metrics = True,
            plot_verbosity = parsed_args['plot_verbosity']
            )

    t_result_stop = time.time()

    logger.debug(f"Making histograms took {(t_result_stop-t_result_start):.2f} seconds.")
    util.reportMemUsage(logger)

    tracemalloc.stop()

def getArgsParser(arguments_list=None, print_help=False):
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', nargs='+',
                        type=str, action=util.ParseEnvVar,
                        help="Observed data root file names")
    parser.add_argument('-s', '--signal', nargs='+',
                        type=str, action=util.ParseEnvVar,
                        help="Signal MC root file names")
    parser.add_argument('-b', '--background', nargs='+',
                        type=str, action=util.ParseEnvVar,
                        help="Background MC root file names")
    parser.add_argument('--bdata',
                        nargs='+', type=str, action=util.ParseEnvVar,
                        help="Background MC files to be mixed with data")
    parser.add_argument('--observables', nargs='+',
                        default=['th_pt', 'th_y', 'th_phi', 'th_e', 'tl_pt', 'tl_y', 'tl_phi', 'tl_e'],
                        help="List of observables to use in training.")
    parser.add_argument('--observables-extra', nargs='*', default=[],
                        #default=['mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht', 'th_eta', 'th_m', 'th_pout', 'tl_eta', 'tl_m', 'tl_pout'],
                        help="List of extra observables to unfold.")
    parser.add_argument('--observable-config', action=util.ParseEnvVar,
                        default='${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json',
                        help="JSON configurations for observables")
    parser.add_argument('-o', '--outputdir',
                        type=str, default='.', action=util.ParseEnvVar,
                        help="Directory for storing outputs")
    parser.add_argument('-t', '--truth-known',
                        action='store_true',
                        help="MC truth is known for 'data' sample")
    parser.add_argument('-r', '--reweight-data',
                        choices=reweight.rw.keys(), default=None,
                        help="Reweight strategy of the input spectrum for stress tests. Requires --truth-known.")
    parser.add_argument('--unfolded-weights',
                        nargs='*', type=str, action=util.ParseEnvVar,
                        help="Unfolded weights file names. If provided, load event weights directly from the files and skip training.")
    parser.add_argument('-v', '--verbose',
                        action='count', default=0,
                        help="Verbosity level")
    parser.add_argument('-g', '--gpu',
                        type=int, default=None,
                        help="Manually select one of the GPUs to run")
    parser.add_argument('-i', '--iterations', type=int, default=4,
                        help="Numbers of iterations for unfolding")
    parser.add_argument("--nruns", type=int, default=1,
                        help="Number of times to run unfolding")
    parser.add_argument("--resample-data", action='store_true',
                        help="If True, fluctuate data weights")
    parser.add_argument("--resample-mc", action='store_true',
                        help="If True, fluctuate MC weights")
    parser.add_argument("--resample-everyrun", action='store_true',
                        help="If True, resample data or MC weights every run. Ignored if neither 'resample_data' nor 'resample_mc' is True.")
    parser.add_argument('-m', '--model-name',
                        type=str, default='dense_100x3',
                        help="Name of the model for unfolding")
    parser.add_argument('-n', '--normalize',
                        action='store_true',
                        help="If True, normalize simulation weights to data")
    parser.add_argument('--batch-size', type=int, default=16384,
                        help="Batch size for training")
    parser.add_argument('-l', '--load-models', type=str, action=util.ParseEnvVar,
                        help="Directory from where to load trained models. If provided, training will be skipped.")
    parser.add_argument('-c', '--correct-acceptance', action='store_true',
                        help="If True, include events that fail truth-level requirements to account for acceptance effects")
    parser.add_argument('--correct-efficiency', action='store_true',
                        help="If True, include events that fail reco-level requirements to account for efficiency effects")
    parser.add_argument('--fast-correction', action='store_true',
                        help="If True, assign an average weight of one for events that are not truth matched for acceptance correction")
    parser.add_argument('--binning-config', type=str, action=util.ParseEnvVar,
                        default="${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json",
                        help="Binning config file for variables")
    parser.add_argument('-p', '--plot-verbosity', action='count', default=0,
                        help="Plot verbose level. '-ppp' to make all plots.")
    parser.add_argument('--run-ibu', action='store_true',
                        help="If True, run unfolding also with IBU for comparison")
    parser.add_argument('--weight-mc', type=str, default='nominal', action=util.ParseEnvVar,
                        help="Type of event weights to retrieve from MC ntuples")
    parser.add_argument('--weight-data', type=str, default='nominal', action=util.ParseEnvVar,
                        help="Type of event weights to retrieve from data ntuples")
    parser.add_argument('--parallel-models', type=int, default=1, help="Number of parallel models, default ot 1")
    parser.add_argument('--lrscheduler-config', type=str, action=util.ParseEnvVar,
                        default="${SOURCE_DIR}/configs/lrs/constant_warm_up.json",
                        help="config file for learning rate scheduler")
    parser.add_argument('--toydata', action='store_true', help="If True, use toy data")
    parser.add_argument('--exclude-flow', action='store_true',
                        help="If True, exclude events in overflow and underflow bins given a binning configuration")
    parser.add_argument('--resume', action='store_true',
                        help="If True, load previously trained models and continue to run more steps if needed")
    parser.add_argument('--match-dR', type=float,
                        help="Require dR between the reco and truth tops less than the provided value")

    if print_help:
        parser.print_help()

    # Deprecated arguments. Keep them for backward compatibility
    parser.add_argument('--nresamples', type=int, default=None,
                        help="Use number of times for resampling to estimate the unfolding uncertainty using the bootstrap method.")
    parser.add_argument('-e', '--error-type',
                        choices=['sumw2','bootstrap_full','bootstrap_model'],
                        default=None, help="Method to evaluate uncertainties")
    parser.add_argument('--dummy-value', type=float,
                        help="Dummy value to fill events that failed selecton. If None (default), only events that pass reco and truth (if apply) level selections are used for unfolding")
    parser.add_argument('--weight-type', type=str, default='nominal',
                        help="Type of event weights to retrieve from MC ntuples. Same as '--weight-mc'")
    parser.add_argument("--preprocessor-config", type=str, default='configs/preprocessor/std.json', help="location of the preprocessor config file")
    #

    args = parser.parse_args(arguments_list)

    # verify truth is known when reweighting
    if args.reweight_data is not None and not args.truth_known:
        raise RuntimeError("--reweight-data requires --truth-known")

    # check if configuration file exist and expand the file path
    fullpath_obsconfig = util.expandFilePath(args.observable_config)
    if fullpath_obsconfig is not None:
        args.observable_config = fullpath_obsconfig
    else:
        raise RuntimeError(f"Cannot find file: {args.observable_config}")

    fullpath_binconfig = util.expandFilePath(args.binning_config)
    if fullpath_binconfig is not None:
        args.binning_config = fullpath_binconfig
    else:
        raise RuntimeError(f"Cannot find file: {args.binning_config}")

    # for backward compatibility
    if args.nresamples is not None:
        logger.warn("The argument '--nresamples' is superceded by '--nruns'")
        args.nruns = args.nresamples + 1

    if args.error_type is not None:
        logger.warn("The argument '--error-type' is superceded by '--resample-data'")
        if args.error_type == 'bootstrap_full':
            args.resample_data = True
            args.resample_everyrun = True
        else:
            args.resample_data = False

    if args.dummy_value is not None:
        logger.warn("The argument '--dummy-value <xx>' is superceded by '--correct-acceptance'")
        args.correct_acceptance = True

    if args.weight_type is not None and args.weight_mc is None:
        args.weight_mc = args.weight_type

    return args

if __name__ == "__main__":

    try:
        args = getArgsParser()
    except Exception as e:
        sys.exit(f"Config Failure: {e}")

    # unfold
    unfold(**vars(args))