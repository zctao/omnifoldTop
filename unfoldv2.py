#!/usr/bin/env python3
import os
import sys
import time
import tracemalloc
import logging

import util
import plotting
import reweight
from omnifoldwbkgv2 import OmniFoldTTbar

def unfold(**parsed_args):
    tracemalloc.start()

    logger = logging.getLogger('Unfold')

    # Print arguments to logger
    for argkey, argvalue in sorted(parsed_args.items()):
        if argvalue is None:
            continue
        logger.debug(f"Argument {argkey}: {argvalue}")

    #################
    # Variables
    #################
    # Dictionary for observable configurations
    observable_dict = util.read_dict_from_json(parsed_args['observable_config'])

    logger.info("Observables used in training: {}".format(' '.join(parsed_args['observables'])))
    varnames_train_reco = [ observable_dict[key]['branch_det'] for key in parsed_args['observables'] ]
    varnames_train_truth = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables'] ]

    if parsed_args['observables_extra']:
        logger.info("Extra observables to unfold: {}".format(' '.join(parsed_args['observables_extra'])))
        varnames_extra_reco = [ observable_dict[key]['branch_det'] for key in parsed_args['observables_extra'] ]
        varnames_extra_truth = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables_extra'] ]
    else:
        varnames_extra_reco = []
        varnames_extra_truth = []

    #################
    # Initialize and load data
    #################

    # reweights
    rw = None
    if parsed_args["reweight_data"]:
        var_lookup = np.vectorize(lambda v: observable_dict[v]["branch_mc"])
        rw = reweight.rw[parsed_args["reweight_data"]]
        rw.variables = var_lookup(rw.variables)

    t_init_start = time.time()
        
    unfolder = OmniFoldTTbar(
        varnames_train_reco,
        varnames_train_truth,
        parsed_args['data'],
        parsed_args['signal'],
        parsed_args['background'],
        parsed_args['bdata'],
        truth_known = parsed_args['truth_known'],
        normalize_to_data = parsed_args['normalize'],
        variables_reco_extra = varnames_extra_reco,
        variables_truth_extra = varnames_extra_truth,
        dummy_value = parsed_args['dummy_value'],
        outputdir = parsed_args["outputdir"],
        data_reweighter = rw
        )

    t_init_done = time.time()
    logger.debug(f"Initializing unfolder and loading input data took {(t_init_done-t_init_start):.2f} seconds.")
    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*10**-6:.1f} MB; Peak usage: {mpeak*10**-6:.1f} MB")

    #################
    # Run unfolding
    #################

    logger.info("Start unfolding ... ")
    t_run_start = time.time()

    if parsed_args['unfolded_weights']:
        # load unfolded event weights from files
        unfolder.load(parsed_args['unfolded_weights'])
    else:
        # configure GPU
        util.configGPUs(parsed_args['gpu'], verbose=parsed_args['verbose'])

        # run unfolding
        unfolder.run(
            niterations = parsed_args['iterations'],
            error_type = parsed_args['error_type'],
            nresamples = parsed_args['nresamples'],
            model_type = parsed_args['model_name'],
            save_models = True,
            load_previous_iteration = True, # TODO check here
            load_models_from = parsed_args['load_models'],
            batch_size = parsed_args['batch_size']
        )

    t_run_done = time.time()
    logger.info("Done!")
    logger.info(f"Unfolding took {t_run_done-t_run_start:.2f} seconds")

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*10**-6:.1f} MB; Peak usage: {mpeak*10**-6:.1f} MB")

    #################
    # Show results
    #################
    t_result_start = time.time()

    for observable in set().union(parsed_args['observables'], parsed_args['observables_extra']):
        logger.info(f"Unfold variable: {observable}")
        varname_reco = observable_dict[observable]['branch_det']
        varname_truth = observable_dict[observable]['branch_mc']

        # get bins
        # TODO different binning at reco and truth level?
        bins_det = util.get_bins(observable, parsed_args['binning_config'])
        bins_mc = util.get_bins(observable, parsed_args['binning_config'])

        # reco level distribution for reference
        # TODO

        # unfolded distribution
        h_uf, h_corr = unfolder.get_unfolded_distribution(varname_truth, bins_mc, normalize=True) # TODO check this

        # prior distribution
        h_gen = unfolder.handle_sig.get_histogram(varname_truth, bins_mc)

        # truth distribution if known
        if parsed_args['truth_known']:
            h_truth = unfolder.handle_obs.get_histogram(varname_truth, bins_mc)
        else:
            h_truth = None

        # IBU: TODO
        h_ibu = None

        # plot
        figname = os.path.join(unfolder.outdir, f"Unfold_{observable}")
        logger.info(f"  Plot unfolded distribution: {figname}")
        # For now
        plotting.plot_results(
            h_gen, h_uf, h_ibu, h_truth, figname=figname,
            **observable_dict[observable]
            )
        

    tracemalloc.stop()

def getArgsParser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', required=True, nargs='+',
                        type=str,
                        help="Observed data npz file names")
    parser.add_argument('-s', '--signal', required=True, nargs='+',
                        type=str,
                        help="Signal MC npz file names")
    parser.add_argument('-b', '--background', nargs='+',
                        type=str,
                        help="Background MC npz file names")
    parser.add_argument('--bdata', nargs='+', type=str, default=None,
                        help="Background MC files to be mixed with data")
    parser.add_argument('--observables', nargs='+',
                        default=['th_pt', 'th_y', 'th_phi', 'th_e', 'tl_pt', 'tl_y', 'tl_phi', 'tl_e'],
                        help="List of observables to use in training.")
    parser.add_argument('--observables-extra', nargs='+',
                        default=['mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht', 'th_eta', 'th_m', 'th_pout', 'tl_eta', 'tl_m', 'tl_pout'],
                        help="List of extra observables to unfold.")
    parser.add_argument('--observable-config',
                        default='configs/observables/vars_ttbardiffXs.json',
                        help="JSON configurations for observables")
    parser.add_argument('-o', '--outputdir', type=str, default='.',
                        help="Directory for storing outputs")
    parser.add_argument('-t', '--truth-known',
                        action='store_true',
                        help="MC truth is known for 'data' sample")
    parser.add_argument('-r', '--reweight-data',
                        choices=reweight.rw.keys(), default=None,
                        help="Reweight strategy of the input spectrum for stress tests. Requires --truth-known.")
    parser.add_argument('-w', '--unfolded-weights',
                        nargs='*', type=str,
                        help="Unfolded weights file names. If provided, load event weights directly from the files and skip training.")
    parser.add_argument('-v', '--verbose',
                        action='count', default=0,
                        help="Verbosity level")
    parser.add_argument('-g', '--gpu',
                        type=int, default=None,
                        help="Manually select one of the GPUs to run")
    parser.add_argument('-i', '--iterations', type=int, default=4,
                        help="Numbers of iterations for unfolding")
    parser.add_argument('-e', '--error-type',
                        choices=['sumw2','bootstrap_full','bootstrap_model'],
                        default='sumw2', help="Method to evaluate uncertainties")
    parser.add_argument('--nresamples', type=int, default=25,
                        help="number of times for resampling to estimate the unfolding uncertainty using the bootstrap method")
    parser.add_argument('-m', '--model-name',
                        type=str, default='dense_100x3',
                        help="Name of the model for unfolding")
    parser.add_argument('-n', '--normalize',
                        action='store_true',
                        help="If True, normalize simulation weights to data")
    parser.add_argument('--batch-size', type=int, default=16384,
                        help="Batch size for training")
    parser.add_argument('-l', '--load-models', type=str,
                        help="Directory from where to load trained models. If provided, training will be skipped.")
    parser.add_argument('--dummy-value', type=float,
                        help="Dummy value to fill events that failed selecton. If None (default), only events that pass reco and truth (if apply) level selections are used for unfolding")
    parser.add_argument('--binning-config', dest='binning_config',
                        default='configs/binning/bins_10equal.json', type=str,
                        help="Binning config file for variables")



    parser.add_argument('-c', '--plot-correlations',
                        action='store_true',
                        help="Plot pairwise correlations of training variables")
    parser.add_argument('--plot-history', dest='plot_history',
                        action='store_true',
                        help="If true, plot intermediate steps of unfolding")
    parser.add_argument('--run-ibu', action='store_true',
                        help="If True, run unfolding using IBU for comparison")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = getArgsParser()

    # Verify truth is known when reweighting
    if args.reweight_data is not None and not args.truth_known:
        print("--reweight-data requires --truth-known", file=sys.stderr)
        sys.exit(2)

    # logging
    logfile = os.path.join(args.outputdir, 'log.txt')
    util.configRootLogger(filename=logfile)
    logger = logging.getLogger('Unfold')
    logger.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)

    # output directory
    if args.outputdir is not None:
        if not os.path.isdir(args.outputdir):
            logger.info(f"Create output directory {args.outputdir}")
            os.makedirs(args.outputdir)

    # check if configuration files exist and expand the file path
    # TODO move these to the reader methods?
    fullpath_obsconfig = util.expandFilePath(args.observable_config)
    if fullpath_obsconfig is not None:
        args.observable_config = fullpath_obsconfig
    else:
        logger.error("Cannot find file: {}".format(args.observable_config))
        sys.exit("Config Failure")

    fullpath_binconfig = util.expandFilePath(args.binning_config)
    if fullpath_binconfig is not None:
        args.binning_config = fullpath_binconfig
    else:
        logger.error("Cannot find file: {}".format(args.binning_config))
        sys.exit("Config Failure")

    # unfold
    unfold(**vars(args))
