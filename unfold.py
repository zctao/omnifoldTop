#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import tensorflow as tf
import time
import tracemalloc

from datahandler import DataHandler
from datahandler_root import DataHandlerROOT
from omnifoldwbkg import OmniFoldwBkg
from omnifoldwbkg import OmniFoldwBkg_negW, OmniFoldwBkg_multi
from ibu import IBU
import reweight
from util import read_dict_from_json, get_bins
from util import configGPUs, expandFilePath, getFilesExtension, configRootLogger
import logging

import metrics

def unfold(**parsed_args):
    tracemalloc.start()

    logger = logging.getLogger('Unfold')

    # log arguments
    for argkey, argvalue in sorted(parsed_args.items()):
        if argvalue is None:
            continue
        logger.info('Argument {}: {}'.format(argkey, argvalue))

    #################
    # Variables
    #################
    # Dictionary for observable configurations
    observable_dict = read_dict_from_json(parsed_args['observable_config'])

    logger.info("Observables used in training: {}".format(', '.join(parsed_args['observables_train'])))
    parsed_args['observables'] = list(set().union(parsed_args['observables'], parsed_args['observables_train']))
    logger.info("Observables to unfold: {}".format(', '.join(parsed_args['observables'])))

    # all variable names at detector level
    vars_det_all = [ observable_dict[key]['branch_det'] for key in parsed_args['observables'] ]
    # all variable names at truth level
    vars_mc_all = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables'] ]

    # detector-level variable names for training
    vars_det_train = [ observable_dict[key]['branch_det'] for key in parsed_args['observables_train'] ]
    # truth-level variable names for training
    vars_mc_train = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables_train'] ] 

    #################
    # Load data
    #################
    def loadData(file_names, reco_only=False):
        # use the proper data handler based on the type of input files
        if getFilesExtension(file_names) == '.root':
            # ROOT files
            # hard code tree names here for now
            tree_reco = 'reco'
            tree_mc = None if reco_only else 'parton'

            if parsed_args['dummy_value'] is None:
                logger.info("Load events that are truth matched and pass both reco and truth level selections")
            else:
                logger.info("Set variables to a dummy value {} for events that fail reco or truth level selections".format(parsed_args['dummy_value']))

            dh = DataHandlerROOT(
                file_names, vars_det_all, vars_mc_all,
                treename_reco = tree_reco, treename_truth = tree_mc,
                dummy_value=parsed_args['dummy_value']
                )
        else:
            # '.npz'
            wname = 'totalWeight_nominal'
            varnames_truth = [] if reco_only else vars_mc_all
            dh = DataHandler(file_names, vars_det_all, varnames_truth, wname)

        return dh

    logger.info("Loading datasets")
    t_data_start = time.time()

    # collision data
    fnames_obs = parsed_args['data']
    logger.info("Data files: {}".format(' '.join(fnames_obs)))
    data_obs = loadData(fnames_obs, reco_only = not parsed_args['truth_known'])

    # mix background simulation for testing if needed
    fnames_obsbkg = parsed_args['bdata']
    if fnames_obsbkg is not None:
        logger.info("Background simulation files to be mixed with data: {}".format(' '.join(fnames_obsbkg)))
        data_obsbkg = loadData(fnames_obsbkg, reco_only=True)
    else:
        data_obsbkg = None

    # signal simulation
    fnames_sig = parsed_args['signal']
    logger.info("Simulation files: {}".format(' '.join(fnames_sig)))
    data_sig = loadData(fnames_sig, reco_only=False)

    # background simulation
    fnames_bkg = parsed_args['background']
    if fnames_bkg is not None:
        logger.info("Background simulation files: {}".format(' '.join(fnames_bkg)))
        data_bkg = loadData(fnames_bkg, reco_only=True)
    else:
        data_bkg = None

    t_data_done = time.time()
    logger.info("Loading dataset took {:.2f} seconds".format(t_data_done-t_data_start))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    #################
    # Unfold
    #################
    if parsed_args['background_mode'] == 'default':
        unfolder = OmniFoldwBkg(
            vars_det_train,
            vars_mc_train,
            iterations=parsed_args["iterations"],
            outdir=parsed_args["outputdir"],
            truth_known=parsed_args["truth_known"],
            model_name=parsed_args['model_name']
        )
    elif parsed_args['background_mode'] == 'negW':
        unfolder = OmniFoldwBkg_negW(
            vars_det_train,
            vars_mc_train,
            iterations=parsed_args["iterations"],
            outdir=parsed_args["outputdir"],
            truth_known=parsed_args["truth_known"],
            model_name=parsed_args['model_name']
        )
    elif parsed_args['background_mode'] == 'multiClass':
        unfolder = OmniFoldwBkg_multi(
            vars_det_train,
            vars_mc_train,
            iterations=parsed_args["iterations"],
            outdir=parsed_args["outputdir"],
            truth_known=parsed_args["truth_known"],
            model_name=parsed_args['model_name']
        )
    else:
        logger.error("Unknown background mode {}".format(parsed_args['background_mode']))

    rw = None
    if parsed_args["reweight_data"]:
        var_lookup = np.vectorize(lambda v: observable_dict[v]["branch_mc"])
        rw = reweight.rw[parsed_args["reweight_data"]]
        rw.variables = var_lookup(rw.variables)

    # prepare input data
    logger.info("Prepare data")
    t_prep_start = time.time()

    unfolder.prepare_inputs(
        data_obs,
        data_sig,
        data_bkg,
        data_obsbkg,
        parsed_args["plot_correlations"],
        reweighter=rw,
    )

    t_prep_done = time.time()
    logger.info("Preparing data took {:.2f} seconds".format(t_prep_done - t_prep_start))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    # start unfolding
    logger.info("Start unfolding")
    t_unfold_start = time.time()

    if parsed_args['unfolded_weights']:
        # load unfolded event weights from the saved files
        unfolder.load(parsed_args['unfolded_weights'],
                      legacy_mode=parsed_args['legacy_weights'])
    else:
        # set up hardware
        configGPUs(parsed_args['gpu'], verbose=parsed_args['verbose'])

        # run training
        unfolder.run(parsed_args['error_type'], parsed_args['nresamples'],
                     load_previous_iteration=False,
                     load_models_from=parsed_args['load_models'],
                     batch_size=parsed_args['batch_size'])

    t_unfold_done = time.time()
    logger.info("Done!")
    logger.info("Unfolding took {:.2f} seconds".format(t_unfold_done - t_unfold_start))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    #################
    # Show results
    #################
    t_result_start = time.time()

    for varname in parsed_args['observables']:
        logger.info("Unfold variable: {}".format(varname))
        varConfig = observable_dict[varname]

        # detector-level distributions
        bins_det = get_bins(varname, parsed_args['binning_config'])
        if bins_det is None:
            bins_det = np.linspace(varConfig['xlim'][0], varConfig['xlim'][1], varConfig['nbins_det']+1)

        unfolder.plot_distributions_reco(varname, varConfig, bins_det)

        # truth-level distributions
        bins_mc = get_bins(varname, parsed_args['binning_config'])
        if bins_mc is None:
            bins_mc = np.linspace(varConfig['xlim'][0], varConfig['xlim'][1], varConfig['nbins_mc']+1)

        # iterative Bayesian unfolding
        if parsed_args['run_ibu']:
            # data
            array_obs = data_obs[varConfig['branch_det']]
            w_obs = data_obs.get_weights()
            if data_obsbkg is not None:
                array_obsbkg = data_obsbkg[varConfig['branch_det']]
                w_obsbkg = data_obsbkg.get_weights()
                array_obs = np.concatenate([array_obs, array_obsbkg])
                w_obs = np.concatenate([w_obs, w_obsbkg])

            # background simulation if needed
            array_simbkg = data_bkg[varConfig['branch_det']] if data_bkg else None
            w_bkg = data_bkg.get_weights() if data_bkg else None

            # signal simulation
            # only truth matched events
            reco_match_truth = data_sig.pass_truth[data_sig.pass_reco]
            array_sim = data_sig[varConfig['branch_det']][reco_match_truth]
            w_sim = data_sig.get_weights()[reco_match_truth]

            truth_match_reco = data_sig.pass_reco[data_sig.pass_truth]
            array_gen = data_sig[varConfig['branch_mc']][truth_match_reco]
            w_gen = data_sig.get_weights(reco_level=False)[truth_match_reco]

            ibu = IBU(varname, bins_det, bins_mc,
                      array_obs, array_sim, array_gen, array_simbkg,
                      w_obs, w_sim, w_gen, w_bkg,
                      iterations=parsed_args['iterations'], # same as OmniFold
                      nresample=25, #parsed_args['nresamples']
                      outdir = unfolder.outdir)
            ibu.run()
        else:
            ibu = None

        unfolder.plot_distributions_unfold(varname, varConfig, bins_mc, ibu=ibu, iteration_history=parsed_args['plot_history'])

        logger.info("  Evaluate metrics")
        metrics.evaluate_all_metrics(varname, varConfig, bins_mc, unfolder, ibu)

    t_result_done = time.time()
    logger.info("Plotting results took {:.2f} seconds ({:.2f} seconds per variable)".format(t_result_done - t_result_start, (t_result_done - t_result_start)/len(parsed_args['observables']) ))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    tracemalloc.stop()

def getArgsParser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--observables-train', dest='observables_train',
                        nargs='+',
                        default=['th_pt', 'th_y', 'th_phi', 'th_e', 'tl_pt', 'tl_y', 'tl_phi', 'tl_e'],
                        help="List of observables to use in training.")
    parser.add_argument('--observables',
                        nargs='+',
                        default=['mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht', 'th_pt', 'th_y', 'th_eta', 'th_phi', 'th_m', 'th_e', 'th_pout', 'tl_pt', 'tl_y', 'tl_eta', 'tl_phi', 'tl_m', 'tl_e', 'tl_pout'],
                        help="List of observables to unfold")
    parser.add_argument('--observable-config', dest='observable_config',
                        default='configs/observables/vars_ttbardiffXs.json',
                        help="JSON configurations for observables")
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
    parser.add_argument('-o', '--outputdir',
                        default='./output',
                        help="Directory for storing outputs")
    parser.add_argument('-t', '--truth-known', dest='truth_known',
                        action='store_true',
                        help="MC truth is known for 'data' sample")
    parser.add_argument('-c', '--plot-correlations', dest='plot_correlations',
                        action='store_true',
                        help="Plot pairwise correlations of training variables")
    parser.add_argument('-i', '--iterations', type=int, default=4,
                        help="Numbers of iterations for unfolding")
    parser.add_argument('-m', '--background-mode', dest='background_mode',
                        choices=['default', 'negW', 'multiClass'],
                        default='default', help="Background mode")
    parser.add_argument('-r', '--reweight-data', dest='reweight_data',
                        choices=reweight.rw.keys(), default=None,
                        help="Reweight strategy of the input spectrum for stress tests. Requires --truth-known.")
    parser.add_argument('-v', '--verbose',
                        action='count', default=0,
                        help="Verbosity level")
    parser.add_argument('-g', '--gpu',
                        type=int, choices=[0, 1], default=None,
                        help="Manually select one of the GPUs to run")
    parser.add_argument('--unfolded-weights', dest='unfolded_weights',
                        nargs='*', type=str,
                        help="Unfolded weights file names. If provided, load event weights directly from the files and skip training.")
    parser.add_argument('--binning-config', dest='binning_config',
                        default='configs/binning/bins_10equal.json', type=str,
                        help="Binning config file for variables")
    parser.add_argument('--plot-history', dest='plot_history',
                        action='store_true',
                        help="If true, plot intermediate steps of unfolding")
    parser.add_argument('--nresamples', type=int, default=25,
                        help="number of times for resampling to estimate the unfolding uncertainty using the bootstrap method")
    parser.add_argument('-e', '--error-type', dest='error_type',
                        choices=['sumw2','bootstrap_full','bootstrap_model'],
                        default='sumw2', help="Method to evaluate uncertainties")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16384,
                        help="Batch size for training")
    parser.add_argument('-l', '--load-models', dest='load_models', type=str,
                        help="Directory from where to load trained models. If provided, training will be skipped.")
    parser.add_argument('--model-name', type=str, default='dense_100x3',
                        help="Name of the model for unfolding")
    parser.add_argument('--legacy-weights', action='store_true',
                        help="If True, load weights in the legacy mode. The unfolded weights read from files are divided by the simulation prior weights. Only useful when --unfolded-weights is not None.")
    parser.add_argument('--run-ibu', action='store_true',
                        help="If True, run unfolding using IBU for comparison")
    parser.add_argument('--dummy-value', type=float,
                        help="Dummy value to fill events that failed selecton. If None (default), only events that pass reco and truth (if apply) level selections are used for unfolding")

    #parser.add_argument('-n', '--normalize',
    #                    action='store_true',
    #                    help="Normalize the distributions when plotting the result")
    #parser.add_argument('--alt-rw', dest='alt_rw',
    #                    action='store_true',
    #                    help="Use alternative reweighting if true")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = getArgsParser()

    # Verify truth is known when reweighting
    if args.reweight_data is not None and not args.truth_known:
        print("--reweight-data requires --truth-known", file=sys.stderr)
        sys.exit(2)

    logfile = os.path.join(args.outputdir, 'log.txt')
    configRootLogger(filename=logfile)
    logger = logging.getLogger('Unfold')
    logger.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)

    #################
    if not os.path.isdir(args.outputdir):
        logger.info("Create output directory {}".format(args.outputdir))
        os.makedirs(args.outputdir)

    # check if configuraiton files exist and expand the file path
    fullpath_obsconfig = expandFilePath(args.observable_config)
    if fullpath_obsconfig is not None:
        args.observable_config = fullpath_obsconfig
    else:
        logger.error("Cannot find file: {}".format(args.observable_config))
        sys.exit("Config Failure")

    fullpath_binconfig = expandFilePath(args.binning_config)
    if fullpath_binconfig is not None:
        args.binning_config = fullpath_binconfig
    else:
        logger.error("Cannot find file: {}".format(args.binning_config))
        sys.exit("Config Failure")

    #with tf.device('/GPU:{}'.format(args.gpu)):
    unfold(**vars(args))
