#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import tensorflow as tf
import time
import tracemalloc

import cli
from datahandler import DataHandler
from omnifoldwbkg import OmniFoldwBkg
from omnifoldwbkg import OmniFoldwBkg_negW, OmniFoldwBkg_multi
from ibu import IBU
import reweight
from util import read_dict_from_json, get_bins
from util import configGPUs, expandFilePath, configRootLogger
import logging

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

    # weight name
    wname = parsed_args['weight']

    #################
    # Load data
    #################
    logger.info("Loading datasets")
    t_data_start = time.time()

    # collision data
    fnames_obs = parsed_args['data']
    logger.info("Data files: {}".format(' '.join(fnames_obs)))
    vars_obs = vars_det_all+vars_mc_all if parsed_args['truth_known'] else vars_det_all
    data_obs = DataHandler(fnames_obs, wname, variable_names=vars_obs)

    # mix background simulation for testing if needed
    fnames_obsbkg = parsed_args['bdata']
    if fnames_obsbkg is not None:
        logger.info("Background simulation files to be mixed with data: {}".format(' '.join(fnames_obsbkg)))
        data_obsbkg = DataHandler(fnames_obsbkg, wname, variable_names = vars_det_all)
    else:
        data_obsbkg = None

    # signal simulation
    fnames_sig = parsed_args['signal']
    logger.info("Simulation files: {}".format(' '.join(fnames_sig)))
    data_sig = DataHandler(fnames_sig, wname, variable_names = vars_det_all+vars_mc_all)

    # background simulation
    fnames_bkg = parsed_args['background']
    if fnames_bkg is not None:
        logger.info("Background simulation files: {}".format(' '.join(fnames_bkg)))
        data_bkg =  DataHandler(fnames_bkg, wname, variable_names = vars_det_all)
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
        )
    elif parsed_args['background_mode'] == 'negW':
        unfolder = OmniFoldwBkg_negW(
            vars_det_train,
            vars_mc_train,
            iterations=parsed_args["iterations"],
            outdir=parsed_args["outputdir"],
            truth_known=parsed_args["truth_known"],
        )
    elif parsed_args['background_mode'] == 'multiClass':
        unfolder = OmniFoldwBkg_multi(
            vars_det_train,
            vars_mc_train,
            iterations=parsed_args["iterations"],
            outdir=parsed_args["outputdir"],
            truth_known=parsed_args["truth_known"],
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
        unfolder.load(parsed_args['unfolded_weights'])
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
        if True: # doIBU
            array_obs = data_obs[varConfig['branch_det']]
            if data_obsbkg is not None:
                array_obsbkg = data_obsbkg[varConfig['branch_det']]
                array_obs = np.concatenate([array_obs, array_obsbkg])

            array_sim = data_sig[varConfig['branch_det']]
            array_gen = data_sig[varConfig['branch_mc']]
            array_simbkg = data_bkg[varConfig['branch_det']] if data_bkg else None
            ibu = IBU(varname, bins_det, bins_mc,
                      array_obs, array_sim, array_gen, array_simbkg,
                      # use the same weights from OmniFold
                      unfolder.datahandle_obs.get_weights(),
                      unfolder.datahandle_sig.get_weights(),
                      unfolder.datahandle_bkg.get_weights() if unfolder.datahandle_bkg is not None else None,
                      iterations=parsed_args['iterations'], # same as OmniFold
                      nresample=25, #parsed_args['nresamples']
                      outdir = unfolder.outdir)
            ibu.run()
        else:
            ibu = None

        unfolder.plot_distributions_unfold(varname, varConfig, bins_mc, ibu=ibu, iteration_history=parsed_args['plot_history'])

    t_result_done = time.time()
    logger.info("Plotting results took {:.2f} seconds ({:.2f} seconds per variable)".format(t_result_done - t_result_start, (t_result_done - t_result_start)/len(parsed_args['observables']) ))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    tracemalloc.stop()

if __name__ == "__main__":
    args = cli.unfold.parse_args()

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
