#!/usr/bin/env python3
import os
import sys
import time
import tracemalloc
import logging
import numpy as np

import util
import plotter
import reweight
import metrics
from OmniFoldTTbar import OmniFoldTTbar
from ibuv2 import run_ibu

def unfold(**parsed_args):
    tracemalloc.start()

    logger = logging.getLogger('Unfold')

    # Print arguments to logger
    for argkey, argvalue in sorted(parsed_args.items()):
        if argvalue is None:
            continue
        logger.info(f"Argument {argkey}: {argvalue}")

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
            batch_size = parsed_args['batch_size'],
            plot_status = parsed_args['plot_verbosity'] >= 2
        )

    t_run_done = time.time()
    logger.info("Done!")
    logger.info(f"Unfolding took {t_run_done-t_run_start:.2f} seconds")

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*10**-6:.1f} MB; Peak usage: {mpeak*10**-6:.1f} MB")

    #################
    # Plot results
    #################
    if parsed_args['plot_verbosity'] == 0:
        # skip plotting
        tracemalloc.stop()
        return

    t_result_start = time.time()

    # Input variable correlations
    if parsed_args['plot_verbosity'] >= 3: # '-ppp'
        logger.info(f"Plot input variable correlations")
        corr_data = unfolder.handle_obs.get_correlations(varnames_train_reco)
        plotter.plot_correlations(
            os.path.join(unfolder.outdir, "InputCorr_data"), corr_data
        )

        corr_sim = unfolder.handle_sig.get_correlations(varnames_train_reco)
        plotter.plot_correlations(
            os.path.join(unfolder.outdir, "InputCorr_sim"), corr_sim
        )

        corr_gen = unfolder.handle_sig.get_correlations(varnames_train_truth)
        plotter.plot_correlations(
            os.path.join(unfolder.outdir, "InputCorr_gen"), corr_gen
        )

    # for each observable
    for observable in set().union(parsed_args['observables'], parsed_args['observables_extra']):
        logger.info(f"Unfold variable: {observable}")
        varname_reco = observable_dict[observable]['branch_det']
        varname_truth = observable_dict[observable]['branch_mc']

        # get bins
        # TODO different binning at reco and truth level?
        bins_det = util.get_bins(observable, parsed_args['binning_config'])
        bins_mc = util.get_bins(observable, parsed_args['binning_config'])

        ####
        # -p 
        # Truth level
        # unfolded distribution
        h_uf, h_uf_corr = unfolder.get_unfolded_distribution(varname_truth, bins_mc, normalize=True) # TODO check this
        # normalize to the prior weightst for now, normalize to data?

        # prior distribution
        h_gen = unfolder.handle_sig.get_histogram(varname_truth, bins_mc)

        # truth distribution if known
        if parsed_args['truth_known']:
            h_truth = unfolder.handle_obs.get_histogram(varname_truth, bins_mc)
        else:
            h_truth = None

        ####
        # IBU
        if parsed_args['run_ibu']:
            # data array
            array_obs = unfolder.handle_obs[varname_reco]
            wobs = unfolder.handle_obs.get_weights()
            if unfolder.handle_obsbkg is not None:
                array_obsbkg = unfolder.handle_obsbkg[varname_reco]
                wobsbkg = unfolder.handle_obsbkg.get_weights()
                array_obs = np.concatenate([array_obs, array_obsbkg])
                wobs = np.concatenate([wobs, wobsbkg])

            # simulation
            array_sim = unfolder.handle_sig[varname_reco]
            wsim = unfolder.handle_sig.get_weights()

            array_gen = unfolder.handle_sig[varname_truth]
            wgen = unfolder.handle_sig.get_weights(reco_level=False)

            if unfolder.handle_bkg is not None:
                array_bkg = unfolder.handle_bkg[varname_reco]
                wbkg = unfolder.handle_bkg.get_weights()
            else:
                array_bkg, wbkg = None, None

            hists_ibu_alliters, h_ibu_corr, response = run_ibu(
                bins_det, bins_mc,
                array_obs, array_sim, array_gen, array_bkg,
                wobs, wsim, wgen, wbkg,
                niterations = parsed_args['iterations'],
                all_iterations=True)
            # hists_ibu_alliters have the same norm as wobs (therefore wsim)
            # TODO: check this. rescale to the prior truth weights for now
            hists_ibu_alliters = [h * wgen.sum() / wobs.sum() for h in hists_ibu_alliters]

            # plot response
            figname_resp = os.path.join(unfolder.outdir, f"Response_{observable}")
            logger.info(f"  Plot detector response: {figname_resp}")
            plotter.plot_response(figname_resp, response, observable)

            h_ibu = hists_ibu_alliters[-1]
            h_ibu_corr = h_ibu_corr[-1]
        else:
            h_ibu = None
            hists_ibu_alliters = None
            h_ibu_corr = None
        ####

        # print metrics on the plot
        texts_chi2 = []
        if parsed_args['plot_verbosity'] >= 2: # -pp
            texts_chi2 = metrics.write_texts_Chi2(
                h_truth, [h_uf, h_ibu, h_gen],
                labels=['MultiFold', 'IBU', 'Prior']
            )

        # plot
        figname_uf = os.path.join(unfolder.outdir, f"Unfold_{observable}")
        logger.info(f"  Plot unfolded distribution: {figname_uf}")
        plotter.plot_distributions_unfold(
            figname_uf, h_uf, h_gen, h_truth, h_ibu,
            xlabel = observable_dict[observable]['xlabel'],
            ylabel = observable_dict[observable]['ylabel'],
            legend_loc = observable_dict[observable]['legend_loc'],
            legend_ncol = observable_dict[observable]['legend_ncol'],
            stamp_loc = observable_dict[observable]['stamp_xy'],
            stamp_texts = texts_chi2
        )

        ####
        # More plots if -pp
        if parsed_args['plot_verbosity'] < 2:
            continue

        ###
        # Reco level distribution
        # observed data
        h_data = unfolder.handle_obs.get_histogram(varname_reco, bins_det)
        if unfolder.handle_obsbkg is not None:
            h_data += unfolder.handle_obsbkg.get_histogram(varname_reco, bins_det)

        # signal simulation
        h_sig = unfolder.handle_sig.get_histogram(varname_reco, bins_det)

        # background simulation if available
        if unfolder.handle_bkg is not None:
            h_bkg = unfolder.handle_bkg.get_histogram(varname_reco, bins_det)
        else:
            h_bkg = None

        figname_reco = os.path.join(unfolder.outdir, f"Reco_{observable}")
        logger.info(f"  Plot detector-level distribution: {figname_reco}")
        plotter.plot_distributions_reco(
            figname_reco, h_data, h_sig, h_bkg,
            xlabel = observable_dict[observable]['xlabel'],
            ylabel = observable_dict[observable]['ylabel'],
            legend_loc = observable_dict[observable]['legend_loc'],
            legend_ncol = observable_dict[observable]['legend_ncol']
        )

        ###
        # Unfolded distribution bin correlations
        if h_uf_corr is not None:
            figname_uf_corr = os.path.join(unfolder.outdir, f"BinCorr_{observable}_OmniFold")
            logger.info(f"  Plot bin correlations: {figname_uf_corr}")
            plotter.plot_correlations(figname_uf_corr, h_uf_corr, bins_mc)

        if h_ibu_corr is not None:
            figname_ibu_corr = os.path.join(unfolder.outdir, f"BinCorr_{observable}_IBU")
            logger.info(f"  Plot bin correlations: {figname_ibu_corr}")
            plotter.plot_correlations(figname_ibu_corr, h_ibu_corr, bins_mc)

        ## Metrics
        if parsed_args['truth_known']:
            mdict = dict()
            mdict[observable] = metrics.evaluate_all_metrics(
                unfolder, varname_truth, bins_mc, hists_ibu_alliters
            )

            metrics_dir = os.path.join(unfolder.outdir, 'Metrics')
            if not os.path.isdir(metrics_dir):
                os.makedirs(metrics_dir)

            # write to json file
            util.write_dict_to_json(mdict, metrics_dir+f"/{observable}.json")

            # make plots
            metrics.plot_all_metrics(
                mdict[observable], metrics_dir+f"/{observable}"
            )

        ####
        # Even more plots if -ppp
        if parsed_args['plot_verbosity'] < 3:
            continue

        ## Resamples
        hists_uf_resample = unfolder.get_unfolded_hists_resamples(
            varname_truth, bins_mc, normalize=False, all_iterations=True)

        if len(hists_uf_resample) > 0:
            resample_dir = os.path.join(unfolder.outdir, 'Resamples')
            if not os.path.isdir(resample_dir):
                logger.info(f"Create directory {resample_dir}")
                os.makedirs(resample_dir)

            # all unfolded distributions from resampling
            figname_rs = os.path.join(resample_dir, f"Unfold_AllResamples_{observable}")
            logger.info(f"  Plot unfolded distributions from all resamples: {figname_rs}")
            plotter.plot_distributions_resamples(
                figname_rs, hists_uf_resample[-1], h_gen, h_truth,
                xlabel = observable_dict[observable]['xlabel'],
                ylabel = observable_dict[observable]['ylabel'])

            # distributions of bin entries
            figname_bindistr = os.path.join(resample_dir, f"Unfold_BinDistr_{observable}")
            logger.info(f"  Plot distributions of bin entries from all resamples: {figname_bindistr}")
            # For now
            plotter.plot_hists_bin_distr(figname_bindistr, hists_uf_resample, h_truth)

        ## Iteration history
        iteration_dir = os.path.join(unfolder.outdir, 'Iterations')
        if not os.path.isdir(iteration_dir):
            logger.info(f"Create directory {iteration_dir}")
            os.makedirs(iteration_dir)

        hists_uf_alliters = unfolder.get_unfolded_distribution(varname_truth, bins_mc, normalize=True, all_iterations=True)[0]

        figname_alliters = os.path.join(iteration_dir, f"Unfold_AllIterations_{observable}")
        logger.info(f"  Plot unfolded distributions at every iteration: {figname_alliters}")
        plotter.plot_distributions_iteration(
            figname_alliters, hists_uf_alliters, h_gen, h_truth,
            xlabel = observable_dict[observable]['xlabel'],
            ylabel = observable_dict[observable]['ylabel'])

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
    parser.add_argument('--observables-extra', nargs='*', default=[],
                        #default=['mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht', 'th_eta', 'th_m', 'th_pout', 'tl_eta', 'tl_m', 'tl_pout'],
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
    parser.add_argument('--unfolded-weights',
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
    parser.add_argument('-p', '--plot-verbosity', action='count', default=0,
                        help="Plot verbose level. '-ppp' to make all plots.")
    parser.add_argument('--run-ibu', action='store_true',
                        help="If True, run unfolding also with IBU for comparison")

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
