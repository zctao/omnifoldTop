#!/usr/bin/env python3
import os
import time
import tracemalloc
import numpy as np

import util
import metrics
import plotter
import histogramming as myhu
from OmniFoldTTbar import load_unfolder
from ibuv2 import run_ibu

import logging
logger = logging.getLogger("make_histograms")

def get_ibu_unfolded_histogram_from_unfolder(
    unfolder,
    vname_reco,
    vname_truth,
    bins_reco,
    bins_truth,
    niterations = None, # number of iterations
    all_iterations = False, # if True, return results at every iteration
    norm = None
    ):

    # Prepare inputs
    # data array
    array_obs = unfolder.handle_obs[vname_reco]
    wobs = unfolder.handle_obs.get_weights()

    if unfolder.handle_obsbkg is not None:
        array_obsbkg = unfolder.handle_obsbkg[vname_reco]
        wobsbkg = unfolder.handle_obsbkg.get_weights()
        array_obs = np.concatenate([array_obs, array_obsbkg])
        wobs = np.concatenate([wobs, wobsbkg])

    # simulation
    array_sim = unfolder.handle_sig[vname_reco]
    wsim = unfolder.handle_sig.get_weights()

    array_gen = unfolder.handle_sig[vname_truth]
    wgen = unfolder.handle_sig.get_weights(reco_level=False)

    if unfolder.handle_bkg is not None:
        array_bkg = unfolder.handle_bkg[vname_reco]
        wbkg = unfolder.handle_bkg.get_weights()
    else:
        array_bkg, wbkg = None, None

    ###
    # run IBU
    if niterations is None:
        # run IBU the same iterations as OmniFold
        niterations = unfolder.unfolded_weights.shape[1]

    hists_ibu, h_ibu_corr, reponse = run_ibu(
        bins_reco, bins_truth,
        array_obs, array_sim, array_gen, array_bkg,
        wobs, wsim, wgen, wbkg,
        niterations = niterations,
        all_iterations = all_iterations,
        norm = norm
        )

    return hists_ibu, h_ibu_corr, reponse

def make_histograms_of_observable(
    unfolder,
    observable, # str, name of the observable
    obsConfig_d, # dict, observable configuration
    binning_config, # str, path to binning config file
    iteration = -1, # int, which iteration to use as the nominal. Default is the last one
    nruns = None, # int, number of runs. Default is to take all that are available
    absoluteValue = False, # If True, use the absolute value to fill the histogram
    normalize = True, # If True, rescale all truth-level histograms to the same normalization
    all_runs = True, # If True, include unfolded histograms of all runs at specified iteration
    all_iterations = False, # If True, include nominal unfolded histograms at all iterations
    all_histograms = False, # If True, include also histograms of every run and every iteration
    include_ibu = False, # If True, include also IBU for comparison
    include_reco = False # If True, include also reco level histograms
    ):

    hists_v_d = {}

    varname_truth = obsConfig_d[observable]['branch_mc']

    # get bin edges
    bins_mc = util.get_bins(observable, binning_config)

    ###
    # The unfolded distributions
    logger.debug(f" Unfolded distributions")
    h_uf, h_uf_corr = unfolder.get_unfolded_distribution(
        varname_truth,
        bins_mc,
        iteration = iteration,
        nresamples = nruns,
        absoluteValue = absoluteValue
        )

    norm_uf = h_uf.sum(flow=True)['value'] if normalize else None

    # set x-axis label
    h_uf.axes[0].label = obsConfig_d[observable]['xlabel']

    hists_v_d['unfolded'] = h_uf
    hists_v_d['unfolded_corr'] = h_uf_corr

    if all_runs:
        # unfolded histograms from every run
        hists_v_d['unfolded_allruns'] = unfolder.get_unfolded_hists_resamples(
            varname_truth,
            bins_mc,
            iteration = iteration,
            nresamples = nruns,
            absoluteValue = absoluteValue
            )

    if all_iterations:
        # unfoldedd histograms at every iteration
        hists_v_d['unfolded_alliters'] = unfolder.get_unfolded_distribution(
            varname_truth,
            bins_mc,
            nresamples = nruns,
            all_iterations=True,
            absoluteValue = absoluteValue
            )[0]

    if all_histograms:
        hists_v_d['unfolded_all'] = unfolder.get_unfolded_hists_resamples(
            varname_truth,
            bins_mc,
            nresamples = nruns,
            all_iterations=True,
            absoluteValue = absoluteValue
            )

    ###
    # Other truth-level distributions
    ##
    # Prior
    logger.debug(f" Prior distribution")
    h_gen = unfolder.handle_sig.get_histogram(
        varname_truth, bins_mc, absoluteValue=absoluteValue)

    if norm_uf:
        h_gen *= (norm_uf/h_gen.sum(flow=True)['value'])

    hists_v_d['prior'] = h_gen

    ##
    # truth distribution if using pseudo data
    if unfolder.handle_obs.data_truth is not None:
        logger.debug(f" Truth distribution")
        h_truth = unfolder.handle_obs.get_histogram(
            varname_truth, bins_mc, absoluteValue=absoluteValue)

        if norm_uf:
            h_truth *= (norm_uf/h_truth.sum(flow=True)['value'])

        hists_v_d['truth'] = h_truth

    ##
    # IBU
    if include_ibu:
        logger.info(f" Run IBU for {observable}")

        varname_reco = obsConfig_d[observable]['branch_det']

        # TODO different binning at reco and truth level
        bins_det = util.get_bins(observable, binning_config)

        hists_ibu_alliters, h_ibu_corr, response = get_ibu_unfolded_histogram_from_unfolder(
            unfolder,
            varname_reco, varname_truth,
            bins_det, bins_mc,
            all_iterations = True,
            norm = norm_uf
        )

        # take the ones at the same iteration as OmniFold
        h_ibu = hists_ibu_alliters[iteration]
        h_ibu_corr = h_ibu_corr[iteration]

        hists_v_d['ibu'] = h_ibu
        hists_v_d['ibu_alliters'] = hists_ibu_alliters
        hists_v_d['ibu_corr'] = h_ibu_corr

    ###
    # Reco level
    if include_reco:
        logger.debug(f" Reco-level distributions")
        varname_reco = obsConfig_d[observable]['branch_det']

        # TODO different binning at reco and truth level
        bins_det = util.get_bins(observable, binning_config)

        # observed data
        h_data = unfolder.handle_obs.get_histogram(
            varname_reco, bins_det, absoluteValue=absoluteValue)

        if unfolder.handle_obsbkg is not None:
            h_data += unfolder.handle_obsbkg.get_histogram(
                varname_reco, bins_det, absoluteValue=absoluteValue)

        hists_v_d['reco_data'] = h_data

        # signal simulation
        hists_v_d['reco_sig'] = unfolder.handle_sig.get_histogram(
            varname_reco, bins_det, absoluteValue=absoluteValue)

        # background simulation if available
        if unfolder.handle_bkg is not None:
            hists_v_d['reco_bkg'] = unfolder.handle_bkg.get_histogram(
                varname_reco, bins_det, absoluteValue=absoluteValue)

    return hists_v_d

def evaluate_metrics(
    observable,
    hists_dict,
    outdir = '.',
    unbinned_metrics = False,
    unfolder = None, # Only needed if unbinned_metrics is True
    varname_truth = '', # Only needed if unbinned_metrics is True
    plot = False
    ):

    mdict = {}
    mdict[observable] = {}

    # binned
    hists_uf_alliters = hists_dict.get('unfolded_alliters')
    h_gen = hists_dict.get('prior')
    h_truth = hists_dict.get('truth')

    if hists_uf_alliters is None or h_gen is None:
        logger.error(f"Cannot compute binned metrics for {observable}")
    else:
        mdict[observable]["nominal"] = metrics.write_all_metrics_binned(
            hists_uf_alliters, h_gen, h_truth)

    hists_uf_all = hists_dict.get('unfolded_all')
    if h_gen is not None and hists_uf_all is not None and len(hists_uf_all) > 1:
        # every run
        mdict[observable]['resample'] = metrics.write_all_metrics_binned(
            hists_uf_all, h_gen, h_truth)

    # IBU if available
    hists_ibu_alliters = hists_dict.get('ibu_alliters')
    if hists_ibu_alliters is not None:
        mdict[observable]['IBU'] = metrics.write_all_metrics_binned(
            hists_ibu_alliters, h_gen, h_truth)

    # unbinned
    if unbinned_metrics:
        if unfolder is None or not vaarname_truth:
            logger.error(f"Cannot compute unbinned metrics for {observable}")
        else:
            mdict_unbinned = metrics.evaluate_unbinned_metrics(
                unfolder, varname_truth)
            for k in mdict_unbinned:
                mdict[observable][k].update(mdict_unbinned[k])

    # Save metrics to JSON file
    if outdir:

        metrics_dir = os.path.join(outdir, 'Metrics')
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir)

        util.write_dict_to_json(mdict, metrics_dir + f"/{observable}.json")

        # plot
        if plot:
            metrics.plot_all_metrics(
                mdict[observable], metrics_dir + f"/{observable}")

    return mdict

def plot_histograms(
    observable,
    hists_dict,
    plot_verbosity,
    outdir = '.',
    # plotting options
    xlabel = '',
    ylabel = '',
    legend_loc = 'best',
    legend_ncol = 1,
    stamp_loc=(0.75, 0.75)
    ):

    if plot_verbosity < 1 or not outdir:
        return

    # plot_verbosity >= 1
    # -p
    h_uf = hists_dict.get('unfolded')
    h_gen = hists_dict.get('prior')
    h_truth = hists_dict.get('truth')
    h_ibu = hists_dict.get('ibu')

    bins_mc = h_uf.axes[0].edges

    ###
    # print metrics on the plot
    texts_chi2 = metrics.write_texts_Chi2(
        h_truth, [h_uf, h_ibu, h_gen], labels = ['MultiFold', 'IBU', 'Prior'])

    figname_uf = os.path.join(outdir, f"Unfold_{observable}")
    logger.info(f" Plot unfolded distribution: {figname_uf}")
    plotter.plot_distributions_unfold(
        figname_uf,
        h_uf, h_gen, h_truth, h_ibu,
        xlabel = xlabel, ylabel = ylabel,
        legend_loc = legend_loc, legend_ncol = legend_ncol,
        stamp_loc = stamp_loc, stamp_texts = texts_chi2
        )

    ###
    # bin correlations
    h_uf_corr = hists_dict.get('unfolded_corr')
    if h_uf_corr is not None:
        figname_uf_corr = os.path.join(outdir, f"BinCorr_{observable}_OmniFold")
        logger.info(f" Plot bin correlations: {figname_uf_corr}")
        plotter.plot_correlations(figname_uf_corr, h_uf_corr, bins_mc)

    h_ibu_corr = hists_dict.get('ibu_corr')
    if h_ibu_corr is not None:
        figname_ibu_corr = os.path.join(outdir, f"BinCorr_{observable}_IBU")
        logger.info(f" Plot bin correlations: {figname_ibu_corr}")
        plotter.plot_correlations(figname_ibu_corr, h_ibu_corr, bins_mc)

    ###
    # Reco level distributions
    h_data = hists_dict.get('reco_data')
    h_sig = hists_dict.get('reco_sig')
    h_bkg = hists_dict.get('reco_bkg')

    figname_reco = os.path.join(outdir, f"Reco_{observable}")
    logger.info(f" Plot detector-level distribution: {figname_reco}")
    plotter.plot_distributions_reco(
        figname_reco,
        h_data, h_sig, h_bkg,
        xlabel = xlabel, ylabel = ylabel,
        legend_loc = legend_loc, legend_ncol = legend_ncol
        )

    ######
    if plot_verbosity < 2:
        return
    # More plots if plot_verbosity >= 2
    # -pp

    ###
    # Iteration history
    hists_uf_alliters = hists_dict.get('unfolded_alliters')
    if hists_uf_alliters:
        iteration_dir = os.path.join(outdir, 'Iterations')
        if not os.path.isdir(iteration_dir):
            logger.info(f"Create directory {iteration_dir}")
            os.makedirs(iteration_dir)

        figname_alliters = os.path.join(
            iteration_dir, f"Unfold_AllIterations_{observable}")
        logger.info(f" Plot unfolded distributions at every iteration: {figname_alliters}")
        plotter.plot_distributions_iteration(
            figname_alliters,
            hists_uf_alliters, h_gen, h_truth,
            xlabel = xlabel, ylabel = ylabel
            )

    ###
    # All runs
    hists_uf_allruns = hists_dict.get("unfolded_allruns")
    if hists_uf_allruns is not None and len(hists_uf_allruns) > 1:
        allruns_dir = os.path.join(outdir, 'AllRuns')
        if not os.path.isdir(allruns_dir):
            logger.info(f"Create directory {allruns_dir}")
            os.makedirs(allruns_dir)

        # unfolded distributions from all runs
        figname_rs = os.path.join(allruns_dir, f"Unfold_AllRuns_{observable}")
        logger.info(f" Plot unfolded distributions from all runs: {figname_rs}")

        plotter.plot_distributions_resamples(
                figname_rs,
                hists_uf_allruns, h_gen, h_truth,
                xlabel = xlabel, ylabel = ylabel
                )

    ######
    if plot_verbosity < 3:
        return

    # Usually skip plotting these unless really necessary
    # -ppp
    ###
    # Distributions of bin entries
    hists_uf_all =  hists_dict.get('unfolded_all')
    if hists_uf_all is not None and len(hists_uf_all) > 1:
        allruns_dir = os.path.join(outdir, 'AllRuns')
        if not os.path.isdir(allruns_dir):
            logger.info(f"Create directory {allruns_dir}")
            os.makedirs(allruns_dir)

        figname_bindistr = os.path.join(allruns_dir, f"Unfold_BinDistr_{observable}")
        logger.info(f" Plot distributions of bin entries from all runs: {figname_bindistr}")
        plotter.plot_hists_bin_distr(figname_bindistr, hists_uf_all, h_truth)

    return

def make_histograms_from_unfolder(
    unfolder,
    binning_config, # path to the binning config file
    observables, # list of observable names
    obsConfig_d, # dict for observable configurations
    iteration = -1, # by default take the last iteration
    nruns = None, # by default take all that are available
    normalize = True, # If True, rescale all truth-level histograms to the same normalization
    outputdir = None, # str, output directory
    outfilename = "histograms.root", # str, output file name
    include_ibu = False, # If True, include also IBU for comparison
    compute_metrics = False, # If True, compute metrics
    plot_verbosity = 0 # int, control how many plots to make
    ):

    # output directory
    if not outputdir:
        outputdir = unfolder.outdir
    elif not os.path.isdir(outputdir):
        logger.info(f"Create directory {outputdir}")
        os.makedirs(outputdir)

    # control flags
    all_runs = True
    all_iterations = compute_metrics or plot_verbosity >= 2
    all_histograms = compute_metrics or plot_verbosity >= 2
    include_reco = plot_verbosity >= 1

    histograms_dict = {}

    for ob in observables:
        logger.info(f"Make histograms for {ob}")

        histograms_dict[ob] = make_histograms_of_observable(
            unfolder,
            ob, 
            obsConfig_d,
            binning_config,
            iteration = iteration,
            nruns = nruns,
            absoluteValue = False, # ob in ['th_y', 'tl_y']
            normalize = normalize,
            all_runs = all_runs,
            all_iterations = all_iterations,
            all_histograms = all_histograms,
            include_ibu = include_ibu, 
            include_reco = include_reco
            )

        # compute metrics
        if compute_metrics:
            evaluate_metrics(
                ob,
                histograms_dict[ob],
                outdir = outputdir,
                unbinned_metrics = False,
                plot = plot_verbosity > 1
                )

        # plot
        plot_histograms(
            ob,
            histograms_dict[ob],
            plot_verbosity,
            outdir = outputdir,
            xlabel = obsConfig_d[ob]['xlabel'],
            ylabel = obsConfig_d[ob]['ylabel'],
            legend_loc = obsConfig_d[ob]['legend_loc'],
            legend_ncol = obsConfig_d[ob]['legend_ncol'],
            stamp_loc =  obsConfig_d[ob]['stamp_xy']
            )

    # save histograms to file
    if outputdir:
        outname_hist = os.path.join(outputdir, outfilename)
        logger.info(f" Write histograms to file: {outname_hist}")
        # hard code here for now
        keys_to_save = ['unfolded', 'unfolded_allruns', 'prior', 'truth', 'ibu']

        hists_to_write = {}
        for ob in histograms_dict:
            hists_to_write[ob] = {}
            for hname in histograms_dict[ob]:
                if hname in keys_to_save:
                    hists_to_write[ob][hname] = histograms_dict[ob][hname]

        myhu.write_histograms_dict_to_file(hists_to_write, outname_hist)

    return histograms_dict

def make_histograms(
    result_dir,
    binning_config,
    observables = [],
    observable_config = '',
    iteration = -1,
    nruns = None,
    normalize = True,
    outputdir = None,
    outfilename = 'histograms.root',
    include_ibu = False,
    compute_metrics = False,
    plot_verbosity = 0,
    verbose = False
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
    # If None, use the same one as in the argument json config
    obsConfig_d = {}
    if observable_config:
        obsConfig_d = util.read_dict_from_json(observable_config)
    # if empty, it will be filled by load_unfolder

    # unfolder
    logger.info(f"Load unfolder from {result_dir} ... ")
    t_load_start = time.time()

    ufdr = load_unfolder(fpath_args_config, observables, obsConfig_d)

    t_load_stop = time.time()
    logger.info(f"Done")
    logger.debug(f"Loading time: {(t_load_stop-t_load_start):.2f} seconds")

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*1e-6:.1f} MB; Peak usage: {mpeak*1e-6:.1f} MB")

    logger.info("Start histogramming")
    t_hist_start = time.time()

    make_histograms_from_unfolder(
        ufdr,
        binning_config,
        observables,
        obsConfig_d,
        iteration = iteration,
        nruns = nruns,
        normalize = True,
        outputdir = outputdir,
        outfilename = outfilename,
        include_ibu = include_ibu,
        compute_metrics = compute_metrics,
        plot_verbosity = plot_verbosity
        )

    t_hist_stop = time.time()
    logger.debug(f"Histogramming time: {(t_hist_stop-t_hist_start):.2f} seconds")

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*1e-6:.1f} MB; Peak usage: {mpeak*1e-6:.1f} MB")

    tracemalloc.stop()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Make, plot, and store histograms from unfolding results')

    parser.add_argument('result_dir', type=str,
                        help="Directory of the unfolding results")
    parser.add_argument("--binning-config", type=str,
                        default='configs/binning/bins_10equal.json',
                        #default='configs/binning/bins_ttdiffxs.json',
                        help="Path to the binning config file for variables.")
    parser.add_argument("--observables", nargs='+', default=[],
                        help="List of observables to make histograms. If not provided, use the same ones from the unfolding results")
    parser.add_argument("--observable_config", type=str,
                        help="Path to the observable config file. If not provided, use the same one from the unfolding results")
    parser.add_argument("-i", "--iteration", type=int, default=-1,
                        help="Use the results at the specified iteration")
    parser.add_argument("-n", "--nruns", type=int,
                        help="Number of runs for making unfolded distributions. If not specified, use all that are available")
    parser.add_argument("-o", "--outputdir", type=str,
                        help="Output directory. If not provided, use result_dir.")
    parser.add_argument("-f", "--outfilename", type=str, default="histograms.root",
                        help="Output file name")
    parser.add_argument('--include-ibu', action='store_true',
                        help="If True, run unfolding also with IBU as well")
    parser.add_argument('--compute-metrics', action='store_true',
                        help="If True, compute metrics of unfolding performance")
    parser.add_argument('-p', '--plot-verbosity', action='count', default=0,
                        help="Plot verbose level. '-ppp' to make all plots.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="If True, set logging level to DEBUG, otherwise INFO")

    args = parser.parse_args()

    util.configRootLogger()

    make_histograms(**vars(args))
