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

from ttbarDiffXsRun2.binnedCorrections import binned_corrections, apply_efficiency_correction

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
    norm = None,
    absoluteValue = False,
    acceptance = None,
    efficiency = None,
    correction_flow = True
    ):

    # Prepare inputs
    # data array
    array_obs = unfolder.handle_obs[vname_reco]
    if absoluteValue:
        array_obs = np.abs(array_obs)

    wobs = unfolder.handle_obs.get_weights()

    if unfolder.handle_obsbkg is not None:
        array_obsbkg = unfolder.handle_obsbkg[vname_reco]
        if absoluteValue:
            array_obsbkg = np.abs(array_obsbkg)

        wobsbkg = unfolder.handle_obsbkg.get_weights()

        array_obs = np.concatenate([array_obs, array_obsbkg])
        wobs = np.concatenate([wobs, wobsbkg])

    # simulation
    # Take only events that are truth matched i.e. pass both reco and truth cuts
    pass_all_sig = unfolder.handle_sig.pass_reco & unfolder.handle_sig.pass_truth

    array_sim = unfolder.handle_sig.get_arrays(vname_reco)[pass_all_sig]
    if absoluteValue:
        array_sim = np.abs(array_sim)

    wsim = unfolder.handle_sig.get_weights(valid_only=False)[pass_all_sig]

    array_gen = unfolder.handle_sig.get_arrays(vname_truth)[pass_all_sig]
    if absoluteValue:
        array_gen = np.abs(array_gen)

    wgen = unfolder.handle_sig.get_weights(reco_level=False, valid_only=False)[pass_all_sig]

    if unfolder.handle_bkg is not None:
        array_bkg = unfolder.handle_bkg[vname_reco]
        if absoluteValue:
            array_bkg = np.abs(array_bkg)

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
        density = False,
        norm = norm,
        acceptance_correction = acceptance,
        efficiency_correction = efficiency,
        flow = correction_flow
        )

    return hists_ibu, h_ibu_corr, reponse

def make_histograms_of_observable(
    unfolder,
    observable, # str, name of the observable
    obsConfig_d, # dict, observable configuration
    binning_d, # dict, binning config
    iteration = -1, # int, which iteration to use as the nominal. Default is the last one
    nruns = None, # int, number of runs. Default is to take all that are available
    all_runs = True, # If True, include unfolded histograms of all runs at specified iteration
    all_iterations = False, # If True, include nominal unfolded histograms at all iterations
    all_histograms = False, # If True, include also histograms of every run and every iteration
    include_reco = False, # If True, include also reco level histograms
    include_ibu = False, # If True, include also IBU for comparison
    binned_correction_d = None, # dict, dictionary of binned corrections
    binned_correction_flow = True # bool, if False, exclude overflow/underflow bins when compute binned corrections
    ):

    logger.info(f"Make histograms for {observable}")

    hists_v_d = {}

    varname_reco = obsConfig_d[observable]['branch_det']
    varname_truth = obsConfig_d[observable]['branch_mc']

    # get bin edges
    # TODO different binning at reco and truth level
    bins_det = binning_d[observable]
    bins_mc = binning_d[observable]

    absValue = "_abs" in observable

    # flags to filter out underflow/overflow events
    if not binned_correction_flow:
        unfolder.reset_underflow_overflow_flags()
        unfolder.update_underflow_overflow_flags(varname_reco, bins_det)
        unfolder.update_underflow_overflow_flags(varname_truth, bins_mc)

        inbins_sig = ~unfolder.handle_sig.is_underflow_or_overflow()
        inbins_sig_truth = inbins_sig[unfolder.handle_sig.pass_truth]

        inbins_obs = ~unfolder.handle_obs.is_underflow_or_overflow()
        inbins_obs_truth = inbins_obs[unfolder.handle_obs.pass_truth]
    else:
        inbins_sig_truth = None
        inbins_obs_truth = None

    ###
    # Get binned corrections if available
    acceptance, efficiency = None, None
    if binned_correction_d:
        acceptance = binned_correction_d[observable]['acceptance']
        efficiency = binned_correction_d[observable]['efficiency']

        # Add to the histogram dict
        hists_v_d['acceptance'] = acceptance
        hists_v_d['efficiency'] = efficiency

    ###
    # The unfolded distributions
    logger.debug(f" Unfolded distributions")
    h_uf, h_uf_correlation = unfolder.get_unfolded_distribution(
        varname_truth,
        bins_mc,
        iteration = iteration,
        nresamples = nruns,
        density = False,
        absoluteValue = absValue,
        extra_cuts=inbins_sig_truth
        )

    # set x-axis label
    h_uf.axes[0].label = obsConfig_d[observable]['xlabel']

    hists_v_d['unfolded'] = h_uf
    hists_v_d['unfolded_correlation'] = h_uf_correlation

    if efficiency:
        h_uf_corrected = apply_efficiency_correction(h_uf, efficiency)
        hists_v_d['unfolded_corrected'] = h_uf_corrected

        # Acceptance corrections have already been accounted for during iterations

        hists_v_d['absoluteDiffXs'] = h_uf_corrected / myhu.get_hist_widths(h_uf_corrected)

        hists_v_d['relativeDiffXs'] = hists_v_d['absoluteDiffXs'].copy()
        myhu.renormalize_hist(hists_v_d['relativeDiffXs'], density=True)

    if all_runs:
        # unfolded histograms from every run
        hists_v_d['unfolded_allruns'] = unfolder.get_unfolded_hists_resamples(
            varname_truth,
            bins_mc,
            iteration = iteration,
            nresamples = nruns,
            density = False,
            absoluteValue = absValue,
            extra_cuts=inbins_sig_truth
            )

    if all_iterations:
        # unfoldedd histograms at every iteration
        hists_v_d['unfolded_alliters'] = unfolder.get_unfolded_distribution(
            varname_truth,
            bins_mc,
            nresamples = nruns,
            all_iterations = True,
            density = False,
            absoluteValue = absValue,
            extra_cuts=inbins_sig_truth
            )[0]

    if all_histograms:
        hists_v_d['unfolded_all'] = unfolder.get_unfolded_hists_resamples(
            varname_truth,
            bins_mc,
            nresamples = nruns,
            density = False,
            all_iterations=True,
            absoluteValue = absValue,
            extra_cuts=inbins_sig_truth
            )

    ###
    # Other truth-level distributions
    ##
    # Prior
    # For now TODO fix norm
    logger.debug(f" Prior distribution")
    hists_v_d['prior'] = unfolder.handle_sig.get_histogram(
        varname_truth,
        bins_mc,
        #weights = unfolder.handle_sig.get_weights(reco_level=True),
        density = False,
        absoluteValue = absValue,
        extra_cuts=inbins_sig_truth
        )

    hist2d_sig = unfolder.handle_sig.get_histogram2d(
        varname_reco,
        varname_truth,
        bins_x = bins_det,
        bins_y = bins_mc,
        absoluteValue_x = absValue,
        absoluteValue_y = absValue
        )

    htemp_prior = hist2d_sig.project(1) # this includes underflow/overflow bins in axis 0
    # Use the binning from the projected histogram but overwrite its content
    # Exclude underflow and overflow bins
    htemp_prior.view()['value'] = hist2d_sig.values().sum(axis=0)
    htemp_prior.view()['variance'] = hist2d_sig.variances().sum(axis=0)
    hists_v_d['prior_noflow'] = htemp_prior

    ##
    # truth distribution if using pseudo data
    if unfolder.handle_obs.data_truth is not None:
        logger.debug(f" Truth distribution")
        hists_v_d['truth'] = unfolder.handle_obs.get_histogram(
            varname_truth,
            bins_mc,
            #weights = unfolder.handle_obs.get_weights(reco_level=True),
            density = False,
            absoluteValue = absValue,
            extra_cuts=inbins_obs_truth
            )

        hist2d_obs = unfolder.handle_obs.get_histogram2d(
            varname_reco,
            varname_truth,
            bins_x = bins_det,
            bins_y = bins_mc,
            absoluteValue_x = absValue,
            absoluteValue_y = absValue
        )

        htemp_truth = hist2d_obs.project(1)
        htemp_truth.view()['value'] = hist2d_obs.values().sum(axis=0)
        htemp_truth.view()['variance'] = hist2d_obs.variances().sum(axis=0)
        hists_v_d['truth_noflow'] = htemp_truth

    ##
    # IBU
    if include_ibu:
        logger.info(f" Run IBU for {observable}")

        varname_reco = obsConfig_d[observable]['branch_det']

        hists_ibu_alliters, h_ibu_correlation, response = get_ibu_unfolded_histogram_from_unfolder(
            unfolder,
            varname_reco, varname_truth,
            bins_det, bins_mc,
            all_iterations = True,
            absoluteValue = absValue,
            acceptance = acceptance,
            efficiency = efficiency,
            correction_flow = binned_correction_flow
        )

        # take the ones at the same iteration as OmniFold
        h_ibu = hists_ibu_alliters[iteration]
        h_ibu_correlation = h_ibu_correlation[iteration]

        hists_v_d['ibu'] = h_ibu
        hists_v_d['ibu_alliters'] = hists_ibu_alliters
        hists_v_d['ibu_correlation'] = h_ibu_correlation
        hists_v_d['response'] = response

        if acceptance and efficiency:
            hists_v_d['absoluteDiffXs_ibu'] = h_ibu / myhu.get_hist_widths(h_ibu)
            hists_v_d['relativeDiffXs_ibu'] = hists_v_d['absoluteDiffXs_ibu'].copy()
            myhu.renormalize_hist(hists_v_d['relativeDiffXs_ibu'], density=True)

    ###
    # Reco level
    if include_reco:
        logger.debug(f" Reco-level distributions")

        # observed data
        h_data = unfolder.handle_obs.get_histogram(
            varname_reco,
            bins_det,
            density = False,
            absoluteValue = absValue
            )

        if unfolder.handle_obsbkg is not None:
            h_data += unfolder.handle_obsbkg.get_histogram(
                varname_reco,
                bins_det,
                density = False,
                absoluteValue = absValue)

        hists_v_d['reco_data'] = h_data

        # signal simulation
        hists_v_d['reco_sig'] = unfolder.handle_sig.get_histogram(
            varname_reco,
            bins_det,
            density = False,
            absoluteValue = absValue
            )

        # background simulation if available
        if unfolder.handle_bkg is not None:
            hists_v_d['reco_bkg'] = unfolder.handle_bkg.get_histogram(
                varname_reco,
                bins_det,
                density = False,
                absoluteValue = absValue
                )

    return hists_v_d

def make_histograms_of_observables_multidim(
    unfolder,
    observables, # observable names in the format of "obs1_vs_obs2_vs_..."
    obsConfig_d, # dict, observable configuration
    binConfig_d, # dict, binning configuration
    iteration = -1, # int, which iteration to use as the nominal. Default is the last one
    nruns = None, # int, number of runs. Default is to take all that are available
    binned_correction_d = None, # dict, dictionary of binned corrections
    binned_correction_flow = True # bool, if False, exclude overflow/underflow bins when compute binned corrections
    ):

    obs_list = observables.split("_vs_")
    ndim = len(obs_list)
    logger.info(f"Make {ndim}D histograms for {obs_list}")

    hists_multidim_d = {}

    varnames_reco = [obsConfig_d[ob]['branch_det'] for ob in obs_list]
    varnames_truth = [obsConfig_d[ob]['branch_mc'] for ob in obs_list]

    absValues = ["_abs" in ob for ob in obs_list]

    # bins dict
    bins_reco_d = binConfig_d[observables]
    bins_truth_d = binConfig_d[observables]

    # binned_correction_flow
    if not binned_correction_flow:
        unfolder.reset_underflow_overflow_flags()
        unfolder.update_underflow_overflow_flags(varnames_reco, bins_reco_d)
        unfolder.update_underflow_overflow_flags(varnames_truth, bins_truth_d)

        inbins_sig = ~unfolder.handle_sig.is_underflow_or_overflow()
        inbins_sig_truth = inbins_sig[unfolder.handle_sig.pass_truth]

        inbins_obs = ~unfolder.handle_obs.is_underflow_or_overflow()
        inbins_obs_truth = inbins_obs[unfolder.handle_obs.pass_truth]
    else:
        inbins_sig_truth = None
        inbins_obs_truth = None

    # binned correction
    acceptance, efficiency = None, None
    if binned_correction_d:
        acceptance = binned_correction_d[observables]['acceptance']
        efficiency = binned_correction_d[observables]['efficiency']

        # Add to the histogram dict
        hists_multidim_d['acceptance'] = acceptance
        hists_multidim_d['efficiency'] = efficiency

    ###
    # The unfolded distributions
    logger.debug(f" Unfolded distributions")
    hists_multidim_d['unfolded'] = unfolder.get_unfolded_distribution_multidim(
        varnames_truth,
        bins_truth_d,
        iteration = iteration,
        nresamples = nruns, # default, take all that are available
        density = False,
        absoluteValues = absValues,
        extra_cuts = inbins_sig_truth
    )

    # set axis labels
    if len(obs_list) >= 2:
        hists_multidim_d['unfolded'].set_xlabel(obsConfig_d[obs_list[0]]['xlabel'])
        hists_multidim_d['unfolded'].set_ylabel(obsConfig_d[obs_list[1]]['xlabel'])
    if len(obs_list) >= 3:
        hists_multidim_d['unfolded'].set_zlabel(obsConfig_d[obs_list[2]]['xlabel'])

    if efficiency:
        # apply binned correction
        hists_multidim_d['unfolded_corrected'] = apply_efficiency_correction(hists_multidim_d['unfolded'], efficiency)

        # Acceptance corrections have already been accounted for during iterations

        hists_multidim_d['absoluteDiffXs'] = hists_multidim_d['unfolded_corrected'].copy()
        hists_multidim_d['absoluteDiffXs'].make_density()

        hists_multidim_d['relativeDiffXs'] = hists_multidim_d['unfolded_corrected'].copy()
        hists_multidim_d['relativeDiffXs'].renormalize(norm=1., density=False, flow=True)
        hists_multidim_d['relativeDiffXs'].make_density()

    ###
    # Prior
    logger.debug(f" Prior distribution")
    hists_multidim_d['prior'] = unfolder.handle_sig.get_histograms_flattened(
        varnames_truth,
        bins_truth_d,
        density=False,
        absoluteValues=absValues,
        extra_cuts = inbins_sig_truth
    )

    if efficiency:
        hists_multidim_d['absoluteDiffXs_prior'] = hists_multidim_d['prior'].copy()
        hists_multidim_d['absoluteDiffXs_prior'].make_density()

        hists_multidim_d['relativeDiffXs_prior'] = hists_multidim_d['prior'].copy()
        hists_multidim_d['relativeDiffXs_prior'].renormalize(norm=1., density=False, flow=True)
        hists_multidim_d['relativeDiffXs_prior'].make_density()

    return hists_multidim_d

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
        if unfolder is None or not varname_truth:
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
    stamp_loc=(0.75, 0.75),
    ratio_lim = None
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
    if plot_verbosity > 2:
        texts_chi2 = metrics.write_texts_Chi2(
            h_truth, [h_uf, h_ibu, h_gen], labels = ['MultiFold', 'IBU', 'Prior'])
    else:
        texts_chi2 = []

    figname_uf = os.path.join(outdir, f"Unfold_{observable}")
    logger.info(f" Plot unfolded distribution: {figname_uf}")
    plotter.plot_distributions_unfold(
        figname_uf,
        h_uf, h_gen, h_truth, h_ibu,
        xlabel = xlabel, ylabel = ylabel,
        legend_loc = legend_loc, legend_ncol = legend_ncol,
        stamp_loc = stamp_loc, stamp_texts = texts_chi2,
        ratio_lim = ratio_lim
        )

    ###
    # bin correlations
    h_uf_corr = hists_dict.get('unfolded_correlation')
    if h_uf_corr is not None:
        figname_uf_corr = os.path.join(outdir, f"BinCorr_{observable}_OmniFold")
        logger.info(f" Plot bin correlations: {figname_uf_corr}")
        plotter.plot_correlations(figname_uf_corr, h_uf_corr.values(), bins_mc)

    h_ibu_corr = hists_dict.get('ibu_correlation')
    if h_ibu_corr is not None:
        figname_ibu_corr = os.path.join(outdir, f"BinCorr_{observable}_IBU")
        logger.info(f" Plot bin correlations: {figname_ibu_corr}")
        plotter.plot_correlations(figname_ibu_corr, h_ibu_corr.values(), bins_mc)

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
    # Response
    resp = hists_dict.get("response")
    if resp is not None:
        figname_resp = os.path.join(outdir, f"Response_{observable}")
        logger.info(f" Plot response: {figname_resp}")
        plotter.plot_response(figname_resp, resp, observable)

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
    if plot_verbosity < 5:
        return

    # Usually skip plotting these unless really necessary
    # -ppppp
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
    outputdir = None, # str, output directory
    outfilename = "histograms.root", # str, output file name
    include_ibu = False, # If True, include also IBU for comparison
    compute_metrics = False, # If True, compute metrics
    plot_verbosity = 0, # int, control how many plots to make
    binned_correction_fpath = None, # str, file path to read histograms for binned corrections
    binned_correction_flow = True, # bool, if False, exclude overflow/underflow bins when compute binned corrections
    observables_multidim = [], # list of observables in the "x_vs_y_vs_.." format for higher dimension distributions
    ):

    # output directory
    if not outputdir:
        outputdir = unfolder.outdir

    # in case it is not the last iteration that is used
    if iteration != -1:
        outputdir = os.path.join(outputdir, f"iter{iteration+1}")
        # +1 because the index 0 of the weights array is for iteration 1

    # in case not all runs are used to make histograms
    if nruns is not None:
        outputdir = os.path.join(outputdir, f"nruns{nruns}")

    if not os.path.isdir(outputdir):
        logger.info(f"Create directory {outputdir}")
        os.makedirs(outputdir)

    # control flags
    all_runs = nruns is None # if number of runs is explicitly specified, no need to include all runs
    include_reco = True
    all_iterations = compute_metrics or plot_verbosity >= 2
    all_histograms = compute_metrics or plot_verbosity >= 2

    # binning config
    binning_d = util.get_bins_dict(binning_config)

    # binned corrections
    binned_corrections_d = binned_corrections(
        myhu.read_histograms_dict_from_file(binned_correction_fpath),
        observables + observables_multidim,
        flow = binned_correction_flow
    )

    histograms_dict = {}

    for ob in observables:

        histograms_dict[ob] = make_histograms_of_observable(
            unfolder,
            ob,
            obsConfig_d,
            binning_d,
            iteration = iteration,
            nruns = nruns,
            all_runs = all_runs,
            all_iterations = all_iterations,
            all_histograms = all_histograms,
            include_ibu = include_ibu,
            include_reco = include_reco,
            binned_correction_d = binned_corrections_d,
            binned_correction_flow = binned_correction_flow
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
            stamp_loc =  obsConfig_d[ob]['stamp_xy'],
            ratio_lim = obsConfig_d[ob].get('ratio_lim')
            )

    for obs in observables_multidim:

        histograms_dict[obs] = make_histograms_of_observables_multidim(
            unfolder,
            obs,
            obsConfig_d,
            binning_d,
            iteration = iteration,
            nruns = nruns,
            binned_correction_d = binned_corrections_d,
            binned_correction_flow = binned_correction_flow
        )

    # save histograms to file
    if outputdir:
        outname_hist = os.path.join(outputdir, outfilename)
        logger.info(f"Write histograms to file: {outname_hist}")
        # hard code here for now
        keys_to_save = [
            'unfolded', 'unfolded_alliters', 'unfolded_allruns', 'unfolded_correlation',
            'unfolded_corrected', 'relativeDiffXs', 'absoluteDiffXs',
            'prior', 'prior_noflow', 'truth', 'truth_noflow',
            'reco_data', 'reco_sig', 'reco_bkg',
            'ibu', 'ibu_alliters', 'ibu_correlation', 'response',
            'relativeDiffXs_ibu', 'absoluteDiffXs_ibu',
            'acceptance', 'efficiency'
            ]

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
    observables_multidim = [],
    observable_config = '',
    iterations = [-1],
    nruns = [None],
    outputdir = None,
    outfilename = 'histograms.root',
    include_ibu = False,
    compute_metrics = False,
    plot_verbosity = 0,
    verbose = False,
    binned_correction_fpath = None,
    binned_correction_noflow = False
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
        # try an alternative config
        fpath_args_config_rw = os.path.join(result_dir, "arguments_rw.json")
        if os.path.isfile(fpath_args_config_rw):
            fpath_args_config = fpath_args_config_rw
        else:
            logger.critical(f"Cannot find argument config {fpath_args_config}")
            return

    # observable config
    # If None, use the same one as in the argument json config
    obsConfig_d = {}
    if observable_config:
        obsConfig_d = util.read_dict_from_json(observable_config)
    # if empty, it will be filled by load_unfolder

    observables_extra = []
    for obs_md in observables_multidim:
        observables_extra += obs_md.split("_vs_")
    observables_extra = list(set(observables_extra))

    # unfolder
    logger.info(f"Load unfolder from {result_dir} ... ")
    t_load_start = time.time()

    ufdr = load_unfolder(
        fpath_args_config,
        observables,
        obsConfig_d,
        args_update = {'observables_extra': observables_extra}
        )

    t_load_stop = time.time()
    logger.info(f"Done")
    logger.debug(f"Loading time: {(t_load_stop-t_load_start):.2f} seconds")

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*1e-6:.1f} MB; Peak usage: {mpeak*1e-6:.1f} MB")

    logger.info("Start histogramming")
    t_hist_start = time.time()

    # check if iterations is a list
    if not isinstance(iterations, list):
        iterations = [iterations]

    # check if nruns is a list
    if not isinstance(nruns, list):
        nruns = [nruns]

    for it in iterations:
        logger.info(f"iteration: {it}")

        for n in nruns:
            logger.info(f" nruns: {n}")

            make_histograms_from_unfolder(
                ufdr,
                binning_config,
                observables,
                obsConfig_d,
                iteration = it,
                nruns = n,
                outputdir = outputdir,
                outfilename = outfilename,
                include_ibu = include_ibu,
                compute_metrics = compute_metrics,
                plot_verbosity = plot_verbosity,
                binned_correction_fpath = binned_correction_fpath,
                binned_correction_flow = not binned_correction_noflow,
                observables_multidim = observables_multidim
                )

    t_hist_stop = time.time()
    logger.debug(f"Histogramming time: {(t_hist_stop-t_hist_start):.2f} seconds")

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*1e-6:.1f} MB; Peak usage: {mpeak*1e-6:.1f} MB")

    tracemalloc.stop()

def make_histograms_bootstrap(
    bootstrap_topdir, # str, top directory of the bootstrap results
    nresamples = None, # int, number of resamples for bootstraping
    histname = "histograms.root", # common name of the histogram root file
    outfilename = None, # str, output file name. If None, use histname
    ):

    logger.debug(f"Collect bootstrap results from {bootstrap_topdir}")

    # file paths to bootstraping histograms
    fpaths_hists_bootstrap = [os.path.join(bootstrap_topdir, d, histname) for d in os.listdir(bootstrap_topdir)]

    if nresamples is not None:
        if nresamples > len(fpaths_hists_bootstrap):
            logger.error(f"Require {nresamples} resamples but only {len(fpaths_hists_bootstrap)} available")
            nresamples = len(fpaths_hists_bootstrap)

        fpaths_hists_bootstrap = fpaths_hists_bootstrap[:nresamples]

    # check if all histogram files in the list exist
    for fpath in fpaths_hists_bootstrap:
        if not os.path.isfile(fpath):
            logger.error(f"Cannot find histogram file {fpath}. Need to run make_histograms.py first")
            # TODO: run make_histogram here?
            return

    #fpaths_hists_bootstrap[:] = [fp for fp in fpaths_hists_bootstrap if os.path.isfile(fp)]

    # Read all histograms
    resample_histograms_l = [myhu.read_histograms_dict_from_file(fpath) for fpath in fpaths_hists_bootstrap]
    if len(resample_histograms_l) < 2:
        logger.error(f"Not enough resampling results")
        return

    histograms_bs_d = {}

    # loop over observables
    for ob in resample_histograms_l[0]:
        histograms_bs_d[ob] = {}

        # loop over histograms
        for hname in resample_histograms_l[0][ob]:

            h0 = resample_histograms_l[0][ob][hname]
            # skip if a list of histograms (from individual runs) to keep things simple
            if isinstance(h0, list):
                continue
            else:
                histograms_bs_d[ob][hname] = h0.copy()

            hists_l = [ hists_d[ob][hname] for hists_d in resample_histograms_l ]

            # Compute bin entry means and standard deviations
            histograms_bs_d[ob][hname] = myhu.average_histograms(hists_l, False)

    # Save to file
    if outfilename is None:
        outfilename = histname
    outfpath = os.path.join(bootstrap_topdir, outfilename)
    logger.info(f"Write histograms to file: {outfpath}")
    myhu.write_histograms_dict_to_file(histograms_bs_d, outfpath)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Make, plot, and store histograms from unfolding results')

    parser.add_argument('result_dir', type=str,
                        help="Directory of the unfolding results")
    parser.add_argument("--binning-config", type=str,
                        default='configs/binning/bins_ttdiffxs.json',
                        help="Path to the binning config file for variables.")
    parser.add_argument("--observables", nargs='+', default=[],
                        help="List of observables to make histograms. If not provided, use the same ones from the unfolding results")
    parser.add_argument("--observables-multidim", nargs='+', default=[],
                        help="List of observables to make multi-dimension histograms.")
    parser.add_argument("--observable_config", type=str,
                        help="Path to the observable config file. If not provided, use the same one from the unfolding results")
    parser.add_argument("-i", "--iterations", type=int, nargs='+', default=[-1],
                        help="Use the results at the specified iteration")
    parser.add_argument("-n", "--nruns", type=int, nargs='+', default=[None],
                        help="Number of runs for making unfolded distributions. If None, use all that are available")
    parser.add_argument("-o", "--outputdir", type=str,
                        help="Output directory. If not provided, use result_dir.")
    parser.add_argument("-f", "--outfilename", type=str, default="histograms.root",
                        help="Output file name")
    parser.add_argument('--include-ibu', action='store_true',
                        help="If True, run unfolding also with IBU")
    parser.add_argument('--compute-metrics', action='store_true',
                        help="If True, compute metrics of unfolding performance")
    parser.add_argument('-p', '--plot-verbosity', action='count', default=0,
                        help="Plot verbose level. '-ppp' to make all plots.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="If True, set logging level to DEBUG, otherwise INFO")
    parser.add_argument('--binned-correction', dest="binned_correction_fpath",
                        type=str,
                        help="File path to read histograms for binned corrections")
    parser.add_argument("--correction-noflow", dest="binned_correction_noflow",
                        action='store_true',
                        help="If True, exclude underflow and overflow bins when computing binned corrections")

    args = parser.parse_args()

    util.configRootLogger()

    make_histograms(**vars(args))
