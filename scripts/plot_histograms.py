#!/usr/bin/env python3
import os

import util
import plotter
import metrics
import histogramming as myhu

import logging
logger = logging.getLogger("plot_histograms")

def set_labels(obs_list, histogram, obsConfig):

    if len(obs_list) == 1:
        histogram.axes[0].label = obsConfig[obs_list[0]]['xlabel']
    elif len(obs_list) >= 2:
        histogram.set_xlabel(obsConfig[obs_list[0]]['xlabel'])
        histogram.set_ylabel(obsConfig[obs_list[1]]['xlabel'])
        if len(obs_list) >= 3:
            histogram.set_zlabel(obsConfig[obs_list[2]]['xlabel'])

def plot_distributions_unfold(
    obs,
    histograms_d,
    obsConfig_d,
    outputdir,
    include_ibu = False,
    plot_verbosity = 1
    ):

    obs_list = obs.split('_vs_')

    hist_uf = histograms_d[obs].get('unfolded')
    hist_gen = histograms_d[obs].get('prior')
    hist_truth = histograms_d[obs].get('truth')
    hist_ibu = histograms_d[obs].get('ibu') if include_ibu else None
    if include_ibu and hist_ibu is None:
        logger.warn("Cannot find histogram 'ibu'")

    ###
    # print metrics on the plot
    if plot_verbosity > 2:
        texts_chi2 = metrics.write_texts_Chi2(
            hist_truth, [hist_uf, hist_ibu, hist_gen], labels = ['MultiFold', 'IBU', 'Prior'])
    else:
        texts_chi2 = []

    figname_uf = os.path.join(outputdir, f"Unfold_{obs}")
    logger.info(f" Plot unfolded distribution: {figname_uf}")

    hists_toplot = []
    draw_options = []

    if hist_gen:
        hists_toplot.append(hist_gen)
        draw_options.append(plotter.gen_style)

    if hist_ibu:
        hists_toplot.append(hist_ibu)
        ibu_opt = plotter.ibu_style.copy()
        ibu_opt.update({'xerr':True})
        draw_options.append(ibu_opt)

    hists_toplot.append(hist_uf)
    omnifold_opt = plotter.omnifold_style.copy()
    omnifold_opt.update({'xerr':True})
    draw_options.append(omnifold_opt)

    # update axis labels
    set_labels(obs_list, hists_toplot[-1], obsConfig_d)

    plotter.plot_histograms_and_ratios(
        figname_uf,
        hists_numerator = hists_toplot,
        hist_denominator = hist_truth,
        draw_options_numerator = draw_options,
        draw_option_denominator = plotter.truth_style,
        xlabel = obsConfig_d[obs_list[0]]['xlabel'],
        ylabel = obsConfig_d[obs_list[0]]['ylabel'],
        ylabel_ratio = 'Ratio to\nTruth',
        log_scale = False,
        legend_loc = obsConfig_d[obs_list[0]]['legend_loc'],
        legend_ncol = obsConfig_d[obs_list[0]]['legend_ncol'],
        stamp_texts = texts_chi2,
        stamp_loc = obsConfig_d[obs_list[0]]['stamp_xy'],
        ratio_lim = obsConfig_d[obs_list[0]].get('ratio_lim')
        )

def plot_distributions_reco(obs, histograms_d, obsConfig_d, outputdir,):

    obs_list = obs.split('_vs_')

    hist_data = histograms_d[obs].get('reco_data')
    hist_sig = histograms_d[obs].get('reco_sig')
    hist_bkg = histograms_d[obs].get('reco_bkg')

    figname_reco = os.path.join(outputdir, f"Reco_{obs}")
    logger.info(f" Plot detector-level distribution: {figname_reco}")

    # simulation
    if hist_bkg is None:
        hist_sim = [hist_sig]
        style_sim = [plotter.sim_style.copy()]
    else:
        hist_sim = [hist_bkg, hist_sig]
        style_sim = [plotter.bkg_style.copy(), plotter.sim_style.copy()]

    # data
    style_data = plotter.data_style.copy()
    style_data.update({"xerr":True})

    # update axis labels
    set_labels(obs_list, hist_sim[-1], obsConfig_d)

    plotter.plot_histograms_and_ratios(
        figname_reco,
        hists_numerator = hist_sim,
        hist_denominator = hist_data,
        draw_options_numerator = style_sim,
        draw_option_denominator = style_data,
        xlabel = obsConfig_d[obs_list[0]]['xlabel'],
        ylabel = obsConfig_d[obs_list[0]]['ylabel'],
        ylabel_ratio = 'Ratio to\nData',
        log_scale = False,
        legend_loc = obsConfig_d[obs_list[0]]['legend_loc'],
        legend_ncol = obsConfig_d[obs_list[0]]['legend_ncol'],
        stack_numerators = True,
        ratio_lim = None
        )

def plot_bin_correlations(obs, histograms_d, outputdir, include_ibu=False):

    huf_corr = histograms_d[obs].get('unfolded_correlation')

    if huf_corr:
        figname_uf_corr = os.path.join(outputdir, f"BinCorr_{obs}_OmniFold")
        logger.info(f" Plot bin correlations: {figname_uf_corr}")
        plotter.plot_correlations(
            figname_uf_corr,
            huf_corr.values(),
            huf_corr.axes[0].edges
        )
    else:
        logger.warning(f"No 'unfolded_correlation' for observable {obs}")

    if include_ibu:
        huf_corr_ibu = histograms_d[obs].get('ibu_correlation')
        if huf_corr_ibu:
            figname_ibu_corr = os.path.join(outputdir, f"BinCorr_{obs}_IBU")
            logger.info(f" Plot bin correlations: {figname_ibu_corr}")
            plotter.plot_correlations(
                figname_ibu_corr,
                huf_corr_ibu.values(),
                huf_corr_ibu.axes[0].edges
            )
        else:
            logger.warning(f"No 'ibu_correlation' for observable {obs}")

def plot_response(obs, histograms_d, outputdir):

    resp = histograms_d[obs].get("response")

    if resp:
        figname_resp = os.path.join(outputdir, f"Response_{obs}")
        logger.info(f" Plot response: {figname_resp}")
        plotter.plot_response(figname_resp, resp, obs, cmap='Blues')
    else:
        logger.warning(f"No 'response' for observable {obs}")

def plot_iteration_history(obs, histograms_d, obsConfig_d, outputdir):

    hists_uf_alliters = histograms_d[obs].get('unfolded_alliters')

    if hists_uf_alliters:
        iteration_dir = os.path.join(outputdir, 'Iterations')
        if not os.path.isdir(iteration_dir):
            logger.info(f"Create directory {iteration_dir}")
            os.makedirs(iteration_dir)

        figname_alliters = os.path.join(iteration_dir, f"Unfold_AllIterations_{obs}")
        logger.info(f" Plot unfolded distributions at every iteration: {figname_alliters}")

        hist_gen = histograms_d[obs].get('prior')
        hist_truth = histograms_d[obs].get('truth')

        obs_list = obs.split('_vs_')

        plotter.plot_distributions_iteration(
            figname_alliters,
            hists_uf_alliters,
            hist_gen,
            hist_truth,
            xlabel = obsConfig_d[obs_list[0]]['xlabel'],
            ylabel = obsConfig_d[obs_list[0]]['ylabel']
        )
    else:
        logger.warning(f"No 'unfolded_alliters' for observable {obs}")

def plot_unfolded_allruns(obs, histograms_d, obsConfig_d, outputdir):
    hists_uf_allruns = histograms_d[obs].get("unfolded_allruns")
    if hists_uf_allruns is not None and len(hists_uf_allruns) > 1:
        allruns_dir = os.path.join(outputdir, 'AllRuns')
        if not os.path.isdir(allruns_dir):
            logger.info(f"Create directory {allruns_dir}")
            os.makedirs(allruns_dir)

        # unfolded distributions from all runs
        figname_rs = os.path.join(allruns_dir, f"Unfold_AllRuns_{obs}")
        logger.info(f" Plot unfolded distributions from all runs: {figname_rs}")

        hist_gen = histograms_d[obs].get('prior')
        hist_truth = histograms_d[obs].get('truth')

        obs_list = obs.split('_vs_')

        plotter.plot_distributions_resamples(
            figname_rs,
            hists_uf_allruns,
            hist_gen,
            hist_truth,
            xlabel = obsConfig_d[obs_list[0]]['xlabel'],
            ylabel = obsConfig_d[obs_list[0]]['ylabel']
            )

def plot_bin_entries_distributions(obs, histograms_d, outputdir):
    hists_uf_all =  histograms_d[obs].get('unfolded_all')
    if hists_uf_all is not None and len(hists_uf_all) > 1:
        allruns_dir = os.path.join(outputdir, 'AllRuns')
        if not os.path.isdir(allruns_dir):
            logger.info(f"Create directory {allruns_dir}")
            os.makedirs(allruns_dir)

        figname_bindistr = os.path.join(allruns_dir, f"Unfold_BinDistr_{obs}")
        logger.info(f" Plot distributions of bin entries from all runs: {figname_bindistr}")

        hist_truth = histograms_d[obs].get('truth')

        plotter.plot_hists_bin_distr(figname_bindistr, hists_uf_all, hist_truth)

def plot_histograms_from_dict(
    histograms_d,
    outputdir,
    observables=[],
    obsConfig_d={}, # dict for observable configurations
    include_ibu=False,
    plot_verbosity=1
    ):

    if plot_verbosity < 1:
        logger.debug(f"plot_verbosity is set to {plot_verbosity}. skip plotting")
        return

    # output directory
    if not os.path.isdir(outputdir):
        logger.info(f"Create output directory {outputdir}")
        os.makedirs(outputdir)

    # observables
    if not observables:
        observables = list(histograms_d.keys())

    # loop over observables
    for obs in observables:
        logger.info(obs)

        if not obs in histograms_d:
            logger.warn(f"No histograms for observable {obs}")
            continue

        ###
        # unfolded distributions
        plot_distributions_unfold(
            obs,
            histograms_d,
            obsConfig_d,
            outputdir = outputdir,
            include_ibu = include_ibu,
            plot_verbosity = plot_verbosity
        )

        ###
        # reco-level distributions
        plot_distributions_reco(
            obs, histograms_d, obsConfig_d, outputdir = outputdir)

        ######
        if plot_verbosity < 2:
            continue
        # More plots if plot_verbosity >= 2
        # -pp

        ###
        # bin correlations
        plot_bin_correlations(
            obs, histograms_d, outputdir=outputdir, include_ibu=include_ibu)

        ###
        # Response
        plot_response(obs, histograms_d, outputdir)

        ###
        # Iteration history
        plot_iteration_history(obs, histograms_d, obsConfig_d, outputdir)

        ###
        # All runs
        plot_unfolded_allruns(obs, histograms_d, obsConfig_d, outputdir)

        ######
        if plot_verbosity < 5:
            continue
        # Usually skip plotting these unless really necessary
        # -ppppp
        ###
        # Distributions of bin entries
        plot_bin_entries_distributions(obs, histograms_d, outputdir)

def plot_histograms(
    fpath_histograms,
    outputdir=None,
    observables=[],
    obsConfig_d={}, # dict for observable configurations
    include_ibu=False,
    plot_verbosity=1
    ):

    if plot_verbosity < 1:
        logger.debug(f"plot_verbosity is set to {plot_verbosity}. skip plotting")
        return

    # output directory
    if not outputdir:
        outputdir = os.path.dirname(fpath_histograms)

    # read histograms from the file
    logger.info(f"Read histograms from {fpath_histograms}")
    histograms_d = myhu.read_histograms_dict_from_file(fpath_histograms)

    plot_histograms_from_dict(
        histograms_d,
        outputdir = outputdir,
        observables = observables,
        obsConfig_d = obsConfig_d,
        include_ibu = include_ibu,
        plot_verbosity = plot_verbosity
    )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Make plots from histograms")

    parser.add_argument("fpath_histograms", type=str, action=util.
    ParseEnvVar, 
                        help="File path to the .root file containing histograms")
    parser.add_argument("-o", "--outputdir", type=str, action=util.ParseEnvVar, 
                        help="Output directory. If not provided, use the direcotry from fpath_histogram.")
    parser.add_argument("--observables", nargs='+', default=[],
                        help="List of observables to plot histograms. If not provided, plot all that are available")
    parser.add_argument('--include-ibu', action='store_true',
                        help="If True, plot IBU result if available")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="If True, set logging level to DEBUG")
    parser.add_argument("--observable-config", type=str, action=util.ParseEnvVar,
                        default='${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json',
                        help="Path to the observable config file.")
    parser.add_argument('-p', '--plot-verbosity', action='count', default=1,
                        help="Plot verbose level. '-ppp' to make all plots.")

    args = parser.parse_args()

    util.configRootLogger()

    # logger
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    plot_histograms(
        fpath_histograms = args.fpath_histograms,
        outputdir = args.outputdir,
        observables = args.observables,
        obsConfig_d = util.read_dict_from_json(args.observable_config),
        include_ibu = args.include_ibu,
        plot_verbosity = args.plot_verbosity
    )