#!/usr/bin/env python3
import os
import numpy as np

import util
import histogramming as myhu

import plotter
from plotter import set_default_colors

import logging
logger = logging.getLogger('EvalUnfoldError')

def get_unfolded_histogram_from_dict(
        observable, # str, observable name
        histograms_dict, # dict, histograms dictionary
        nensembles = None # int, number of ensembles for stablizing unfolding
    ):

    if nensembles is None:
        # return the one from dict computed from all available ensembles
        return histograms_dict[observable].get('unfolded')
    else:
        hists_allruns = histograms_dict[observable].get('unfolded_allruns')
        if nensembles > len(hists_allruns):
            logger.error(f"The required number of ensembles {nensembles} is larger than what is available: {len(hists_allruns)}")
            return None

        # Use a subset of the histograms from all runs
        h_unfolded = myhu.average_histograms(hists_allruns[:nensembles])
        h_unfolded.axes[0].label = histograms_dict[observable]['unfolded'].axes[0].label

        return h_unfolded

def compute_bin_uncertainties(
        hists_resample, # list of hist objects
        hist_nominal = None, # hist, nominal unfolded histogram
        hist_ibu = None, # hist, unfolded histogram from IBU
    ):

    # Each element of the list hists_resample is the unfolded histogram given a
    # toy dataset by fluctuating data weights.
    # The bin uncertainties of each element in hists_resample is estimated by
    # re-running the unfolding multiple times with the same input datsets, i.e.
    # "model uncertainty". The mean and standard error of the mean in each bin of
    # all runs are taken as the bin entry and error, respectively.

    assert(len(hists_resample) > 0)

    errors_d = {}

    # statistical bin uncertainties from bootstrap
    mean_resample = myhu.get_mean_from_hists(hists_resample)
    sigma_resample = myhu.get_sigma_from_hists(hists_resample)

    # relative statistical bin errors
    h_norm = mean_resample if hist_nominal is None else hist_nominal.values()

    errors_d["statistical"] = sigma_resample / h_norm

    # also record the model uncertainties
    if hist_nominal:
        val_model, sigma_model = myhu.get_values_and_errors(hist_nominal)
        errors_d["model"] = sigma_model / val_model

    vals_model, sigmas_model = myhu.get_values_and_errors(hists_resample)
    assert(len(vals_model) == len(hists_resample))
    assert(len(sigmas_model) == len(hists_resample))

    errors_d["model_allruns"] = np.asarray(sigmas_model) / np.asarray(vals_model)

    # compare to IBU if it is available
    if hist_ibu:
        val_ibu, err_ibu = myhu.get_values_and_errors(hist_ibu)
        errors_d["ibu"] = err_ibu / val_ibu

    return errors_d

def plot_hists_resample(
        figname,
        histograms_resample,
        xlabel,
        colors = None
    ):

    plotter.plot_multiple_histograms(
        figname,
        histograms_resample,
        xlabel = xlabel,
        log_scale = False,
        colors = colors,
        alpha = 0.8, marker='.', markersize=1,
        )

def plot_bin_errors(
        figname,
        errors_dict,
        xlabel,
        bin_edges,
        plot_allruns = False,
        colors = None
    ):

    relerrors, draw_options = [], []

    # stat
    relerrors.append(errors_dict["statistical"])
    draw_options.append({
        "label": "Stat.", "edgecolor": "black", "facecolor":"none"})

    # model
    if 'model' in errors_dict:
        relerrors.append(errors_dict['model'])
        draw_options.append({
            "label": "Model", "edgecolor": "tab:red", "facecolor":"none"})

    if plot_allruns:
        if colors is None:
            colors = set_default_colors( len(errors_dict["model"]) )
        else:
            assert(len(colors) == len(errors_dict["model"]))

        for i, (err, c) in enumerate(zip(errors_dict["model"], colors)):
            relerrors.append(err)
            draw_options.append({
                "label": "Model (all runs)" if i==0 else None,
                "edgecolor": c, "facecolor": "none", "ls": "--"
                })

    # IBU bootstrap
    if 'ibu' in errors_dict:
        relerrors.append(errors_dict['ibu'])
        draw_options.append({
            "label": "IBU bootstrap",
            "edgecolor": "grey", "facecolor": "none", "ls": "-."
            })

    # make the plot
    plotter.plot_uncertainties(
        figname = figname,
        bins = bin_edges,
        uncertainties = relerrors,
        draw_options = draw_options,
        xlabel = xlabel,
        ylabel = "Uncertainty"
    )

def evaluate_unfolding_stats_uncertainty(
        resample_topdir,
        nresamples,
        outdir = '.',
        resample_prefix = "resample",
        histograms_name = "histograms.root",
        include_ibu = False,
        nominal_dir = None,
        nruns_model = None
    ):

    hists_dict_resamples = []
    for i in range(nresamples):
        # path to the histogram root file
        fpath_hists = os.path.join(
            resample_topdir, f"{resample_prefix}{i}", histograms_name
            )

        # read the file into dict
        hists_dict_resamples.append(
            myhu.read_histograms_dict_from_file(fpath_hists)
            )

    if nominal_dir:
        fpath_hists_nominal = os.path.join(nominal_dir, histograms_name)
        hists_dict_nominal = myhu.read_histograms_dict_from_file(
            fpath_hists_nominal
            )
    else:
        hists_dict_nominal = None

    # loop over observables
    for ob in hists_dict_resamples[0]:

        # collect histograms from all resamples for this observable
        hs_resample = []
        for hists_d in hists_dict_resamples:
            hs_resample.append(
                get_unfolded_histogram_from_dict(ob, hists_d, nruns_model)
                )

        # nominal histogram if available
        if hists_dict_nominal:
            h_nominal = get_unfolded_histogram_from_dict(
                ob, hists_dict_nominal, nruns_model
                )
        else:
            h_nominal = None

        h_ibu = None
        if include_ibu:
            if hists_dict_nominal:
                h_ibu = hists_dict_nominal[ob].get('ibu')
            else:
                h_ibu = hists_dict_resamples[0][ob].get('ibu')

            if h_ibu is None:
                logger.warn(f"IBU unfolded histogram is not available for {ob}")

        errors_ob = compute_bin_uncertainties(
            hs_resample, h_nominal, h_ibu)

        logger.info(f"Plot bin errors for {ob}")

        colors = set_default_colors( len(hs_resample) )

        plot_bin_errors(
            os.path.join(outdir, f"errors_{ob}"),
            errors_ob,
            xlabel = hs_resample[0].axes[0].label,
            bin_edges = hs_resample[0].axes[0].edges,
            colors = colors
            )

        logger.info(f"Plot resampled histograms of {ob}")
        plot_hists_resample(
            os.path.join(outdir, f"unfolded_resamples_{ob}"),
            hs_resample,
            xlabel = hs_resample[0].axes[0].label,
            colors = colors
            )

def evaluate_unfolding_model_uncertainty(
        nominal_dir,
        nensembles_model,
        outdir = '.',
        histograms_name = "histograms.root",
    ):
    # plot model uncertainties with different nensembles

    # read histograms
    hists_dict_nominal = myhu.read_histograms_dict_from_file(
         os.path.join(nominal_dir, histograms_name))

    # default list of colors
    colors = set_default_colors(len(nensembles_model))

    # loop over observables
    for ob in hists_dict_nominal:

        # unfolded distributions for different nensembles
        xlabel = None
        bin_edges = None

        relerrors, draw_options = [], []
        for n, c in zip(nensembles_model, colors):
            h_nominal_n = get_unfolded_histogram_from_dict(
                ob, hists_dict_nominal, n
                )

            if xlabel is None:
                xlabel = h_nominal_n.axes[0].label
            if bin_edges is None:
                bin_edges = h_nominal_n.axes[0].edges

            val, sigma = myhu.get_values_and_errors(h_nominal_n)
            relerrors.append( sigma / val )
            draw_options.append({
                "label": f"N = {n}", "edgecolor": c, "facecolor": "none"})

        # plot
        logger.info(f"Plot model uncertainties for {ob}")
        plotter.plot_uncertainties(
            figname = os.path.join(outdir, f"model_uncertainty_{ob}"),
            bins = bin_edges,
            uncertainties = relerrors,
            draw_options = draw_options,
            xlabel = xlabel,
            ylabel = "Uncertainty"
            )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--nominal-dir', type=str,
                        help="Directory of the nominal unfolding result")
    parser.add_argument('--resample-topdir', type=str,
                        help="Top directory that contains unfolding results from resampling")
    parser.add_argument('--nresamples', type=int,
                        help="Number of resamples/toy datasets for estimating statistical uncertainties")
    parser.add_argument('--resample-prefix', type=str, default="resample",
                        help="Prefix of the resample directories under top directory")
    parser.add_argument('--histograms-name', type=str, default='histograms.root',
                        help="Name of the root file containing histograms")
    parser.add_argument('-o', '--outdir', type=str, default='UnfoldErrors',
                        help="Output directory")
    parser.add_argument('--include-ibu', action='store_true',
                        help="If True, include IBU bootstrap uncertainties")
    parser.add_argument('--nensembles-model', type=int, nargs='*',
                        help="Number of ensembles for model uncertainty")

    args = parser.parse_args()

    util.configRootLogger()

    if not os.path.isdir(args.outdir):
        logger.info(f"Create output directory: {args.outdir}")
        os.makedirs(args.outdir)

    if args.resample_topdir and args.nresamples:
        # statistical uncertainty
        evaluate_unfolding_stats_uncertainty(
            args.resample_topdir,
            args.nresamples,
            resample_prefix = args.resample_prefix,
            histograms_name = args.histograms_name,
            outdir = args.outdir,
            include_ibu = args.include_ibu,
            nominal_dir = args.nominal_dir,
            nruns_model = max(args.nensembles_model) if args.nensembles_model else None
            )

    if args.nominal_dir and args.nensembles_model:
        # model uncertainties with different number of ensembles
        evaluate_unfolding_model_uncertainty(
            args.nominal_dir,
            args.nensembles_model,
            outdir = args.outdir,
            histograms_name = args.histograms_name
            )
