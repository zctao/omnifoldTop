#!/usr/bin/env python3
import os

import util
import histogramming as myhu
import plotter
from FlattenedHistogram import FlattenedHistogram2D, FlattenedHistogram3D

import logging
logger = logging.getLogger('compareAcceptance')
util.configRootLogger()

draw_opt_after = {"label":"With corrections", "color":"black", "histtype":"step"}
draw_opt_before = {"label":"Without corrections", "color":"tab:red", "histtype":"step"}

# Compare the effect of acceptance corrections in OmniFold to IBU
def compareAcceptance(
    fpath_histograms_with_correction, # str
    fpath_histograms_without_correction, # str
    observables = [], # list of str; if empty, take all that are availabe from the histograms
    histkey_omnifold = "unfolded",
    histkey_ibu = "ibu",
    outputdir = ".", # str; output directory
    ):

    if not os.path.isdir(outputdir):
        logger.info(f"Create output directory {outputdir}")
        os.makedirs(outputdir)

    # read histograms
    logger.debug(f"Load histograms from with acceptance corrections from {fpath_histograms_with_correction}")
    histograms_after_d = myhu.read_histograms_dict_from_file(fpath_histograms_with_correction)

    logger.debug(f"Load histograms without acceptance corrections from {fpath_histograms_without_correction}")
    histograms_before_d = myhu.read_histograms_dict_from_file(fpath_histograms_without_correction)

    # Take observables from the histograms if none is specified
    if not observables:
        observables = set(histograms_after_d.keys()) & set(histograms_before_d.keys())
        logger.debug(f"Observables: {observables}")

    # loop over observables
    for obs in observables:
        logger.info(obs)

        hist_omf_after = histograms_after_d[obs][histkey_omnifold]
        if isinstance(hist_omf_after, FlattenedHistogram2D) or isinstance(hist_omf_after, FlattenedHistogram3D):
            hist_omf_after = hist_omf_after.flatten()

        hist_omf_before = histograms_before_d[obs][histkey_omnifold]
        if isinstance(hist_omf_before, FlattenedHistogram2D) or isinstance(hist_omf_before, FlattenedHistogram3D):
            hist_omf_before = hist_omf_before.flatten()

        hist_ibu_after = histograms_after_d[obs][histkey_ibu]
        if isinstance(hist_ibu_after, FlattenedHistogram2D) or isinstance(hist_ibu_after, FlattenedHistogram3D):
            hist_ibu_after = hist_ibu_after.flatten()

        hist_ibu_before = histograms_before_d[obs][histkey_ibu]
        if isinstance(hist_ibu_before, FlattenedHistogram2D) or isinstance(hist_ibu_before, FlattenedHistogram3D):
            hist_ibu_before = hist_ibu_before.flatten()

        hist_omf_ratio = myhu.divide(hist_omf_after, hist_omf_before)
        hist_ibu_ratio = myhu.divide(hist_ibu_after, hist_ibu_before)

        ndim_obs = len(obs.split('_vs_'))
        uf_label = "MultiFold" if ndim_obs > 1 else "UniFold"

        # plot
        logscale_x = obs in ['ptt_vs_mtt', 'th_pt_vs_mtt']

        plotter.plot_histograms_and_ratios(
            figname = os.path.join(outputdir, f"{obs}_omnifold"),
            hists_numerator = [hist_omf_after],
            hist_denominator = hist_omf_before,
            draw_options_numerator = [draw_opt_after],
            draw_option_denominator = draw_opt_before,
            xlabel = obs,
            ylabel = "Events",
            ylabel_ratio = "Ratio",
            title = uf_label,
            log_xscale = logscale_x
        )

        plotter.plot_histograms_and_ratios(
            figname = os.path.join(outputdir, f"{obs}_ibu"),
            hists_numerator = [hist_ibu_after],
            hist_denominator = hist_ibu_before,
            draw_options_numerator = [draw_opt_after],
            draw_option_denominator = draw_opt_before,
            xlabel = obs,
            ylabel = "Events",
            ylabel_ratio = "Ratio",
            title = "IBU",
            log_xscale = logscale_x
        )

        plotter.plot_histograms_and_ratios(
            figname = os.path.join(outputdir, f"{obs}_ratio"),
            hists_numerator = [hist_omf_ratio],
            hist_denominator = hist_ibu_ratio,
            draw_options_numerator = [{'label':uf_label,'color':'tab:red','histtype':'step'}],
            draw_option_denominator = {'label':'IBU','color':'grey','histtype':'step'},
            xlabel = obs,
            ylabel = "Ratio",
            ylabel_ratio = f"{uf_label} / IBU",
            log_xscale = logscale_x,
            #ratio_lim = [0,2],
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("fpath_histograms_with_correction", type=str,
                        help="File path to histograms with corrections")
    parser.add_argument("fpath_histograms_without_correction", type=str,
                        help="File path to histograms without corrections")
    parser.add_argument("--observables", type=str, nargs="+",
                        help="List of observables")
    parser.add_argument("-o", "--outputdir", default='.', type=str,
                        help="Output directory")

    args = parser.parse_args()

    compareAcceptance(**vars(args))