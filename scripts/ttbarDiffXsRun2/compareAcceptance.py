#!/usr/bin/env python3
import os

import util
import histogramming as myhu
import plotter

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
        hist_omf_before = histograms_before_d[obs][histkey_omnifold]
        hist_ibu_after = histograms_after_d[obs][histkey_ibu]
        hist_ibu_before = histograms_before_d[obs][histkey_ibu]

        hist_omf_ratio = myhu.divide(hist_omf_after, hist_omf_before)
        hist_ibu_ratio = myhu.divide(hist_ibu_after, hist_ibu_before)

        # plot
        plotter.plot_histograms_and_ratios(
            figname = os.path.join(outputdir, f"{obs}_omnifold"),
            hists_numerator = [hist_omf_after],
            hist_denominator = hist_omf_before,
            draw_options_numerator = [draw_opt_after],
            draw_option_denominator = draw_opt_before,
            xlabel = obs,
            ylabel = "Events",
            ylabel_ratio = "Ratio",
            title = "UniFold"
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
            title = "IBU"
        )

        plotter.plot_histograms_and_ratios(
            figname = os.path.join(outputdir, f"{obs}_ratio"),
            hists_numerator = [hist_omf_ratio],
            hist_denominator = hist_ibu_ratio,
            draw_options_numerator = [{'label':'UniFold','color':'tab:red','histtype':'step'}],
            draw_option_denominator = {'label':'IBU','color':'grey','histtype':'step'},
            xlabel = obs,
            ylabel = "Ratio",
            ylabel_ratio = "UniFold / IBU",
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