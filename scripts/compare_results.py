import os

import histogramming as myhu
import plotter
import util

import logging
logger = logging.getLogger("compare_results")

hnames_to_compare = [
    'unfolded', 'unfolded_corrected', 'relativeDiffXs', 'absoluteDiffXs',
    'prior', 'prior_noflow', 'truth', 'truth_noflow',
    'reco_data', 'reco_sig', 'reco_bkg',
    'ibu', 'ibu_corrected', 'relativeDiffXs_ibu', 'absoluteDiffXs_ibu',
    'acceptance', 'efficiency'
]

def compare_results(
        fpaths_histogram,
        labels = [],
        observables = [],
        outputdir = '.',
        keyword_filter = '',
        observable_config = 'configs/observables/vars_ttbardiffXs_pseudotop.json',
        verbose = False
    ):

    logging.basicConfig(level=logging.INFO)
    if verbose:
        logger.setLevel(logging.DEBUG)

    if not labels:
        labels = [f"label_{i}" for i in range(len(fpaths_histogram))]
    else:
        assert(len(fpaths_histogram)==len(labels))

    # take the first in the list as the denominator
    logger.info(f"Read histograms from {fpaths_histogram[0]} as the reference")
    histograms_denom = myhu.read_histograms_dict_from_file(fpaths_histogram[0])
    label_denom = labels[0]

    logger.info(f"Read histograms from {fpaths_histogram[1:]}")
    histograms_numer = [myhu.read_histograms_dict_from_file(fp) for fp in fpaths_histogram[1:]]
    labels_numer = labels[1:]

    colors = plotter.get_default_colors(len(labels))
    color_denom = colors[0]
    colors_numer = colors[1:]

    if not observables:
        observables = list(histograms_denom.keys())

    # observable config
    obsCfg_d = util.read_dict_from_json(observable_config)

    for obs in observables:
        logger.info(obs)

        obs_dir = os.path.join(outputdir, obs)
        if not os.path.isdir(obs_dir):
            logger.info(f"Create output directory {obs_dir}")
            os.makedirs(obs_dir)

        if not obs in histograms_denom:
            logger.warn(f"No histograms for observable {obs}")
            continue

        for hname in hnames_to_compare:
            if not keyword_filter in hname:
                continue

            if not hname in histograms_denom[obs]:
                logger.debug(f"No histogram {hname}")
                continue

            # make comparison plot
            figname = os.path.join(obs_dir, hname)
            logger.info(f"Create plot {figname}")

            # plot styles
            draw_opts_denom = {'label':label_denom, 'color':color_denom, 'xerr':True, 'histtype':'step', 'linewidth':1}
            draw_opts_numer = [{'label':ln, 'color':cn, 'xerr':True, 'histtype':'step', 'linewidth':1} for ln, cn in zip(labels_numer, colors_numer)]

            plotter.plot_histograms_and_ratios(
                figname,
                hists_numerator = [hist_d[obs][hname] for hist_d in histograms_numer],
                draw_options_numerator = draw_opts_numer,
                hist_denominator = histograms_denom[obs][hname],
                draw_option_denominator = draw_opts_denom,
                xlabel = ' vs '.join([obsCfg_d[ob]['xlabel'] for ob in obs.split('_vs_')]),
                ylabel = "",
                ylabel_ratio = f"Ratio to {label_denom}"
            )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("fpaths_histogram", nargs='+', type=str,
                        help="List of file paths to histograms for comparison")
    parser.add_argument("-l", "--labels", nargs='+', type=str,
                        help="List of labels for histograms")
    parser.add_argument("--observables", nargs='+', type=str,
                        help="List of observable names. If empty, use all that are available in the histogram files")
    parser.add_argument("-o", "--outputdir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("-k", "--keyword-filter", type=str, default="",
                        help="Keyword to filter histograms for comparison")
    parser.add_argument("--observable-config", type=str, 
                        default="configs/observables/vars_ttbardiffXs_pseudotop.json",
                        help="Path to the observable configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="If True, raise the logging level to debug, otherwise info")

    args = parser.parse_args()

    compare_results(**vars(args))