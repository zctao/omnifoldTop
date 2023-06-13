import os

import histogramming as myhu
import plotter
import util
from metrics import compute_Chi2, compute_Delta

import logging
logger = logging.getLogger("compare_histograms")

hnames_to_compare = [
    'unfolded', 'unfolded_corrected', 'relativeDiffXs', 'absoluteDiffXs',
    'prior', 'prior_noflow', 'truth', 'truth_noflow',
    'reco_data', 'reco_sig', 'reco_bkg',
    'ibu', 'ibu_corrected', 'relativeDiffXs_ibu', 'absoluteDiffXs_ibu',
    'acceptance', 'efficiency'
]

def compare_histograms(
        fpaths_histogram,
        labels = [],
        observables = [],
        outputdir = '.',
        keyword_filter = '',
        observable_config = 'configs/observables/vars_ttbardiffXs_pseudotop.json',
        verbose = False,
        compute_chi2 = False,
        compute_delta = False
    ):

    for fpath_hist in fpaths_histogram:
        if not os.path.isfile(fpath_hist):
            logger.error(f"Cannot read file {fpath_hist}")
            return

    logging.basicConfig(format='%(asctime)s %(levelname)-7s %(name)-15s %(message)s')
    logger.setLevel(logging.DEBUG) if verbose else logger.setLevel(logging.INFO)

    if not labels:
        labels = [f"label_{i}" for i in range(len(fpaths_histogram))]
    else:
        #print(fpaths_histogram)
        #print(labels)
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

    metrics_d = {}

    for obs in observables:
        logger.info(obs)

        obs_dir = os.path.join(outputdir, obs)
        if not os.path.isdir(obs_dir):
            logger.info(f"Create output directory {obs_dir}")
            os.makedirs(obs_dir)

        if not obs in histograms_denom:
            logger.warning(f"No histograms for observable {obs}")
            continue

        obs_in_hists_num = all([obs in hist_d for hist_d in histograms_numer])
        if not obs_in_hists_num:
            logger.warning(f"Not all files have histograms for observable {obs}")
            continue

        metrics_d[obs] = {}

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

            hists_numerator = [hist_d[obs][hname] for hist_d in histograms_numer]
            hist_denominator = histograms_denom[obs][hname]

            texts = ["$\\chi^2$/NDF"] if compute_chi2 else []

            metrics_d[obs][hname] = {}

            if compute_chi2 or compute_delta:
                for hnum, dopt in zip(hists_numerator, draw_opts_numer):
                    comp_key = f"{dopt['label']}_vs_{label_denom}"
                    metrics_d[obs][hname][comp_key] = {}

                    if compute_chi2:
                        chi2, ndf = compute_Chi2(hnum, hist_denominator)
                        metrics_d[obs][hname][comp_key]['ndf'] = ndf
                        metrics_d[obs][hname][comp_key]['chi2'] = chi2
                        texts.append(f"{comp_key}: {chi2:.3f}/{ndf}")

                    if compute_delta:
                        metrics_d[obs][hname][comp_key]['delta'] = compute_Delta(hnum, hist_denominator)

            plotter.plot_histograms_and_ratios(
                figname,
                hists_numerator = hists_numerator,
                draw_options_numerator = draw_opts_numer,
                hist_denominator = hist_denominator,
                draw_option_denominator = draw_opts_denom,
                xlabel = ' vs '.join([obsCfg_d[ob]['xlabel'] for ob in obs.split('_vs_')]),
                ylabel = "",
                ylabel_ratio = f"Ratio to {label_denom}",
                stamp_texts = texts
            )

    if compute_chi2 or compute_delta: # write to file
        util.write_dict_to_json(metrics_d, os.path.join(outputdir, "compare_hists.json"))

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
    parser.add_argument("--compute-chi2", action="store_true",
                        help="If True, compute Chi2s")
    parser.add_argument("--compute-delta", action="store_true",
                        help="If True, compute Deltas")

    args = parser.parse_args()

    compare_histograms(**vars(args))