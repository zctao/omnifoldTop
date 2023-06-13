import os
import numpy as np
import h5py
from scipy import stats

import histogramming as myhu
import plotter
import util

import logging
logger = logging.getLogger("compare_weights")

def ks_ordered_weights(weights1, weights2):
    # metrics.ks_2samp_weighted, but without data1 and data2
    # weights1 and weights2 are for the same events
    # they are essentiall ordered by event index
    assert(len(weights1) == len(weights2))

    w1 = weights1/np.mean(weights1)
    w2 = weights2/np.mean(weights2)

    n1 = sum(w1)
    n2 = sum(w2)

    cw1 = np.hstack([0, np.cumsum(w1)/n1])
    cw2 = np.hstack([0, np.cumsum(w2)/n2])

    ks = np.max(np.abs(cw1 - cw2))

    en = np.sqrt(n1 * n2 / (n1 + n2))
    prob = stats.kstwobign.sf(ks * en)

    return ks, prob

def compare_weights(
    fpaths_weights,
    labels = [],
    outputdir = '.',
    array_name="unfolded_weights",
    nbins=25,
    logy=False,
    verbose=False,
    do_ks=False,
    nruns=None
    ):

    for fpath_w in fpaths_weights:
        if not os.path.isfile(fpath_w):
            logger.error(f"Cannot read weights {fpath_w}")
            return

    logging.basicConfig(format='%(asctime)s %(levelname)-7s %(name)-15s %(message)s')
    logger.setLevel(logging.DEBUG) if verbose else logger.setLevel(logging.INFO)

    if not labels:
        labels = [f"label_{i}" for i in range(len(fpaths_weights))]
    else:
        assert(len(fpaths_weights)==len(labels))

    # Read the files
    logger.info(f"Read weights files from: {fpaths_weights}")
    wfiles = [ h5py.File(fp) for fp in fpaths_weights ]

    # Get weights
    logger.debug("Read weight arrays")
    if nruns is None:
        weights_list = [ wf[array_name][:,-1,:].ravel() for wf in wfiles ]
    else:
        weights_list = [ wf[array_name][:nruns,-1,:].ravel() for wf in wfiles ]

    # Make histograms
    # determine binning
    xmin = min(np.min(warr) for warr in weights_list)
    xmax = max(np.max(warr) for warr in weights_list)
    bin_edges = np.linspace(xmin, xmax, nbins+1)

    logger.debug("Make histograms")
    hists_weights = [ myhu.calc_hist(warr, bin_edges) for warr in weights_list ]

    # Plot
    colors = plotter.get_default_colors(len(wfiles))

    figname = os.path.join(outputdir, "weights_allruns")

    logger.info(f"Create plot {figname}")
    plotter.plot_histograms_and_ratios(
        figname = figname,
        hists_numerator = hists_weights,
        draw_options_numerator = [{'label':l, 'color':c, 'histtype':'step'} for l,c in zip(labels, colors)],
        xlabel = "weights",
        log_scale = logy
    )

    if do_ks:
        logger.info("Do KS tests")
        ks_results = {}
        ks_results_ev = {}

        warr_ref = weights_list[0]
        label_ref = labels[0]

        for warr_test, label_test in zip(weights_list[1:], labels[1:]):

            ks_key = f"{label_test}_vs_{label_ref}"
            logger.info(ks_key)

            ks_results[ks_key] = {}
            ks_results_ev[ks_key] = {}

            re_ks = stats.kstest(warr_ref, warr_test)
            ks_results[ks_key]['statistic'] = re_ks.statistic
            ks_results[ks_key]['pvalue'] = re_ks.pvalue

            # ordered by event index
            if len(warr_ref)==len(warr_test):
                re_ks_ev = ks_ordered_weights(warr_ref, warr_test)
                ks_results_ev[ks_key]['statistic'] = re_ks_ev[0]
                ks_results_ev[ks_key]['pvalue'] = re_ks_ev[1]

        # write to file
        logger.info("Write KS test statistic to file")
        if ks_results:
            util.write_dict_to_json(ks_results, os.path.join(outputdir, "weights_kstest.json"))

        if ks_results_ev:
            util.write_dict_to_json(ks_results_ev, os.path.join(outputdir, "weights_kstest_ev.json"))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("fpaths_weights", nargs='+', type=str,
                        help="List of file paths to weights files for comparison")
    parser.add_argument("-l", "--labels", nargs='+', type=str,
                        help="List of labels for histograms")
    parser.add_argument("-o", "--outputdir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--array-name", type=str, default="unfolded_weights",
                        help="Name of the weight array")
    parser.add_argument("--nbins", type=int, default=25,
                        help="Number of bins for the weight histograms")
    parser.add_argument("--logy", action="store_true", 
                        help="If True, plot with log scale for y-axis")
    parser.add_argument("--nruns", type=int,
                        help="Number of runs of weights to use. If None, use all available ")
    parser.add_argument("--do-ks", action="store_true",
                        help="If True, do KS tests")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="If True, raise the logging level to debug, otherwise info")

    args = parser.parse_args()

    compare_weights(**vars(args))