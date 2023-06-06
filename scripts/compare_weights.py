import os
import numpy as np
import h5py

import histogramming as myhu
import plotter

import logging
logger = logging.getLogger("compare_weights")

def compare_weights(
    fpaths_weights,
    labels = [],
    outputdir = '.',
    array_name="unfolded_weights",
    nbins=25,
    logy=False,
    verbose=False
    ):

    for fpath_w in fpaths_weights:
        if not os.path.isfile(fpath_w):
            logger.error(f"Cannot read weights {fpath_w}")
            return

    logging.basicConfig(level=logging.INFO)
    if verbose:
        logger.setLevel(logging.DEBUG)

    if not labels:
        labels = [f"label_{i}" for i in range(len(fpaths_weights))]
    else:
        assert(len(fpaths_weights)==len(labels))

    # Read the files
    wfiles = [ h5py.File(fp) for fp in fpaths_weights ]

    # Get weights
    weights_list = [ wf[array_name][:,-1,:].ravel() for wf in wfiles ]

    # Make histograms
    # determine binning
    xmin = min(np.min(warr) for warr in weights_list)
    xmax = max(np.max(warr) for warr in weights_list)
    bin_edges = np.linspace(xmin, xmax, nbins+1)

    hists_weights = [ myhu.calc_hist(warr, bin_edges) for warr in weights_list ]

    # Plot
    colors = plotter.get_default_colors(len(wfiles))

    plotter.plot_histograms_and_ratios(
        figname = os.path.join(outputdir, "weights_allruns"),
        hists_numerator = hists_weights,
        draw_options_numerator = [{'label':l, 'color':c, 'histtype':'step'} for l,c in zip(labels, colors)],
        xlabel = "weights",
        log_scale = logy
    )

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
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="If True, raise the logging level to debug, otherwise info")

    args = parser.parse_args()

    compare_weights(**vars(args))