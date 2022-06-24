#!/usr/bin/env python3
import os
import numpy as np

import util
import histogramming as myhu
import plotter
from createRun2Config import syst_dict

import logging
logger = logging.getLogger('EvalSystError')

def collect_all_histograms(
        results_topdir,
        nominal_dir = None,
        hist_filename = 'histograms.root',
        systematics = ['all'],
        resdir_name = 'output_run2'
    ):

    histograms_dict_all = {}

    # read nominal unfolded histogams
    logger.debug("Read nominal histograms")

    if nominal_dir is None:
        nominal_dir = os.path.join(results_topdir, "nominal", resdir_name)

    fpath_hists_nominal = os.path.join(nominal_dir, hist_filename)

    if not os.path.isfile(fpath_hists_nominal):
        logger.error(f"No file found: {fpath_hists_nominal}")
        return

    histograms_dict_all["nominal"] = myhu.read_histograms_dict_from_file(
        fpath_hists_nominal)

    # systematic uncertainties
    include_all_syst = systematics==['all']

    for k in syst_dict:
        prefix = syst_dict[k]["prefix"]
        for s in syst_dict[k]["uncertainties"]:
            syst_name = f"{prefix}_{s}"

            if not include_all_syst and not syst_name in systematics:
                # skip this one
                continue

            histograms_dict_all[syst_name] = {}

            for v in syst_dict[k]["variations"]:
                syst_name_v = f"{prefix}_{s}_{v}"
                logger.debug(f"Read histograms from {syst_name_v}")

                # result directory of syst_name_v 
                syst_dir = os.path.join(results_topdir, syst_name_v, resdir_name)

                # path to the histogram file
                fpath_hists_syst = os.path.join(syst_dir, hist_filename)

                if not os.path.isfile(fpath_hists_syst):
                    logger.error(f"No file found: {fpath_hists_syst}")
                    continue

                histograms_dict_all[syst_name][v] = myhu.read_histograms_dict_from_file(fpath_hists_syst)

    return histograms_dict_all

def compute_bin_uncertainties(hist_nominal, hist_syst_up, hist_syst_down):

    relerr_up = hist_syst_up.values() / hist_nominal.values() - 1.
    relerr_down = hist_syst_down.values() / hist_nominal.values() - 1.

    return relerr_up, relerr_down

def compute_bin_uncertainties_allruns(
        hists_nominal_allruns,
        hists_syst_up_allruns,
        hists_syst_down_allruns,
        average = False
    ):

    if hists_nominal_allruns is None or hists_syst_up_allruns is None or hists_syst_down_allruns is None:
        logger.warn("Histograms of all runs are not available")
        return

    # list of relative bin errors
    relerr_up_allruns = []
    relerr_down_allruns = []

    for h_nom, h_up, h_down in zip(hists_nominal_allruns, hists_syst_up_allruns, hists_syst_down_allruns):
        relerr_up_allruns.append(h_up.values() / h_nom.values() - 1.)
        relerr_down_allruns.append(h_down.values() / h_nom.values() - 1.)

    if average:
        relerr_up = np.mean(relerr_up_allruns, axis=0)
        relerr_down = np.mean(relerr_down_allruns, axis=0)

        return relerr_up, relerr_down

    else:
        return [(err_u, err_d) for err_u, err_d in zip(relerr_up_allruns, relerr_down_allruns)]

def plot_systematic_bin_errors(
        observable,
        histograms_d,
        outdir,
        highlight = '',
        label_total = "Detector calibration"
    ):

    # nominal histogram
    h_nominal = histograms_d['nominal'][observable]['unfolded']

    # systematic uncertainties

    # for total bin errors
    sqsum_up = 0
    sqsum_down = 0

    nSyst = 0
    last_syst = ''

    relerrs_highlight = None

    for syst_name in histograms_d:
        if syst_name == 'nominal':
            continue

        nSyst += 1
        last_syst = syst_name

        hists_syst = []
        for variation in histograms_d[syst_name]:
            hists_syst.append(
                histograms_d[syst_name][variation][observable]['unfolded']
                )
        assert(len(hists_syst)==2)

        relerrs_syst = compute_bin_uncertainties(h_nominal, *hists_syst)
        # (relerr_var1, relerr_var2)

        if syst_name == highlight:
            relerrs_highlight = relerrs_syst

        # add to total uncertainties
        relerr_syst_up = np.maximum(*relerrs_syst)
        relerr_syst_down = np.minimum(*relerrs_syst)
        #print(syst_name, relerr_syst_up, relerr_syst_down)
        #assert( np.all(relerr_syst_up>=0.) and np.all(relerr_syst_down<=0.) )

        sqsum_up += relerr_syst_up * relerr_syst_up
        sqsum_down += relerr_syst_down * relerr_syst_down

    # Total bin uncertainties
    relerrs_syst_total = (np.sqrt(sqsum_up), -1*np.sqrt(sqsum_down))
    #print("total", *relerrs_syst_total)

    ####
    # Make plot
    errors_toplot = []
    draw_options = []

    errors_toplot.append(relerrs_syst_total)

    if nSyst == 1: # only one systematic uncertainty to plot:
        draw_options.append({
            'label': last_syst, 'edgecolor':'tab:blue', 'facecolor':'none',
            'hatch':'///'
            })
    else:
        draw_options.append({
            'label': label_total, 'edgecolor':'none', 'facecolor':'tab:blue'
            })

    if relerrs_highlight:
        errors_toplot.append(relerrs_highlight)
        draw_options.append({
            'label': highlight, 'edgecolor':'black', 'facecolor':'none',
            'hatch':'///'
            })

    figname = os.path.join(outdir, f"errors_syst_{observable}")
    logger.info(f"  Make plot: {figname}")

    plotter.plot_uncertainties(
        figname = figname,
        bins = h_nominal.axes[0].edges,
        uncertainties = errors_toplot,
        draw_options = draw_options,
        xlabel = h_nominal.axes[0].label,
        ylabel = 'Uncertainty'
        )

def compare_syst_computations(
        observable,
        histograms_d,
        outdir,
        syst_name
    ):

    if not syst_name:
        return

    # nominal
    h_nominal = histograms_d['nominal'][observable]['unfolded']
    h_nominal_allruns = histograms_d['nominal'][observable].get('unfolded_allruns')

    # syst
    h_syst = []
    h_syst_allruns = []
    for variation in histograms_d[syst_name]:
        h_syst.append(
            histograms_d[syst_name][variation][observable]['unfolded']
            )

        h_syst_allruns.append(
            histograms_d[syst_name][variation][observable].get('unfolded_allruns')
            )

    assert(len(h_syst)==2)

    # option 1: compute from the central unfolded histograms
    relerrs = compute_bin_uncertainties(h_nominal, *h_syst)

    # option 2: compute from all runs
    relerrs_allruns = compute_bin_uncertainties_allruns(
        h_nominal_allruns, *h_syst_allruns
        )

    # also the average
    relerrs_avg = compute_bin_uncertainties_allruns(
        h_nominal_allruns, *h_syst_allruns, average = True
        )

    # plot
    errors_toplot, draw_options = [], []

    errors_toplot.append(relerrs)
    draw_options.append({
        'label':'Baseline', 'edgecolor':'black', 'facecolor':'none', 'lw':1
        })

    colors = plotter.get_random_colors(len(relerrs_allruns))
    for i, (relerr, c) in enumerate(zip(relerrs_allruns, colors)):
        errors_toplot.append(relerr)
        draw_options.append({
            'label': "All Runs" if i==0 else None,
            'edgecolor':c, 'facecolor':'none', 'ls':'--', 'alpha':0.5, 'lw':0.5
            })

    errors_toplot.append(relerrs_avg)
    draw_options.append({
        'label': "Average all runs", 'edgecolor':'tab:red', 'facecolor':'none',
        'lw':0.5
        })

    figname = os.path.join(outdir, f"compare_errors_syst_{observable}")
    logger.info(f"  Make plot: {figname}")

    plotter.plot_uncertainties(
        figname = figname,
        bins = h_nominal.axes[0].edges,
        uncertainties = errors_toplot,
        draw_options = draw_options,
        xlabel = h_nominal.axes[0].label,
        ylabel = 'Uncertainty'
        )

def evaluate_systematic_uncertainties(
        results_topdir,
        nominal_dir = None,
        hist_filename = 'histograms.root',
        resdir_name = 'output_run2',
        outdir = 'SystErrors',
        systematics = ['all'],
        highlight = '',
        compare_methods = False
    ):

    # Get nominal histograms as well as ones for systematic uncertainties
    histograms_dict = collect_all_histograms(
        results_topdir,
        nominal_dir,
        hist_filename,
        systematics,
        resdir_name
        )

    # loop over observables
    for ob in histograms_dict["nominal"]:
        logger.info(f"{ob}")

        plot_systematic_bin_errors(
            ob,
            histograms_dict,
            outdir,
            highlight
            )

        if compare_methods:
            compare_syst_computations(
                ob,
                histograms_dict,
                outdir = outdir,
                syst_name = highlight
                )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("results_topdir", type=str,
                        help="Top directory of the unfolding results")
    parser.add_argument("--resdir-name", type=str, default="output_run2",
                        help="Unfolding result directory name")
    parser.add_argument("--nominal-dir", type=str,
                        help="Path to the nominal directory. If not provided, deduce from results_topdir and resdir_name")
    parser.add_argument("--hist-filename", type=str, default="histograms.root",
                        help="Name of the histogram file")
    parser.add_argument("-o", "--outdir", type=str, default="SystErrors",
                        help="Output directory")
    parser.add_argument("-s", "--systematics", type=str, nargs="+",
                        default=['all'],
                        help="List of systematic uncertainties to evaluate. A special case: 'all' includes all systematics")
    parser.add_argument("--highlight", type=str, default='',
                        help="Name of the systematic uncertainty to plot individually")
    parser.add_argument("--compare-methods", action='store_true',
                        help="If True, compare methods to compute systematic uncertainties")

    args = parser.parse_args()

    util.configRootLogger()

    if not os.path.isdir(args.outdir):
        logger.info(f"Create output directory {args.outdir}")
        os.makedirs(args.outdir)

    evaluate_systematic_uncertainties(**vars(args))
