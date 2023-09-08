#!/usr/bin/env python3
import os

import histogramming as myhu
import plotter
import util
import numpy as np

# systematic uncertainties
from ttbarDiffXsRun2.systematics import get_systematics

import logging
logger = logging.getLogger('EvaluateUncertainties')
util.configRootLogger()

def get_unfolded_histogram_from_dict(
    observable, # str, observable name
    histograms_dict, # dict, histograms dictionary
    ibu = False, # bool, if True, read IBU unfolded distribution
    hist_key = 'absoluteDiffXs', # str, unfolded histogram name. 'unfolded', "absoluteDiffXs", or "relativeDiffXs"
    nensembles = None, # int, number of ensembles for stablizing unfolding
    ):

    if ibu:
        if hist_key == 'unfolded':
            hist_key = 'ibu'
        elif hist_key in ["absoluteDiffXs", "relativeDiffXs"]:
            hist_key = f"{hist_key}_ibu"
        else:
            logger.error(f"Unknown hist_key {hist_key}")

    if nensembles is None:
        # return the one from dict computed from all available ensembles
        huf = histograms_dict[observable].get(hist_key)
        if huf is None:
            logger.critical(f"histogram {hist_key} is not available")

        return huf

    else:

        hist_key = f"{hist_key}_allruns"

        hists_allruns = histograms_dict[observable].get(hist_key)
        if hists_allruns is None:
            logger.critical(f"histogram {hist_key} is not available")
            return None

        if nensembles > len(hists_allruns):
            logger.warn(f"The required number of ensembles {nensembles} is larger than what is available: {len(hists_allruns)}")
            nensembles = len(hists_allruns)

        # Use a subset of the histograms from all runs
        h_unfolded = myhu.average_histograms(hists_allruns[:nensembles])
        h_unfolded.axes[0].label = histograms_dict[observable]['unfolded'].axes[0].label

        return h_unfolded

def extract_bin_uncertainties_from_histograms(
    result_dir, # str, directory of unfolding results
    uncertainty_label, # str, label assigned to this uncertainty
    histograms_nominal_d = None, # dict, nominal unfolded histograms. If None, use the ones from result_dir,
    hist_filename = "histograms.root",
    ibu = False, # bool, if True, read IBU unfolded distribution
    hist_key = 'absoluteDiffXs'
    ):

    fpath_hists = os.path.join(result_dir, hist_filename)
    logger.info(f" Read histograms from {fpath_hists}")
    hists_d = myhu.read_histograms_dict_from_file(fpath_hists)

    # bin uncertainties
    unc_d = dict()

    # loop over observables
    for ob in hists_d:
        logger.debug(ob)
        unc_d[ob] = dict()

        h_uf = get_unfolded_histogram_from_dict(ob, hists_d, ibu, hist_key=hist_key)

        values, sigmas = myhu.get_values_and_errors(h_uf)

        if histograms_nominal_d is not None:
            # get the nominal histogram values
            h_nominal = get_unfolded_histogram_from_dict(ob, histograms_nominal_d, ibu, hist_key=hist_key)
            values = h_nominal.values()

        # relative errors
        relerrs = sigmas / values

        # store as a histogram
        unc_d[ob][uncertainty_label] = h_uf.copy()
        myhu.set_hist_contents(unc_d[ob][uncertainty_label], relerrs)
        unc_d[ob][uncertainty_label].view()['variance'] = 0.

    return unc_d

def compute_systematic_uncertainties(
    systematics_topdir,
    histograms_nominal_d,
    systematics_keywords = [],
    hist_filename = "histograms.root",
    every_run = False,
    ibu = False,
    hist_key = 'unfolded',
    normalize = False
    ):

    logger.debug("Compute systematic bin uncertainties")
    syst_unc_d = dict()

    logger.debug("Loop over systematic uncertainty variations")
    for syst_variation in get_systematics(systematics_keywords):
        logger.debug(syst_variation)

        fpath_hist_syst = os.path.join(systematics_topdir, syst_variation, hist_filename)

        # read histograms
        try:
            hists_syst_d = myhu.read_histograms_dict_from_file(fpath_hist_syst)
        except:
            logger.debug(f"No histograms found for {syst_variation}: cannot open {fpath_hist_syst}")
            continue

        # loop over observables
        for ob in hists_syst_d:
            logger.debug(ob)
            if not ob in syst_unc_d:
                syst_unc_d[ob] = dict()

            # get the unfolded distributions
            h_syst = get_unfolded_histogram_from_dict(ob, hists_syst_d, ibu=ibu, hist_key=hist_key)
            h_nominal = get_unfolded_histogram_from_dict(ob, histograms_nominal_d, ibu=ibu, hist_key=hist_key)

            if normalize:
                if hist_key in ["absoluteDiffXs", "relativeDiffXs"]:
                    logger.warn(f"Skip renormalizing histogram {hist_key}!")
                else:
                    myhu.renormalize_hist(h_syst, norm=myhu.get_hist_norm(h_nominal))

            # compute relative bin errors
            relerr_syst = h_syst.values() / h_nominal.values() - 1.

            # store as a histogram
            syst_unc_d[ob][syst_variation] = h_syst.copy()
            myhu.set_hist_contents(syst_unc_d[ob][syst_variation], relerr_syst)
            syst_unc_d[ob][syst_variation].view()['variance'] = 0.
            # set name?

            if every_run: # compute the systematic variation for every run
                syst_unc_d[ob][f"{syst_variation}_allruns"] = list()

                h_syst_allruns = hists_syst_d[ob].get('unfolded_allruns')
                h_nominal_allruns = histograms_nominal_d[ob].get('unfolded_allruns')

                for h_syst_i, h_nominal_i in zip(h_syst_allruns, h_nominal_allruns):
                    if normalize:
                        myhu.renormalize_hist(h_syst_i, norm=myhu.get_hist_norm(h_nominal_i))

                    relerr_syst_i = h_syst_i.values() / h_nominal_i.values() - 1.

                    # store as a histogram
                    herrtmp_i = h_syst_i.copy()
                    myhu.set_hist_contents(herrtmp_i, relerr_syst_i)
                    herrtmp_i.view()['variance'] = 0.

                    syst_unc_d[ob][f"{syst_variation}_allruns"].append(herrtmp_i)

    return syst_unc_d

def plot_fractional_uncertainties(
    bin_uncertainties_dict,
    uncertainties = [], # list of uncertainty labels to plot
    outname_prefix = 'bin_uncertainties',
    highlight = '' #TODO
    ):

    # for plotting
    colors = plotter.get_default_colors(len(uncertainties))

    # loop over observables
    for ob in bin_uncertainties_dict:
        errors_toplot = []
        draw_opts = []

        # loop over uncertainties
        for unc, color in zip(uncertainties, colors):
            if isinstance(unc, tuple):
                # down, up variations
                unc_var1, unc_var2 = unc

                hist_var1 = bin_uncertainties_dict[ob].get(unc_var1)
                if hist_var1 is None:
                    logger.error(f"No entry found for uncertainty {unc_var1}")
                    continue

                hist_var2 = bin_uncertainties_dict[ob].get(unc_var2)
                if hist_var2 is None:
                    logger.error(f"No entry found for uncertainty {unc_var2}")
                    continue

                component_name = os.path.commonprefix([unc_var1, unc_var2])
            else:
                # symmetric
                hist_var1 = bin_uncertainties_dict[ob].get(unc)
                if hist_var1 is None:
                    logger.error(f"No entry found for uncertainty {unc}")
                    continue

                hist_var2 = hist_var1 * -1.

                component_name = unc

            component_name = component_name.strip('_')

            relerr_var1 = myhu.get_values_and_errors(hist_var1)[0] * 100.
            relerr_var2 = myhu.get_values_and_errors(hist_var2)[0] * 100.

            errors_toplot.append((relerr_var1, relerr_var2))
            draw_opts.append({'label': component_name, 'edgecolor': color, "facecolor": 'none'})

        figname = f"{outname_prefix}_{ob}"
        logger.info(f"Make uncertainty plot: {figname}")

        plotter.plot_uncertainties(
            figname = figname,
            bins = hist_var1.axes[0].edges,
            uncertainties = errors_toplot,
            draw_options = draw_opts,
            xlabel = hist_var1.axes[0].label,
            ylabel = 'Uncertainty [%]'
        )

def plot_uncertainties_from_file(fpath):
    unc_d = myhu.read_histograms_dict_from_file(fpath)
    plot_fractional_uncertainties(unc_d)

def compute_total_uncertainties(
    bin_uncertainties_dict,
    uncertainties = [], # list of uncertainty labels to plot
    group_label = 'total'
    ):

    total_uncertainties_d = {}

    # loop over observables
    for ob in bin_uncertainties_dict:

        var_up = 0.
        var_down = 0.

        # loop over uncertainties
        for unc in uncertainties:
            if isinstance(unc, tuple):
                # up and down variations
                unc_var1, unc_var2 = unc

                hist_var1 = bin_uncertainties_dict[ob].get(unc_var1)
                if hist_var1 is None:
                    logger.error(f"No entry found for uncertainty {unc_var1}")
                    continue

                hist_var2 = bin_uncertainties_dict[ob].get(unc_var2)
                if hist_var2 is None:
                    logger.error(f"No entry found for uncertainty {unc_var2}")
                    continue
            else:
                # symmetric
                hist_var1 = bin_uncertainties_dict[ob].get(unc)
                if hist_var1 is None:
                    logger.error(f"No entry found for uncertainty {unc}")
                    continue

                hist_var2 = hist_var1 * -1.

            relerr_var1 = myhu.get_values_and_errors(hist_var1)[0]
            relerr_var2 = myhu.get_values_and_errors(hist_var2)[0]

            relerr_up = np.max([relerr_var1, relerr_var2], axis=0)
            relerr_down = np.min([relerr_var1, relerr_var2], axis=0)

            var_up += relerr_up ** 2
            var_down += relerr_down ** 2

        total_uncertainties_d[ob] = {
            f"{group_label}_up" : hist_var1.copy(),
            f"{group_label}_down" : hist_var1.copy()
        }

        myhu.set_hist_contents(total_uncertainties_d[ob][f"{group_label}_up"], np.sqrt(var_up))
        total_uncertainties_d[ob][f"{group_label}_up"].view()['variance'] = 0.

        myhu.set_hist_contents(total_uncertainties_d[ob][f"{group_label}_down"], -1*np.sqrt(var_down))
        total_uncertainties_d[ob][f"{group_label}_down"].view()['variance'] = 0.

    return total_uncertainties_d

def evaluate_uncertainties(
    nominal_dir, # str, directory of the nominal unfolding results
    bootstrap_topdir = None, # str, top directory of the results for bootstraping
    bootstrap_mc_topdir = None, # str, top directory of MC bootstraping results
    systematics_topdir = None, # str, top directory of the results for systemaic uncertainties
    network_error_dir = None, # str, directory to extract network uncertainty
    output_dir = '.', # str, output directory
    nensembles_model = None, # int, number of runs to compute bin uncertainties. If None, use all available
    systematics_keywords = [], # list of str, keywords for selecting a subset of systematic uncertainties
    systematics_everyrun = False, # boolen
    hist_filename = "histograms.root", # str, name of the histogram root file
    ibu = False,
    plot = False,
    hist_key = 'unfolded',
    normalize = False
    ):

    bin_uncertainties_d = dict()
    uncertainties_all = list()

    if nensembles_model is not None:
        # use the histograms produced with the specified nensembles
        hist_filename = os.path.join(f"nruns{nensembles_model}", hist_filename)

    # Read nominal results
    fpath_histograms_nominal = os.path.join(nominal_dir, hist_filename)
    logger.info(f"Read nominal histograms from {fpath_histograms_nominal}")
    hists_nominal_d = myhu.read_histograms_dict_from_file(fpath_histograms_nominal)

    # prepare uncertainty dictionary
    for ob in hists_nominal_d:
        bin_uncertainties_d[ob] = {}

    # systematic uncertainties
    if systematics_topdir:
        logger.info(f"Read results for systematic uncertainty variations from {systematics_topdir}")

        bin_errors_syst_d = compute_systematic_uncertainties(
            systematics_topdir,
            hists_nominal_d,
            systematics_keywords = systematics_keywords,
            hist_filename = hist_filename,
            every_run = systematics_everyrun,
            ibu = ibu,
            hist_key = hist_key,
            normalize = normalize
            )

        for ob in bin_uncertainties_d:
            bin_uncertainties_d[ob].update(bin_errors_syst_d[ob])

        uncertainties_all += get_systematics(systematics_keywords, list_of_tuples=True)

    # statistical uncertainty from bootstraping
    if bootstrap_topdir:
        logger.info(f"Data stat.")

        bin_errors_Dstat_d = extract_bin_uncertainties_from_histograms(
            bootstrap_topdir,
            uncertainty_label = "Data stat.",
            histograms_nominal_d = None,
            hist_filename = hist_filename,
            ibu = ibu,
            hist_key = hist_key
            )

        for ob in bin_uncertainties_d:
            bin_uncertainties_d[ob].update(bin_errors_Dstat_d[ob])

        uncertainties_all.append("Data stat.")

    if bootstrap_mc_topdir:
        logger.info(f"MC stat.")

        bin_errors_MCstat_d = extract_bin_uncertainties_from_histograms(
            bootstrap_mc_topdir,
            uncertainty_label = "MC stat.",
            histograms_nominal_d = None,
            hist_filename = hist_filename,
            ibu = ibu,
            hist_key = hist_key
            )

        for ob in bin_uncertainties_d:
            bin_uncertainties_d[ob].update(bin_errors_MCstat_d[ob])

        uncertainties_all.append("MC stat.")

    # network uncertainty
    if not ibu and network_error_dir is not None:

        logger.info("network")

        bin_errors_network_d = extract_bin_uncertainties_from_histograms(
            network_error_dir,
            uncertainty_label = "network",
            histograms_nominal_d = None,
            hist_filename = hist_filename,
            hist_key = hist_key
            )

        for ob in bin_uncertainties_d:
            bin_uncertainties_d[ob].update(bin_errors_network_d[ob])

        uncertainties_all.append('network')

    # compute grouped and total uncertainties
    bin_errors_total = compute_total_uncertainties(
        bin_uncertainties_d,
        uncertainties = uncertainties_all,
        group_label = 'total'
    )

    # compute other uncertainty groups here
    # bin_errors_detector = compute_total_uncertainties(
    #    bin_uncertainties_d,
    #    uncertainties = uncertainties_detector,
    #    group_label = 'total'
    # )

    # Add to the uncertainty dict
    for ob in bin_uncertainties_d:
        bin_uncertainties_d[ob].update(bin_errors_total[ob])

    # save to file
    output_name = os.path.join(output_dir, 'bin_uncertainties.root')
    logger.info(f"Write to output file {output_name}")
    myhu.write_histograms_dict_to_file(bin_uncertainties_d, output_name)

    if plot:
        plot_fractional_uncertainties(
            bin_uncertainties_d,
            uncertainties = uncertainties_all,
            outname_prefix = os.path.splitext(output_name)[0]
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("nominal_dir", type=str,
                        help="Directory of the nominal unfolding results")
    parser.add_argument("-b", "--bootstrap-topdir", type=str,
                        help="Top directory of the unfolding results for bootstraping")
    parser.add_argument("-m", "--bootstrap-mc-topdir", type=str,
                        help="Top directory of the unfolding outputs for MC bootstraping")
    parser.add_argument("-s", "--systematics-topdir", type=str,
                        help="Top directory of the unfolding results for systematic uncertainty variations")
    parser.add_argument("-t", "--network-error-dir", type=str,
                        help="Directory of unfolding results to extract uncertainty from network initialization and training.")
    parser.add_argument("-o", "--output-dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("-n", "--nensembles-model", type=int,
                        help="Number of runs for evaluating model uncertainty. If None, use all available runs")
    parser.add_argument("-k", "--systematics-keywords", nargs='*', type=str,
                        help="Keywords for selecting a subset of systematic uncertainties")
    parser.add_argument("--systematics-everyrun", action='store_true',
                        help="If True, compute the systematic bin uncertainties for every unfolding run")
    parser.add_argument("--hist-filename", type=str, default="histograms.root",
                        help="Name of the unfolding histogram file")
    parser.add_argument("--ibu", action='store_true',
                        help="If True, use unfolded distributions from IBU for comparison")
    parser.add_argument("--normalize", action='store_true',
                        help="If True, normalize histograms to the nominal before computing the bin uncertainties")
    parser.add_argument("-p", "--plot", action='store_true',
                        help="If True, make plots")


    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        logger.info(f"Create output directory {args.output_dir}")
        os.makedirs(args.output_dir)

    evaluate_uncertainties(**vars(args))