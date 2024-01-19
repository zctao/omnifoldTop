#!/usr/bin/env python3
import os

import histogramming as myhu
import plotter
import util
import numpy as np

# systematic uncertainties
from ttbarDiffXsRun2.systematics import get_systematics, syst_groups

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
            logger.warning(f"The required number of ensembles {nensembles} is larger than what is available: {len(hists_allruns)}")
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
    hist_key = 'absoluteDiffXs',
    observables = []
    ):

    fpath_hists = os.path.join(result_dir, hist_filename)
    logger.debug(f" Read histograms from {fpath_hists}")
    hists_d = myhu.read_histograms_dict_from_file(fpath_hists)

    # bin uncertainties
    unc_d = dict()

    # loop over observables
    if not observables:
        observables = list(hists_d.keys())

    for ob in observables:
        logger.debug(f" {ob}")
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

def compute_total_uncertainty_hist(hists_tuple_list):

    var_up, var_down = 0., 0.

    for hist_var1, hist_var2 in hists_tuple_list:
        relerr_var1 = myhu.get_values_and_errors(hist_var1)[0]
        relerr_var2 = myhu.get_values_and_errors(hist_var2)[0]

        relerr_up = np.max([relerr_var1, relerr_var2], axis=0)
        relerr_down = np.min([relerr_var1, relerr_var2], axis=0)

        var_up += relerr_up ** 2
        var_down += relerr_down ** 2

    # return total up and down variations as histograms
    hist_total_up = hist_var1.copy()
    hist_total_down = hist_var1.copy()

    myhu.set_hist_contents(hist_total_up, np.sqrt(var_up))
    hist_total_up.view()['variance'] = 0.

    myhu.set_hist_contents(hist_total_down, -1*np.sqrt(var_down))
    hist_total_down.view()['variance'] = 0.

    return hist_total_up, hist_total_down

def compute_total_uncertainty(
    uncertainty_names, # list of str or list of tuple of str
    bin_uncertainties_d, # dict
    label = 'total',
    group = None
    ):

    logger.debug(f" Compute total uncertainty")

    # Initialize total_uncertainties_d
    total_uncertainties_d = { obs : {} for obs in bin_uncertainties_d}

    for obs in bin_uncertainties_d:
        bin_unc_obs_d = bin_uncertainties_d[obs] if group is None else bin_uncertainties_d[obs][group]

        # collect histograms of this observable
        hists_unc_obs = []
        for unc in uncertainty_names:
            if isinstance(unc, tuple):
                # up and down variation
                unc_var1, unc_var2 = unc
                hist_var1 = bin_unc_obs_d.get(unc_var1)
                if hist_var1 is None:
                    logger.warning(f"Cannot find uncertainty: {unc_var1}")
                    logger.debug(f"bin_unc_obs_d.keys() = {bin_unc_obs_d.keys()}")
                    continue

                hist_var2 = bin_unc_obs_d.get(unc_var2)
                if hist_var2 is None:
                    logger.warning(f"Cannot find uncertainty: {unc_var2}")
                    logger.debug(f"bin_unc_obs_d.keys() = {bin_unc_obs_d.keys()}")
                    continue
            else:
                # symmetric
                hist_var1 = bin_unc_obs_d.get(unc)
                if hist_var1 is None:
                    logger.warning(f"Cannot find uncertainty: {unc}")
                    logger.debug(f"bin_unc_obs_d.keys() = {bin_unc_obs_d.keys()}")
                    continue

                hist_var2 = hist_var1 * -1.

            hists_unc_obs.append((hist_var1, hist_var2))

        h_total_up, h_total_down = compute_total_uncertainty_hist(hists_unc_obs)

        total_uncertainties_d[obs].update(
            {f"{label}_up" : h_total_up, f"{label}_down" : h_total_down}
        )

    # end of obs loop

    return total_uncertainties_d

def update_dict_with_group_label(target_dict, component_dict, group_label):
    for obs in component_dict:
        if not obs in target_dict:
            target_dict[obs] = dict()

        if not group_label in target_dict[obs]:
            target_dict[obs][group_label] = dict()

        target_dict[obs][group_label].update(component_dict[obs])

    return target_dict

def compute_systematic_uncertainties(
    uncertainty_list,
    systematics_topdir,
    histograms_nominal_d,
    hist_filename = "histograms.root",
    every_run = False,
    ibu = False,
    hist_key = 'unfolded',
    normalize = False,
    observables = []
    ):

    syst_unc_d = dict()

    logger.debug("Loop over systematic uncertainty variations")
    for syst_variation in uncertainty_list:
        logger.info(f"{syst_variation}")

        fpath_hist_syst = os.path.join(systematics_topdir, syst_variation, hist_filename)

        # read histograms
        try:
            hists_syst_d = myhu.read_histograms_dict_from_file(fpath_hist_syst)
        except:
            logger.debug(f" No histograms found for {syst_variation}: cannot open {fpath_hist_syst}")
            continue

        # loop over observables
        if not observables:
            observables = list(histograms_nominal_d.keys())

        for ob in observables:
            logger.debug(f" {ob}")
            if not ob in syst_unc_d:
                syst_unc_d[ob] = dict()

            # get the unfolded distributions
            h_syst = get_unfolded_histogram_from_dict(ob, hists_syst_d, ibu=ibu, hist_key=hist_key)
            h_nominal = get_unfolded_histogram_from_dict(ob, histograms_nominal_d, ibu=ibu, hist_key=hist_key)

            if normalize:
                if hist_key in ["absoluteDiffXs", "relativeDiffXs"]:
                    logger.warning(f"Skip renormalizing histogram {hist_key}!")
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

def compare_errors(errors1, errors2):
    err1_up, err1_down = errors1
    err2_up, err2_down = errors2

    errsum1 = np.abs(np.asarray(err1_up) - np.asarray(err1_down)).sum()
    errsum2 = np.abs(np.asarray(err2_up) - np.asarray(err2_down)).sum()

    return errsum1 < errsum2

def plot_fractional_uncertainties(
    figname,
    hists_uncertainty_total,
    hists_uncertainty_compoments,
    label_total,
    labels_component,
    color_total = None,
    colors_component = [],
    highlight_dominant = False,
    ):

    errors_toplot = []
    draw_opts = []

    # Total
    h_grp_up, h_grp_down = hists_uncertainty_total
    errors_toplot.append((
        myhu.get_values_and_errors(h_grp_up)[0]*100.,
        myhu.get_values_and_errors(h_grp_down)[0]*100.
        ))

    if color_total is None:
        color_total = 'black'

    draw_opts.append({'label': label_total, 'edgecolor': color_total, "facecolor": 'none'})

    # components
    if not colors_component:
        colors_component = plotter.get_default_colors(len(hists_uncertainty_compoments))

    for (h_comp_up, h_comp_down), lcomp, ccomp in zip(hists_uncertainty_compoments, labels_component, colors_component):
        relerrs_comp = (
            myhu.get_values_and_errors(h_comp_up)[0]*100.,
            myhu.get_values_and_errors(h_comp_down)[0]*100.
            )

        opt_comp = {'label':lcomp, 'edgecolor':ccomp, 'facecolor':'none'}

        if not highlight_dominant or len(errors_toplot) < 2: # first component
            errors_toplot.append(relerrs_comp)
            draw_opts.append(opt_comp)
        else:
            # make a comparison and replace if needed
            if compare_errors(errors_toplot[-1], relerrs_comp):
                errors_toplot[-1] = relerrs_comp
                draw_opts[-1] = opt_comp

    # plot
    logger.info(f"Make uncertainty plot: {figname}")
    plotter.plot_uncertainties(
        figname = figname,
        bins = h_grp_up.axes[0].edges,
        uncertainties = errors_toplot,
        draw_options = draw_opts,
        xlabel = h_grp_up.axes[0].label,
        ylabel = 'Uncertainty [%]'
        )

def plot_uncertainties(
    bin_uncertainties_dict,
    outname_prefix = 'bin_uncertainties',
    highlight_dominant=False # If True, only plot the dominant component, otherwise plot all components
    ):

    for obs in bin_uncertainties_dict:

        groups = [grp for grp in bin_uncertainties_dict[obs] if grp != 'Total']

        # Each uncertainty group
        for group in groups:

            if not group in syst_groups:
                logger.debug(f"{group} not in syst_groups. Skip.")
                continue

            # group total
            h_grp_up = bin_uncertainties_dict[obs]["Total"][f"{group}_up"]
            h_grp_down = bin_uncertainties_dict[obs]["Total"][f"{group}_down"]

            # components
            hists_uncertainty_compoments = []
            component_labels = []

            components_grp = get_systematics(syst_groups[group]['filters'], list_of_tuples=True)
            for comp_up, comp_down in components_grp:
                h_comp_up = bin_uncertainties_dict[obs][group].get(comp_up)
                h_comp_down = bin_uncertainties_dict[obs][group].get(comp_down)

                if h_comp_up is None or h_comp_down is None:
                    #logger.debug(f"No histograms found for {(comp_up, comp_down)}")
                    continue
                #else:
                #    logger.debug(f"Add component {(comp_up, comp_down)}")

                hists_uncertainty_compoments.append((h_comp_up, h_comp_down))
                component_labels.append(os.path.commonprefix([comp_up, comp_down]).strip('_'))

            plot_fractional_uncertainties(
                figname = f"{outname_prefix}_{obs}_{group}",
                hists_uncertainty_total = (h_grp_up, h_grp_down),
                hists_uncertainty_compoments = hists_uncertainty_compoments,
                label_total = group,
                labels_component = component_labels,
                color_total = syst_groups[group].get('color', 'black'),
                highlight_dominant = highlight_dominant
            )

        # Total
        plot_fractional_uncertainties(
            figname = f"{outname_prefix}_{obs}_total",
            hists_uncertainty_total = (bin_uncertainties_dict[obs]['Total']['total_up'], bin_uncertainties_dict[obs]['Total']['total_down']),
            hists_uncertainty_compoments = [(bin_uncertainties_dict[obs]['Total'][f'{grp}_up'], bin_uncertainties_dict[obs]['Total'][f'{grp}_down']) for grp in groups],
            label_total = "Syst. + Stat.",
            labels_component = groups,
            color_total = 'black',
            colors_component = [], # [syst_groups[grp]['color'] for grp in groups]
            highlight_dominant = False
        )

def evaluate_uncertainties(
    nominal_dir, # str, directory of the nominal unfolding results
    bootstrap_topdir = None, # str, top directory of the results for bootstraping
    bootstrap_mc_topdir = None, # str, top directory of MC bootstraping results
    systematics_topdir = None, # str, top directory of the results for systemaic uncertainties
    network_error_dir = None, # str, directory to extract network uncertainty
    output_dir = '.', # str, output directory
    nensembles_network = None, # int, number of runs to compute bin uncertainties. If None, use all available
    systematics_groups = [], # list of str, systematic groups
    systematics_keywords = [], # list of str, keywords for selecting a subset of systematic uncertainties
    systematics_everyrun = False, # boolen
    hist_filename = "histograms.root", # str, name of the histogram root file
    ibu = False,
    plot = False,
    hist_key = 'unfolded',
    normalize = False,
    observables = [],
    verbose = False
    ):

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    bin_uncertainties_d = dict()

    if nensembles_network is not None:
        # use the histograms produced with the specified nensembles
        hist_filename = os.path.join(f"nruns{nensembles_network}", hist_filename)

    # Read nominal results
    fpath_histograms_nominal = os.path.join(nominal_dir, hist_filename)
    logger.info(f"Read nominal histograms from {fpath_histograms_nominal}")
    hists_nominal_d = myhu.read_histograms_dict_from_file(fpath_histograms_nominal)

    if not observables:
        observables = list(hists_nominal_d.keys())
    else:
        # TODO: check if all observables are available
        pass

    # Initialize uncertainty dictionary
    bin_uncertainties_d = { obs : {} for obs in observables}

    ######
    # Systematic uncertainties
    if systematics_topdir:
        logger.info(f"Read results for systematic uncertainty variations from {systematics_topdir}")

        # All systematic uncertainties under consideration if there is a global keyword list
        if systematics_keywords:
            all_systs = get_systematics(systematics_keywords)
            all_systs_pair = get_systematics(systematics_keywords, list_of_tuples=True)
        else:
            all_systs = None
            all_systs_pair = None

        # Compute syst uncertainties in groups
        if not systematics_groups:
            systematics_groups = list(syst_groups.keys())

        for grp in systematics_groups:
            if not grp in syst_groups:
                logger.error(f"Unknow uncertainty group {grp}")
                continue

            logger.info(f"Uncertainty group: {grp}")

            # Use the keyword filters of the group to select the uncertainties
            grp_systs = get_systematics(syst_groups[grp]["filters"])
            grp_systs_pair = get_systematics(syst_groups[grp]["filters"], list_of_tuples=True)

            # In case there is a global keyword list, take the intersection
            if all_systs is not None:
                grp_systs = list( set(grp_systs) & set(all_systs) )
                grp_systs_pair = list( set(grp_systs_pair) & set(all_systs_pair) )

            bin_err_grp_d = compute_systematic_uncertainties(
                grp_systs,
                systematics_topdir,
                hists_nominal_d,
                hist_filename = hist_filename,
                every_run = systematics_everyrun,
                ibu = ibu,
                hist_key = hist_key,
                normalize = normalize,
                observables = observables
            )

            # Add the group uncertainties to bin_uncertainties_d
            update_dict_with_group_label(bin_uncertainties_d, bin_err_grp_d, grp)

            # Group total uncertainty
            bin_err_grp_tot_d = compute_total_uncertainty(grp_systs_pair, bin_err_grp_d, label=grp)

            update_dict_with_group_label(bin_uncertainties_d, bin_err_grp_tot_d, 'Total')

        # end of grp loop

        # Special cases
        # Network uncertainty
        if not ibu and network_error_dir is not None:
            logger.info("Uncertainty: Network")

            bin_err_nn_d = extract_bin_uncertainties_from_histograms(
                network_error_dir,
                uncertainty_label = "network",
                histograms_nominal_d = None,
                hist_filename = hist_filename,
                hist_key = hist_key,
                observables = observables
            )

            update_dict_with_group_label(bin_uncertainties_d, bin_err_nn_d, 'Network')

            # Also add to sub-directory "Total"
            bin_err_nn_tot_d = compute_total_uncertainty(['network'], bin_err_nn_d, label='Network')
            update_dict_with_group_label(bin_uncertainties_d, bin_err_nn_tot_d, 'Total')

        # Compute the total systematic uncertainty
        bin_err_syst_d = compute_total_uncertainty(
                [(f"{grp}_up", f"{grp}_down") for grp in systematics_groups] + [('Network_up', 'Network_down')],
                bin_uncertainties_d,
                label = 'syst_total',
                group = 'Total'
            )

        update_dict_with_group_label(bin_uncertainties_d, bin_err_syst_d, "Total")

    ######
    # Statistical uncertainties
    stat_unc_comp = []
    bin_err_stat_d = {}

    if bootstrap_topdir:
        logger.info(f"Uncertainty: Data stat.")

        bin_errors_Dstat_d = extract_bin_uncertainties_from_histograms(
            bootstrap_topdir,
            uncertainty_label = "data_stat",
            histograms_nominal_d = None,
            hist_filename = hist_filename,
            ibu = ibu,
            hist_key = hist_key,
            observables = observables
            )

        for obs in bin_errors_Dstat_d:
            if not obs in bin_err_stat_d:
                bin_err_stat_d[obs] = dict()
            bin_err_stat_d[obs].update(bin_errors_Dstat_d[obs])

        stat_unc_comp.append("data_stat")

    if bootstrap_mc_topdir:
        logger.info(f"Uncertainty: MC stat.")

        bin_errors_MCstat_d = extract_bin_uncertainties_from_histograms(
            bootstrap_mc_topdir,
            uncertainty_label = "mc_stat",
            histograms_nominal_d = None,
            hist_filename = hist_filename,
            ibu = ibu,
            hist_key = hist_key,
            observables = observables
            )

        for obs in bin_errors_MCstat_d:
            if not obs in bin_err_stat_d:
                bin_err_stat_d[obs] = dict()
            bin_err_stat_d[obs].update(bin_errors_MCstat_d[obs])

        stat_unc_comp.append("mc_stat")

    update_dict_with_group_label(bin_uncertainties_d, bin_err_stat_d, "Stat")

    # Total stat uncertainty
    if stat_unc_comp:
        bin_err_stat_tot_d = compute_total_uncertainty(stat_unc_comp, bin_err_stat_d, label="stat_total")

        update_dict_with_group_label(bin_uncertainties_d, bin_err_stat_tot_d, "Total")

    # Total syst+stat uncertainty
    logger.info("Total uncertainty")
    bin_err_tot_d = compute_total_uncertainty(
        [('syst_total_up','syst_total_down'), ('stat_total_up','stat_total_down')],
        bin_uncertainties_d,
        label = 'total',
        group = "Total"
    )

    update_dict_with_group_label(bin_uncertainties_d, bin_err_tot_d, "Total")

    ######
    # save to file
    output_name = os.path.join(output_dir, 'bin_uncertainties.root')
    logger.info(f"Write to output file {output_name}")
    myhu.write_histograms_dict_to_file(bin_uncertainties_d, output_name)

    if plot:
        plot_uncertainties(
            bin_uncertainties_d,
            outname_prefix = os.path.splitext(output_name)[0],
            highlight_dominant=False
        )

def plot_uncertainties_from_file(fpath):
    unc_d = myhu.read_histograms_dict_from_file(fpath)
    plot_uncertainties(
        unc_d,
        outname_prefix = os.path.splitext(fpath)[0],
        highlight_dominant=False
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
    parser.add_argument("-n", "--nensembles-network", type=int,
                        help="Number of runs for evaluating model uncertainty. If None, use all available runs")
    parser.add_argument("-g", "--systematics-groups", nargs='*', type=str,
                        help="Groups of systematic uncertainties to evaluate")
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
    parser.add_argument("--observables", type=str, nargs='*',
                        help="List of observables to evaluate bin uncertainties")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="If True, set logging level to debug, else info")

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        logger.info(f"Create output directory {args.output_dir}")
        os.makedirs(args.output_dir)

    evaluate_uncertainties(**vars(args))