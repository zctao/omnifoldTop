import os
import numpy as np

import util
import plotter
import histogramming as myhu
import FlattenedHistogram as fh

# systematic uncertainties
from ttbarDiffXsRun2.systematics import get_systematics, syst_groups

import logging
logger = logging.getLogger('PlotUncertainties')
util.configRootLogger()

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

    if h_grp_up is None or h_grp_down is None:
        logger.warning("Total uncertainty is None")
        return
    
    if isinstance(h_grp_up, fh.FlattenedHistogram):
        h_grp_up = h_grp_up.flatten()
    if isinstance(h_grp_down, fh.FlattenedHistogram):
        h_grp_down = h_grp_down.flatten()

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

        if h_comp_up is None or h_comp_down is None:
            logger.debug(f"Component {lcomp} is None")
            continue

        if isinstance(h_comp_up, fh.FlattenedHistogram):
            h_comp_up = h_comp_up.flatten()
        if isinstance(h_comp_down, fh.FlattenedHistogram):
            h_comp_down = h_comp_down.flatten()

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

def plot_uncertainties_from_dict(
    bin_uncertainties_dict,
    output_dir = '.',
    outname_prefix = 'bin_uncertainties',
    groups = [],
    plot_allcomponents = False, # If True, include all components in the plots, otherwise only plot the largest component of the uncertainty group
    ):

    for obs in bin_uncertainties_dict:

        if not groups:
            groups = [grp for grp in bin_uncertainties_dict[obs] if grp != 'Total']

        # For each uncertainty group
        for group in groups:

            if not group in syst_groups:
                logger.debug(f"{group} not in syst_groups. Skip.")
                continue

            # group total
            h_grp_up = bin_uncertainties_dict[obs]["Total"][f"{group}_up"]
            h_grp_down = bin_uncertainties_dict[obs]["Total"][f"{group}_down"]

            if h_grp_up is None or h_grp_down is None:
                logger.debug(f"Group {group} total is None")
                continue

            # components
            hists_uncertainty_compoments = []
            component_labels = []

            for comp_unc in get_systematics(syst_groups[group]['filters'], list_of_tuples=True):
                if len(comp_unc) == 1:
                    # symmetric
                    comp_var = comp_unc[0]
                    h_comp_up = bin_uncertainties_dict[obs][group].get(comp_var)
                    h_comp_down = h_comp_up * -1. if h_comp_up is not None else None

                    comp_label = comp_var.strip('_')
                else:
                    # unpack up and down variations
                    comp_up, comp_down = comp_unc
                    h_comp_up = bin_uncertainties_dict[obs][group].get(comp_up)
                    h_comp_down = bin_uncertainties_dict[obs][group].get(comp_down)

                    comp_label = os.path.commonprefix([comp_up, comp_down]).strip('_')

                if h_comp_up is None or h_comp_down is None:
                    logger.debug(f"No histograms found for {comp_unc}")
                    continue
                #else:
                #    logger.debug(f"Add component {comp_unc}")

                hists_uncertainty_compoments.append((h_comp_up, h_comp_down))
                component_labels.append(comp_label)

            plot_fractional_uncertainties(
                figname = os.path.join(output_dir, obs, f"{outname_prefix}_{obs}_{group}"),
                hists_uncertainty_total = (h_grp_up, h_grp_down),
                hists_uncertainty_compoments = hists_uncertainty_compoments,
                label_total = group,
                labels_component = component_labels,
                color_total = syst_groups[group].get('color', 'black'),
                highlight_dominant = True
            )

            if plot_allcomponents:
                plot_fractional_uncertainties(
                figname = os.path.join(output_dir, obs, f"{outname_prefix}_{obs}_{group}_allcomp"),
                hists_uncertainty_total = (h_grp_up, h_grp_down),
                hists_uncertainty_compoments = hists_uncertainty_compoments,
                label_total = group,
                labels_component = component_labels,
                color_total = syst_groups[group].get('color', 'black'),
                highlight_dominant = False
            )

        # Total
        plot_fractional_uncertainties(
            figname = os.path.join(output_dir, obs, f"{outname_prefix}_{obs}_total"),
            hists_uncertainty_total = (bin_uncertainties_dict[obs]['Total']['total_up'], bin_uncertainties_dict[obs]['Total']['total_down']),
            hists_uncertainty_compoments = [(bin_uncertainties_dict[obs]['Total'][f'{grp}_up'], bin_uncertainties_dict[obs]['Total'][f'{grp}_down']) for grp in groups],
            label_total = "Total Syst.", #"Syst. + Stat.",
            labels_component = groups,
            color_total = 'black',
            colors_component = [], # [syst_groups[grp]['color'] for grp in groups]
            highlight_dominant = False
        )

def plot_uncertainties_from_file(
        fpath_uncertainties,
        output_dir = None,
        outname_prefix = None,
        groups = [],
        plot_allcomponents = False
        ):
    uncertainty_d = myhu.read_histograms_dict_from_file(fpath_uncertainties)

    if output_dir is None:
        output_dir = os.path.dirname(fpath_uncertainties)

    if outname_prefix is None:
        outname_prefix = os.path.basename(os.path.splitext(fpath_uncertainties)[0])

    plot_uncertainties_from_dict(
        uncertainty_d,
        output_dir = output_dir,
        outname_prefix = outname_prefix,
        groups = groups,
        plot_allcomponents = plot_allcomponents
    )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("fpath_uncertainties", type=str,
                        help="file path to bin uncertainties")
    parser.add_argument("-o", "--output-dir", type=str,
                        help="Output directory. If not specified, use the directory of fpath_uncertainties")
    parser.add_argument("-n", "--outname-prefix", type=str,
                        help="Plot file name prefix. If not specified, use the file name from fpath_uncertainties")
    parser.add_argument("-g", "--groups", nargs='+', type=str,
                        help="List of systematic groups. If not specified, use all available in fpath_uncertainties")
    parser.add_argument("-a", "--plot-allcomponents", action='store_true',
                        help="If True, plot all components of each uncertainty group")

    args = parser.parse_args()

    plot_uncertainties_from_file(**vars(args))