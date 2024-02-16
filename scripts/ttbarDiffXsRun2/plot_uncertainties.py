import os
import numpy as np
import matplotlib.pyplot as plt

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

def compare_hist_errors(h_errors1, h_errors2):
    # unpack
    h_err1_up, h_err1_down = h_errors1
    h_err2_up, h_err2_down = h_errors2

    # assume all input histograms are of the same type
    if isinstance(h_err1_up, fh.FlattenedHistogram):
        val1_up = h_err1_up.flatten().values()
        val1_down = h_err1_down.flatten().values()
        val2_up = h_err2_up.flatten().values()
        val2_down = h_err2_down.flatten().values()
    else:
        val1_up = h_err1_up.values()
        val1_down = h_err1_down.values()
        val2_up = h_err2_up.values()
        val2_down = h_err2_down.values()

    return compare_errors((val1_up, val1_down), (val2_up, val2_down))

def plot_uncertainties_1D(
    figname,
    hist_errors_toplot, #
    draw_options,
    draw_error_on_error=False
    ):

    fig, ax = plt.subplots()

    xlabel=hist_errors_toplot[0][0].axes[0].label
    ylabel='Uncertainty [%]'

    plotter.draw_uncertainties_hist(
        ax,
        hist_errors_toplot,
        draw_options,
        xlabel=xlabel,
        ylabel=ylabel,
        draw_error_on_error = draw_error_on_error
    )

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches="tight")
    plt.close(fig)

def draw_uncertainties_2D(
    axes,
    hist_errors_toplot,
    draw_options,
    draw_error_on_error=False,
    outerbin_label=None
    ):

    fh_rep = hist_errors_toplot[0][0]
    assert isinstance(fh_rep, fh.FlattenedHistogram2D)

    categories = fh_rep.get_y_category_labels()
    xbin_labels = fh_rep.get_x_category_labels()

    ylabel='Uncertainty [%]'

    for iy, ybin in enumerate(fh_rep):
        islast = iy == len(fh_rep)-1

        hist_todraw = [(fh_up[ybin], fh_down[ybin]) for fh_up, fh_down in hist_errors_toplot]

        if outerbin_label:
            stamp_texts = [outerbin_label, categories[iy]]
        else:
            stamp_texts = [categories[iy]]

        plotter.draw_uncertainties_hist(
            axes[iy],
            hist_todraw,
            draw_options,
            ylabel = ylabel if iy==0 else None,
            bin_labels = xbin_labels[iy],
            stamps = stamp_texts,
            draw_legend = islast,
            draw_error_on_error = draw_error_on_error
        )

    return axes

def plot_uncertainties_2D(
    figname,
    hist_errors_toplot,
    draw_options,
    draw_error_on_error=False
    ):

    fh_rep = hist_errors_toplot[0][0]
    assert isinstance(fh_rep, fh.FlattenedHistogram2D)

    fig, axes = plt.subplots(nrows=1, ncols=len(fh_rep), sharey=True, figsize=(3.6*len(fh_rep), 3.6))
    fig.subplots_adjust(wspace=0)

    draw_uncertainties_2D(
        axes,
        hist_errors_toplot,
        draw_options,
        draw_error_on_error = draw_error_on_error
    )

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_uncertainties_3D(
    figname,
    hist_errors_toplot,
    draw_options,
    draw_error_on_error=False
    ):

    fh_rep = hist_errors_toplot[0][0]
    assert isinstance(fh_rep, fh.FlattenedHistogram3D)

    fig = plt.figure(figsize=(18, 3.6*len(fh_rep)))
    subfigs = fig.subfigures(len(fh_rep), 1, hspace=0.05)

    for iz, zbin in enumerate(fh_rep):

        axes = subfigs[iz].subplots(nrows=1, ncols=len(fh_rep[zbin]), sharey=True)
        fig.subplots_adjust(wspace=0)

        zbin_text = fh_rep.get_z_category_labels()[iz]

        hists_todraw = [(fh_up[zbin], fh_down[zbin]) for fh_up, fh_down in hist_errors_toplot]

        draw_uncertainties_2D(
            axes,
            hists_todraw,
            draw_options,
            draw_error_on_error = draw_error_on_error,
            outerbin_label = zbin_text
        )

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_fractional_uncertainties(
    figname,
    hists_uncertainty_total,
    hists_uncertainty_compoments,
    label_total,
    labels_component,
    draw_opt_total = {},
    draw_opts_comp = [],
    highlight_dominant = False,
    uncertainty_on_uncertainty = False
    ):

    h_errors_toplot = []
    draw_opts = []
    draw_error_on_error = []

    # Total
    h_grp_up, h_grp_down = hists_uncertainty_total

    if h_grp_up is None or h_grp_down is None:
        logger.warning("Total uncertainty is None")
        return

    # make the values percentage
    h_errors_toplot.append((h_grp_up * 100., h_grp_down * 100.))

    if not draw_opt_total:
        draw_opt_total = {'edgecolor': 'black', "facecolor": 'none', 'label': label_total}
    else:
        draw_opt_total.update({'label': label_total})

    draw_opts.append(draw_opt_total)

    draw_error_on_error.append(uncertainty_on_uncertainty)

    # components
    colors_iter = iter(plotter.get_default_colors(len(hists_uncertainty_compoments)))

    if not draw_opts_comp:
        draw_opts_comp = [{}] * len(hists_uncertainty_compoments)

    for (h_comp_up, h_comp_down), lcomp, ocomp in zip(hists_uncertainty_compoments, labels_component, draw_opts_comp):

        if h_comp_up is None or h_comp_down is None:
            logger.debug(f"Component {lcomp} is None")
            continue

        opt_comp = {'label':lcomp}
        opt_comp.update(ocomp)
        if not 'color' in opt_comp and not 'edgecolor' in opt_comp:
            opt_comp.update({'edgecolor':next(colors_iter), 'facecolor':'none'})

        h_comp_pair = (h_comp_up * 100., h_comp_down * 100.)

        if not highlight_dominant or len(h_errors_toplot) < 2: # first component
            h_errors_toplot.append(h_comp_pair)
            draw_opts.append(opt_comp)
            draw_error_on_error.append(False)
        else:
            # make a comparison and replace if needed
            if compare_hist_errors(h_errors_toplot[-1], h_comp_pair):
                h_errors_toplot[-1] = h_comp_pair
                draw_opts[-1] = opt_comp
                draw_error_on_error[-1] = False

    # plot
    logger.info(f"Make uncertainty plot: {figname}")
    if isinstance(h_grp_up, fh.FlattenedHistogram3D):
        f_plot_uncertainties = plot_uncertainties_3D
    elif isinstance(h_grp_up, fh.FlattenedHistogram2D):
        f_plot_uncertainties = plot_uncertainties_2D
    else:
        f_plot_uncertainties = plot_uncertainties_1D

    f_plot_uncertainties(
        figname,
        h_errors_toplot,
        draw_opts,
        draw_error_on_error = draw_error_on_error
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
            h_grp_up = bin_uncertainties_dict[obs]["Total"].get(f"{group}_up")
            h_grp_down = bin_uncertainties_dict[obs]["Total"].get(f"{group}_down")

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
                highlight_dominant = True,
                uncertainty_on_uncertainty = False
            )

            plot_fractional_uncertainties(
                figname = os.path.join(output_dir, obs, f"{outname_prefix}_{obs}_{group}_err"),
                hists_uncertainty_total = (h_grp_up, h_grp_down),
                hists_uncertainty_compoments = hists_uncertainty_compoments,
                label_total = group,
                labels_component = component_labels,
                highlight_dominant = True,
                uncertainty_on_uncertainty = True
            )

            if plot_allcomponents:
                plot_fractional_uncertainties(
                figname = os.path.join(output_dir, obs, f"{outname_prefix}_{obs}_{group}_allcomp"),
                hists_uncertainty_total = (h_grp_up, h_grp_down),
                hists_uncertainty_compoments = hists_uncertainty_compoments,
                label_total = group,
                labels_component = component_labels,
                highlight_dominant = False
            )

        # Total
        plot_fractional_uncertainties(
            figname = os.path.join(output_dir, obs, f"{outname_prefix}_{obs}_total"),
            hists_uncertainty_total = (bin_uncertainties_dict[obs]['Total']['total_up'], bin_uncertainties_dict[obs]['Total']['total_down']),
            hists_uncertainty_compoments = [(bin_uncertainties_dict[obs]['Total'][f'{grp}_up'], bin_uncertainties_dict[obs]['Total'][f'{grp}_down']) for grp in groups],
            label_total = "Total Syst.", #"Syst. + Stat.",
            labels_component = groups,
            #draw_opt_total = {},
            draw_opts_comp = [syst_groups[grp].get('style',{}) for grp in groups],
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