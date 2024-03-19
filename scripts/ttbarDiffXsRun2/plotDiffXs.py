import os
import logging
import numpy as np

import util
import histogramming as myhu
import plotter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from ttbarDiffXsRun2.systematics import uncertainty_groups

util.configRootLogger()
logger = logging.getLogger("plotDiffXs")

default_markers = ['o','s','^','v','d','<','>','p','h','*','x','1','2','3','4']

rescale_oom = {
    "th_pt_vs_mtt": [8,6,4,2,0],
    "ptt_vs_mtt": [8,6,4,2,0],
    "ytt_abs_vs_mtt": [4,2,0],
    "mtt_vs_ytt_abs": [6,4,2,0],
    "ptt_vs_ytt_abs": [6,4,2,0],
    "mtt_vs_ptt_vs_ytt_abs": [12,9,6,3,0],
    "mtt_vs_th_pt_vs_th_y_abs": [12,9,6,3,0],
    "mtt_vs_th_pt_vs_ytt_abs": [12,9,6,3,0],
    "mtt_vs_th_y_abs_vs_ytt_abs": [9,6,3,0],
}

ATLAS_stamp = [
#    "ATLAS WIP",
    "OmniFold",
    "$\sqrt{s}=13$ TeV, 140 fb$^{-1}$",
    "Full phase space"
]

def get_error_arrays(hist_unc_down, hist_unc_up, hist_central=None):
    if hist_unc_up is not None and hist_unc_down is None:
        hist_unc_down = hist_unc_up * -1.
    elif hist_unc_up is None and hist_unc_down is not None:
        hist_unc_up = hist_unc_down * -1.

    if hist_unc_up is not None and hist_unc_down is not None:
        unc_arr = np.stack([
            hist_unc_down.values(), hist_unc_up.values()
            ])
        if hist_central is None:
            return unc_arr
        else:
            # The errors are percentage
            return  unc_arr * hist_central.values()
    else:
        return None

def get_error_arrays_2D(fhist_unc_down, fhist_unc_up, fhist_central=None):
    if fhist_unc_up is not None and fhist_unc_down is None:
        fhist_unc_down = fhist_unc_up * -1.
    elif fhist_unc_up is None and fhist_unc_down is not None:
        fhist_unc_up = fhist_unc_down * -1.

    if fhist_unc_up is not None and fhist_unc_down is not None:
        assert(len(fhist_unc_up)==len(fhist_unc_down))
        errors = []
        for ybin in fhist_unc_down:
            errors.append(
                np.stack([
                    fhist_unc_down[ybin].values(), fhist_unc_up[ybin].values()
                ])
            )

            if fhist_central is not None: # percentage error
                errors[-1] *= fhist_central[ybin].values()
    else:
        errors = None

    return errors

def plot_diffXs_1D(
    figname,
    observable_label,
    histogram_data,
    histograms_mc,
    label_nominal,
    labels_mc,
    hist_relerrs_total = None,
    hist_relerrs_stat = None,
    ylabel = '',
    ylabel_ratio = '',
    log_obs = False,
    log_diffXs = False,
    stamp_texts = []
    ):

    # unfolded measurement
    draw_opt_data = {
        'label': label_nominal, 'color': 'black', 'histtype': 'errorbar',
        'marker': 'o', 'markersize': 3, 'xerr': True, 'yerr': False
    }

    histograms_werr = []
    draw_opt_werr = []

    # total uncertainty
    if hist_relerrs_total:
        h_relerrs_tot_up, h_relerrs_tot_down = hist_relerrs_total
        errors_tot = get_error_arrays(h_relerrs_tot_up, h_relerrs_tot_down, histogram_data)

        if errors_tot is None:
            logger.warning("No Total Uncertainty")
        else:
            histograms_werr.append(histogram_data)

            error_band_tot_opt = {'label': uncertainty_groups['Total'].get('label')}
            error_band_tot_opt.update(uncertainty_groups['Total'].get('style'))
            draw_opt_werr.append({
                'yerr' : errors_tot,
                'style_error_band' : error_band_tot_opt,
                'skip_central' : True
            })

    if hist_relerrs_stat:
        h_relerrs_stat_up, h_relerrs_stat_down = hist_relerrs_stat
        errors_stat = get_error_arrays(h_relerrs_stat_up, h_relerrs_stat_down, histogram_data)
        if errors_stat is None:
            logger.warning("No Stat. Uncertainty")
        else:
            histograms_werr.append(histogram_data)

            error_band_stat_opt = {'label' : uncertainty_groups['stat_total'].get('label')}
            error_band_stat_opt.update(uncertainty_groups['stat_total'].get('style'))
            draw_opt_werr.append({
                'yerr' : errors_stat,
                'style_error_band' : error_band_stat_opt,
                'skip_central' : True
            })

    # MC truth
    colors_mc = plotter.get_default_colors(len(histograms_mc))
    draw_opts_mc = [
        {'label': l_mc, 'color': c_mc, 'histtype': 'step', 'xerr': True}
        for l_mc, c_mc in zip(labels_mc, colors_mc)
    ]

    xlabel = observable_label[0] if isinstance(observable_label, list) else observable_label

    plotter.plot_histograms_and_ratios(
        figname,
        hists_numerator = histograms_werr + histograms_mc,
        hist_denominator = histogram_data,
        draw_options_numerator = draw_opt_werr + draw_opts_mc,
        draw_option_denominator = draw_opt_data,
        xlabel = xlabel,
        ylabel = ylabel,
        ylabel_ratio = ylabel_ratio,
        log_xscale = log_obs,
        log_yscale = log_diffXs,
        stamp_texts=stamp_texts,
        stamp_loc='upper left',
        stamp_opt={"prop":{"fontsize":"medium"}},
        y_lim = 'x2',
        ratio_lim = (0.5, 1.5),
        height_ratios = (5,1),
        figsize = [6.4, 6.4]
    )

def get_color_sequence(colormap, ncolors):
    return [colormap(i) for i in np.linspace(0.9, 0.4, ncolors)]

def get_color_for_legend(colormap):
    return colormap(0.7)

def draw_error_band_2D(
    ax,
    fhist_relerrs,
    fhist_central,
    error_group,
    rescales_order_of_magnitude = None,
    legend_off = True
    ):
    if fhist_relerrs is None:
        return

    fhist_relerrs_up, fhist_relerrs_down = fhist_relerrs
    errors_arr = get_error_arrays_2D(fhist_relerrs_up, fhist_relerrs_down, fhist_central)

    if errors_arr is None:
        logger.warning(f"No {error_group} Uncertainty")
    else:
        error_band_opt = {'label': uncertainty_groups[error_group].get('label')}
        error_band_opt.update(uncertainty_groups[error_group].get('style'))

        fhist_central.draw(
            ax,
            common_styles = {
                'style_error_band' : error_band_opt, 'skip_central' : True
            },
            rescales_order_of_magnitude = rescales_order_of_magnitude,
            errors = errors_arr,
            legend_off = legend_off
        )

def draw_diffXs_distr_2D(
    ax,
    fhistogram_data,
    fhistograms_mc,
    label_nominal,
    labels_mc,
    fhist_relerrs_total = None,
    fhist_relerrs_stat = None,
    xlabel = '',
    ylabel = '',
    title = '',
    log_xscale = False,
    log_yscale = False,
    rescales_order_of_magnitude = None,
    stamp_texts = []
    ):

    fhistogram_data.draw(
        ax,
        markers = default_markers[:len(fhistogram_data)],
        colors = 'black',
        common_styles = {'histtype': 'errorbar'},
        rescales_order_of_magnitude = rescales_order_of_magnitude,
        errors = None,
        legend_off = True,
        stamp_texts = stamp_texts,
        stamp_loc = 'upper left',
        stamp_opt = {"prop":{"fontsize":"medium"}}
    )

    leg_handles, leg_labels = ax.get_legend_handles_labels()

    # MC histograms
    # Get a color map for each MC
    cmaps_mc = plotter.get_default_colormaps(len(fhistograms_mc))

    for fhist_mc, cm_mc in zip(fhistograms_mc, cmaps_mc):
        colors_mc = get_color_sequence(cm_mc, len(fhist_mc))

        fhist_mc.draw(
            ax,
            colors = colors_mc,
            common_styles = {'histtype': 'step'},
            rescales_order_of_magnitude = rescales_order_of_magnitude,
            legend_off = True
        )

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    if log_xscale:
        ax.set_xscale('log')

    if log_yscale:
        ax.set_yscale('log')

    # y limits
    ylim_bot, ylim_top = ax.get_ylim()
    if log_yscale:
        ylim_top *= 10**10
    else:
        ylim_top *= 3
    ax.set_ylim(ylim_bot, ylim_top)

    # re-draw legends
    # remove error bar and add line marker for each MC
    new_handles = [(lh[0],) for lh in leg_handles]

    for fhist_mc, cm_mc in zip(fhistograms_mc, cmaps_mc):
        colors_mc = get_color_sequence(cm_mc, len(fhist_mc))
        assert(len(new_handles)==len(colors_mc))
        for i, c in enumerate(colors_mc):
            new_handles[i] = (mlines.Line2D([], [], color=c),) + new_handles[i]

    new_handles.append(mlines.Line2D ([], [], color='black', linestyle='None', marker='o', markersize=3))
    leg_labels.append(label_nominal)

    for cm_mc, l_mc in zip(cmaps_mc, labels_mc):
        new_handles.append(mlines.Line2D([], [], color=get_color_for_legend(cm_mc)))
        leg_labels.append(l_mc)

    ax.legend(new_handles, leg_labels, loc="upper right", fontsize='small', numpoints=1)

def draw_diffXs_ratio_2D(
    axes,
    fhistogram_data,
    fhistograms_mc,
    label_nominal,
    labels_mc,
    fhist_relerrs_total = None,
    fhist_relerrs_stat = None,
    xlabel = '',
    ylabel = '',
    log_xscale = False,
    legend_off = False,
    stamp_texts = [],
    outer_bin_label = None
    ):

    for ax in axes:
        ax.yaxis.grid()

        if xlabel:
            ax.set_xlabel(xlabel)

        if log_xscale:
            ax.set_xscale('log')

        ax.set_ylim(0.0, 2.0)

    if ylabel:
        axes[0].set_ylabel(ylabel)

    ratio_one = fhistogram_data.copy()
    ratio_one.divide(fhistogram_data)

    # errors
    draw_error_band_2D(
        axes,
        fhist_relerrs_total,
        ratio_one,
        "Total"
        )

    draw_error_band_2D(
        axes,
        fhist_relerrs_stat,
        ratio_one,
        "stat_total"
        )

    # MC
    cmaps_mc = plotter.get_default_colormaps(len(fhistograms_mc))

    for fhist_mc, cm_mc, l_mc in zip(fhistograms_mc, cmaps_mc, labels_mc):

        ratio_mc = fhist_mc.copy()
        ratio_mc.divide(fhistogram_data)
        ratio_mc.draw(
            axes,
            colors = cm_mc(0.7),
            common_styles = {'histtype':'step', 'yerr':False, 'label': l_mc},
            legend_off = True
            )

    # data
    ratio_one.draw(
        axes,
        colors = "black",
        common_styles = {'histtype':'step', 'yerr':False},
        stamp_texts = [] if outer_bin_label is None else [outer_bin_label],
        stamp_loc = "lower right",
        stamp_opt = {"prop":{"fontsize":"medium"}},
        legend_off = False
    )

    if not legend_off:
        handles, labels = axes[-1].get_legend_handles_labels()

        # append the legend for data
        handles.append(mlines.Line2D([], [], color="black"))
        labels.append(label_nominal)

        axes[-1].legend(handles, labels, loc='lower right', fontsize='medium', ncols=2, bbox_to_anchor=(1,1))

    if stamp_texts:
        plotter.draw_text(axes[0], stamp_texts, loc='lower left', prop={'fontsize':'medium'}, frameon=False, bbox_to_anchor=(0,1), bbox_transform=axes[0].transAxes)

def plot_diffXs_2D(
    figname,
    observable_labels,
    fhistogram_data,
    fhistograms_mc,
    label_nominal,
    labels_mc,
    fhist_relerrs_total = None,
    fhist_relerrs_stat = None,
    ylabel = '',
    ylabel_ratio = '',
    log_obs = False,
    log_diffXs = False,
    title = '',
    title_ratio = '',
    rescales_order_of_magnitude = None,
    stamp_texts = []
    ):

    fig, ax = plt.subplots()

    if observable_labels:
        assert(len(observable_labels)==2)
        fhistogram_data.set_xlabel(observable_labels[0])
        fhistogram_data.set_ylabel(observable_labels[1])
    else:
        observable_labels = [
            fhistogram_data.get_xlabel(), fhistogram_data.get_ylabel()
        ]

    xlabel = observable_labels[0]

    draw_diffXs_distr_2D(
        ax,
        fhistogram_data,
        fhistograms_mc,
        label_nominal,
        labels_mc,
        fhist_relerrs_total = fhist_relerrs_total,
        fhist_relerrs_stat = fhist_relerrs_stat,
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        log_xscale = log_obs,
        log_yscale = log_diffXs,
        rescales_order_of_magnitude = rescales_order_of_magnitude,
        stamp_texts = stamp_texts
    )

    # save plot
    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    ######
    # draw the ratio in a separate plot
    fig_r, ax_r = plt.subplots(1, len(fhistogram_data), sharey=True, figsize = (3.6*len(fhistogram_data),4.0))
    plt.subplots_adjust(wspace=0)

    if title_ratio:
        fig_r.suptitle(title_ratio)

    draw_diffXs_ratio_2D(
        ax_r,
        fhistogram_data,
        fhistograms_mc,
        label_nominal,
        labels_mc,
        fhist_relerrs_total = fhist_relerrs_total,
        fhist_relerrs_stat = fhist_relerrs_stat,
        xlabel = xlabel,
        ylabel = ylabel_ratio,
        log_xscale = log_obs,
        stamp_texts = stamp_texts
    )

    # save plot
    fig_r.savefig(figname+'_ratio.png', dpi=300, bbox_inches='tight')
    plt.close(fig_r)

def plot_diffXs_3D(
    figname,
    observable_labels,
    fhistogram_data,
    fhistograms_mc,
    label_nominal,
    labels_mc,
    fhist_relerrs_total = None,
    fhist_relerrs_stat = None,
    ylabel = '',
    ylabel_ratio = '',
    log_obs = False,
    log_diffXs = False,
    title = '',
    title_ratio = '',
    rescales_order_of_magnitude = None,
    stamp_texts = []
    ):

    if np.asarray(rescales_order_of_magnitude).ndim < 2:
        rescales_order_of_magnitude = [rescales_order_of_magnitude] * len(fhistogram_data)

    if observable_labels:
        assert(len(observable_labels)==3)
        fhistogram_data.set_xlabel(observable_labels[0])
        fhistogram_data.set_ylabel(observable_labels[1])
        fhistogram_data.set_zlabel(observable_labels[2])
    else:
        observable_labels = [
            fhistogram_data.get_xlabel(),
            fhistogram_data.get_ylabel(),
            fhistogram_data.get_zlabel()
        ]

    xlabel = observable_labels[0]

    fig = plt.figure(layout='constrained', figsize=(4.8, 3.6*len(fhistogram_data)))
    subfigs = fig.subfigures(len(fhistogram_data), 1)

    if title:
        fig.suptitle(title)

    # zbin edges
    zbin_edges = fhistogram_data.get_zbin_edges()
    zobs = observable_labels[2]

    def get_fhist_relerrs_zbin(fhist_relerrs, zbin):
        if fhist_relerrs is None:
            return None

        fhist_relerrs_up, fhist_relerrs_down = fhist_relerrs
        if fhist_relerrs_up is None or fhist_relerrs_down is None:
            return None

        return (fhist_relerrs_up[zbin], fhist_relerrs_down[zbin])

    for i, zbin_label in enumerate(fhistogram_data):

        ax = subfigs[i].subplots(1)

        zbin_text = f"{zbin_edges[i]}$\\leq${zobs}$<${zbin_edges[i+1]}"

        draw_diffXs_distr_2D(
            ax,
            fhistogram_data[zbin_label],
            [fhist_mc[zbin_label] for fhist_mc in fhistograms_mc],
            label_nominal,
            labels_mc,
            fhist_relerrs_total = get_fhist_relerrs_zbin(fhist_relerrs_total, zbin_label),
            fhist_relerrs_stat = get_fhist_relerrs_zbin(fhist_relerrs_stat, zbin_label),
            xlabel = xlabel,
            ylabel = ylabel if i==0 else '',
            log_xscale = log_obs,
            log_yscale = log_diffXs,
            rescales_order_of_magnitude = rescales_order_of_magnitude[i],
            stamp_texts = stamp_texts + [zbin_text]
        )

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ratio
    fig_r = plt.figure(figsize=(18, 3.6*len(fhistogram_data)))
    subfigs_r = fig_r.subfigures(len(fhistogram_data), 1, hspace=0.01)

    if title_ratio:
        fig_r.suptitle(title_ratio)

    for i, zbin_label in enumerate(fhistogram_data):

        axes_r = subfigs_r[i].subplots(1, len(fhistogram_data[zbin_label]), sharey=True)
        fig_r.subplots_adjust(wspace=0)

        zbin_text = f"{zbin_edges[i]}$\\leq${zobs}$<${zbin_edges[i+1]}"

        draw_diffXs_ratio_2D(
            axes_r,
            fhistogram_data[zbin_label],
            [fhist_mc[zbin_label] for fhist_mc in fhistograms_mc],
            label_nominal,
            labels_mc,
            fhist_relerrs_total = get_fhist_relerrs_zbin(fhist_relerrs_total, zbin_label),
            fhist_relerrs_stat = get_fhist_relerrs_zbin(fhist_relerrs_stat, zbin_label),
            xlabel = xlabel,
            ylabel = ylabel_ratio,
            log_xscale = log_obs,
            legend_off = False if i==0 else True,
            stamp_texts = stamp_texts if i==0 else [],
            outer_bin_label = zbin_text
        )

    fig_r.savefig(figname+'_ratio.png', dpi=300, bbox_inches='tight')
    plt.close(fig_r)

def plotDiffXs(
    fpath_nominal,
    fpath_uncertainties = None,
    fpaths_otherMC = [],
    isRelative = False,
    outputdir = None,
    observables = [],
    label_nominal = "MC",
    labels_otherMC = [],
    skip_nominal_MC = False,
    observable_config = 'configs/observables/vars_ttbardiffXs_pseudotop.json'
    ):

    # output directory
    if not outputdir:
        logger.info("Use the histogram directory as the output directory")
        outputdir = os.path.dirname(fpath_nominal)

    if not os.path.isdir(outputdir):
        logger.debug(f"Create output directory {outputdir}")
        os.makedirs(outputdir)

    # observable config
    logger.debug(f"Read observable config from {observable_config}")
    obsConfig_d = util.read_dict_from_json(observable_config)

    # read histograms from the file
    logger.debug(f"Read nominal histograms from {fpath_nominal}")
    hists_nominal = myhu.read_histograms_dict_from_file(fpath_nominal)

    labels_MC = labels_otherMC if skip_nominal_MC else [label_nominal] + labels_otherMC

    # observables
    # get the observable list from hists_nominal if not specified
    if not observables:
        for hkey in hists_nominal:
            observables.append(hkey)

    # read other MC histograms if available
    logger.debug(f"Read other MC predictions from {fpaths_otherMC}")
    hists_others = [
        myhu.read_histograms_dict_from_file(fp_other) for fp_other in fpaths_otherMC
    ]
    assert(len(hists_others)==len(labels_otherMC))

    # read uncertainties if available
    if fpath_uncertainties:
        logger.debug(f"Read uncertainties from {fpath_uncertainties}")
        unc_d = myhu.read_histograms_dict_from_file(fpath_uncertainties)
    else:
        logger.warning(f"No uncertainty is provided")
        unc_d = {}

    # histogram name
    histname = 'relativeDiffXs' if isRelative else 'absoluteDiffXs'
    logger.debug(f"histogram name: {histname}")

    # stamp
    if isRelative:
        stamps = ATLAS_stamp + ["Relative cross-section"]
    else:
        stamps = ATLAS_stamp + ["Absolute cross-section"]

    # loop over observables
    for obs in observables:
        logger.info(obs)

        # predictions
        hists_MC = [h_d[obs][f'{histname}_MC'] for h_d in hists_others]
        if not skip_nominal_MC:
            hists_MC = [hists_nominal[obs][f'{histname}_MC']] + hists_MC

        # measurement
        hist_meas = hists_nominal[obs][histname]

        # uncertainties
        hist_relerr_tot_up, hist_relerr_tot_down = None, None
        hist_relerr_stat_up, hist_relerr_stat_down = None, None
        if unc_d:
            if not obs in unc_d:
                logger.warning(f"No uncertainties found for observable {obs}")
            elif not 'Total' in unc_d[obs]:
                logger.warning(f"Total uncertainties not available for observable {obs}")
            else:
                # systematic + statistical
                hist_relerr_tot_up = unc_d[obs]['Total'].get('total_up')
                hist_relerr_tot_down = unc_d[obs]['Total'].get('total_down')

                # statistical
                hist_relerr_stat_up = unc_d[obs]['Total'].get('stat_total_up')
                hist_relerr_stat_down = unc_d[obs]['Total'].get('stat_total_down')

        # plot
        obs_list = obs.split('_vs_')

        obs_labels = util.get_obs_label(obs_list, obsConfig_d)
        ylabel = util.get_diffXs_label(obs_list, obsConfig_d, isRelative, "pb")
        ylabel_ratio = "$\dfrac{\mathrm{Prediction}}{\mathrm{Data}}$"
        yscale_log = obsConfig_d[obs_list[0]].get("log_scale", False)

        if len(obs_list) == 1:
            f_plot_diffXs = plot_diffXs_1D
            extra_args = {}
        elif len(obs_list) == 2:
            f_plot_diffXs = plot_diffXs_2D
            extra_args = {
                'rescales_order_of_magnitude': rescale_oom.get(obs),
                }
            yscale_log = True
        elif len(obs_list) == 3:
            f_plot_diffXs = plot_diffXs_3D
            extra_args = {
                'rescales_order_of_magnitude': rescale_oom.get(obs),
                }
            yscale_log = True
        else:
            raise RuntimeError(f"Don't know how to plot observable {obs}")

        figname = os.path.join(outputdir, obs, f'{obs}_{histname}')

        f_plot_diffXs(
            figname,
            obs_labels,
            hist_meas,
            hists_MC,
            'Data',
            labels_MC,
            (hist_relerr_tot_up, hist_relerr_tot_down),
            (hist_relerr_stat_up, hist_relerr_stat_down),
            ylabel = ylabel,
            ylabel_ratio = ylabel_ratio,
            log_obs = False, # for now
            log_diffXs = yscale_log,
            stamp_texts = stamps,
            **extra_args
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('fpath_nominal', type=str, action=util.ParseEnvVar,
                        help="File path to the nominal results")
    parser.add_argument('-a', '--uncertainties-abs', type=str, 
                        action=util.ParseEnvVar, help="File path to uncertainties of absolute differential cross sections")
    parser.add_argument('-r', '--uncertainties-rel', type=str, 
                        action=util.ParseEnvVar, help="File path to uncertainties of relative differential cross sections")
    parser.add_argument('-m', '--fpaths-otherMC', type=str, nargs='+', default=[],
                        action=util.ParseEnvVar, help="File paths to other MC predictions")
    parser.add_argument('-o', '--outputdir', type=str, 
                        action=util.ParseEnvVar, help="Output directory")
    parser.add_argument('--observables', type=str, nargs='+', default=[],
                        help="List of observables to plot")
    parser.add_argument('--label-nominal', type=str, default="MC",
                        help="Label of the nominal MC prediction")
    parser.add_argument('--labels-otherMC', type=str, nargs='+', default=[],
                        help="Labels of other MC predictions")
    parser.add_argument('--skip-nominalMC', action='store_true',
                        help="If True, do not plot the nominal MC prediction")
    parser.add_argument('--observable-config', action=util.ParseEnvVar,
                        default='${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json',
                        help="JSON configurations for observables")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="If True, set logging level to DEBUG")

    args = parser.parse_args()

    # logger
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # absolute differential cross sections
    plotDiffXs(
        args.fpath_nominal,
        fpath_uncertainties = args.uncertainties_abs,
        fpaths_otherMC = args.fpaths_otherMC,
        isRelative = False,
        outputdir = args.outputdir,
        observables = args.observables,
        label_nominal = args.label_nominal,
        labels_otherMC = args.labels_otherMC,
        skip_nominal_MC = args.skip_nominalMC,
        observable_config = args.observable_config
    )

    # relative differential cross sections
    plotDiffXs(
        args.fpath_nominal,
        fpath_uncertainties = args.uncertainties_rel,
        fpaths_otherMC = args.fpaths_otherMC,
        isRelative = True,
        outputdir = args.outputdir,
        observables = args.observables,
        label_nominal = args.label_nominal,
        labels_otherMC = args.labels_otherMC,
        skip_nominal_MC = args.skip_nominalMC,
        observable_config = args.observable_config
    )