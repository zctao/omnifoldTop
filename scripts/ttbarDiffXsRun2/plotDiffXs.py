import os
import logging
import numpy as np

import util
import histogramming as myhu
import plotter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

util.configRootLogger()
logger = logging.getLogger("plotDiffXs")

default_markers = ['o','s','^','v','d','<','>','p','h','*','x','1','2','3','4']

rescale_oom = {
    "ptt_vs_mtt": [1,1,1,1,1]
}

ATLAS_stamp = ["ATLAS WIP", "$\sqrt{s}=13$ TeV, 140 fb$^{-1}$"]

def get_error_arrays(hist_unc_down, hist_unc_up, hist_central=None):
    if hist_unc_up is not None and hist_unc_down is None:
        hist_unc_down = hist_unc_up
    elif hist_unc_up is None and hist_unc_down is not None:
        hist_unc_up = hist_unc_down

    if hist_unc_up is not None and hist_unc_down is not None:
        unc_arr = np.stack([
            np.abs(hist_unc_down.values()), np.abs(hist_unc_up.values())
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
        fhist_unc_down = fhist_unc_up
    elif fhist_unc_up is None and fhist_unc_down is not None:
        fhist_unc_up = fhist_unc_down

    if fhist_unc_up is not None and fhist_unc_down is not None:
        assert(len(fhist_unc_up)==len(fhist_unc_down))
        errors = []
        for ybin in fhist_unc_down:
            errors.append(
                np.stack([
                    np.abs(fhist_unc_down[ybin].values()),
                    np.abs(fhist_unc_up[ybin].values())
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
    hist_unc_up = None,
    hist_unc_down = None,
    ylabel = '',
    ylabel_ratio = '',
    log_obs = False,
    log_diffXs = False
    ):

    # unfolded measurement
    draw_opt_data = {
        'label': label_nominal, 'color': 'black', 'histtype': 'errorbar',
        'marker': 'o', 'markersize': 3, 'xerr': True
    }

    errors = get_error_arrays(hist_unc_down, hist_unc_up, histogram_data)
    if errors is not None:
        draw_opt_data['yerr'] = errors

    # MC truth
    colors_mc = plotter.get_default_colors(len(histograms_mc))
    draw_opts_mc = [
        {'label': l_mc, 'color': c_mc, 'histtype': 'step', 'xerr': True}
        for l_mc, c_mc in zip(labels_mc, colors_mc)
    ]

    xlabel = observable_label[0] if isinstance(observable_label, list) else observable_label

    plotter.plot_histograms_and_ratios(
        figname,
        hists_numerator = histograms_mc,
        hist_denominator = histogram_data,
        draw_options_numerator = draw_opts_mc,
        draw_option_denominator = draw_opt_data,
        xlabel = xlabel,
        ylabel = ylabel,
        ylabel_ratio = ylabel_ratio,
        log_xscale = log_obs,
        log_yscale = log_diffXs,
        stamp_texts=ATLAS_stamp,
        stamp_loc='upper left',
        stamp_opt={"prop":{"fontsize":"medium"}},
        y_lim = 'x2',
        ratio_lim = (0.5, 1.5)
    )

def draw_diffXs_distr_2D(
    ax,
    fhistogram_data,
    fhistograms_mc,
    label_nominal,
    labels_mc,
    fhist_unc_up = None,
    fhist_unc_down = None,
    xlabel = '',
    ylabel = '',
    title = '',
    log_xscale = False,
    log_yscale = False,
    rescales_order_of_magnitude = None
    ):

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

    # errors
    errors = get_error_arrays_2D(fhist_unc_down, fhist_unc_up, fhistogram_data)

    fhistogram_data.draw(
        ax,
        markers = default_markers[:len(fhistogram_data)],
        colors = 'black',
        common_styles = {'histtype': 'errorbar'},
        rescales_order_of_magnitude = rescales_order_of_magnitude,
        errors = errors,
        legend_off = True
    )

    leg_handles, leg_labels = ax.get_legend_handles_labels()

    # MC histograms
    colors_mc = plotter.get_default_colors(len(fhistograms_mc))
    for fhist_mc, c_mc in zip(fhistograms_mc, colors_mc):
        fhist_mc.draw(
            ax,
            colors = c_mc,
            common_styles = {'histtype': 'step'},
            rescales_order_of_magnitude = rescales_order_of_magnitude,
            legend_off = True,
            stamp_texts = [],
            stamp_loc = 'upper left',
        )

    # re-draw legends
    leg_handles.append(mlines.Line2D ([], [], color='black', linestyle='None', marker='o'))
    leg_labels.append(label_nominal)

    for c_mc, l_mc in zip(colors_mc, labels_mc):
        leg_handles.append(mlines.Line2D([], [], color=c_mc))
        leg_labels.append(l_mc)

    ax.legend(leg_handles, leg_labels)

def draw_diffXs_ratio_2D(
    axes,
    fhistogram_data,
    fhistograms_mc,
    label_nominal,
    labels_mc,
    fhist_unc_up = None,
    fhist_unc_down = None,
    xlabel = '',
    ylabel = '',
    log_xscale = False
    ):

    for ax in axes:
        ax.yaxis.grid()

        if xlabel:
            ax.set_xlabel(xlabel)

        if log_xscale:
            ax.set_xscale('log')

    if ylabel:
        axes[0].set_ylabel(ylabel)

    # legend
    handles, labels = [], []

    # errors
    if fhist_unc_up is not None and fhist_unc_down is None:
        fhist_unc_down = fhist_unc_up.copy().scale(-1.)
    elif fhist_unc_up is None and fhist_unc_down is not None:
        fhist_unc_up = fhist_unc_down.copy().scale(-1.)

    if fhist_unc_up is not None or fhist_unc_down is not None:
        assert(len(fhist_unc_up)==len(fhist_unc_down))
        ratio_unc_up = fhist_unc_up.divide(fhistogram_data)
        ratio_unc_down = fhist_unc_down.divide(fhistogram_data)
        # plot
        ratio_unc_up.draw(
            axes,
            colors = 'grey',
            common_styles = {'histtype':'fill'},
            legend_off = True
        )

        ratio_unc_down.draw(
            axes,
            colors = 'grey',
            common_styles = {'histtype':'fill'},
            legend_off = True
        )

        handles.append(mpatches.Patch(color='grey'))
        labels.append(label_nominal)

    # MC
    colors_mc = plotter.get_default_colors(len(fhistograms_mc))
    for fhist_mc, c_mc, l_mc in zip(fhistograms_mc, colors_mc, labels_mc):
        ratio_mc = fhist_mc.divide(fhistogram_data)
        ratio_mc.draw(
            axes,
            colors = c_mc,
            common_styles = {'histtype':'step'},
            legend_off = True
            )

        handles.append(mlines.Line2D([], [], color=c_mc))
        labels.append(l_mc)

    return handles, labels

def plot_diffXs_2D(
    figname,
    observable_labels,
    fhistogram_data,
    fhistograms_mc,
    label_nominal,
    labels_mc,
    fhist_unc_up = None,
    fhist_unc_down = None,
    ylabel = '',
    ylabel_ratio = '',
    log_obs = False,
    log_diffXs = False,
    title = '',
    title_ratio = '',
    rescales_order_of_magnitude = None
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
        fhist_unc_up = fhist_unc_up,
        fhist_unc_down = fhist_unc_down,
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        log_xscale = log_obs,
        log_yscale = log_diffXs,
        rescales_order_of_magnitude = rescales_order_of_magnitude
    )

    # save plot
    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    ######
    # draw the ratio in a separate plot
    fig_r, ax_r = plt.subplots(1, len(fhistogram_data), sharey=True)

    if title_ratio:
        fig_r.suptitle(title_ratio)

    rhandles, rlabels = draw_diffXs_ratio_2D(
        ax_r,
        fhistogram_data,
        fhistograms_mc,
        label_nominal,
        labels_mc,
        fhist_unc_up = fhist_unc_up,
        fhist_unc_down = fhist_unc_down,
        xlabel = xlabel,
        ylabel = ylabel_ratio,
        log_xscale = log_obs
    )

    # legend
    if rhandles:
        fig_r.legend(rhandles, rlabels, loc='outside upper right')

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
    fhist_unc_up = None,
    fhist_unc_down = None,
    ylabel = '',
    ylabel_ratio = '',
    log_obs = False,
    log_diffXs = False,
    title = '',
    title_ratio = '',
    rescales_order_of_magnitude = None
    ):

    fig, axes = plt.subplots(1, len(fhistogram_data))

    if title:
        fig.suptitle(title)

    # for ratio
    #fig_r, axes_r = plt.subplots(len(fhistogram_data), 1, sharey='row')
    fig_r = plt.figure(layout='constrained')
    subfigs_r = fig.subfigures(len(fhistogram_data), 1)

    if title_ratio:
        fig_r.suptitle(title_ratio)

    if not isinstance(rescales_order_of_magnitude, list):
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

    for i, zbin_label in enumerate(fhistogram_data):

        xlabel = observable_labels[0]

        draw_diffXs_distr_2D(
            axes[i],
            fhistogram_data[zbin_label],
            [fhist_mc[zbin_label] for fhist_mc in fhistograms_mc],
            label_nominal,
            labels_mc,
            fhist_unc_up = fhist_unc_up[zbin_label] if fhist_unc_up is not None else None,
            fhist_unc_down = fhist_unc_down[zbin_label] if fhist_unc_down is not None else None,
            xlabel = xlabel,
            ylabel = ylabel if i==0 else '',
            log_xscale = log_obs,
            log_yscale = log_diffXs,
            rescales_order_of_magnitude = rescales_order_of_magnitude[i],
        )

        # ratio
        axes_r = subfigs_r[i].subplots(1, len(fhistogram_data[zbin_label]), sharey=True)
        draw_diffXs_ratio_2D(
            axes_r,
            fhistogram_data[zbin_label],
            [fhist_mc[zbin_label] for fhist_mc in fhistograms_mc],
            label_nominal,
            labels_mc,
            fhist_unc_up = fhist_unc_up[zbin_label] if fhist_unc_up is not None else None,
            fhist_unc_down = fhist_unc_down[zbin_label] if fhist_unc_down is not None else None,
            xlabel = xlabel,
            ylabel = ylabel_ratio,
            log_xscale = log_obs
        )

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    #fig_r.savefig(figname+'_ratio.png', dpi=300, bbox_inches='tight')
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
        unc_d = {}

    # histogram name
    histname = 'relativeDiffXs' if isRelative else 'absoluteDiffXs'
    logger.debug(f"histogram name: {histname}")

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
        hist_unc_up, hist_unc_down = None, None
        if unc_d:
            # for now
            if not obs in unc_d:
                logger.warning(f"No uncertainties found for observable {obs}")
            else:
                hist_unc_up = unc_d[obs]['total_up']
                hist_unc_down = unc_d[obs]['total_down']

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
        elif len(obs_list) == 3:
            f_plot_diffXs = plot_diffXs_3D
            extra_args = {
                'rescales_order_of_magnitude': rescale_oom.get(obs),
                }
        else:
            raise RuntimeError(f"Don't know how to plot observable {obs}")

        figname = os.path.join(outputdir, obs, f'{obs}_{histname}')

        f_plot_diffXs(
            figname,
            obs_labels,
            hist_meas,
            hists_MC,
            'data',
            labels_MC,
            hist_unc_up,
            hist_unc_down,
            ylabel = ylabel,
            ylabel_ratio = ylabel_ratio,
            log_obs = False, # for now
            log_diffXs = yscale_log,
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
    parser.add_argument('--observables', type=str, nargs='+',
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