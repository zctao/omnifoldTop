import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import colormaps
import mplhep as hep
import functools

from numpy.random import default_rng
rng = default_rng()

import util
import histogramming as myhu

import logging
logger = logging.getLogger('plotter')

# styles
#hep.style.use(hep.style.ATLAS)
data_style = {'color': 'black', 'label': 'Data', 'histtype': 'errorbar', 'marker': 'o', 'markersize': 3}
sim_style = {'color': 'orange', 'label': 'Sim.', 'histtype': 'fill'}
bkg_style = {'color': 'cyan', 'label': 'Bkg.', 'histtype': 'fill'}
gen_style = {'color': 'blue', 'label': 'Gen.', 'histtype': 'step', 'lw':0.8}
truth_style = {'edgecolor': 'green', 'facecolor': (0.75, 0.875, 0.75), 'label': 'Truth', 'histtype': 'fill'}
omnifold_style = {'color': 'tab:red', 'label':'MultiFold', 'histtype': 'errorbar', 'marker': 's', 'markersize': 2}
ibu_style = {'color': 'gray', 'label':'IBU', 'histtype': 'errorbar', 'marker': 'o', 'markersize': 2}
error_style = {'lw': 1, 'capsize': 1.5, 'capthick': 1, 'markersize': 1.5}

def draw_ratio(ax, hist_denom, hists_numer, color_denom, colors_numer, label_denom=None, labels_numer=[], yerr_denom=None, yerrs_numer=[]):
    """
    Plot ratios of several numerator histograms to a denominator histogram.

    The denominator error is plotted around the line at y = 1. Numerators
    are drawn as points in the middle of the bin. If y error bars are
    provided, the x errors are +/- half the bin width.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    hist_denom : Hist object.
        Histogram for the denominator.
    hists_numer : sequence of Hist objects
        Each numerator is a sequence of histograms. num / denom will be
        plotted for each bin.
    color_denom : color
        Colour of the denominator
    colors_numer : sequence of colors
        Colour of each numerator series.
    """
    ax.yaxis.grid()

    bin_edges = hist_denom.axes[0].edges

    # denominator uncertainty
    values_denom, errors_denom = myhu.get_values_and_errors(hist_denom)
    # use the provided y errors if available
    if yerr_denom is not None:
        errors_denom = yerr_denom

    if errors_denom is not None:
        relerrs_denom = np.divide(errors_denom, values_denom, out=np.zeros_like(errors_denom), where=(values_denom!=0))

        if relerrs_denom.ndim == 1:
            relerrs_denom = np.append(relerrs_denom, relerrs_denom[-1])
            relerrs_denom_down = 1 - relerrs_denom
            relerrs_denom_up = 1 + relerrs_denom
        elif relerrs_denom.ndim == 2:
            relerrs_denom_down, relerrs_denom_up = relerrs_denom
            relerrs_denom_down = 1 - np.append(relerrs_denom_down, relerrs_denom_down[-1])
            relerrs_denom_up = 1 + np.append(relerrs_denom_up, relerrs_denom_up[-1])
        else:
            raise RuntimeError(f"draw_ratio: unknown error shape {errors_denom.shape}")

        ax.fill_between(bin_edges, relerrs_denom_down, relerrs_denom_up, step='post', facecolor=color_denom, alpha=0.3, label=label_denom)

    # in case hists_numer and colors_numer are not lists
    if not isinstance(hists_numer, list):
        hists_numer = [hists_numer]
    if not isinstance(colors_numer, list):
        colors_numer = [colors_numer]

    if not labels_numer:
        labels_numer = [None] * len(hists_numer)
    elif not isinstance(labels_numer, list):
        labels_numer = [labels_numer]

    if not yerrs_numer:
        yerrs_numer = [None] * len(hists_numer)
    else:
        assert(len(yerrs_numer)==len(hists_numer))

    for hnum, cnum, lnum, yerr_n in zip(hists_numer, colors_numer, labels_numer, yerrs_numer):
        if hnum is None:
            continue

        values_num, errors_num = myhu.get_values_and_errors(hnum)
        # use the provided error if available
        if yerr_n is not None:
            errors_num = yerr_n

        ratio = np.divide(values_num, values_denom, out=np.zeros_like(values_denom), where=(values_denom!=0))

        if errors_num is not None:
            ratio_errs = np.divide(errors_num, values_denom, out=np.zeros_like(errors_num), where=(values_denom!=0))
            ratio_errs = np.abs(ratio_errs)
        else:
            ratio_errs = None

        hep.histplot(ratio, bin_edges, yerr=ratio_errs, histtype='errorbar', xerr=True, color=cnum, ax=ax, label=lnum, **error_style)

def draw_stamp(ax, texts, x=0.5, y=0.5, dy=0.045, **opts):
    """
    Add a series of text to the axis, one line per text.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    texts : iterable of str
        Each item in the iterable will be drawn on its own line.
    x, y : float, default: 0.5
        Position of the text in data coordinates.
    dy : float, default: 0.045
        Separation between texts. Note that this will not set interline
        separation if a text includes a \n.
    """
    textopts = {'horizontalalignment': 'left',
                'verticalalignment': 'center',
                'fontsize': 5.,
                'transform': ax.transAxes}

    textopts.update(opts)

    for i, txt in enumerate(texts):
        if txt is not None:
            ax.text(x, y-i*dy, txt, **textopts)

def draw_text(ax, texts, loc='center', prop={'size':5}, frameon=False, **kwargs):
    if isinstance(texts, list):
        txt_str = '\n'.join(texts)
    else:
        txt_str = texts

    at = AnchoredText(txt_str, loc=loc, prop=prop, frameon=frameon, **kwargs)
    ax.add_artist(at)

def draw_error_band(
    ax,
    h_central, # hist.Hist
    binerrors, # numpy.ndarray
    **draw_options
    ):

    bin_edges = h_central.axes[0].edges

    if binerrors.ndim == 2:
        binerrors_up = binerrors[1]
        binerrors_down = binerrors[0]
    elif binerrors.ndim == 1:
        binerrors_up = binerrors
        binerrors_down = binerrors * -1.
    else:
        raise RuntimeError(f"Cannot handle bin errors of shape: {binerrors.shape()}")

    values_up = h_central.values() + binerrors_up
    values_down = h_central.values() + binerrors_down

    values_up = np.append(values_up, values_up[-1])
    assert len(values_up)==len(bin_edges)

    values_down = np.append(values_down, values_down[-1])
    assert len(values_down)==len(bin_edges)

    ax.fill_between(bin_edges, values_down, values_up, step='post', **draw_options)

    return ax

def draw_ratio_new(
    ax,
    histogram_denom,
    histograms_numer,
    draw_option_denom={},
    draw_optinos_numer=[],
    xlabel = None,
    ylabel = None
    ):

    ax.yaxis.grid()

    hists_ratio = []
    draw_options_ratio = []

    histogram_ref = histogram_denom.copy()
    # set variance to 0: the error on the ratio plot = original error / denominator
    histogram_ref.view()['variance'] = 0.

    # draw denominator error bar
    hists_ratio.append(myhu.divide(histogram_denom, histogram_ref))

    draw_options_ratio.append(draw_option_denom.copy())
    # update yerr if needed
    if 'yerr' in draw_options_ratio[-1] and draw_options_ratio[-1]['yerr'] is not None:
        draw_options_ratio[-1]['yerr'] = draw_options_ratio[-1]['yerr'] / histogram_ref.values()

    if not 'style_error_band' in draw_options_ratio[-1]:
        draw_options_ratio[-1]['style_error_band'] = {
            'edgecolor':'none',
            'facecolor': get_color_from_draw_options(draw_options_ratio[-1]),
            'alpha' : 0.3
        }

    # do not draw donominator central
    draw_options_ratio[-1]['skip_central'] = True

    # numerators
    if not draw_optinos_numer:
        draw_optinos_numer = [{}] * len(histograms_numer)

    for hist_numer, draw_opt in zip(histograms_numer, draw_optinos_numer):
        hists_ratio.append(myhu.divide(hist_numer, histogram_ref))

        dopt = draw_opt.copy()
        if 'yerr' in dopt and dopt['yerr'] is not None:
            dopt['yerr'] = dopt['yerr'] / histogram_ref.values()
        dopt.update(error_style)
        dopt['histtype'] = 'errorbar'
        dopt['xerr'] = True
        draw_options_ratio.append(dopt)

    draw_histograms(
        ax,
        hists_ratio,
        draw_options_ratio,
        xlabel = xlabel,
        ylabel = ylabel,
        legend_loc = None # do not draw legend on the ratio plot
    )

    return ax

def draw_histograms(
    ax,
    histograms, # list of Hist
    draw_options, # list of dict
    xlabel = None,
    ylabel = None,
    log_scale = False,
    legend_loc="best",
    legend_ncol=1,
    stamp_texts=[],
    stamp_loc=(0.75, 0.75),
    stamp_opt={},
    title = '',
    stack = False
    ):

    if not histograms:
        logger.warn("No histograms to plot")
        return

    # x-axis limits and labels
    bin_edges = histograms[0].axes[0].edges
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    if stack:
        style_keys = list(draw_options[-1].keys())
        style_stack = {k: [dopt[k] for dopt in draw_options] for k in style_keys}
        # histtype does not allow a list of styles for the stacked histograms
        style_stack['histtype'] = 'fill'

        hep.histplot(histograms, yerr=False, stack=True, ax=ax, **style_stack)
    else:
        for h, opt in zip(histograms, draw_options):
            opt_tmp = opt.copy()

            # bin errors: use 'yerr' in opt if provided, otherwise extract from h
            binerrs = opt_tmp.pop('yerr', myhu.get_values_and_errors(h)[1])

            # error bar style
            error_band_style = opt_tmp.pop('style_error_band', {})
            if error_band_style:
                # draw error band
                draw_error_band(ax, h, binerrs, **error_band_style)

                # draw central
                if not opt_tmp.pop('skip_central', False):
                    hep.histplot(h, ax=ax, yerr=False, **opt_tmp)
            else:
                # draw normally
                hep.histplot(h, ax=ax, yerr=binerrs, **opt_tmp)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if log_scale:
        ax.set_yscale('log')

    if title:
        ax.set_title(title)

    # legend
    if legend_loc is not None:
        ax.legend(loc=legend_loc, ncol=legend_ncol, frameon=False)

    # stamp
    if stamp_texts:
        if isinstance(stamp_loc, str):
            draw_text(ax, stamp_texts, loc=stamp_loc, **stamp_opt)
        else:
            draw_stamp(ax, stamp_texts, *stamp_loc, **stamp_opt)

def get_default_colors(ncolors):
    """
    Get up to the first `ncolors` of the default colour cycle.

    Parameters
    ----------
    ncolors : non-negative int

    Returns
    -------
    sequence of str
         The first `ncolors` items in the matplotlib colour cycle, as a
         sequence of 6-digit RGB hex strings (i.e. "#rrggbb"). Returns a
         number of colours equal to the shorter of (`ncolors`, length of
         the cycle).
    """
    if ncolors <= 10:
        return plt.rcParams['axes.prop_cycle'].by_key()['color'][:ncolors]
    else:
        return get_random_colors(ncolors)

def get_random_colors(ncolors, alpha=1):
    colors = [ tuple(rng.random(3)) + (alpha,) for i in range(ncolors)]
    return colors

def get_default_colormaps(nmaps):
    cmap_names = [
        'Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        ]

    cmaps = [colormaps[cn] for cn in cmap_names[:nmaps]]

    return cmaps

def plot_graphs(
        figname,
        data_arrays,
        error_arrays=None,
        labels=None,
        title='',
        xlabel='',
        ylabel='',
        xscale=None,
        yscale=None,
        colors=None,
        markers=None,
        **style
):
    """
    Create scatter plots of several data series on the same x and y axes.

    The generated figure is saved to "{figname}.png".

    Parameters
    ----------
    figname : str
        Path to save the figure, excluding the file extension.
    data_arrays : sequence of n (xdata, ydata) pairs where xdata, ydata are array-like
        Data positions to plot. Each element is one data series.
    error_arrays : sequence of n floats or 2-tuples, optional
        Error bars. Sequence of floats is interpreted as y error bars;
        2-tuples are interpreted as (x error bars, y error bars).
    labels : sequence of n str, optional
         Data series labels. If provided, the plot will contain a legend.
    title : str, default: ""
        Title of the figure.
    xlabel, ylabel : str, default: ""
        Axis labels.
    xscale, yscale : {"log", "log2"}, optional
        Set the axis scale to log base 10 or log base 2. If omitted, the
        axis is linear.
    colors : sequence of n colors, optional
        Line/marker colours for each data series. See also matplotlib.colors.
    markers : sequence of n str or matplotlib.markers.MarkerStyle, optional
        Marker style for each series. See also matplotlib.markers.
    **style : dict, optional
        Additional keyword arguments passed to matplotlib.axes.Axes.errorbar.
    """
    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xscale=='log':
        ax.set_xscale('log')
    elif xscale=='log2':
        ax.set_xscale('log', basex=2)

    if yscale=='log':
        ax.set_yscale('log')
    elif yscale=='log2':
        ax.set_yscale('log', basey=2)

    if colors is None:
        if len(data_arrays) <=10:
            colors = get_default_colors(len(data_arrays))
        else:
            colors = get_random_colors(len(data_arrays))
    else:
        assert(len(data_arrays)==len(colors))

    if error_arrays is not None:
        assert(len(error_arrays)==len(data_arrays))

    for i, (x, y) in enumerate(data_arrays):
        label = None if labels is None else labels[i]
        marker = None if markers is None else markers[i]
        color = colors[i%len(colors)]

        xerr, yerr = None, None
        if error_arrays is not None:
            error = error_arrays[i]
            if isinstance(error, tuple):
                xerr, yerr = error
            else:
                yerr = error

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, label=label, marker=marker, color=color, **style)

    # plot legend if needed
    if labels is not None:
        ax.legend()

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def plot_hist(
    figname,
    data_arrs,
    weight_arrs=None,
    labels=None,
    nbins=20,
    xlabel=None,
    ylabel=None,
    title=None,
    bin_margin=0.1
    ):
    """
    Plot a series of datasets as 1d histograms.

    Each dataset uses the same binning, which is evenly distributed across
    the full range of all combined datasets. The generated figure is saved
    to "{figname}.png".

    Parameters
    ----------
    figname : str
        Path to save the figure, excluding the file extension.
    data_arrs : sequence of n array-like
        Unbinned data.
    weight_arrs : sequence of n array-like, optional
        Each weight array should be the same shape as the corresponding
        data. Each data point contributes only its associated weight to the
        bin count instead of 1.
    labels : sequence of str, optional
        Labels for data arrays
    nbins : positive int, default: 20
        Number of bins in the histogram.
    xlabel: str, optional
        X-axis label
    ylabel: str, optional
        Y-axis label
    title: str, optional
        Histogram title
    bin_margin: float, optional
        Margin in percentage for determining bin edges
    """
    xmax = max([np.nanmax(data) for data in data_arrs])
    xmin = min([np.nanmin(data) for data in data_arrs])
    margin = (xmax - xmin) * bin_margin
    bins = np.linspace(xmin-margin, xmax+margin, nbins+1)

    if weight_arrs is None:
        weight_arrs = [None] * len(data_arrs)
    else:
        assert(len(weight_arrs) == len(data_arrs))

    if labels is None:
        labels = [None] * len(data_arrs)
    else:
        assert(len(labels) == len(data_arrs))

    fig, ax = plt.subplots()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    for data, w, l in zip(data_arrs, weight_arrs, labels):
        ax.hist(data, bins=bins, weights=w, histtype='step', label=l)

    ax.legend()

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def get_color_from_draw_options(draw_opt):
    c = draw_opt.get('color')

    if c is None:
        c = draw_opt.get('edgecolor')

    if c is None:
        c = draw_opt.get('facecolor')

    if c is None:
        logger.warn("Color is not set")

    if isinstance(c, list):
        c = c[-1]

    return c

def plot_histogram_and_function(
    figname,
    histogram,
    draw_option_histogram,
    function,
    draw_option_function,
    xlabel = None,
    ylabel = None,
    title = None
    ):

    fig, ax = plt.subplots()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # draw the histogram
    yerr = myhu.get_values_and_errors(histogram)[1]
    hep.histplot(histogram, yerr=yerr, ax=ax, **draw_option_histogram)

    # draw the fit function
    binedges = histogram.axes.edges[0]
    xp = np.linspace(binedges[0], binedges[-1],100)
    ax.plot(xp, function(xp), **draw_option_function)

    draw_text(ax, str(function), loc='lower left', prop={'size':6}, frameon=True)

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def plot_histograms_and_ratios(
    figname,
    hists_numerator, # list of Hist
    hist_denominator = None, # Hist; If None, don't plot the ratio
    draw_options_numerator = None, # list of dict
    draw_option_denominator = {}, # dict
    xlabel = '',
    ylabel = '',
    ylabel_ratio = 'Ratio',
    log_xscale = False,
    log_yscale = False,
    log_scale = False, # same as log_yscale
    legend_loc="best",
    legend_ncol=1,
    stamp_texts=[],
    stamp_loc=(0.75, 0.75),
    stamp_opt={},
    denominator_ratio_only = False,
    y_lim = None,
    ratio_lim = None,
    title = '',
    stack_numerators = False,
    height_ratios = (4,1),
    **fig_kw
    ):

    if not hists_numerator:
        logger.warn("No histograms to plot")
        return

    fig, axes = plt.subplots(
        2 if hist_denominator else 1,
        sharex = True,
        gridspec_kw = {
            'height_ratios': height_ratios if hist_denominator else (1,),
            'hspace': 0.15
        },
        **fig_kw
    )

    if hist_denominator:
        ax = axes[0]
        ax_ratio = axes[1]
    else:
        ax = axes
        ax_ratio = None

    if log_xscale:
        ax.set_xscale('log')
    if log_yscale or log_scale:
        ax.set_yscale('log')

    if isinstance(y_lim, tuple):
        ax.set_ylim(*y_lim)

    # draw histograms
    if hist_denominator and not denominator_ratio_only:
        # use 'yerr' in draw_option_denominator if provided
        # otherwise extract from the histogram object
        if 'yerr' in draw_option_denominator:
            hep.histplot(hist_denominator, ax=ax, **draw_option_denominator)
        else:
            yerr_d = myhu.get_values_and_errors(hist_denominator)[1]
            hep.histplot(hist_denominator, ax=ax, yerr=yerr_d, **draw_option_denominator)

    if not draw_options_numerator:
        draw_options_numerator = [{}] * len(hists_numerator)
    else:
        assert(len(hists_numerator) == len(draw_options_numerator))

    draw_histograms(
        ax,
        histograms = hists_numerator,
        draw_options = draw_options_numerator,
        xlabel = xlabel if ax_ratio is None else '',
        ylabel = ylabel,
        legend_loc = legend_loc,
        legend_ncol = legend_ncol,
        stamp_texts = stamp_texts,
        stamp_loc = stamp_loc,
        stamp_opt=stamp_opt,
        title = title,
        stack = stack_numerators
    )

    # special case
    if isinstance(y_lim, str): # e.g. "x2"
        try:
            yfactor = float(y_lim.lstrip("x"))
        except ValueError:
            logger.error(f"Cannot convert y_lim {y_lim}")
            yfactor = 1.

        ylim_bot, ylim_top = ax.get_ylim()
        if log_yscale or log_scale:
            ylim_top *= 10**yfactor
        else:
            ylim_top *= yfactor
        ax.set_ylim(ylim_bot, ylim_top)

    # draw_ratios
    if hist_denominator:

        assert(ax_ratio is not None)
        ax_ratio.set_ylabel(ylabel_ratio)
        ax_ratio.set_xlabel(xlabel)
        ax_ratio.set_xlim(*ax.get_xlim())
        if ratio_lim:
            ax_ratio.set_ylim(*sorted(ratio_lim))

        if log_xscale:
            ax_ratio.set_xscale('log')

        # denominator
        draw_opt_den_r = draw_option_denominator.copy()

        # numerator
        if stack_numerators:
            hists_num_r = [functools.reduce(lambda x,y: x+y, hists_numerator)]
            draw_opt_num_r = [draw_options_numerator[-1].copy()]
            # stack yerr
            yerrs_num = [opt.get('yerr') for opt in draw_options_numerator]
            yerrs_tot_num = functools.reduce(lambda x,y: np.sqrt(x*x+y*y), yerrs_num) if not any(e is None for e in yerrs_num) else None
            draw_opt_num_r[-1]['yerr'] = yerrs_tot_num
        else:
            hists_num_r = hists_numerator
            draw_opt_num_r = draw_options_numerator.copy()

        draw_ratio_new(
            ax_ratio,
            hist_denominator,
            hists_num_r,
            draw_opt_den_r,
            draw_opt_num_r
        )

    # save plot
    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_distributions_reco(
    figname,
    hist_data,
    hist_sig,
    hist_bkg=None,
    xlabel='',
    ylabel='Events',
    ylabel_ratio='Ratio to\nData',
    legend_loc="best",
    legend_ncol=1,
    ratio_lim = None,
    ):
    """
    Plot detector-level variable distributions.

    Parameters
    ----------
    figname : str
        Path to save the generated figure, excluding the file extension.
    hist_data, hist_sig, hist_bkg : Hist objects
        Histograms for observables, signal, and
        background (optional), respectively.
    xlabel : str
        Label for x axis
    ylabel : str, default: 'Events'
        Label for y axis
    ylabel_ratio : str, default: 'Ratio to\nData'
        Label for y axis of the ratio plot
    legend_loc : str, edefault: 'best'
        Legend location
    legend_ncol : int, default: 1
        Number of column of legend
    """

    # simulation
    if hist_bkg is None:
        hist_sim = [hist_sig]
        style_sim = [sim_style.copy()]
    else:
        hist_sim = [hist_bkg, hist_sig]
        style_sim = [bkg_style.copy(), sim_style.copy()]

    style_data = data_style.copy()
    style_data.update({"xerr":True})

    plot_histograms_and_ratios(
        figname,
        hists_numerator = hist_sim,
        hist_denominator = hist_data,
        draw_options_numerator = style_sim,
        draw_option_denominator = style_data,
        xlabel = xlabel,
        ylabel = ylabel,
        ylabel_ratio = ylabel_ratio,
        log_scale = False,
        legend_loc = legend_loc,
        legend_ncol = legend_ncol,
        stack_numerators = True,
        ratio_lim = ratio_lim
        )

def plot_distributions_unfold(
    figname,
    hist_unfold,
    hist_prior = None,
    hist_truth = None,
    hist_ibu = None,
    xlabel='',
    ylabel='Events',
    ylabel_ratio='Ratio to\nTruth',
    legend_loc="best",
    legend_ncol=1,
    stamp_texts=[],
    stamp_loc=(0.75, 0.75),
    ratio_lim = None,
    ):
    """
    Plot and compare the unfolded distributions.

    Parameters
    ----------
    figname : str
        Path to save the generated figure, excluding the file extension.
    hist_unfold, hist_prior, hist_truth, hist_ibu : Hist objects
        Histograms for unfolded distribution with OmniFold, the prior 
        distribution (optional), unfolded distributioin with IBU (optional), 
        and the truth distribution (optional), respectively.
    xlabel : str
        Label for x axis
    ylabel : str, default: 'Events'
        Label for y axis
    ylabel_ratio : str, default: 'Ratio to\nData'
        Label for y axis of the ratio plot
    legend_loc : str, edefault: 'best'
        Legend location
    legend_ncol : int, default: 1
        Number of column of legend
    stamp_texts : sequence of str, default: []
        Additional text to draw on the axes.
    stamp_loc : tuple of (float, float)
        Position of the additional text in data coordinates.
    ratio_lim : tuple of (float, float)
        Y-axis limit of the ratio plot
    """

    hists_toplot = []
    draw_options = []

    if hist_prior:
        hists_toplot.append(hist_prior)
        draw_options.append(gen_style)

    if hist_ibu is not None:
        hists_toplot.append(hist_ibu)
        ibu_opt = ibu_style.copy()
        ibu_opt.update({'xerr':True})
        draw_options.append(ibu_opt)

    hists_toplot.append(hist_unfold)
    omnifold_opt = omnifold_style.copy()
    omnifold_opt.update({'xerr':True})
    draw_options.append(omnifold_opt)

    plot_histograms_and_ratios(
        figname,
        hists_numerator = hists_toplot,
        hist_denominator = hist_truth,
        draw_options_numerator = draw_options,
        draw_option_denominator = truth_style,
        xlabel = xlabel,
        ylabel = ylabel,
        ylabel_ratio = ylabel_ratio,
        log_scale = False,
        legend_loc = legend_loc,
        legend_ncol = legend_ncol,
        stamp_texts = stamp_texts,
        stamp_loc = stamp_loc,
        ratio_lim = ratio_lim
        )

def plot_correlations(figname, correlations, bins=None, print_bincontents=False):
    fig, ax = plt.subplots()

    if bins is None:
        bins = np.linspace(0, len(correlations), len(correlations)+1)
        ax.tick_params(axis='both', labelsize='small')
        ax.tick_params(axis='x', top=False, labeltop=False, bottom=True, labelbottom=True, labelrotation=30)
        ticks = np.arange(0.5, len(correlations)+0.5, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(correlations.columns)
        ax.set_yticklabels(correlations.columns)

    hep.hist2dplot(
        correlations, xbins=bins, ybins=bins,
        cbar=True, cmin=-1, cmax=1, cmap='coolwarm',
        ax=ax)

    if print_bincontents:
        if isinstance(correlations, pd.DataFrame):
            indices = correlations.columns
        else:
            indices = list(range(len(correlations)))

        fontsize = min(10, 40/(len(bins)-1))

        centers = (bins[:-1]+bins[1:])/2
        for ix, xc in zip(indices, centers):
            for iy, yc in zip(indices, centers):
                bin_content = f"{correlations[ix][iy]:.2f}"
                ax.text(xc, yc, bin_content, ha='center', va='center', fontsize=fontsize)

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=200)
    plt.close(fig)

def plot_distributions_resamples(
    figname,
    hists_uf_resample,
    hist_prior,
    hist_truth = None,
    xlabel='',
    ylabel=''
    ):

    # colors
    colors_rs = get_random_colors(len(hists_uf_resample), alpha=0.5)

    draw_options_rs = []

    for i, (huf, color) in enumerate(zip(hists_uf_resample, colors_rs)):
        draw_opt_i = {"histtype":"step", "ls":"--", "lw":1, "color":color}
        if i==0:
            draw_opt_i.update({"label":"Resample"})

        draw_options_rs.append(draw_opt_i)

    if hist_truth:
        hists_numer = hists_uf_resample + [hist_prior]
        draw_opt_numer = draw_options_rs + [gen_style]
        hist_denom = hist_truth
        draw_opt_denom = {'histtype':'step', 'label':truth_style['label'], 'color':truth_style['edgecolor']}
        ylabel_ratio = "Ratio to \nTruth"
    else:
        hists_numer = hists_uf_resample
        draw_opt_numer = draw_options_rs
        hist_denom = hist_prior
        draw_opt_denom = gen_style
        ylabel_ratio = "Ratio to \nPrior"

    plot_histograms_and_ratios(
        figname,
        hists_numerator = hists_numer,
        hist_denominator = hist_denom,
        draw_options_numerator = draw_opt_numer,
        draw_option_denominator = draw_opt_denom,
        xlabel = xlabel,
        ylabel = ylabel,
        ylabel_ratio = ylabel_ratio
        )

def plot_distributions_iteration(
    figname,
    hists_unfold,
    hist_prior,
    hist_truth=None,
    nhistmax=7,
    xlabel='',
    ylabel=''
    ):

    # add prior to the head of the unfolded hist list i.e. as iteration 0
    hists_all = [hist_prior] + list(hists_unfold)

    # if there are more than nhistmax histograms, plot at most nhistmax
    assert(nhistmax<=10) # plt.rcParams['axes.prop_cycle'] provides max 10 colors
    if (len(hists_all) > nhistmax):
        selected_i = np.linspace(1, len(hists_all)-2, nhistmax-2).astype(int).tolist()
        # the first [0] and the last [-1] are always plotted
        selected_i = [0] + selected_i + [len(hists_all)-1]
    else:
        selected_i = list(range(len(hists_all)))

    hists_toplot = [hists_all[i] for i in selected_i]

    colors = get_default_colors(len(selected_i))

    draw_options_num = []
    for i, h, c in zip(selected_i, hists_toplot, colors):
        draw_options_num.append({
            'histtype':'step',
            'color':c, 'alpha':0.8,
            'label':f'iteration {i}'
            })

    if hist_truth:
        # ratio to truth
        ylabel_ratio = "Ratio to\n Truth"
        hist_denom = hist_truth
        draw_opt_denom = truth_style
    else:
        # ratio to prior
        ylabel_ratio = "Ratio to\n Prior"
        hist_denom = hist_prior
        draw_opt_denom = draw_options_num[0]

    plot_histograms_and_ratios(
        figname,
        hists_numerator = hists_toplot,
        hist_denominator = hist_denom,
        draw_options_numerator = draw_options_num,
        draw_option_denominator = draw_opt_denom,
        xlabel = xlabel,
        ylabel = ylabel,
        ylabel_ratio = ylabel_ratio,
        denominator_ratio_only = True
        )

def plot_hists_bin_distr(figname, histograms_list, histogram_ref=None):

    histograms_arr = np.asarray(histograms_list)['value']
    # histograms_arr can be either dimension 3 if all iterations are included
    # or dimension 2 if only the final unfolded ones for all trials

    nbins = histograms_arr.shape[-1]
    niterations = histograms_arr.shape[1] if histograms_arr.ndim > 2 else 1

    fig, ax = plt.subplots(nrows=niterations, ncols=nbins,
                            sharey='row', sharex='col',
                            figsize=((nbins+1)*1.1, niterations*1.1),
                            constrained_layout=True)

    # loop over bins
    for ibin in range(nbins):
        nentries = histograms_arr[..., ibin]
        # nentries is of the shape (nresamples, niterations) if include iteration history
        # otherwise it is of the shape (nresamples,)

        #mu = np.mean(nentries, axis=0)
        #sigma = np.std(nentries, ddof=1, axis=0)
        #pulls = (nentries - mu) / sigma

        # compare and standardize to ref
        ref = histogram_ref.values()[ibin] if histogram_ref else 0
        nentries_ref = nentries - ref

        # plot
        if niterations == 1: # only the final results
            h, b = ax[ibin].hist(nentries_ref, density=True)[:2]
            plot_gaussian(ax[ibin], h, b, dofit=False)
            ax[ibin].set_xlabel("Bin {}".format(ibin+1))
            ax[ibin].xaxis.set_label_position('top')
            ax[ibin].tick_params(labelsize=7)
            ax[ibin].ticklabel_format(style='sci', scilimits=(-2,2), useMathText=True)
            ax[ibin].xaxis.get_offset_text().set_fontsize(7)
        else: # all iterations
            for it in range(niterations):
                h, b = ax[it][ibin].hist(nentries_ref[...,it], density=False)[:2]
                plot_gaussian(ax[it][ibin], h, b, dofit=False)
                # labels
                if ibin == 0:
                    ax[it][ibin].set_ylabel("Iter {}".format(it+1))
                if it == 0:
                    ax[it][ibin].set_xlabel("Bin {}".format(ibin+1))
                    ax[it][ibin].xaxis.set_label_position('top')
                ax[it][ibin].tick_params(labelsize=7)
                ax[it][ibin].ticklabel_format(style='sci', scilimits=(-2,2), useMathText=True)
                ax[it][ibin].xaxis.get_offset_text().set_fontsize(7)

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_gaussian(ax, histogram, binedges, dofit=False):
    """
    Draw a Gaussian fit to a histogram on the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    histogram : (n,) array-like
        Bin heights in a histogram.
    binedges : (n + 1,) array-like
        Location of bin edges for `histogram`.
    dofit : bool, default: False
        If True, fit the Gaussian to the histogram using least-squares. If
        False, use a Gaussian with the same mean, standard deviation, and
        maximum as the histogram.
    """
    A, mu, sigma = util.fit_gaussian_to_hist(histogram, binedges, dofit)

    x = np.linspace(binedges[0], binedges[-1], 50)
    y = util.gaus(x, A, mu, sigma)
    style = '-' if dofit else '--'
    ax.plot(x, y, style)

    #yref = gaus(x, sum(histogram)/math.sqrt(2*math.pi), 0, 1)
    #ax.plot(x, yref, '--')

def plot_LR_distr(figname, ratios, labels=None, xlabel="Likelihood ratio", logy=False, nbins=25):
    """
    Plot the distribution of likelihood ratios.

    Parameters
    ----------
    figname : str
        Path to save the figure, excluding the file extension.
    ratios : sequence of array-like
        Datasets of unbinned likelihood ratios.
    labels : sequence of str, optional
        Labels for each set of likelihood ratios in `ratios`.
    """
    #np.linspace(min(r.min() for r in ratios)*0.9, max(r.max() for r in ratios)*1.1, 50)

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale('log')

    for i,r in enumerate(ratios):
        hr = myhu.calc_hist(r, nbins, density=False)
        l = labels[i] if labels is not None else None
        hep.histplot(hr, ax=ax, label=l)

    if labels is not None:
        ax.legend()

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def plot_training_vs_validation(
        figname,
        predictions_train,
        labels_train,
        weights_train,
        predictions_val,
        labels_val,
        weights_val,
        nbins=100
):
    """
    Compare the model's performance on the training vs. validation set

    Parameters
    ----------
    figname : str
        Path to save the figure, excluding the file extension.
    predictions_train : (n_pred,) array-like of floats in [0, 1]
        Probability that each event in the training set is in category 1.
    labels_train : (n_pred,) or (n_pred, m) array-like
        Labels for events in the test set, either as a 1d array or categorical.
    weights_train : (n_pred,) array-like
        Weight per event in the training set. Each event contributes its
        associated weight to its bin instead of 1.
    predictions_val : (n_val,) array-like of floats in [0, 1]
        Probability that each event in the validation set is in category 1.
    labels_val : (n_val,) or (n_val, m) array-like
        Labels for events in the validation set, either as a 1d array or categorical.
    weights_val : (n_val,) array-like
        Weight per event in the validation set. Each event contributes its
        associated weight to its bin instead of 1.
    nbins : positive int, default: 100
        Number of bins in the generated histogram.
    """
    # determine bins of histograms to plot
    bins_min = math.floor(min(predictions_train.min(), predictions_val.min())*10)/10
    bins_max = math.ceil(max(predictions_train.max(), predictions_val.max())*10)/10
    bins_preds = np.linspace(bins_min, bins_max, nbins)

    if labels_train.ndim == 1: # label array is simply a 1D array
        preds_cat1_t = predictions_train[labels_train==1]
        preds_cat0_t = predictions_train[labels_train==0]
        w_cat1_t = weights_train[labels_train==1]
        w_cat0_t = weights_train[labels_train==0]
    else: # label array is categorical
        preds_cat1_t = predictions_train[labels_train.argmax(axis=1)==1]
        preds_cat0_t = predictions_train[labels_train.argmax(axis=1)==0]
        w_cat1_t = weights_train[labels_train.argmax(axis=1)==1]
        w_cat0_t = weights_train[labels_train.argmax(axis=1)==0]

    hist_preds_cat1_t = myhu.calc_hist(preds_cat1_t, bins_preds, weights=w_cat1_t, density=True)
    hist_preds_cat0_t = myhu.calc_hist(preds_cat0_t, bins_preds, weights=w_cat0_t, density=True)

    # validation data
    if labels_val.ndim == 1: # label array is simply a 1D array
        preds_cat1_v = predictions_val[labels_val==1]
        preds_cat0_v = predictions_val[labels_val==0]
        w_cat1_v = weights_val[labels_val==1]
        w_cat0_v = weights_val[labels_val==0]
    else: # label array is categorical
        preds_cat1_v = predictions_val[labels_val.argmax(axis=1)==1]
        preds_cat0_v = predictions_val[labels_val.argmax(axis=1)==0]
        w_cat1_v = weights_val[labels_val.argmax(axis=1)==1]
        w_cat0_v = weights_val[labels_val.argmax(axis=1)==0]

    hist_preds_cat1_v = myhu.calc_hist(preds_cat1_v, bins_preds, weights=w_cat1_v, density=True)
    hist_preds_cat0_v = myhu.calc_hist(preds_cat0_v, bins_preds, weights=w_cat0_v, density=True)

    fig, ax = plt.subplots()
    ax.set_xlabel("Prediction (y = 1)")

    hep.histplot(hist_preds_cat1_t, ax=ax, label='y = 1 (training)', histtype='step')
    hep.histplot(hist_preds_cat0_t, ax=ax, label='y = 0 (training)', histtype='step')
    hep.histplot(hist_preds_cat1_v, ax=ax, label='y = 1 (validation)', histtype='errorbar', markersize=3, marker='+')
    hep.histplot(hist_preds_cat0_v, ax=ax, label='y = 0 (validation)', histtype='errorbar', markersize=3, marker='+')

    ax.legend()

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def plot_train_log(csv_file, plot_name=None):
    """
    Generate a plot of training loss from a training history file.

    Parameters
    ----------
    csv_file : str
        Path to the training history CSV file.
    plot_name : str, optional
        Path to save the figure, excluding the file extension. If omitted,
        the generated figure will have the same name as the input file with
        the file extension replaced.
    """
    df = pd.read_csv(csv_file)

    if plot_name is None:
        plot_name = csv_file.replace('.csv', '_loss')

    #plot_train_loss(plot_name, df['loss'], df['val_loss'])

    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('loss')

    ax.plot(df['loss'], label='loss')
    ax.plot(df['val_loss'], label='val loss')

    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_useMathText(True)
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-3,4))

    ax.legend()

    plt.savefig(plot_name+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def init_subplots_grid(
    nrows,
    ncols,
    figsize_per_grid = (5.5,3.5),
    xlabels = None,
    ylabels = None,
    **subplots_args
    ):

    fig, ax = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize = (ncols*figsize_per_grid[0], nrows*figsize_per_grid[1]),
        **subplots_args
        )

    ax = ax.reshape(nrows, ncols)

    # margin
    mg_l = 0.17/ncols
    mg_r = 1 - 0.05/ncols
    mg_t = 1 - 0.15/nrows
    mg_b = 0.15/nrows
    fig.subplots_adjust(left=mg_l, right=mg_r, top=mg_t, bottom=mg_b)

    if xlabels is not None:
        assert(len(xlabels)==ncols)
        for c, xl in enumerate(xlabels):
            ax[0][c].set_xlabel(xl)
            ax[0][c].xaxis.set_label_position('top')

    if ylabels is not None:
        assert(len(ylabels)==nrows)
        for r, yl in enumerate(ylabels):
            ax[r][0].set_ylabel(yl)

    return fig, ax

def init_training_input_ratio_plot(niterations, feature_names):
    nfeatures = len(feature_names)

    fig, ax = init_subplots_grid(
        nrows = niterations,
        ncols = nfeatures,
        figsize_per_grid = (5.5,3.5),
        xlabels = feature_names,
        ylabels = [f"Iteration {i+1}" for i in range(niterations)],
        sharex = 'col',
        constrained_layout = True
        )

    fig.text(0.02/nfeatures, 0.5, "Ratio", va='center', rotation='vertical')

    return fig, ax

def draw_training_inputs_ratio(axes, features_array, label_array, weights):
    # axes: list of axes from plt.subplots(); len(axes) = nfeatures
    # features_array: ndarray of shape (nevents, nfeatures)
    # label_array: ndarray of shape (nevents,)
    # weights: ndarray of shape (nevents,)

    # number of bins
    nbins=20

    for ax, feature_arr in zip(axes, features_array.T):
        # bin edges
        xmin = feature_arr.min()
        xmax = feature_arr.max()
        bin_edges = np.linspace(xmin, xmax, nbins+1)

        hist_label1 = myhu.calc_hist(feature_arr[label_array==1], bins=bin_edges, weights=weights[label_array==1], norm=1.)
        hist_label0 = myhu.calc_hist(feature_arr[label_array==0], bins=bin_edges, weights=weights[label_array==0], norm=1.)

        draw_ratio(ax, hist_label1, hist_label0, *get_default_colors(2), label_denom='label 1', labels_numer='label 0')

def draw_inputs_ratio(axes, X_0, w_0, X_1, w_1, X_b=None, w_b=None):
    # axes: list of axes from plt.subplots(); len(axes) = nfeatures
    # X_0: feature arrays of label 0; ndarray of shape (nevents_0, nfeatures)
    # w_0: weight array of label 0; ndarray of shape (nevents_0,)
    # X_1: feature arrays of label 1; ndarray of shape (nevents_1, nfeatures)
    # w_1: weight array of label 1; ndarray of shape (nevents_1,)
    # X_b: feature arrays of background to be merged with X_1, optional
    # w_b: weight array of background to be subtracted from w_1, optional

    # only plot the first parallel run
    if w_0.ndim > 1:
        w_0 = w_0[0]

    if w_1.ndim > 1:
        w_1 = w_1[0]

    if X_b is not None and w_b is not None:
        if w_b.ndim > 1:
            w_b = w_b[0]

        X_1 = np.concatenate([X_1, X_b])
        w_1 = np.concatenate([w_1, -1*w_b])

    # number of bins
    nbins=20

    for ax, arr0, arr1 in zip(axes, X_0.T, X_1.T):
        # bin edges
        xmin = min(arr0.min(),arr1.min())
        xmax = max(arr0.max(),arr1.max())
        bin_edges = np.linspace(xmin, xmax, nbins+1)

        hist_label1 = myhu.calc_hist(arr1, bins=bin_edges, weights=w_1, norm=1.)
        hist_label0 = myhu.calc_hist(arr0, bins=bin_edges, weights=w_0, norm=1.)

        draw_ratio(ax, hist_label1, hist_label0, *get_default_colors(2), label_denom='label 1', labels_numer='label 0')

def plot_training_inputs_step1(
        figname_prefix,
        variable_names,
        Xobs, Xsim, Xbkg,
        wobs, wsim, wbkg
        ):
    # features
    logger.info("Plot distributions of input variables for step 1")

    Xdata = Xobs if Xbkg is None else np.concatenate([Xobs, Xbkg])
    wdata = wobs if Xbkg is None else np.concatenate([wobs, -1*wbkg])

    for vname, vdata, vsim in zip(variable_names, Xdata.T, Xsim.T):
        logger.debug(f"  Plot variable {vname}")
        plot_hist(
            figname_prefix+f"_{vname}",
            [ vdata, vsim ],
            weight_arrs = [wdata, wsim],
            labels = ['Data', 'Sim.'],
            xlabel = vname,
            title = "Step-1 training inputs")

    # weights
    logger.info("Plot distributions of the prior weights for step 1")
    plot_hist(
        figname_prefix+f"_weights",
        [wdata, wsim],
        labels = ['Data', 'Sim.'],
        xlabel = 'w (training)',
        title = "Step-1 prior weights at reco level")

def plot_training_inputs_step2(figname_prefix, variable_names, Xgen, wgen):
    # features
    logger.info("Plot distributions of input variables for step 2")
    for vname, vgen in zip(variable_names, Xgen.T):
        logger.debug(f"  Plot variable {vname}")
        plot_hist(
            figname_prefix+f"_{vname}",
            [vgen], weight_arrs = [wgen],
            labels = ['Gen.'],
            xlabel = vname,
            title = "Step-2 training inputs")

    # weights
    logger.info("Plot distributions of prior weights for step 2")
    plot_hist(
        figname_prefix+f"_weights",
        [wgen],
        labels = ['Gen.'],
        xlabel = 'w (training)',
        title = "Step-2 prior weights at truth level")

def plot_response(figname, histogram2d, variable, title='Detector Response', percentage=True, **pcolormesh_args):
    """
    Plot the unfolded detector response matrix.

    Parameters
    ----------
    figname : str
        Path to save the generated figure, excluding the file extension.
    histogram2d :Hist object. 2D histogram.
        Detector response matrix. Entries are percentages in the range [0, 1].
    variable : str
        Name of the plotted variable.
    """
    h2d = histogram2d.values()
    xedges = histogram2d.axes[0].edges
    yedges = histogram2d.axes[1].edges

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(f'Detector-level {variable}')
    ax.set_ylabel(f'Truth-level {variable}')

    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, h2d.T*100 if percentage else h2d.T, **pcolormesh_args)
    fig.colorbar(im, ax=ax, label = "%" if percentage else '')

    # label bin content
    xcenter =(xedges[:-1]+xedges[1:])/2
    ycenter = (yedges[:-1]+yedges[1:])/2
    for i, xc in enumerate(xcenter):
        for j, yc in enumerate(ycenter):
            bin_content = round(h2d[i, j],2)
            if bin_content != 0:
                if percentage:
                    bin_text = f"{int(bin_content*100)}"
                else:
                    bin_text = f"{bin_content:.2f}"
                ax.text(xc, yc, bin_text, ha='center', va='center', fontsize=3)

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def draw_uncertainties_hist(
    ax,
    hists_uncertainty, # list of tuple of hist.Hist
    draw_options=None,
    bin_labels=None,
    xlabel=None,
    ylabel=None,
    stamps=[],
    draw_legend=True,
    draw_error_on_error=False
    ):

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if not draw_options:
        draw_options = [{}]*len(hists_uncertainty)

    if not hasattr(draw_error_on_error, '__len__'):
        draw_error_on_error = [draw_error_on_error] * len(hists_uncertainty)
    else:
        assert len(draw_error_on_error) == len(hists_uncertainty)

    for hist_err, opt, draw_err in zip(hists_uncertainty, draw_options, draw_error_on_error):
        if not isinstance(hist_err, tuple):
            # if symmetric
            h_err_t = (hist_err, -1*hist_err)
        else:
            h_err_t = hist_err

        err1 = h_err_t[0].values()
        err2 = h_err_t[1].values()
        if draw_err:
            err1_err = np.sqrt(h_err_t[0].variances())
            err2_err = np.sqrt(h_err_t[1].variances())

        # append the error array with its last entry
        err1 = np.append(err1, err1[-1])
        err2 = np.append(err2, err2[-1])
        if draw_err:
            err1_err = np.append(err1_err, err1_err[-1])
            err2_err = np.append(err2_err, err2_err[-1])

        if bin_labels is None:
            xpoints = h_err_t[0].axes[0].edges
        else:
            # equal bin size
            xpoints = np.array(range(len(bin_labels)+1))

        if draw_err:
            ax.fill_between(xpoints, err1-err1_err, err1+err1_err, step='post', facecolor='none', edgecolor='gray', hatch='///', alpha=0.5)
            ax.fill_between(xpoints, err2-err2_err, err2+err2_err, step='post', facecolor='none', edgecolor='gray', hatch='///', alpha=0.5)

        ax.fill_between(xpoints, err1, err2, step='post', **opt)

    # horizontal line at zero
    ax.axhline(y=0., color='black', linestyle='--', alpha=0.3)

    if draw_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if stamps:
        draw_text(ax, stamps, prop={'size':7}, loc='upper left')

    if bin_labels:
        # Hide original major tick labels
        ax.set_xticklabels('')
        # Add the bin labels as minor tick labels at the center of each bin
        ax.set_xticks(
            xpoints[:-1] + (xpoints[1:]-xpoints[:-1])/2,
            labels = bin_labels,
            minor = True,
            rotation = 90,
            fontsize=7
            )

    return ax

def draw_uncertainties(
    ax,
    bins,
    uncertainties,
    draw_options=None,
    xlabel=None,
    ylabel=None,
    stamps=[]
    ):

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if not draw_options:
        draw_options = [{}]*len(uncertainties)

    for err, opt in zip(uncertainties, draw_options):

        # if symmetric
        if not isinstance(err, tuple):
            err_t = (np.asarray(err), -1*np.asarray(err))
        else:
            err_t = err

        assert len(err_t)>=2
        if len(err_t[0]) == len(bins) - 1:
            # append the error array with its last entry
            err_t = tuple(np.append(e, e[-1]) for e in err_t)

        # draw
        # error band if needed
        if len(err) > 2:
            err_1 = err_t[2]
            if len(err) > 3:
                err_2 = err_t[3]
            else:
                err_2 = err_1

            ax.fill_between(bins, err_t[0]-err_1, err_t[0]+err_1, step='post', facecolor='none', edgecolor='gray', hatch='///', alpha=0.5)
            ax.fill_between(bins, err_t[1]-err_2, err_t[1]+err_2, step='post', facecolor='none', edgecolor='gray', hatch='///', alpha=0.5)

        ax.fill_between(bins, err_t[0], err_t[1], step='post', **opt)

    # horizontal line at zero
    ax.axhline(y=0., color='black', linestyle='--', alpha=0.3)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if stamps:
        draw_text(ax, stamps, loc='lower right')

    return ax

def plot_uncertainties(
    figname,
    bins, # bin edges
    uncertainties, # list of (error_up, error_donw) for uncertainties
    draw_options=None,
    xlabel=None,
    ylabel=None,
    stamps=[]
    ):

    fig, ax = plt.subplots()

    draw_uncertainties(
        ax, bins, uncertainties,
        draw_options=draw_options,
        xlabel=xlabel,
        ylabel=ylabel,
        stamps=stamps
        )

    if not os.path.isdir(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    fig.savefig(figname+'.png', dpi=300, bbox_inches="tight")
    plt.close(fig)