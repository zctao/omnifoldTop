import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import mplhep as hep

import util
from histogramming import get_values_and_errors, get_mean_from_hists, get_sigma_from_hists, get_hist, calc_hist

import logging
logger = logging.getLogger('plotter')

# styles
#hep.style.use(hep.style.ATLAS)
data_style = {'color': 'black', 'label': 'Data', 'histtype': 'errorbar', 'marker': 'o', 'markersize': 3}
sim_style = {'color': 'orange', 'label': 'Sim.', 'histtype': 'step'}
bkg_style = {'color': 'cyan', 'label': 'Bkg.', 'histtype': 'fill'}
gen_style = {'color': 'blue', 'label': 'Gen.', 'histtype': 'step', 'lw':0.8}
truth_style = {'edgecolor': 'green', 'facecolor': (0.75, 0.875, 0.75), 'label': 'Truth', 'histtype': 'fill'}
omnifold_style = {'color': 'tab:red', 'label':'MultiFold', 'histtype': 'errorbar', 'marker': 's', 'markersize': 2}
ibu_style = {'color': 'gray', 'label':'IBU', 'histtype': 'errorbar', 'marker': 'o', 'markersize': 2}
error_style = {'lw': 1, 'capsize': 1.5, 'capthick': 1, 'markersize': 1.5}

def draw_ratio(ax, hist_denom, hists_numer, color_denom, colors_numer):
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
    values_denom, errors_denom = get_values_and_errors(hist_denom)
    if errors_denom is not None:
        relerrs_denom = np.divide(errors_denom, values_denom, out=np.zeros_like(values_denom), where=(values_denom!=0))
        relerrs_denom = np.append(relerrs_denom, relerrs_denom[-1])

        ax.fill_between(bin_edges, 1-relerrs_denom, 1+relerrs_denom, step='post', facecolor=color_denom, alpha=0.3)

    for hnum, cnum in zip(hists_numer, colors_numer):
        if hnum is None:
            continue

        values_num, errors_num = get_values_and_errors(hnum)
        ratio = np.divide(values_num, values_denom, out=np.zeros_like(values_denom), where=(values_denom!=0))

        if errors_num is not None:
            ratio_errs = np.divide(errors_num, values_denom, out=np.zeros_like(values_denom), where=(values_denom!=0))
        else:
            ratio_errs = None

        hep.histplot(ratio, bin_edges, yerr=ratio_errs, histtype='errorbar', xerr=True, color=cnum, ax=ax, **error_style)

def draw_stamp(ax, texts, x=0.5, y=0.5, dy=0.045):
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

    for i, txt in enumerate(texts):
        if txt is not None:
            ax.text(x, y-i*dy, txt, **textopts)

def set_default_colors(ncolors):
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
    return plt.rcParams['axes.prop_cycle'].by_key()['color'][:ncolors]

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
        colors = set_default_colors(len(data_arrays))
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

    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def plot_hist(
    figname,
    data_arrs,
    weight_arrs=None,
    labels=None,
    nbins=20,
    xlabel='',
    ylabel='',
    title=''
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
    """
    xmax = max([np.max(data) for data in data_arrs])
    xmin = min([np.min(data) for data in data_arrs])
    margin = (xmax - xmin) * 0.1
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
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    for data, w, l in zip(data_arrs, weight_arrs, labels):
        ax.hist(data, bins=bins, histtype='step', label=l)

    ax.legend()

    fig.savefig(figname+'.png', dpi=300)
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
    legend_ncol=1
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

    fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (3.5,1), 'hspace': 0.0})

    # label
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel(ylabel_ratio)
    axes[1].set_xlabel(xlabel)

    # x limits
    bin_edges = hist_data.axes[0].edges
    for ax in axes:
        ax.set_xlim(bin_edges[0], bin_edges[-1])

    # simulation
    if hist_bkg is None:
        hist_sim = hist_sig
        label_sim = "Sim."
        style_sim = sim_style
    else:
        hist_sim = [hist_bkg, hist_sig]
        label_sim = ["Bkg.", "Sim."]
        style_sim = { k: [bkg_style[k], sim_style[k]] for k in sim_style}

    hep.histplot(hist_sim, yerr=False, stack=True, ax=axes[0], **style_sim)

    # data
    hep.histplot(hist_data, yerr=True, xerr=True, ax=axes[0], **data_style)

    # legend
    axes[0].legend(loc=legend_loc, ncol=legend_ncol, frameon=False)

    # ratio
    draw_ratio(
        axes[1],
        hist_data,
        [hist_sig] if hist_bkg is None else [hist_sig + hist_bkg],
        data_style['color'],
        [sim_style['color']]
    )

    # save plot
    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

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
    """
    fig, axes = plt.subplots(
        2 if hist_truth is not None else 1,
        sharex = True,
        gridspec_kw={
            'height_ratios': (3.5,1) if hist_truth is not None else (1,),
            'hspace': 0.0
        }
    )

    if hist_truth is not None:
        ax = axes[0]
        ax_rp = axes[1]
    else:
        ax = axes
        ax_rp = None

    # labels and x limits
    bin_edges = hist_unfold.axes[0].edges

    ax.set_ylabel(ylabel)
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    if ax_rp is not None:
        ax_rp.set_xlim(bin_edges[0], bin_edges[-1])
        ax_rp.set_ylabel(ylabel_ratio)
        ax_rp.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)

     # MC truth if known
    if hist_truth is not None:
        hep.histplot(hist_truth, ax=ax, **truth_style)

    # Prior
    if hist_prior is not None:
        hep.histplot(hist_prior, ax=ax, **gen_style)

    # IBU if available
    if hist_ibu is not None:
        hep.histplot(hist_ibu, ax=ax, yerr=True, xerr=True, **ibu_style)

    # Unfold
    hep.histplot(hist_unfold, ax=ax, yerr=True, xerr=True, **omnifold_style)

    # legend
    ax.legend(loc=legend_loc, ncol=legend_ncol, frameon=False)

    # ratio
    if ax_rp is not None:
        draw_ratio(ax_rp, hist_truth, [hist_prior, hist_ibu, hist_unfold],
            truth_style['edgecolor'],
            [gen_style['color'], ibu_style['color'], omnifold_style['color']]
        )

    # metric stamp
    if stamp_texts:
        draw_stamp(ax, stamp_texts, *stamp_loc)

    # save plot
    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_correlations(figname, correlations, bins=None):
    fig, ax = plt.subplots()

    if bins is None:
        bins = np.linspace(0, len(correlations), len(correlations)+1)
        ax.tick_params(axis='both', labelsize='small')
        ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, labelrotation=30)
        ticks = np.arange(0.5, len(correlations)+0.5, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(correlations.columns)
        ax.set_yticklabels(correlations.columns)

    hep.hist2dplot(
        correlations, xbins=bins, ybins=bins, vmin=-1, vmax=1,
        cbar=True, cmin=-1, cmax=1, cmap='coolwarm',
        ax=ax)

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

    fig, axes = plt.subplots(
        2, sharex=True, gridspec_kw={'height_ratios': (3.5,1), 'hspace': 0.0}
    )

    axes[0].set_ylabel(ylabel)
    axes[1].set_xlabel(xlabel)

    # assign colors
    alpha=0.5
    colors_rs = []
    for i in range(len(hists_uf_resample)):
        colors_rs.append( tuple(np.random.random(3))+(alpha,) )

    # plot
    for i, (huf, color) in enumerate(zip(hists_uf_resample, colors_rs)):
        label = 'Resample' if i==0 else None
        hep.histplot(huf, histtype='step', yerr=True, label=label, ax=axes[0], ls='--', lw=1)

#    bin_edges = hist_prior.axes[0].edges
#
#    # mean of each bin
#    hmean = get_mean_from_hists(hists_uf_resample)
#    hsigma = get_sigma_from_hists(hists_uf_resample)
#    hist_mean = get_hist(bin_edges, hmean, hsigma)
#
#    hep.histplot(hist_mean, ax=axes[0], label="Mean", histtype='step', color=omnifold_stype['color'])

    # prior distribution
    hep.histplot(hist_prior, ax=axes[0], **gen_style)

    # truth distribution
    if hist_truth is not None:
        hep.histplot(hist_truth, ax=axes[0], histtype='step', label=truth_style['label'], color=truth_style['edgecolor'])

    # ratio
    if hist_truth is None:
        # ratio to the prior
        draw_ratio(axes[1], hist_prior, hists_uf_resample, gen_style['color'], colors_rs)
        axes[1].set_ylabel("Ratio to \nPrior")
    else:
        # ratio to MC truth
        draw_ratio(axes[1], hist_truth, hists_uf_resample, truth_style['edgecolor'], colors_rs)
        axes[1].set_ylabel("Ratio to \nTruth")

    # legend
    axes[0].legend(frameon=False, handlelength=2, numpoints=2)

    # save plot
    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_distributions_iteration(
    figname,
    hists_unfold,
    hist_prior,
    hist_truth=None,
    nhistmax=7,
    xlabel='',
    ylabel=''
    ):

    fig, axes = plt.subplots(
        2, sharex=True, gridspec_kw={'height_ratios': (3.5,1), 'hspace': 0.0}
    )

    axes[0].set_ylabel(ylabel)
    axes[1].set_xlabel(xlabel)

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

    colors = set_default_colors(len(selected_i))

    for i, h, c in zip(selected_i, hists_toplot, colors):
        hep.histplot(
            h, ax=axes[0], histtype='errorbar', yerr=True, xerr=True, color=c,
            label=f"iteration {i}", alpha=0.8, marker='o', markersize=2)

    # ratio
    if hist_truth is None:
        # ratio to prior
        axes[1].set_ylabel("Ratio to\n Prior")
        draw_ratio(axes[1], hists_toplot[0], hists_toplot[1:], colors[0], colors[1:])
    else:
        # ratio to truth
        axes[1].set_ylabel("Ratio to\n Truth")
        draw_ratio(axes[1], hist_truth, hists_toplot, truth_style['edgecolor'], colors)

    # legend
    axes[0].legend(frameon=False)

    # save plots
    fig.savefig(figname+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_hists_bin_distr(figname, histograms_list, histogram_ref):
    # histograms_list can be either dimension 3 if all iterations are included
    # or dimension 2 if only the final unfolded ones for all trials
    histograms_arr = np.asarray(histograms_list)['value']

    nbins = histogram_ref.axes[0].size
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
        ref = histogram_ref.values()[ibin]
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

def plot_LR_distr(figname, ratios, labels=None):
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
    bins_r = np.linspace(min(r.min() for r in ratios)*0.9, max(r.max() for r in ratios)*1.1, 50)

    fig, ax = plt.subplots()
    ax.set_xlabel("Likelihood ratio")
    #ax.set_yscale('log')

    for i,r in enumerate(ratios):
        hr = calc_hist(r, bins_r, density=False)
        l = labels[i] if labels is not None else None
        hep.histplot(hr, ax=ax, label=l)

    if labels is not None:
        ax.legend()

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

    hist_preds_cat1_t = calc_hist(preds_cat1_t, bins_preds, weights=w_cat1_t, density=True)
    hist_preds_cat0_t = calc_hist(preds_cat0_t, bins_preds, weights=w_cat0_t, density=True)

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

    hist_preds_cat1_v = calc_hist(preds_cat1_v, bins_preds, weights=w_cat1_v, density=True)
    hist_preds_cat0_v = calc_hist(preds_cat0_v, bins_preds, weights=w_cat0_v, density=True)

    fig, ax = plt.subplots()
    ax.set_xlabel("Prediction (y = 1)")

    hep.histplot(hist_preds_cat1_t, ax=ax, label='y = 1 (training)', histtype='step')
    hep.histplot(hist_preds_cat0_t, ax=ax, label='y = 0 (training)', histtype='step')
    hep.histplot(hist_preds_cat1_v, ax=ax, label='y = 1 (validation)', histtype='errorbar')
    hep.histplot(hist_preds_cat0_v, ax=ax, label='y = 0 (validation)', histtype='errorbar')

    ax.legend()

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

def plot_training_inputs_step1(
        figname_prefix,
        variable_names,
        Xdata, Xsim,
        wdata, wsim
        ):
    # features
    logger.info("Plot distributions of input variables for step 1")
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
