import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from histogramming import get_values_and_errors, get_mean_from_hists, get_sigma_from_hists, get_hist

#import logging
#logger = logging.getLogger('plotter')

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

def plot_correlations(figname, correlations, bins):
    fig, ax = plt.subplots()

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
    axes[0].legend(frameon=False)

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

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(selected_i)]

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
