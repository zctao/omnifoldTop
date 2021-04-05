import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import math
import external.OmniFold.modplot as modplot

from util import add_histograms, compute_chi2, compute_diff_chi2
from util import gaus, fit_gaussian_to_hist

# plotting styles
hist_style = {'histtype': 'step', 'density': False, 'lw': 1, 'zorder': 2}
graph_style = {'fmt': 'o', 'lw': 1.5, 'capsize': 1.5, 'capthick': 1, 'markersize': 1.5}
leg_style = {'handlelength': 2.0, 'loc': 'best', 'frameon': False, 'numpoints': 1, 'fontsize': 'small'}

data_style = {'color': 'black', 'label':  'Data', **hist_style}
sim_style = {'color': 'orange', 'label':  'Sim.', **hist_style}
bkg_style = {'color': 'cyan', 'label': 'Bkg.', **hist_style}

gen_style = {'linestyle': '--', 'color': 'blue', 'lw': 1.15, 'label': 'Gen.'}

truth_style = {'edgecolor': 'green', 'facecolor': (0.75, 0.875, 0.75), 'lw': 1.25, 'zorder': 0, 'label': 'Truth'}

ibu_style = {'ls': '-', 'marker': 'o', 'ms': 2.5, 'color': 'gray', 'zorder': 1, 'label':'IBU'}

omnifold_style = {'ls': '-', 'marker': 's', 'ms': 2.5, 'color': 'tab:red', 'zorder': 3, 'label':'MultiFold'}

def init_fig(title='', xlabel='', ylabel=''):
    fig, ax = plt.subplots()

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return fig, ax

def set_default_colors(ncolors):
    return plt.rcParams['axes.prop_cycle'].by_key()['color'][:ncolors]

def draw_ratios(ax, bins, hist_denom, hists_numer, hist_denom_unc=None, hists_numer_unc=None, color_denom_line='tomato', color_denom_fill='silver', colors_numer=None):
    midbins = (bins[:-1] + bins[1:]) / 2
    binwidths = bins[1:] - bins[:-1]

    # horizontal line at y=1
    ax.plot([np.min(bins), np.max(bins)], [1, 1], '-', color=color_denom_line, lw=0.75)

    if hist_denom_unc is not None:
        denom_unc_ratio = np.divide(hist_denom_unc, hist_denom, out=np.zeros_like(hist_denom), where=(hist_denom!=0))
        denom_unc_ratio = np.append(denom_unc_ratio, denom_unc_ratio[-1])
        ax.fill_between(bins, 1-denom_unc_ratio, 1+denom_unc_ratio, facecolor=color_denom_fill, zorder=-2, step='post')

    if colors_numer is not None:
        assert(len(colors_numer)==len(hists_numer))
    else:
        colors_numer = set_default_colors(len(hists_numer))

    for i, hist_num in enumerate(hists_numer):
        if hist_num is None:
            continue
        ymin, ymax = ax.get_ylim()
        ratio = np.divide(hist_num, hist_denom, out=np.ones_like(hist_denom)*ymin, where=(hist_denom!=0))

        ratio_unc = None
        if hists_numer_unc is not None:
            assert(len(hists_numer_unc)==len(hists_numer))
            if hists_numer_unc[i] is not None:
                ratio_unc = np.divide(hists_numer_unc[i], hist_denom, out=np.zeros_like(hist_denom), where=(hist_denom!=0))

        ax.errorbar(midbins, ratio, xerr=binwidths/2, yerr=ratio_unc, color=colors_numer[i], **modplot.style('errorbar'))

def draw_legend(ax, **config):
    loc = config.get('legend_loc', 'best')
    ncol = config.get('legend_ncol', 2)
    #order = [3, 4, 2, 5, 0, 1] if ncol==2 else [3, 5, 4, 0, 2, 1]
    modplot.legend(ax=ax, loc=loc, ncol=ncol, frameon=False, fontsize='x-small')

def draw_stamp(ax, texts, x=0.5, y=0.5, dy=0.045):
    textopts = {'horizontalalignment': 'left',
                'verticalalignment': 'center',
                'fontsize': 5.,
                'transform': ax.transAxes}

    for i, txt in enumerate(texts):
        if txt is not None:
            ax.text(x, y-i*dy, txt, **textopts)

def draw_histogram(ax, bin_edges, hist, hist_unc=None, **styles):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.hist(midbins, bin_edges, weights=hist, **styles)
    # TODO: uncertainty hist_unc

def draw_stacked_histograms(ax, bin_edges, hists, hists_unc=None, labels=None,
                            colors=None, stacked=True):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2

    if colors is None:
        colors = set_default_colors(len(hists))
    assert(len(colors)==len(hists))

    if labels is None:
        labels = [str(i) for i in range(len(hists))]
    assert(len(labels)==len(hists))

    ax.hist(np.stack([midbins]*len(hists), axis=1), bin_edges,
            weights=np.stack([h for h in hists], axis=1),
            color=colors, label=labels,
            stacked = stacked, histtype='bar', fill=True)
    # TODO: uncertainty

def draw_hist_fill(ax, bin_edges, hist, hist_unc=None, **styles):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.hist(midbins, bin_edges, weights=hist, histtype='step', fill=True, **styles)
    # TODO: uncertainty?

def draw_hist_as_graph(ax, bin_edges, hist, hist_unc=None, **styles):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2
    binwidths = bin_edges[1:] - bin_edges[:-1]

    yerr = hist_unc
    xerr = None if yerr is None else binwidths/2

    ax.errorbar(midbins, hist, xerr=xerr, yerr=yerr, **styles)

def plot_graphs(figname, data_arrays, error_arrays=None, labels=None, title='', xlabel='', ylabel='', xscale=None, yscale=None, colors=None, markers=None, **style):
    fig, ax = init_fig(title, xlabel, ylabel)

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

        xerr, yerr = None, None
        if error_arrays is not None:
            error = error_arrays[i]
            if isinstance(error, tuple):
                xerr, yerr = error
            else:
                yerr = error

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, label=label, marker=marker, color=colors[i], **style)

    # plot legend if needed
    if labels is not None:
        ax.legend(**leg_style)

    fig.savefig(figname+'.png', dpi=300)
    #fig.savefig(figname+'.pdf')
    plt.close(fig)

def plot_histograms1d(figname, bins, hists, hists_err=None, labels=None, title="", xlabel="", ylabel="", colors=None, plottypes=None, marker='o'):
    fig, ax = init_fig(title, xlabel, ylabel)

    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((-3,4))

    if colors is None:
        colors = set_default_colors(len(hists))
    else:
        assert(len(colors)==len(hists))

    if labels is None:
        labels = [""]*len(hists)
    else:
        assert(len(labels)==len(hists))

    if hists_err is not None:
        assert(len(hists_err)==len(hists))
    else:
        hists_err = [np.zeros_like(h) for h in hists]

    if plottypes is None:
        plottypes = ['h']*len(hists)
    else:
        assert(len(plottypes)==len(hists))

    for i, h in enumerate(hists):
        label = None if labels is None else labels[i]
        if plottypes[i] == "g":
            draw_hist_as_graph(ax, bins, h, hists_err[i], label=label, color=colors[i], marker=marker, **graph_style)
        elif plottypes[i] == "h":
            draw_histogram(ax, bins, h, hists_err[i], label=label, color=colors[i], fill=False, **hist_style)
        else:
            raise RuntimeError("Unknown plot type {}".format(plottypes[i]))

    # plot legend if needed
    if labels is not None:
        ax.legend(**leg_style)

    fig.savefig(figname+'.png', dpi=300)
    #fig.savefig(figname+'.pdf')
    plt.close(fig)

def plot_data_arrays(figname, data_arrs, weight_arrs=None, nbins=20, **plotstyle):
    xmax = max([np.max(data) for data in data_arrs]) * 1.2
    xmin = min([np.min(data) for data in data_arrs]) * 0.8
    bins = np.linspace(xmin, xmax, nbins+1)

    if weight_arrs is None:
        weight_arrs = [None] * len(data_arrs)
    else:
        assert(len(weight_arrs) == len(data_arrs))

    histograms = []
    for data, w in zip(data_arrs, weight_arrs):
        h = np.histogram(data, bins=bins, weights=w, density=True)[0]
        histograms.append(h)

    plot_histograms1d(figname, bins, histograms, **plotstyle)

def plot_reco_variable(bins, histogram_obs, histogram_sig,
                        histogram_bkg=(None,None),
                        figname='var_reco', log_scale = False, **config):
    """
    Plot detector-level variable distributions
    """
    hist_obs, hist_obs_unc = histogram_obs
    hist_sig, hist_sig_unc = histogram_sig
    hist_bkg, hist_bkg_unc = histogram_bkg

    # use the plotting tools from the original omnifold package
    fig, axes = modplot.axes(ratio_plot=True, ylabel_ratio='Data \/\nMC', **config)
    ax0 = axes[0]
    ax1 = axes[1]

    # x limits
    ax0.set_xlim(bins[0],bins[-1])
    ax1.set_xlim(bins[0],bins[-1])

    # yscale
    if log_scale:
        ax0.set_yscale('log')
    elif config.get('yscale') is not None:
        ax0.set_yscale(config['yscale'])

    # y limits
    ymax = max(hist_obs.max(), hist_sig.max())
    if hist_bkg is not None:
        ymax = max(hist_bkg.max(), ymax)

    ymin = 1e-4 if log_scale else 0
    ymax = ymax*10 if log_scale else ymax*1.2

    ax0.set_ylim(ymin, ymax*1.2)

    hists_stack = [hist_sig]
    labels = [sim_style['label']]
    colors = [sim_style['color']]
    if hist_bkg is not None:
        hists_stack = [hist_bkg, hist_sig]
        labels = [bkg_style['label'], sim_style['label']]
        colors = [bkg_style['color'], sim_style['color']]

    draw_stacked_histograms(ax0, bins, hists_stack, labels=labels, colors=colors)
    draw_histogram(ax0, bins, hist_obs, **data_style)

    # data/mc ratio
    hist_mc, hist_mc_unc = add_histograms(hist_sig, hist_bkg, hist_sig_unc, hist_bkg_unc)
    draw_ratios(ax1, bins, hist_mc, [hist_obs], hist_mc_unc, [hist_obs_unc], colors_numer=[data_style['color']])

    # legend
    draw_legend(ax0, **config)

    # save plot
    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    #fig.savefig(figname+'.pdf', bbox_inches='tight')

    plt.close(fig)

def plot_results(bins_gen, histogram_gen, histogram_of, histogram_ibu=(None,None), histogram_truth=(None,None), figname='unfolded', texts=[], **config):
    """
    Plot and compare the unfolded distributions
    """
    ymax = 0.

    # use the plotting tools from the original omnifold package
    truth_known = histogram_truth[0] is not None
    fig, axes = modplot.axes(ratio_plot = truth_known, gridspec_update={'height_ratios': (3.5,2) if truth_known else (1,)}, **config)

    # set xaxis limit according to bin edges
    for ax in axes:
        ax.set_xlim(bins_gen[0], bins_gen[-1])

    ax0 = axes[0]
    ax1 = axes[1] if truth_known else None

    if config.get('yscale') is not None:
        ax0.set_yscale(config['yscale'])

    # generator-level
    # signal prior
    hist_gen, hist_gen_unc = histogram_gen
    ymax = max(hist_gen.max(), ymax)
    draw_hist_as_graph(ax0, bins_gen, hist_gen, **gen_style)

    # if truth is known
    hist_truth, hist_truth_unc = histogram_truth
    if hist_truth is not None:
        ymax = max(hist_truth.max(), ymax)
        draw_hist_fill(ax0, bins_gen, hist_truth, **truth_style)

    # unfolded distributions
    # omnifold
    hist_of, hist_of_unc = histogram_of
    ymax = max(hist_of.max(), ymax)
    draw_hist_as_graph(ax0, bins_gen, hist_of, **omnifold_style)

    # iterative Bayesian unfolding
    hist_ibu, hist_ibu_unc = histogram_ibu
    if hist_ibu is not None:
        ymax = max(hist_ibu.max(), ymax)
        draw_hist_as_graph(ax0, bins_gen, hist_ibu, **ibu_style)

    # update y-axis limit
    ax0.set_ylim((0, ymax*1.2))

    if ax1:
        #  ratios of the unfolded distributions to truth
        hists_numerator = [hist_ibu, hist_of]
        hists_unc_numerator = [hist_ibu_unc, hist_of_unc]
        colors_numerator = [ibu_style['color'], omnifold_style['color']]
        if config.get('draw_prior_ratio') is not None:
            if config['draw_prior_ratio']:
                hists_numerator = [hist_gen] + hists_numerator
                hists_unc_numerator = [hist_gen_unc] + hists_unc_numerator
                colors_numerator = [gen_style['color']] + colors_numerator

        draw_ratios(ax1, bins_gen, hist_truth, hists_numerator,
                    hist_truth_unc, hists_unc_numerator,
                    color_denom_line = truth_style['edgecolor'],
                    color_denom_fill= truth_style['facecolor'],
                    colors_numer = colors_numerator)

    draw_legend(ax0, **config)

    draw_stamp(ax0, texts, config['stamp_xy'][0], config['stamp_xy'][1])

    # save plot
    fig.savefig(figname+'.png', dpi=300, bbox_inches='tight')
    #fig.savefig(figname+'.pdf', bbox_inches='tight')

    plt.close(fig)

def plot_response(figname, h2d, xedges, yedges, variable):
    fig, ax = init_fig(
        title='Detector Response',
        xlabel='Detector-level {}'.format(variable),
        ylabel='Truth-level {}'.format(variable)
    )
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, h2d.T*100, cmap='Greens')
    fig.colorbar(im, ax=ax, label="%")

    # label bin content
    xcenter =(xedges[:-1]+xedges[1:])/2
    ycenter = (yedges[:-1]+yedges[1:])/2
    for i, xc in enumerate(xcenter):
        for j, yc in enumerate(ycenter):
            bin_content = round(h2d[i, j]*100)
            if bin_content != 0:
                ax.text(xc, yc, str(int(bin_content)), ha='center', va='center', fontsize=3)

    fig.savefig(figname+'.png', dpi=300)
    #fig.savefig(figname+'.pdf')
    plt.close(fig)

def plot_iteration_distributions(figname, binedges, histograms, histograms_err, histogram_truth=None, histogram_truth_err=None, nhistmax=7, **config):
    # plot intermediate unfolded distributions of all iterations
    fig, axes = modplot.axes(ratio_plot=True, gridspec_update={'height_ratios': (3.5,2)}, **config)
    for ax in axes:
        ax.set_xlim(binedges[0], binedges[-1])

    ax0 = axes[0]

    if config.get('yscale') is not None:
        ax0.set_yscale(config['yscale'])

    # if there are more than nhistmax histograms, plot at most nhistmax histograms
    assert(nhistmax<=10) # plt.rcParams['axes.prop_cycle'] provides max 10 colors
    if len(histograms) > nhistmax:
        selected_i = np.linspace(1, len(histograms)-2, nhistmax-2).astype(int).tolist()
        # the first [0] and the last [-1] are always plotted
        selected_i = [0] + selected_i + [len(histograms)-1]
    else:
        selected_i = list(range(len(histograms)))

    histograms_toplot = [histograms[i] for i in selected_i]
    histograms_err_toplot = [histograms_err[i] for i in selected_i]

    styles = ibu_style.copy()
    colors = set_default_colors(len(selected_i))

    ymax = 0.
    for i, hist, color in zip(selected_i, histograms_toplot, colors):
        styles.update({'color': color, 'label': 'iteration {}'.format(i)})
        ymax = max(hist.max(), ymax)
        draw_hist_as_graph(ax0, binedges, hist, alpha=0.8, **styles)

    # set yaxis range
    ax0.set_ylim((0, ymax*1.2))

    # ratio
    if histogram_truth is not None:
        axes[1].set_ylabel("Ratio to Truth", fontsize=8)
        # Draw ratio to truth
        draw_ratios(axes[1], binedges,
                    histogram_truth, histograms_toplot,
                    histogram_truth_err, histograms_err_toplot,
                    color_denom_line = 'black', colors_numer = colors)
    else:
        # Draw ratio to prior
        axes[1].set_ylabel("Ratio to Prior", fontsize=8)
        draw_ratios(axes[1], binedges,
                    histograms_toplot[0], histograms_toplot[1:],
                    histograms_err_toplot[0], histograms_err_toplot[1:],
                    color_denom_line = colors[0], colors_numer = colors[1:])

    draw_legend(ax0, **config)

    # save plot
    fig.savefig(figname+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_iteration_chi2s(figname, histogram_ref, histogram_err_ref,
                         histograms_arr, histograms_err_arr, labels=None,
                         **style):
    # chi2 between the truth distribution and each unfolding iteration
    fig, ax = init_fig(title='', xlabel='Iteration', ylabel='$\\chi^2$/NDF w.r.t. truth')

    if labels is None:
        labels = [None]*len(histograms_arr)

    for hists, hists_err, label in zip(histograms_arr, histograms_err_arr, labels):
        if hists is None:
            continue

        Chi2s = []
        for h, herr in zip(hists, hists_err):
            chi2, ndf = compute_chi2(h, histogram_ref, herr, histogram_err_ref)
            Chi2s.append(chi2/ndf)

        iters = list(range(len(Chi2s)))

        if label is None:
            ax.plot(iters, Chi2s, marker='o', **style)
        else:
            ax.plot(iters, Chi2s, marker='o', label=label, **style)
            ax.legend()

    fig.savefig(figname+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_iteration_diffChi2s(figname, histograms_arr, histograms_err_arr, labels):
    # chi2s between iterations
    fig, ax = init_fig(title='', xlabel='Iteration', ylabel='$\\Delta\\chi^2$/NDF')
    for hists, hists_err, label in zip(histograms_arr, histograms_err_arr, labels):
        if hists is None:
            continue

        dChi2s = compute_diff_chi2(hists, hists_err)
        iters = list(range(1, len(dChi2s)+1))

        ax.plot(iters, dChi2s, marker='*', label=label)
        ax.legend()

    fig.savefig(figname+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_train_log(csv_file, plot_name=None):
    df = pd.read_csv(csv_file)

    fig, ax = init_fig(title='', xlabel='Epochs', ylabel='loss')

    ax.plot(df['epoch'], df['loss'], label='loss')
    ax.plot(df['epoch'], df['val_loss'], label='val loss')

    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_useMathText(True)
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-3,4))

    ax.legend(**leg_style)

    if plot_name is None:
        plot_name = csv_file.replace('.csv', '_loss')
    plt.savefig(plot_name+'.png', dpi=300, bbox_inches='tight')
    #plt.savefig(plot_name+'.pdf', bbox_inches='tight')
    plt.close(fig)

def plot_correlations(correlations, figname):
    fig, ax = plt.subplots()
    im = ax.imshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    fig.colorbar(im, ax=ax)
    ax.tick_params(axis='both', labelsize='small')
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, labelrotation=30)
    ticks = np.arange(0, len(correlations), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(correlations.columns)
    ax.set_yticklabels(correlations.columns)

    fig.savefig(figname+'.png', dpi=200)
    #fig.savefig(figname+'.pdf')

    plt.close(fig)

def plot_LR_func(figname, bins, f_lr, f_lr_unc=None):
    # Likelihood ratio as a function of model prediction
    x = (bins[:-1] + bins[1:]) / 2
    xerr = (bins[1:] - bins[:-1]) / 2
    #plot_graphs(figname, [(x, f_lr)], [(xerr, f_lr_unc)], xlabel='Prediction (y = 1)', ylabel='LR')

    fig, ax = init_fig(title='', xlabel='Prediction (y = 1)', ylabel='LR')

    ax.errorbar(x, f_lr, xerr=xerr, yerr=f_lr_unc, label='Binned LR', **graph_style)

    # likelihood ratio directly computed from predictions
    xtrim = x[(x>0.1)&(x<0.85)]
    ax.plot(xtrim, xtrim / (1 - xtrim +10**-50), label='f(x) = x/(1-x)')

    ax.legend(**leg_style)

    fig.savefig(figname+'.png', dpi=300)
    #fig.savefig(figname+'.pdf')
    plt.close(fig)

def plot_LR_distr(figname, ratios, labels=None):
    bins_r = np.linspace(0, max(r.max() for r in ratios), 51)

    hists, hists_unc = [], []
    for r in ratios:
        hist_r, hist_r_unc = modplot.calc_hist(r, bins=bins_r, density=True)[:2]
        hists.append(hist_r)
        hists_unc.append(hist_r_unc)

    plot_histograms1d(figname, bins_r, hists, hists_unc, labels, xlabel='r')

def plot_training_vs_validation(figname, predictions_train, labels_train, weights_train, predictions_val, labels_val, weights_val, nbins=100):
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

    hist_preds_cat1_t = np.histogram(preds_cat1_t, bins=bins_preds, weights=w_cat1_t, density=True)[0]
    hist_preds_cat0_t = np.histogram(preds_cat0_t, bins=bins_preds, weights=w_cat0_t, density=True)[0]

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

    hist_preds_cat1_v = np.histogram(preds_cat1_v, bins=bins_preds, weights=w_cat1_v, density=True)[0]
    hist_preds_cat0_v = np.histogram(preds_cat0_v, bins=bins_preds, weights=w_cat0_v, density=True)[0]

    plot_histograms1d(
        figname, bins_preds,
        [hist_preds_cat1_t, hist_preds_cat0_t, hist_preds_cat1_v, hist_preds_cat0_v],
        labels=['y = 1 (training)', 'y = 0 (training)', 'y = 1 (validation)', 'y = 0 (validation)'],
        xlabel = 'Prediction (y = 1)',  plottypes=['h','h','g','g'], marker='+')

def plot_hists_resamples(figname, bins, histograms, hist_prior, hist_truth=None,
                         **config):

    fig, axes = modplot.axes(ratio_plot=True, ylabel_ratio='Ratio to\nPrior', **config)
    # set x axis limit
    for ax in axes:
        ax.set_xlim(bins[0], bins[-1])
    ax0, ax1 = axes

    ymax=0
    alpha=0.5
    for i,hist in enumerate(histograms):
        ymax = max(hist.max(), ymax)
        color=tuple(np.random.random(3))+(alpha,)
        label='Resampled' if i==0 else None
        draw_hist_as_graph(ax0, bins, hist, ls='--', lw=1, color=color, label=label)

    # mean of each bin
    hist_mean = np.mean(np.asarray(histograms), axis=0)
    draw_hist_as_graph(ax0, bins, hist_mean, ls='-', lw=1, color='black', label='Mean')

    # the prior distribution
    draw_hist_as_graph(ax0, bins, hist_prior, ls='-', lw=1, color='blue', label='Prior')
    ymax = max(hist_prior.max(), ymax)

    # the truth distribution
    if hist_truth is not None:
        draw_hist_as_graph(ax0, bins, hist_prior, ls='-', lw=1, color='green', label='Truth')
        ymax = max(hist_truth.max(), ymax)

    ax0.set_ylim(0, ymax*1.2)

    # standard deviation of each bin
    hist_std = np.std(np.asarray(histograms), axis=0, ddof=1)

    # ratio
    if hist_truth is None:
        draw_ratios(ax1, bins, hist_prior, [hist_mean], hists_numer_unc=[hist_std], colors_numer=['black'], color_denom_line='blue')
    else:
        # draw ratio to truth
        ax1.set_ylabel("Ratio to\nTruth", fontsize=8)
        draw_ratios(ax1, bins, hist_truth, [hist_mean], hists_numer_unc=[hist_std], colors_numer=['black'], color_denom_line='green')

    draw_legend(ax0, **config)

    # save plot
    fig.savefig(figname+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_hists_bin_distr(figname, binedges, histograms_list, histogram_ref):
    # histograms_list can be either dimension 3 if all iterations are included
    # or dimension 2 if only the final unfolded ones for all trials
    histograms_arr = np.asarray(histograms_list)

    nbins = len(binedges) - 1
    assert(nbins == len(histogram_ref))
    assert(nbins == histograms_arr.shape[-1])
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
        ref = histogram_ref[ibin]
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
    A, mu, sigma = fit_gaussian_to_hist(histogram, binedges, dofit)

    x = np.linspace(binedges[0], binedges[-1], 50)
    y = gaus(x, A, mu, sigma)
    style = '-' if dofit else '--'
    ax.plot(x, y, style)

    #yref = gaus(x, sum(histogram)/math.sqrt(2*math.pi), 0, 1)
    #ax.plot(x, yref, '--')
