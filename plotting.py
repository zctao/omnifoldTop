import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import external.OmniFold.modplot as modplot

from util import add_histograms

# plotting styles
hist_style = {'histtype': 'step', 'density': False, 'lw': 1, 'zorder': 2}
graph_style = {'fmt': 'o', 'lw': 1.5, 'capsize': 1.5, 'capthick': 1, 'markersize': 1.5}
leg_style = {'handlelength': 2.0, 'loc': 'best', 'frameon': False, 'numpoints': 1, 'fontsize': 'small'}

data_style = {'color': 'black', 'label':  'Data', **hist_style}
sim_style = {'color': 'orange', 'label':  'Sim.', **hist_style}
bkg_style = {'color': 'cyan', 'label': 'Bkg.', **hist_style}

gen_style = {'linestyle': '--', 'color': 'blue', 'lw': 1.15, 'label': 'Gen.'}

truth_style = {'step': 'mid', 'edgecolor': 'green', 'facecolor': (0.75, 0.875, 0.75), 'lw': 1.25, 'zorder': 0, 'label': 'Truth'}

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
    binwidth = bins[1] - bins[0]

    # horizontal line at y=1
    ax.plot([np.min(midbins), np.max(midbins)], [1, 1], '-', color=color_denom_line, lw=0.75)

    if hist_denom_unc is not None:
        denom_unc_ratio = np.divide(hist_denom_unc, hist_denom, out=np.zeros_like(hist_denom), where=(hist_denom!=0))
        ax.fill_between(midbins, 1-denom_unc_ratio, 1+denom_unc_ratio, facecolor=color_denom_fill, zorder=-2)

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

        ax.errorbar(midbins, ratio, xerr=binwidth/2, yerr=ratio_unc, color=colors_numer[i], **modplot.style('errorbar'))

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
                            colors=None):
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
            stacked = True, histtype='step', fill=True)
    # TODO: uncertainty

def draw_hist_fill(ax, bin_edges, hist, hist_unc=None, **styles):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.fill_between(midbins, hist, **styles)
    # TODO: uncertainty?

def draw_hist_as_graph(ax, bin_edges, hist, hist_unc=None, **styles):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2
    binwidth = bin_edges[1] - bin_edges[0]

    yerr = hist_unc
    xerr = None if yerr is None else binwidth/2

    ax.errorbar(midbins, hist, xerr=xerr, yerr=yerr, **styles)

def plot_graphs(figname, data_arrays, labels=None, title='', xlabel='', ylabel='', colors=None, markers=None):
    fig, ax = init_fig(title, xlabel, ylabel)

    if colors is None:
        colors = set_default_colors(len(data_arrays))
    else:
        assert(len(data_arrays)==len(colors))

    # TODO: error
    for i, (x, y) in enumerate(data_arrays):
        label = None if labels is None else labels[i]
        marker = None if markers is None else markers[i]
        ax.errorbar(x, y, label=label, marker=marker, color=colors[i], **graph_style)

    # plot legend if needed
    if labels is not None:
        ax.legend(**leg_style)

    fig.savefig(figname)
    plt.close(fig)

def plot_histograms1d(figname, bins, hists, hists_err=None, labels=None, title="", xlabel="", ylabel="", colors=None, plottypes=None, marker='o'):
    fig, ax = init_fig(title, xlabel, ylabel)

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

    fig.savefig(figname)
    plt.close(fig)

def plot_reco_variable(bins, histogram_obs, histogram_sig,
                        histogram_bkg=(None,None),
                        figname='var_reco.pdf', log_scale = False, **config):
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
    fig.savefig(figname, bbox_inches='tight')

    plt.close(fig)

def plot_results(bins_gen, histogram_gen, histogram_of, histogram_ibu=(None,None), histogram_truth=(None,None), figname='unfolded.pdf', texts=[], **config):
    """
    Plot and compare the unfolded distributions
    """
    ymax = 0.

    # use the plotting tools from the original omnifold package
    truth_known = histogram_truth[0] is not None
    fig, axes = modplot.axes(ratio_plot = truth_known, **config)
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
        draw_ratios(ax1, bins_gen, hist_truth, [hist_ibu, hist_of],
                    hist_truth_unc, [hist_ibu_unc, hist_of_unc],
                    truth_style['edgecolor'], truth_style['facecolor'],
                    [ibu_style['color'], omnifold_style['color']])

    draw_legend(ax0, **config)

    draw_stamp(ax0, texts, config['stamp_xy'][0], config['stamp_xy'][1])

    # save plot
    fig.savefig(figname, bbox_inches='tight')

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

    fig.savefig(figname)
    plt.close(fig)

def plot_train_log(csv_file, plot_name=None):
    df = pd.read_csv(csv_file)

    plt.figure()
    plt.plot(df['epoch'], df['loss']*1000, label='loss')
    plt.plot(df['epoch'], df['val_loss']*1000, label='val loss')
    plt.legend(loc='best')
    plt.ylabel('loss ($%s$)'%('\\times 10^{-3}'))
    plt.xlabel('Epochs')
    if plot_name is None:
        plot_name = csv_file.replace('.csv', '_loss.pdf')
    plt.savefig(plot_name)
    plt.clf()
    plt.close()

def plot_correlations(data, variables, figname):
    df = pd.DataFrame(data, columns=variables)
    correlations = df.corr()

    fig, ax = plt.subplots()
    im = ax.imshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    fig.colorbar(im, ax=ax)
    ax.tick_params(axis='both', labelsize='small')
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, labelrotation=30)
    ticks = np.arange(0, len(variables), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(variables)
    ax.set_yticklabels(variables)

    fig.savefig(figname)

    #plt.figure()
    #pd.plotting.scatter_matrix(df, alpha=0.5)
    #plt.savefig(figname)
    #plt.close()

    plt.close(fig)
