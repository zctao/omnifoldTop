import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import external.OmniFold.modplot as modplot

# plotting styles
hist_style = {'histtype': 'step', 'density': False, 'lw': 1, 'zorder': 2}

data_style = {'color': 'black', 'label':  'Data', **hist_style}
sim_style = {'color': 'orange', 'label':  'Sim.', **hist_style}
bkg_style = {'color': 'cyan', 'label': 'Bkg.', **hist_style}

gen_style = {'linestyle': '--', 'color': 'blue', 'lw': 1.15, 'label': 'Gen.'}

truth_style = {'step': 'mid', 'edgecolor': 'green', 'facecolor': (0.75, 0.875, 0.75), 'lw': 1.25, 'zorder': 0, 'label': 'Truth'}

ibu_style = {'ls': '-', 'marker': 'o', 'ms': 2.5, 'color': 'gray', 'zorder': 1, 'label':'IBU'}

omnifold_style = {'ls': '-', 'marker': 's', 'ms': 2.5, 'color': 'tab:red', 'zorder': 3, 'label':'MultiFold'}


def plot_ratios(ax, bins, hist_truth, hists_unfolded, colors, hist_truth_unc=None, hists_unfolded_unc=None):
    midbins = (bins[:-1] + bins[1:]) / 2
    binwidth = bins[1] - bins[0]

    # horizontal line at y=1
    ax.plot([np.min(midbins), np.max(midbins)], [1, 1], '-', color=truth_style['edgecolor'], lw=0.75)
    
    if hist_truth_unc is not None:
        truth_unc_ratio = np.divide(hist_truth_unc, hist_truth, out=np.zeros_like(hist_truth), where=(hist_truth!=0))
        ax.fill_between(midbins, 1-truth_unc_ratio, 1+truth_unc_ratio, facecolor=truth_style['facecolor'], zorder=-2)
        
    for i, hist_uf in enumerate(hists_unfolded):
        if hist_uf is None:
            continue
        ymin, ymax = ax.get_ylim()
        ratio = np.divide(hist_uf, hist_truth, out=np.ones_like(hist_truth)*ymin, where=(hist_truth!=0))

        ratio_unc = None
        if hists_unfolded_unc is not None:
            if hists_unfolded_unc[i] is not None:
                ratio_unc = np.divide(hists_unfolded_unc[i], hist_truth, out=np.zeros_like(hist_truth), where=(hist_truth!=0))

        ax.errorbar(midbins, ratio, xerr=binwidth/2, yerr=ratio_unc, color=colors[i], **modplot.style('errorbar'))


def plot_legend(ax, **config):
    loc = config.get('legend_loc', 'best')
    ncol = config.get('legend_ncol', 2)
    #order = [3, 4, 2, 5, 0, 1] if ncol==2 else [3, 5, 4, 0, 2, 1]
    modplot.legend(ax=ax, loc=loc, ncol=ncol, frameon=False, fontsize='x-small')

#def plot_stamp(ax):

def plot_histogram(ax, bin_edges, hist, hist_unc=None, **styles):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.hist(midbins, bin_edges, weights=hist, **styles)
    # TODO: uncertainty hist_unc

def plot_stacked_histograms(ax, bin_edges, hists, hists_unc=None, labels=None,
                            colors=None):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(hists)]
    assert(len(colors)==len(hists))

    if labels is None:
        labels = [str(i) for i in range(len(hists))]
    assert(len(labels)==len(hists))

    ax.hist(np.stack([midbins]*len(hists), axis=1), bin_edges,
            weights=np.stack([h for h in hists], axis=1),
            color=colors, label=labels,
            stacked = True, histtype='step', fill=True)
    # TODO: uncertainty

def plot_hist_as_graph(ax, bin_edges, hist, hist_unc=None, **styles):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.plot(midbins, hist, **styles)
    # TODO: uncertainty hist_unc

def plot_hist_fill(ax, bin_edges, hist, hist_unc=None, **styles):
    midbins = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.fill_between(midbins, hist, **styles)
    # TODO: uncertainty?

def plot_reco_variable(bins, histogram_obs, histogram_sig,
                        histogram_bkg=(None,None),
                        figname='var_reco.pdf', **config):
    """
    Plot detector-level variable distributions
    """
    # use the plotting tools from the original omnifold package
    fig, axes = modplot.axes(ratio_plot=True, ylabel_ratio='Data \/\nMC', **config)
    ax0 = axes[0]
    ax1 = axes[1]
    if config.get('yscale') is not None:
        ax0.set_yscale(config['yscale'])

    hist_obs, hist_obs_unc = histogram_obs
    hist_sig, hist_sig_unc = histogram_sig
    hist_bkg, hist_bkg_unc = histogram_bkg

    ymax = max(hist_obs.max(), hist_sig.max())
    if hist_bkg is not None:
        ymax = max(hist_bkg.max(), ymax)
    ax0.set_ylim(0, ymax*1.2)

    hists_stack = [hist_sig]
    labels = [sim_style['label']]
    colors = [sim_style['color']]
    if hist_bkg is not None:
        hists_stack = [hist_bkg, hist_sig]
        labels = [bkg_style['label'], sim_style['label']]
        colors = [bkg_style['color'], sim_style['color']]

    plot_stacked_histograms(ax0, bins, hists_stack, labels=labels, colors=colors)
    plot_histogram(ax0, bins, hist_obs, **data_style)

    # data/mc ratio
    hist_mc = hist_sig if hist_bkg is None else hist_sig + hist_bkg
    hist_mc_unc = hist_sig_unc if hist_bkg_unc is None else np.sqrt(hist_sig_unc**2 + hist_bkg_unc**2)
    plot_ratios(ax1, bins, hist_mc, [hist_obs], [data_style['color']], hist_mc_unc, [hist_obs_unc])

    # legend
    plot_legend(ax0, **config)

    # save plot
    fig.savefig(figname, bbox_inches='tight')

    plt.close(fig)

def plot_results(bins_gen, histogram_gen, histogram_of, histogram_ibu=(None,None), histogram_truth=(None,None), figname='unfolded.pdf', **config):
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
    plot_hist_as_graph(ax0, bins_gen, hist_gen, **gen_style)

    # if truth is known
    hist_truth, hist_truth_unc = histogram_truth
    if hist_truth is not None:
        ymax = max(hist_truth.max(), ymax)
        plot_hist_fill(ax0, bins_gen, hist_truth, **truth_style)

    # unfolded distributions
    # omnifold
    hist_of, hist_of_unc = histogram_of
    ymax = max(hist_of.max(), ymax)
    plot_hist_as_graph(ax0, bins_gen, hist_of, **omnifold_style)

    # iterative Bayesian unfolding
    hist_ibu, hist_ibu_unc = histogram_ibu
    if hist_ibu is not None:
        ymax = max(hist_ibu.max(), ymax)
        plot_hist_as_graph(ax0, bins_gen, hist_ibu, **ibu_style)

    # update y-axis limit
    ax0.set_ylim((0, ymax*1.2))

    if ax1:
        #  ratios of the unfolded distributions to truth
        plot_ratios(ax1, bins_gen, hist_truth, [hist_ibu,hist_of],
                    [ibu_style['color'], omnifold_style['color']],
                    hist_truth_unc, [hist_ibu_unc, hist_of_unc])

    plot_legend(ax0, **config)

    # plot_stamp(ax0)

    # save plot
    fig.savefig(figname, bbox_inches='tight')

    plt.close(fig)

def plot_histogram2d(figname, h2d, xedges, yedges, variable):
    fig, ax = plt.subplots()
    ax.set_title('Detector Response')
    ax.set_xlabel('Detector-level {}'.format(variable))
    ax.set_ylabel('Truth-level {}'.format(variable))
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
