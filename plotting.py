import matplotlib.pyplot as plt
import numpy as np
import OmniFold.modplot as modplot

# plotting styles
hist_style = {'histtype': 'step', 'density': False, 'lw': 1, 'zorder': 2}

gen_style = {'linestyle': '--', 'color': 'blue', 'lw': 1.15, 'label': 'Gen.'}

truth_style = {'step': 'mid', 'edgecolor': 'green', 'facecolor': (0.75, 0.875, 0.75), 'lw': 1.25, 'zorder': 0, 'label': 'Truth'}

ibu_style = {'ls': '-', 'marker': 'o', 'ms': 2.5, 'color': 'gray', 'zorder': 1, 'label':'IBU'}

omnifold_style = {'ls': '-', 'marker': 's', 'ms': 2.5, 'color': 'tab:red', 'zorder': 3, 'label':'MultiFold'}


def plot_ratios(ax, midbins, binwidth, hist_truth, hists_unfolded, colors, hist_truth_unc=None, hists_unfolded_unc=None):
    # horizontal line at y=1
    ax.plot([np.min(midbins), np.max(midbins)], [1, 1], '-', color=truth_style['edgecolor'], lw=0.75)
    
    if hist_truth_unc is not None:
        truth_unc_ratio = hist_truth_unc/(hist_truth + 10**-50)
        ax.fill_between(midbins, 1-truth_unc_ratio, 1+truth_unc_ratio, facecolor=truth_style['facecolor'], zorder=-2)
        
    for i, hist_uf in enumerate(hists_unfolded):
        if hist_uf is None:
            continue
        ratio = hist_uf/(hist_truth + 10**-50)

        ratio_unc = None
        if hists_unfolded_unc is not None:
            if hists_unfolded_unc[i] is not None:
                ratio_unc = hists_unfolded_unc[i]/(hist_truth + 10**-50)

        ax.errorbar(midbins, ratio, xerr=binwidth/2, yerr=ratio_unc, color=colors[i], **modplot.style('errorbar'))
    
    
#def plot_legend(ax):

#def plot_stamp(ax):

def plot_results(variable_name, bins_det, bins_gen, histogram_obs, histogram_sim, histogram_gen, histogram_of, histogram_ibu=None, histogram_truth=None, outdir='', **config):
    """
    TODO: add descriptions
    """
    midbins_det = (bins_det[:-1] + bins_det[1:])/2
    midbins_gen = (bins_gen[:-1] + bins_gen[1:])/2
    binwidth_det = bins_det[1] - bins_det[0]
    binwidth_gen = bins_gen[1] - bins_gen[0]

    # use the plotting tools from the original omnifold package
    fig, [ax0, ax1] = modplot.axes(**config)
    if config.get('yscale') is not None:
        ax0.set_yscale(config['yscale'])

    # detector-level
    hist_obs, hist_obs_unc = histogram_obs
    ax0.hist(midbins_det, bins_det, weights=hist_obs, color='black', label='Data', **hist_style)

    hist_sim, hist_sim_unc = histogram_sim
    ax0.hist(midbins_det, bins_det, weights=hist_sim, color='orange', label='Sim.', **hist_style)

    # generator-level
    # signal prior
    hist_gen, hist_gen_unc = histogram_gen
    ax0.plot(midbins_gen, hist_gen, **gen_style)

    # if truth is known
    hist_truth, hist_truth_unc = histogram_truth
    if hist_truth is not None:
        ax0.fill_between(midbins_gen, hist_truth, **truth_style)

    # unfolded distributions
    # omnifold
    hist_of, hist_of_unc = histogram_of
    ax0.plot(midbins_gen, hist_of, **omnifold_style)

    # iterative Bayesian unfolding
    hist_ibu, hist_ibu_unc = histogram_ibu
    if hist_ibu is not None:
        ax0.plot(midbins_gen, hist_ibu, **ibu_style)

    if hist_truth is not None:
        #  ratios of the unfolded distributions to truth
        plot_ratios(ax1, midbins_gen, binwidth_gen, hist_truth, [hist_ibu,hist_of],
                    [ibu_style['color'], omnifold_style['color']],
                    hist_truth_unc, [hist_ibu_unc, hist_of_unc])

    #plot_legend(ax0)

    # plot_stamp(ax0)

    # save plot
    outdir = outdir.strip('/')+'/'
    fig.savefig(outdir+'MultiFold_{}.pdf'.format(variable_name), bbox_inches='tight')
    plt.show()
