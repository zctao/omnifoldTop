#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

import plotter
import histogramming as myhu

def plot_bin_Y_vs_X(
        figname,
        X, # list
        Ys, # list of np array, or list of list of np array
        YsErr = None, # list, or list of list
        labels = None, # list of str
        xlabel = "",
        ylabel = "",
        **plt_args
    ):

    Ys = np.asarray(Ys)

    if Ys.ndim == 2:
        Ys = np.expand_dims(Ys, 0)
    assert(Ys.ndim == 3)

    nlabels, nX, nbins = Ys.shape

    if YsErr is None:
        YsErr = np.full_like(Ys, None)
    else:
        YsErr = np.asarray(YsErr)
        if YsErr.ndim == 2:
            YsErr = np.expand_dims(YsErr, 0)
    assert(Ys.shape == YsErr.shape)

    if labels is None:
        labels = [None] * nlabels
    elif isinstance(labels, str):
        labels = [labels]
    assert(len(labels) == nlabels)

    fig, ax = plt.subplots(
        nrows = nbins, ncols = 1,
        sharey = 'row', sharex = 'col',
        figsize = (1.1*nX, 1.1*nbins),
        constrained_layout = True
        )

    # for each collection to plot
    for y, yerr, l in zip(Ys, YsErr, labels):

        # loop over each bin
        for b in range(nbins):
            ax[b].errorbar(
                X, y[:,b], yerr = yerr[:,b],
                label = l,
                capsize = 3,
                **plt_args
                )

            ax[b].ticklabel_format(style='sci', axis='y', scilimits=(-2,2), useMathText=True)

            # x label
            if xlabel and b == nbins - 1:
                ax[b].set_xlabel(xlabel)

            # reserve some whitespace
            ax[b].set_ylabel("  ")

            # title
            ax[b].set_title(f"Bin {b+1}")

    #fig.subplots_adjust(left=0.1)
            
    if ylabel:
        fig.text(0.01, 0.5, ylabel, va="center", rotation='vertical')

    # legend
    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    
    # save the plot
    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def compare_full_and_model_bootstrap_errors(
        histograms_d_full,
        histograms_d_model,
        list_of_nensembles,
        outdir
    ):

    for ob in histograms_d_model:
        print(f"Plot for {ob}")

        hists_allruns_merr = histograms_merr_d[ob]['unfolded_allruns']
        hists_allruns_ferr = histograms_ferr_d[ob]['unfolded_allruns']

        binVar_m = []
        binVar_std_m = []
        binVar_f = []
        binVar_std_f = []
        statVar = []
        statVar_std = []

        for N in list_of_nensembles:

            # Bin variance and variance of bin variance
            hvar_merr, hvar_var_merr = myhu.get_variance_from_hists(
                hists_allruns_merr[:N]
                )
            hvar_ferr, hvar_var_ferr = myhu.get_variance_from_hists(
                hists_allruns_ferr[:N]
                )

            binVar_m.append(hvar_merr)
            binVar_std_m.append(np.sqrt(hvar_var_merr))

            binVar_f.append(hvar_ferr)
            binVar_std_f.append(np.sqrt(hvar_var_ferr))

            # Difference between the two
            stat_var = hvar_ferr - hvar_merr

            # Variance of the difference
            stat_var_var = hvar_var_ferr + hvar_var_merr # TODO: check covariance

            statVar.append(stat_var)
            statVar_std.append(np.sqrt(stat_var_var))

        # plot bin variances vs N
        plot_bin_Y_vs_X(
            os.path.join(outdir, f"binVariance_{ob}"),
            list_of_nensembles,
            [binVar_f, binVar_m],
            [binVar_std_f, binVar_std_m],
            labels = ["full", "model"],
            ylabel = "Bin Variance",
            xlabel = "Number of runs"
            )

        # plot difference of the variances
        plot_bin_Y_vs_X(
            os.path.join(outdir, f"statVariance_{ob}"),
            list_of_nensembles,
            statVar,
            statVar_std,
            labels = "stat.",
            ylabel = "Bin Variance",
            xlabel = "Number of runs"
            )

        # Plot the uncertainties in bins of the observable for the largest N
        # i.e. from the nominal unfolded distribution
        hist_nominal_ferr = histograms_ferr_d[ob]['unfolded']
        val_f, err_f = myhu.get_values_and_errors(hist_nominal_ferr)
        relerr_f = err_f / val_f

        hist_nominal_merr = histograms_merr_d[ob]['unfolded']
        val_m, err_m = myhu.get_values_and_errors(hist_nominal_merr)
        relerr_m = err_m / val_m

        color_f, color_m = plotter.set_default_colors(2)

        plotter.plot_uncertainties(
            figname = os.path.join(outdir, f"relerr_{ob}"),
            uncertainties = [
                relerr_f,
                relerr_m
                ],
            draw_options = [
                {"label":"Full bootstrap", "edgecolor":color_f, "facecolor":"none"},
                {"label":"Model only", "edgecolor":color_m, "facecolor":"none"}
                ],
            bins = hist_nominal_merr.axes[0].edges,
            xlabel = hist_nominal_merr.axes[0].label,
            ylabel = "Uncertainty"
            )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("fpath_full_error", type=str,
                        help="Path to the unfolding results with full bootstrap uncertainty")
    # '~/data/OmniFoldOutputs/Run2UncTest/nominal_run2_rs100/output_run2_ejets/ferr'

    parser.add_argument("fpath_model_error", type=str,
                        help="Path to the unfolding results with only model uncertainty")
    # '~/data/OmniFoldOutputs/Run2UncTest/nominal_run2_rs100/output_run2_ejets/merr'

    parser.add_argument('-n', '--nensembles', type=int, nargs='+',
                        help="List of number of ensembles")
    # [5,10,20,30,40,50,60,70,80,90,100]

    parser.add_argument('--histograms-name', type=str, default='histograms.root',
                        help="Name of the root file containing histograms")
    # unfolded_histograms.root
    
    parser.add_argument("-o", "--outdir", type=str, default="UnfoldErrorPlots",
                        help="Output directory")

    args = parser.parse_args()

    # read histograms from file
    fpath_hists_ferr = os.path.join(args.fpath_full_error, args.histograms_name)
    histograms_ferr_d = myhu.read_histograms_dict_from_file(fpath_hists_ferr)

    fpath_hists_merr = os.path.join(args.fpath_model_error, args.histograms_name)
    histograms_merr_d = myhu.read_histograms_dict_from_file(fpath_hists_merr)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    compare_full_and_model_bootstrap_errors(
        histograms_ferr_d,
        histograms_merr_d,
        args.nensembles,
        outdir = args.outdir
    )
