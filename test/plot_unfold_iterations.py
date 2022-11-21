import os
import matplotlib.pyplot as plt

import plotter
import histogramming as myhu

def plot_unfold_iterations(
    fname_histograms,
    observables,
    figname,
    plot_ratio=False,
    ratio_to_ibu=True,
    iterations_to_plot=[]
    ):
    if not os.path.isfile(fname_histograms):
        print(f"ERROR: cannot find file {fname_histograms}")
        return

    if len(observables) == 0:
        print(f"ERROR: no observables are provided")
        return

    # Get histograms from the input file
    histograms_d = myhu.read_histograms_dict_from_file(fname_histograms)

    nfeatures = len(observables)

    # get the number of iterations based on the first observable histograms
    niterations = 0
    for ob in observables:
        h_unfolded_alliters = histograms_d[ob].get('unfolded_alliters')
        if h_unfolded_alliters is not None:
            niterations = len(h_unfolded_alliters)

    print(f"nfeatures = {nfeatures}")
    print(f"niterations = {niterations}")

    if iterations_to_plot:
        niterations = len(iterations_to_plot)
        print(f"plot {niterations} iterations")
    else:
        iterations_to_plot = list(range(niterations))

    # prepare plots
    fig, axes = plotter.init_subplots_grid(
        nrows = niterations, 
        ncols = nfeatures,
        figsize_per_grid = (5.5,3.5),
        #xlabels = observables,
        ylabels = [f"Iteration {i+1}" for i in iterations_to_plot],
        sharex = 'col',
        squeeze =  False
    )

    # plot one column at a time
    for c, ob in enumerate(observables):

        # prior distribution of the simulation at the truth level
        h_prior = histograms_d[ob].get('prior')
        assert(h_prior is not None)

        # truth distribution (target) if pseudo data
        h_truth = histograms_d[ob].get('truth')

        # unfolded distributions at all iterations
        h_unfolded_alliters = histograms_d[ob].get('unfolded_alliters')
        assert(h_unfolded_alliters)

        # unfolded distributions from IBU at all iterations if available
        h_ibu_alliters = histograms_d[ob].get('ibu_alliters')

        ytitle = "Number of events"

        if h_truth is not None:
            hist_ref = h_truth
            if plot_ratio:
                ytitle = "Ratio to Truth" 
            style_ref = plotter.truth_style
        else:
            hist_ref = h_prior
            if plot_ratio:
                ytitle = "Ratio to Gen"
            style_ref = plotter.gen_style

        for r, it in enumerate(iterations_to_plot):

            hists_toplot = []
            hists_styles = []

            if h_ibu_alliters:
                if plot_ratio and ratio_to_ibu:
                    hist_ref = h_ibu_alliters[it]
                    ytitle = "Ratio to IBU"
                    style_ref = plotter.ibu_style.copy()
                    style_ref.update({'xerr':True})
                else:
                    hists_toplot.append(h_ibu_alliters[it])
                    ibu_opt = plotter.ibu_style.copy()
                    ibu_opt.update({'xerr':True})
                    hists_styles.append(ibu_opt)

            hists_toplot.append(h_unfolded_alliters[it])
            omnifold_opt = plotter.omnifold_style.copy()
            omnifold_opt.update({'xerr':True})
            hists_styles.append(omnifold_opt)

            if plot_ratio:
                # draw ratio
                colors_numer = [plotter.get_color_from_draw_options(opt) for opt in hists_styles]
                labels_numer = [opt['label'] for opt in hists_styles]

                plotter.draw_ratio(
                    axes[r][c],
                    hist_denom = hist_ref,
                    hists_numer = hists_toplot,
                    color_denom = plotter.get_color_from_draw_options(style_ref), 
                    colors_numer = colors_numer,
                    label_denom = style_ref['label'], 
                    labels_numer = labels_numer
                )

                axes[r][c].set_xlabel(ob)
            else:
                # draw distributions
                plotter.draw_histograms(
                    axes[r][c],
                    histograms = [hist_ref] + hists_toplot,
                    draw_options = [style_ref] + hists_styles,
                    xlabel = ob,
                    legend_loc = None
                )

                if c==0:
                    axes[r][c].set_ylabel(f"Iteration {it+1}")

    # common y title
    fig.text(0.02/nfeatures, 0.5, ytitle, va='center', rotation='vertical')

    # legend
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    # save to file
    fig.savefig(figname+'.png')
    plt.close(fig)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Plot distributions at every unfolding iteration'
        )

    parser.add_argument("fname_histograms", type=str, 
                        help="file path to the input histograms")
    parser.add_argument("-v", "--observables", nargs='+', type=str,
                        help="List of observables to plot")
    parser.add_argument("-o", "--output-name", type=str, default="unfolded_alliters",
                        help="Output plot name")
    parser.add_argument("-r", "--plot-ratio", action='store_true',
                        help="If True, plot ratios. Otherwise plot distributions")
    parser.add_argument("--ibu-ratio", action='store_true',
                        help="If True and if plot_ratio is True, use IBU as the denominator in the ratio plot")
    parser.add_argument("-i", "--iterations-toplot", type=int, nargs='+',
                        help="List of iterations to plot. If not provided, plot all")

    args = parser.parse_args()

    plot_unfold_iterations(
        args.fname_histograms, 
        args.observables, 
        args.output_name,
        args.plot_ratio,
        ratio_to_ibu = args.ibu_ratio,
        iterations_to_plot = args.iterations_toplot,
        )