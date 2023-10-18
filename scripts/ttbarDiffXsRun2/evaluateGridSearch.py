import os
import numpy as np
import matplotlib.pyplot as plt

import util

def get_metrics_and_std(metrics_d, metrics_str, value_str, iteration):
    value_list = metrics_d['nominal'][metrics_str][value_str]

    if iteration=='best':
        value = min(value_list)
        iteration = value_list.index(value)
    else:
        iteration = int(iteration)
        value = value_list[iteration]

    # standard deviation
    if "resample" in metrics_d:
        values_resamples = metrics_d["resample"][metrics_str][value_str]
        metrics_std = np.std(np.asarray(values_resamples), axis=0)
        value_std = metrics_std[iteration]
    else:
        value_std = 0

    return value, value_std

def plot_metrics(
    figname,
    metrics_name,
    metrics_arr,
    widths,
    depths,
    title
    ):

    fig, ax = plt.subplots()

    if title:
        ax.set_title(title, loc='left')

    ax.set_xlabel("width")
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(['']+widths)

    ax.set_ylabel("depth")
    ax.set_yticklabels(['']+depths)

    im = ax.matshow(metrics_arr, cmap='viridis')
    fig.colorbar(im, ax=ax, label=metrics_name)

    fig.savefig(figname+'.png', dpi=300)
    plt.close(fig)

def evaluateGridSearch(
    topdir,
    test_type, # stress_data, stress_bump, stress_th_pt
    observable, 
    grids = ['dense_10x1', 'dense_100x3', 'dense_1000x10'],
    metrics_dir = "Metrics",
    iteration = -1, # or 'best'
    outputdir = '.'
    ):

    # determine the number of columns and rows
    widths = set()
    depths = set()
    for network in grids:
        wxd = network.split("_")[-1]
        widths.add(int(wxd.split('x')[0]))
        depths.add(int(wxd.split('x')[1]))

    widths = list(widths)
    widths.sort()

    depths = list(depths)
    depths.sort()

    # metrics to plot
    deltas = np.full((len(depths), len(widths)), np.nan)
    deltas_std = np.full((len(depths), len(widths)), np.nan)

    chi2s = np.full((len(depths), len(widths)), np.nan)
    chi2s_std = np.full((len(depths), len(widths)), np.nan)

    for network in grids:
        wxd = network.split("_")[-1]
        w = int(wxd.split('x')[0])
        d = int(wxd.split('x')[1])

        index_w = np.searchsorted(widths, w)
        index_d = np.searchsorted(depths, d)

        fpath_metrics = os.path.join(topdir, test_type, network, metrics_dir, f"{observable}.json")
        metrics_d = util.read_dict_from_json(fpath_metrics)

        try:
            delta_val, delta_std = get_metrics_and_std(
                metrics_d[observable],'Delta', 'delta', iteration
            )
        except IndexError:
            print(f"No result @ iteration {iteration}")
            return

        try:
            chi2_val, chi2_std = get_metrics_and_std(
                metrics_d[observable], 'Chi2', 'chi2/ndf', iteration
            )
        except IndexError:
            print(f"No result @ iteration {iteration}")
            return

        deltas[index_d][index_w] = delta_val
        deltas_std[index_d][index_w] = delta_std

        chi2s[index_d][index_w] = chi2_val
        chi2s_std[index_d][index_w] = chi2_std

    # plot
    plot_metrics(
        figname = os.path.join(outputdir, 'deltas'),
        metrics_name = "$\\Delta$",
        metrics_arr = deltas,
        widths = widths,
        depths = depths,
        title = f"{test_type} {observable}"
    )

    plot_metrics(
        figname = os.path.join(outputdir, 'deltas_std'),
        metrics_name = "$\\sigma(\\Delta)$",
        metrics_arr = deltas_std,
        widths = widths,
        depths = depths,
        title = f"{test_type} {observable}"
    )
    plot_metrics(
        figname = os.path.join(outputdir, 'chi2'),
        metrics_name = "$\\chi^2$/NDF",
        metrics_arr = chi2s,
        widths = widths,
        depths = depths,
        title = f"{test_type} {observable}"
    )

    plot_metrics(
        figname = os.path.join(outputdir, 'chi2_std'),
        metrics_name = "$\\sigma(\\chi^2\\mathrm{/NDF})$",
        metrics_arr = chi2s_std,
        widths = widths,
        depths = depths,
        title = f"{test_type} {observable}"
    )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("topdir", type=str,
                        help="Top directory of grid search outputs")
    parser.add_argument("networks", type=str, nargs="+", 
                        help="Network names")
    parser.add_argument("--test-types", type=str, nargs='+',
                        default=['stress_data','stress_bump','stress_th_pt'])
    parser.add_argument("--observables", type=str, nargs='+',
                        default=['mtt','ptt','th_pt','tl_pt','ytt','th_y', 'tl_y'])
    parser.add_argument("-o", "--outputdir", type=str, default='.',
                        help="Output directory")
    parser.add_argument("--iterations", nargs='+', default=[-1, 'best'],
                        help="Iteration of unfolding")

    args = parser.parse_args()

    # loop over test type
    for test in args.test_types:
        # loop over obvervables
        print(test)
        for obs in args.observables:
            print(f" {obs}")
            # loop over iterations
            for it in args.iterations:
                print(f"  {it}")

                outdir = os.path.join(
                    args.outputdir, test, f"iter_{it}", obs)
                
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)

                evaluateGridSearch(
                    topdir = args.topdir,
                    test_type = test,
                    observable = obs,
                    grids = args.networks,
                    iteration = it,
                    outputdir = outdir
                )