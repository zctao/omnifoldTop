import os

import util
from plotter import plot_uncertainties, set_default_colors

# plot bin errors of unfolded distributions based on metrics json files
def plot_unfolding_uncertainty(
        result_dirs,
        result_labels,
        plot_dir=".",
        iterations=[-1],
        metrics_subdir="Metrics"
    ):

    # A dictionary to collect values to plot
    errors_d = {}
    #errors_d = {
    #    "variable": {
    #        "bins": [...],
    #        "errors": [[...],[...],...],
    #        "labels": [...]
    #    },
    #}

    # directories to look for json files
    json_dirs = [os.path.join(rdir, metrics_subdir) for rdir in result_dirs]

    for d, l in zip(json_dirs, result_labels):
        print(f"Reading metrics json file from {d}")

        for f in os.listdir(d):
            if not os.path.isfile(os.path.join(d,f)):
                continue
            varname, ext = os.path.splitext(f)
            if ext != '.json':
                continue

            if not varname in errors_d:
                errors_d[varname] = {
                    "bins": [], "errors": [], "labels": []
                }

            # open the json file
            print(f"Open file {f}")
            metrics_d = util.read_dict_from_json(os.path.join(d, f))

            # bin edges
            if not errors_d[varname]["bins"]:
                errors_d[varname]["bins"] = metrics_d[varname]['nominal']['BinErrors']['bin edges']

            if len(iterations) > 1: # if more than one iterations
                label = l+" iter-{}"
            else:
                label = l
            label = label.strip()

            # loop over iterations
            for i in iterations:
                iteration = metrics_d[varname]['nominal']['BinErrors']['iterations'][i]
                errors = metrics_d[varname]['nominal']['BinErrors']['percentage'][i]
                errors_d[varname]["errors"].append(errors)
                errors_d[varname]["labels"].append(label.format(iteration))

    # make plots
    colors = set_default_colors(len(result_labels)*len(iterations))

    for vname in errors_d:
        print(f"Plot {vname}")

        draw_options = []
        for l, c in zip(errors_d[vname]["labels"], colors):
            draw_options.append({'label':l, 'edgecolor': c, 'facecolor': 'none'})

        plot_uncertainties(
            figname = os.path.join(plot_dir, f"relerr_{vname}"),
            bins = errors_d[vname]["bins"],
            uncertainties = errors_d[vname]["errors"],
            draw_options = draw_options,
            xlabel = vname,
            ylabel = 'Uncertainty'
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--result-dirs", nargs='+', type=str,
                        help="List of unfolding result directories")
    parser.add_argument("-l", "--result-labels", nargs='+', type=str,
                        help="List of labels given to each result")
    parser.add_argument("-o", "--output-dir", type=str, default='.',
                        help="Output plot directory")
    parser.add_argument("-i", "--iterations", nargs='+', type=int, default=[-1],
                        help="List of number of iterations")

    args = parser.parse_args()

    assert(len(args.result_dirs)==len(args.result_labels))

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    plot_unfolding_uncertainty(
        args.result_dirs,
        args.result_labels,
        args.output_dir,
        args.iterations,
    )
