#!/usr/bin/env python3
import os
import numpy as np
import uproot

import util
import histogramming as myhu
from OmniFoldTTbar import load_unfolder
from createRun2Config import syst_dict
from plotter import plot_uncertainties

def get_unfolded_histograms_from_unfolder(
    fpath_result,
    binning_config,
    observables,
    obsConfig_d,
    iteration = -1,
    nresamples = None, # if None, use all that are available in weights
    nresamples_nominal = None, # for unfolding uncertainties in nominal case
    nominal = False
    ):

    # path to argument json config
    fpath_args_config = os.path.join(fpath_result, "arguments.json")
    if not os.path.isfile(fpath_args_config):
        print(f"ERROR: cannot open argument config {fpath_args_config}")
        return

    unfolder = load_unfolder(
        fpath_args_config,
        observables,
        obsConfig_d
    )

    histograms_d = {}

    for ob in observables:
        print(f"Make histograms for {ob}")
        vname_mc = obsConfig_d[ob]['branch_mc']

        # bin edges
        bins_mc = util.get_bins(ob, binning_config)

        # x-axis label
        xlabel = obsConfig_d[ob]['xlabel']

        histograms_d[ob] = {}

        histograms_d[ob]["unfolded_rs"] = unfolder.get_unfolded_hists_resamples(
            vname_mc,
            bins_mc,
            iteration = iteration,
            nresamples = nresamples,
            absoluteValue = ob in ['th_y', 'tl_y']
        )

        if nominal:
            hist_nominal = unfolder.get_unfolded_distribution(
                vname_mc,
                bins_mc,
                iteration=iteration,
                nresamples = nresamples_nominal,
                absoluteValue = ob in ['th_y', 'tl_y']
            )[0]

            # set x-axis label
            hist_nominal.axes[0].label = xlabel

            histograms_d[ob]['unfolded'] = hist_nominal

    return histograms_d

def write_dict_uproot(file_to_write, obj_dict, top_dir=''):
    for k, v in obj_dict.items():
        if isinstance(v, dict):
            write_dict_uproot(
                file_to_write, v, os.path.join(top_dir, k)
                )
        else:
            if isinstance(v, list):
                for iv, vv in enumerate(v):
                    file_to_write[os.path.join(top_dir, f"{k}-list{iv}")] = vv
            else:
                file_to_write[os.path.join(top_dir, k)] = v

def write_histograms_dict_to_file(hists_dict, file_name):
    with uproot.recreate(file_name) as f:
        write_dict_uproot(f, hists_dict)

def fill_dict_from_path(obj_dict, paths_list, obj):
    if not paths_list:
        return

    p0 = paths_list[0]

    if len(paths_list) == 1:
        # list of objects are denoted by e.g. <obj_name>-list0
        pp = p0.split('-list')
        if len(pp) == 1: # does not contain '-list'
            obj_dict[p0] = obj
        else: # should be added to list
            common_name = pp[0]
            if isinstance(obj_dict.get(common_name), list):
                obj_dict[common_name].append(obj)
            else:
                obj_dict[common_name] = [obj]
    else:
        if not p0 in obj_dict:
            obj_dict[p0] = {}

        fill_dict_from_path(obj_dict[p0], paths_list[1:], obj)

def read_histograms_dict_from_file(file_name):
    histograms_d = {}
    with uproot.open(file_name) as f:
        for k, v in f.classnames().items():
            if not v.startswith("TH"):
                continue

            # create nested dictionary based on directories
            paths = k.split(';')[0].split(os.sep)
            fill_dict_from_path(histograms_d, paths, f[k].to_hist())

    return histograms_d

def compute_bin_errors_systematics(hists_nominal, hists_up, hists_down):
    # list of relative bin errors
    relerr_up_rs = []
    relerr_down_rs = []

    for h_nom, h_up, h_down in zip(hists_nominal, hists_up, hists_down):
        relerr_up_rs.append( h_up.values() / h_nom.values() - 1. )
        relerr_down_rs.append( h_down.values() / h_nom.values() - 1. )

    # mean
    relerr_up = np.mean(relerr_up_rs, axis=0)
    relerr_down = np.mean(relerr_down_rs, axis=0)

    return relerr_up, relerr_down

def compute_bin_errors_unfolding(hist_nominal):
    hval, herr = myhu.get_values_and_errors(hist_nominal)
    relerr = herr / hval
    return relerr, -1*relerr

def make_unfolded_histograms(
    results_top_dir, # top directory to look for unfolding results
    observables, # list of observables to unfold
    fpath_obs_config, # path to observable config
    fpath_bin_config, # path to binning config
    systematics = ['all'], # list of systematic uncertainties
    iteration = -1, # which iteration to extract the unfolding result
    nresamples = None, # number of resamples. If None, use all that are available
    nresamples_nominal = None, # number of resamples for unfolding uncertainty. If None, use all that are available
    results_common_name="output_run2",
    output_dir="." # directory to save histograms
    ):

    # observable config
    obsConfig_d = util.read_dict_from_json(fpath_obs_config)

    histograms_all = {}

    print("nominal")
    fpath_result_nominal = os.path.join(
        results_top_dir, "nominal", results_common_name)

    histograms_all["nominal"] = get_unfolded_histograms_from_unfolder(
        fpath_result_nominal,
        binning_config = fpath_bin_config,
        observables = observables,
        obsConfig_d = obsConfig_d,
        iteration = iteration,
        nresamples = nresamples,
        nresamples_nominal = nresamples_nominal,
        nominal = True
    )

    # systematic uncertainties
    include_all_syst = systematics==['all']

    for k in syst_dict:
        prefix = syst_dict[k]["prefix"]
        for s in syst_dict[k]["uncertainties"]:
            syst_name = f"{prefix}_{s}"

            if not include_all_syst and not syst_name in systematics:
                # skip this one
                continue

            histograms_all[syst_name] = {}

            for v in syst_dict[k]["variations"]:
                syst_name_v = f"{prefix}_{s}_{v}"
                print(syst_name_v)

                fpath_result_syst = os.path.join(
                    results_top_dir, syst_name_v, results_common_name)

                histograms_all[syst_name][v] = get_unfolded_histograms_from_unfolder(
                    fpath_result_syst,
                    binning_config = fpath_bin_config,
                    observables = observables,
                    obsConfig_d = obsConfig_d,
                    iteration = iteration,
                    nresamples = nresamples,
                    nominal = False
                )

    # write histograms to file
    write_histograms_dict_to_file(
        histograms_all, os.path.join(output_dir, 'histograms.root')
        )

    return histograms_all

def compute_bin_uncertainties(observable, histograms_dict, highlight=''):

    errors = []
    draw_options = []

    # histograms for nominal
    h_nominal = histograms_d['nominal'][ob]['unfolded']
    h_nominal_rs = histograms_d['nominal'][ob]['unfolded_rs']

    ####
    # systematic uncertainties

    # for total bin errors
    sqsum_up = 0
    sqsum_down = 0

    nSyst = 0
    last_syst = ''
    for syst_name in histograms_d:
        if syst_name == 'nominal':
            continue
        nSyst += 1
        last_syst = syst_name

        hists_syst = []
        for variation in histograms_d[syst_name]:
            hists_syst.append(
                histograms_d[syst_name][variation][ob]['unfolded_rs']
                )
        assert(len(hists_syst)==2)

        relerr_syst = compute_bin_errors_systematics(h_nominal_rs, *hists_syst)
        # (relerr_var1, relerr_var2)

        if syst_name == highlight:
            errors.append(relerr_syst)
            draw_options.append({
                'label': syst_name, 'hatch':'///', 'edgecolor':'black',
                'facecolor':'none'
                })

        # compute total uncertainty
        relerr_syst_up = np.maximum(*relerr_syst)
        relerr_syst_down = np.minimum(*relerr_syst)
        #print(syst_name, relerr_syst_up, relerr_syst_down)

        sqsum_up += relerr_syst_up * relerr_syst_up
        sqsum_down += relerr_syst_down * relerr_syst_down

    # relerr_syst_total_up, relerr_syst_total_down
    relerr_syst_total = (np.sqrt(sqsum_up), -1*np.sqrt(sqsum_down))
    #print("total", *relerr_syst_total)

    # Add total error to the beginning of the list so it is drawn first
    errors.insert(0, relerr_syst_total)

    if nSyst==1: # only one systematic uncertainty to plot
        draw_options.insert(0, {
            'label': last_syst, 'hatch':'///', 'edgecolor':'tab:blue',
            'facecolor':'none'
            })
    else: # plot total
        draw_options.insert(0, {
            'label': "Detector calibration", 'edgecolor':'none',
            'facecolor':'tab:blue'
            })

    ####
    # unfolding uncertainties
    relerr_uf = compute_bin_errors_unfolding(h_nominal)
    errors.append(relerr_uf)
    draw_options.append({
        'label':'OmniFold', 'edgecolor':'tab:red', 'facecolor':'none'
        })

    return errors, draw_options

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("results_dir", type=str,
                        help="Top directory to look for unfolding results")
    parser.add_argument("-o", "--outdir", type=str, default='systPlot',
                        help="Output plot directory")
    parser.add_argument('--observables', nargs='+', default=[],
                        help="List of observables to evaluate.")
    parser.add_argument('--observable-config', type=str,
                        default="configs/observables/vars_ttbardiffXs_pseudotop.json",
                        help="Path to observable config file")
    parser.add_argument('--binning-config', type=str,
                        default='configs/binning/bins_ttdiffxs.json',
                        help="Binning config file for variables")
    parser.add_argument('--iteration', type=int, default=-1,
                        help="Use unfolded weights at the specified iteration")
    parser.add_argument('--load-hists', type=str,
                        help="If provided, load histograms from the file")
    parser.add_argument("-n", "--unfold-outname", type=str,
                        default="output_run2_ejets",
                        help="Common name for unfolding results directory")
    parser.add_argument("-s", "--systematics", type=str, nargs="*",
                        default=['all'],
                        help="List of systematic uncertainties to evaluate. A special case: 'all' includes all systematics")
    parser.add_argument("--highlight", type=str, default='',
                        help="Name of the systematic uncertainty to plot individually")

    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        print(f"Create output directory: {args.outdir}")
        os.makedirs(args.outdir)

    ####
    # Get unfolded histograms
    if args.load_hists:
        if not os.path.isfile(args.load_hists):
            print(f"ERROR: cannot find file {args.load_hists}")
            sys.exit(1)
        histograms_d = read_histograms_dict_from_file(args.load_hists)
    else:
        # collect histograms from unfolding results
        histograms_d = make_unfolded_histograms(
            args.results_dir,
            observables = args.observables,
            fpath_obs_config = args.observable_config,
            fpath_bin_config = args.binning_config,
            systematics = args.systematics,
            iteration = args.iteration,
            nresamples = 5,
            nresamples_nominal = 25,
            results_common_name = args.unfold_outname,
            output_dir = args.outdir
        )

    ####
    # Evaluate uncertainties and make plots
    if len(args.systematics)==1 and args.systematics != ['all']:
        # only one systematic uncertainty
        syst_highlight = ''
    else:
        syst_highlight = args.highlight

    for ob in histograms_d["nominal"]:
        print(f"{ob}")

        h_nominal = histograms_d['nominal'][ob]['unfolded']
        binEdges = h_nominal.axes[0].edges
        xLabel = h_nominal.axes[0].label

        binErrors, drawOpts = compute_bin_uncertainties(
            ob, histograms_d,
            highlight = syst_highlight
            )

        # plot
        plot_uncertainties(
            figname = os.path.join(args.outdir, f"relerr_{ob}"),
            bins = binEdges,
            uncertainties = binErrors,
            draw_options = drawOpts,
            xlabel = xLabel,
            ylabel = 'Uncertainty'
        )
