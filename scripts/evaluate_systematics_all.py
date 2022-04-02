#!/usr/bin/env python3
import os
import sys
import numpy as np
import uproot
import hist

import util
import histogramming as myhu
from OmniFoldTTbar import OmniFoldTTbar
from plotter import plot_uncertainties

syst_dict = {
    "jet" : {
        "prefix" : "CategoryReduction_JET",
        "uncertainties" : [
            "BJES_Response",
            "EffectiveNP_Detector1",
            "EffectiveNP_Detector2",
            "EffectiveNP_Mixed1",
            "EffectiveNP_Mixed2",
            "EffectiveNP_Mixed3",
            "EffectiveNP_Modelling1",
            "EffectiveNP_Modelling2",
            "EffectiveNP_Modelling3",
            "EffectiveNP_Modelling4",
            "EffectiveNP_Statistical1",
            "EffectiveNP_Statistical2",
            "EffectiveNP_Statistical3",
            "EffectiveNP_Statistical4",
            "EffectiveNP_Statistical5",
            "EffectiveNP_Statistical6",
            "EtaIntercalibration_Modelling",
            "EtaIntercalibration_NonClosure_2018data",
            "EtaIntercalibration_NonClosure_highE",
            "EtaIntercalibration_NonClosure_negEta",
            "EtaIntercalibration_NonClosure_posEta",
            "EtaIntercalibration_TotalStat",
            "Flavor_Composition",
            "Flavor_Response",
            "JER_DataVsMC_MC16",
            "JER_EffectiveNP_1",
            "JER_EffectiveNP_2",
            "JER_EffectiveNP_3",
            "JER_EffectiveNP_4",
            "JER_EffectiveNP_5",
            "JER_EffectiveNP_6",
            "JER_EffectiveNP_7restTerm",
            "Pileup_OffsetMu",
            "Pileup_OffsetNPV",
            "Pileup_PtTerm",
            "Pileup_RhoTopology",
            "PunchThrough_MC16",
            "SingleParticle_HighPt"
        ],
        "variations" : ["_1down", "_1up"]
    },
    "egamma" : {
        "prefix" : "EG",
        "uncertainties" : [
            "RESOLUTION_ALL",
            "SCALE_AF2",
            "SCALE_ALL"
        ],
        "variations" : ["_1down", "_1up"]
    },
    "muon" : {
        "prefix" : "MUON",
        "uncertainties" : [
            "ID",
            "MS",
            "SAGITTA_RESBIAS",
            "SAGITTA_RHO",
            "SCALE"
        ],
        "variations" : ["_1down", "_1up"]
    },
    "met" : {
        "prefix" : "MET",
        "uncertainties" : ["SoftTrk_Scale"],
        "variations" : ["_1down", "_1up"]
    },
    "met_res" : {
        "prefix" : "MET",
        "uncertainties" : ["SoftTrk"],
        "variations" : ["ResoPara", "ResoPerp"]
    }
}

def get_unfolded_histograms_from_unfolder(
    fpath_arguments,
    observables=[],
    fpath_bin_configs='',
    iteration = -1,
    nominal = False
    ):

    print(f"Read arguments from {fpath_arguments}")
    args_d = util.read_dict_from_json(fpath_arguments)

    # observables
    if not observables:
        observables = args_d['observables'] + args_d['observables_extra']

    observable_config = args_d['observable_config']
    print(f"Get observable config from {observable_config}")
    observable_d = util.read_dict_from_json(observable_config)
    assert(observable_d)

    varnames_reco = [observable_d[k]['branch_det'] for k in observables]
    varnames_truth = [observable_d[k]['branch_mc'] for k in observables]

    print("Construct unfolder")
    unfolder = OmniFoldTTbar(
        varnames_reco,
        varnames_truth,
        filepaths_obs = args_d['data'],
        filepaths_sig = args_d['signal'],
        filepaths_bkg = args_d['background'],
        normalize_to_data = args_d['normalize'],
        dummy_value = args_d['dummy_value'],
        weight_type = args_d['weight_type']
    )

    # load weights
    fnames_uw = [os.path.join(args_d['outputdir'], 'weights.npz')]
    if args_d['error_type'] != 'sumw2':
        fnames_uw.append( os.path.join(args_d['outputdir'], f"weights_resample{args_d['nresamples']}.npz") )

    print(f"Load weights from {fnames_uw}")
    unfolder.load(fnames_uw)

    # Get histograms
    if not fpath_bin_configs:
        fpath_bin_configs = args_d['binning_config']

    histograms_d = {}

    for ob in observables:
        print(f"Make histograms for {ob}")
        bins_truth = util.get_bins(ob, fpath_bin_configs)

        vname_truth = observable_d[ob]['branch_mc']

        histograms_d[ob] = {}

        histograms_d[ob]["unfolded_rs"] = unfolder.get_unfolded_hists_resamples(
            vname_truth, bins_truth, iteration=iteration,
            absoluteValue = ob in ['th_y', 'tl_y'])

        if nominal:
            hist_nominal = unfolder.get_unfolded_distribution(
                vname_truth, bins_truth, iteration=iteration,
                absoluteValue = ob in ['th_y', 'tl_y']
                )[0]

            # set x-axis label
            hist_nominal.axes[0].label = observable_d[ob]['xlabel']

            histograms_d[ob]['unfolded'] = hist_nominal

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

def produce_histograms(
        results_top_dir,
        category, # "ejets" or "mjets" for now
        out_dir='.',
        observables = [],
        fpath_bin_configs = '',
        iteration = -1,
        outdir_prefix = "output_run2"
    ):

    if not os.path.isdir(out_dir):
        print(f"Create directory {out_dir}")
        os.makedirs(out_dir)
    
    histograms_all = {}

    print(category)

    # nominal
    print("nominal")
    fpath_args_nominal = os.path.join(results_top_dir, "nominal", f"{outdir_prefix}_{category}", "arguments.json")
    histograms_all['nominal'] = get_unfolded_histograms_from_unfolder(
        fpath_args_nominal,
        observables = observables,
        fpath_bin_configs = fpath_bin_configs,
        iteration = iteration,
        nominal = True)

    # systematic uncertainty variations
    for k in syst_dict:
        prefix = syst_dict[k]["prefix"]
        for s in syst_dict[k]["uncertainties"]:
            syst_name = f"{prefix}_{s}"
            histograms_all[syst_name] = {}

            for v in syst_dict[k]["variations"]:
                syst_name_v = f"{prefix}_{s}_{v}"
                print(syst_name_v)

                fpath_args_syst = os.path.join(results_top_dir, syst_name_v, f"{outdir_prefix}_{category}", "arguments.json")

                if not os.path.isfile(fpath_args_syst):
                    print(f"Cannot find {fpath_args_syst}")
                    # for now
                    continue

                histograms_all[syst_name][v] = get_unfolded_histograms_from_unfolder(
                    fpath_args_syst,
                    observables = observables,
                    fpath_bin_configs = fpath_bin_configs,
                    iteration = iteration)

    # save histograms to file
    write_histograms_dict_to_file(histograms_all, os.path.join(out_dir, 'histograms.root'))

    return histograms_all

def evaluate_systematics(
        histograms_dict,
        plot_dir,
        draw_this_syst = ''
    ):

    # either histograms_dict or fpath_histograms should be provided
    if histograms_dict is None:
        # read histograms from file
        assert(os.path.isfile(fpath_histograms))
        histograms_dict = read_histograms_dict_from_file(fpath_histograms)

    if not os.path.isdir(plot_dir):
        print(f"Create directory {plot_dir}")
        os.makedirs(plot_dir)

    # compute relative bin uncertainties
    for ob in histograms_dict['nominal']:
        print(f"Plot {ob}")

        h_nominal = histograms_dict['nominal'][ob]['unfolded']
        h_nominal_rs = histograms_dict['nominal'][ob]['unfolded_rs']
        bin_edges = h_nominal.axes[0].edges

        # unfolding uncertainties
        relerr_uf = compute_bin_errors_unfolding(h_nominal)

        # systematic uncertainties
        relerr_syst_d = {}

        for syst_name in histograms_dict:
            if syst_name == 'nominal':
                continue

            hists_syst = []
            for variation in histograms_dict[syst_name]:
                hists_syst.append( histograms_dict[syst_name][variation][ob]['unfolded_rs'] )

            assert(len(hists_syst)==2)
            relerrs = compute_bin_errors_systematics(h_nominal_rs, *hists_syst)

            relerr_syst_d[syst_name] = relerrs

        # compute total bin errors
        sqsum_up = 0 # actually an ndarray of zeros with the len of nbins
        sqsum_down = 0
        for syst_name in relerr_syst_d:
            relerrs = relerr_syst_d[syst_name] # (relerr_var1, relerr_var2)

            relerr_up = np.maximum(*relerrs)
            relerr_down = np.minimum(*relerrs)
            #print(syst_name, relerr_up, relerr_down)

            sqsum_up += relerr_up * relerr_up
            sqsum_down += relerr_down * relerr_down

        relerr_total_up = np.sqrt(sqsum_up)
        relerr_total_down = -1*np.sqrt(sqsum_down)

        relerr_total = (relerr_total_up, relerr_total_down)
        #print("total", relerr_total_up, relerr_total_down)

        # make plot
        errors = []
        draw_options = []

        # total
        errors.append(relerr_total)
        draw_options.append({
            'label': "Detector calibration", 'edgecolor':'none', 'facecolor':'tab:blue'
            })

        # high light one
        if draw_this_syst:
            errors.append(relerr_syst_d[draw_this_syst])
            draw_options.append({
                'label': draw_this_syst, 'hatch':'///', 'edgecolor':'black', 'facecolor':'none'
                })

        # unfolding
        errors.append(relerr_uf)
        draw_options.append({
            'label':'OmniFold', 'edgecolor':'tab:red', 'facecolor':'none'
            })

        plot_uncertainties(
            figname = os.path.join(plot_dir, f"relerr_{ob}"),
            bins = bin_edges,
            uncertainties = errors,
            draw_options = draw_options,
            xlabel = h_nominal.axes[0].label,
            ylabel = 'Uncertainty'
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("results_dir", type=str,
                        help="Top directory to look for unfolding results")
    parser.add_argument("-o", "--outdir", type=str, default='syst_plots',
                        help="Output plot directory")
    parser.add_argument("-c", "--category", choices=["ejets", "mjets"],
                        default="ejets", help="Category")
    parser.add_argument('--observables', nargs='+', default=[],
                        help="List of observables to evaluate.")
    parser.add_argument('--binning-config', type=str,
                        default='configs/binning/bins_ttdiffxs.json',
                        help="Binning config file for variables")
    parser.add_argument('--iteration', type=int, default=-1,
                        help="Use unfolded weights at the specified iteration")
    parser.add_argument('--load-hists', type=str,
                        help="If provided, load histograms from the file")

    args = parser.parse_args()

    if args.load_hists:
        if not os.path.isfile(args.load_hists):
            print(f"ERROR: cannot find file {args.load_hists}")
            sys.exit(1)
        hists_d = read_histograms_dict_from_file(args.load_hists)
    else:
        hists_d = produce_histograms(
            args.results_dir,
            args.category,
            out_dir = os.path.join(args.outdir, args.category),
            observables = args.observables,
            fpath_bin_configs = args.binning_config,
            iteration = args.iteration,
            outdir_prefix = "output_run2"
            )

    evaluate_systematics(
        hists_d,
        plot_dir = os.path.join(args.outdir, args.category),
        draw_this_syst = 'CategoryReduction_JET_Pileup_RhoTopology'
        )
