#!/usr/bin/env python3
import os
import glob
import util

# systematics dictionary
from ttbarDiffXsRun2.systematics import get_systematics
#from ttbarDiffXsRun2.systematics import syst_dict, select_systematics

def subCampaigns_to_years(subcampaigns):
    years = []
    for e in subcampaigns:
        if e == "mc16a":
            years += [2015, 2016]
        elif e == "mc16d":
            years += [2017]
        elif e == "mc16e":
            years += [2018]
        else:
            raise RuntimeError(f"Unknown MC subcampaign {e}")

    return years

def get_samples_data(
    sample_dir, # top direcotry to look for sample files
    category = 'ljets', # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    check_exist = True, # If True, check if the files exist
    ):

    years = subCampaigns_to_years(subcampaigns)

    if isinstance(category, str):
        category = [category]

    data = [os.path.join(sample_dir, f"obs/{y}/data_0_pseudotop_{c}.root") for c in category for y in years]

    assert(data)
    if check_exist:
        for d in data:
            assert(os.path.isfile(d))

    return data

def get_samples_signal(
    sample_dir, # top direcotry to look for sample files
    category = 'ljets', # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    sample_type = 'systCRL', # systCRL or detNP or mcWAlt
    sample_suffix = 'nominal',
    check_exist = True, # If True, check if the files exist
    ):

    if isinstance(category, str):
        category = [category]

    sample_name = f"{sample_type}/ttbar_{sample_suffix}"

    samples_sig = []
    for e in subcampaigns:
        for c in category:
            s = glob.glob(os.path.join(sample_dir, f"{sample_name}/{e}/ttbar_*_pseudotop_parton_{c}.root"))
            s.sort()
            samples_sig += s

    assert(samples_sig)
    if check_exist:
        for f in samples_sig:
            assert(os.path.isfile(f))

    return samples_sig

def get_samples_backgrounds(
    sample_dir, # top direcotry to look for sample files
    category = 'ljets', # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    backgrounds = ['fakes', 'Wjets', 'Zjets', 'singleTop', 'ttH', 'ttV', 'VV'],
    sample_type = 'systCRL', # systCRL or detNP or mcWAlt
    sample_suffix = 'nominal',
    check_exist = True, # If True, check if the files exist
    ):

    if isinstance(category, str):
        category = [category]

    samples_bkg = []

    for bkg in backgrounds:
        if bkg.lower() == "fakes":
            # QCD
            years = subCampaigns_to_years(subcampaigns)
            samples_bkg += [os.path.join(sample_dir, f"fakes/{y}/data_0_pseudotop_{c}.root") for c in category for y in years]
        else:
            if bkg in ["Wjets", "Zjets"]:
                sample_name = f"systCRL/{bkg}_nominal" # only one that is available
            else:
                sample_name = f"{sample_type}/{bkg}_{sample_suffix}"

            samples_bkg += [os.path.join(sample_dir, f"{sample_name}/{e}/{bkg}_0_pseudotop_{c}.root") for c in category for e in subcampaigns]

    assert(samples_bkg)
    if check_exist:
        for f in samples_bkg:
            assert(os.path.isfile(f))

    return samples_bkg

def write_config_nominal(
    sample_local_dir,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("nominal")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, category, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_nominal = os.path.join(output_top_dir, "nominal")

    # config
    nominal_cfg = common_cfg.copy()
    nominal_cfg.update({
        "data": data_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_nominal,
        "plot_verbosity": 2
        })

    # write run configuration to file
    outname_config_nominal = f"{outname_config}_nominal.json"
    util.write_dict_to_json(nominal_cfg, outname_config_nominal)

def write_config_bootstrap(
    sample_local_dir,
    nresamples,
    start_index = 0,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("bootstrap data")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, category, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_bs = os.path.join(output_top_dir, "bootstrap")
    outdir_bs_dict = {
        f"resamples{n}": outdir_bs for n in range(start_index, start_index+nresamples)
    }

    # config
    bootstrap_cfg = common_cfg.copy()
    bootstrap_cfg.update({
        "data": data_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_bs_dict,
        "resample_data": True,
        "run_ibu": False
        })

    # write run configuration to file
    outname_config_bootstrap = f"{outname_config}_bootstrap.json"
    util.write_dict_to_json(bootstrap_cfg, outname_config_bootstrap)

def write_config_bootstrap_mc(
    sample_local_dir,
    nresamples,
    start_index = 0,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("bootstrap mc")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, category, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_bs = os.path.join(output_top_dir, "bootstrap_mc")
    outdir_bs_dict = {
        f"resamples{n}": outdir_bs for n in range(start_index, start_index+nresamples)
    }

    # config
    bootstrap_mc_cfg = common_cfg.copy()
    bootstrap_mc_cfg.update({
        "data": data_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_bs_dict,
        "resample_data": False,
        "resample_mc": True,
        "run_ibu": False
        })

    # write run configuration to file
    outname_config_bootstrap_mc = f"{outname_config}_bootstrap_mc.json"
    util.write_dict_to_json(bootstrap_mc_cfg, outname_config_bootstrap_mc)

def write_config_bootstrap_mc_clos(
    sample_local_dir,
    nresamples,
    start_index = 0,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("bootstrap mc closure")

    # nominal samples:
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_bs = os.path.join(output_top_dir, "bootstrap_mc_clos")
    outdir_bs_dict = {
        f"resamples{n}": outdir_bs for n in range(start_index, start_index+nresamples)
    }

    # config
    bootstrap_mc_cfg = common_cfg.copy()
    bootstrap_mc_cfg.update({
        "data": sig_nominal,
        "bdata": bkg_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_bs_dict,
        "resample_data": True,
        "resample_mc": False,
        "run_ibu": False,
        "correct_acceptance" : False,
        #"resample_everyrun" : True
        })

    # write run configuration to file
    outname_config_bootstrap_mc = f"{outname_config}_bootstrap_mc_clos.json"
    util.write_dict_to_json(bootstrap_mc_cfg, outname_config_bootstrap_mc)

def write_config_mc_clos(
    sample_local_dir,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("mc closure")

    # nominal samples:
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_clos = os.path.join(output_top_dir, "mc_clos")

    # config
    mc_clos_cfg = common_cfg.copy()
    mc_clos_cfg.update({
        "data": sig_nominal,
        "bdata": bkg_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_clos,
        "resample_data": True,
        "resample_mc": False,
        "run_ibu": False,
        "correct_acceptance" : False,
        #"resample_everyrun" : True
        })

    # write run configuration to file
    outname_config_mc_clos = f"{outname_config}_mc_clos.json"
    util.write_dict_to_json(mc_clos_cfg, outname_config_mc_clos)

def write_config_systematics(
    sample_local_dir,
    systematics_keywords = [],
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):

    cfg_dict_list = []

    # nominal samples:
    sig_nom = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nom = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # systematics as alternative sets of events in TTrees
    for syst in get_systematics(systematics_keywords, syst_type="Branch"):
        print(syst)

        # samples
        # varied samples as pseudo data
        sig_syst = get_samples_signal(
            sample_local_dir, category, subcampaigns,
            sample_type = 'detNP',
            sample_suffix = syst
            )

        # background samples to be mixed with the above signal samples to make pseudo data
        bkg_syst = get_samples_backgrounds(
            sample_local_dir, category, subcampaigns,
            sample_type = 'detNP',
            sample_suffix = syst
            )

        # unfold using the nominal samples

        # output directory
        outdir_syst = os.path.join(output_top_dir, syst)

        syst_cfg = common_cfg.copy()
        syst_cfg.update({
            "data": sig_syst,
            "bdata": bkg_syst,
            "signal": sig_nom,
            "background": bkg_nom,
            "outputdir": outdir_syst,
            "correct_acceptance" : False,
            #"load_models": ?
            #"nruns": ?
            })

        cfg_dict_list.append(syst_cfg)

    # systematics as scale factor variations
    for syst, wtype in zip(*get_systematics(systematics_keywords, syst_type="ScaleFactor", get_weight_types=True)):
        print(syst)

        # output directory
        outdir_syst = os.path.join(output_top_dir, syst)

        syst_cfg = common_cfg.copy()

        # use nominal samples but different weights as pseudo data
        syst_cfg.update({
            "data": sig_nom,
            "bdata": bkg_nom,
            "signal": sig_nom,
            "background": bkg_nom,
            "weight_data": wtype,
            "weight_mc": "nominal",
            "outputdir": outdir_syst,
            "correct_acceptance" : False,
            #"load_models": ?
            #"nruns": ?
            #"unfolded_weights": ?
            })

        cfg_dict_list.append(syst_cfg)

    # TODO modelling uncertainties

    print(f"Number of run configs for systematics: {len(cfg_dict_list)}")

    # write run configs to file
    outname_config_syst = f"{outname_config}_syst.json"
    util.write_dict_to_json(cfg_dict_list, outname_config_syst)

def getSamples(
    sample_dir, # top directory to read sample files
    category = 'ljets', # "ejets" or "mjets" or "ljets"
    systematics = 'nominal', # type of systematics
    alternative_sample = '', # str, alternative sample to replace the nominal e.g. ttbar_hw_nominal
    subcampaigns = ["mc16a", "mc16d", "mc16e"]
    ):

    years = []
    for e in subcampaigns:
        if e == "mc16a":
            years += [2015, 2016]
        elif e == "mc16d":
            years += [2017]
        elif e == "mc16e":
            years += [2018]
        else:
            raise RuntimeError(f"Unknown MC subcampaign {e}")

    if category == "ljets":
        channels = ["ljets"] #["ejets", "mjets"]
    elif category == "ejets" or category == "mjets":
        channels = [category]

    ###
    # observed data
    data = [os.path.join(sample_dir, f"obs/{y}/data_0_pseudotop_{c}.root") for c in channels for y in years]

    ###
    # background
    backgrounds = []

    # Fakes
    backgrounds += [os.path.join(sample_dir, f"fakes/{y}/data_0_pseudotop_{c}.root") for c in channels for y in years]

    # W+jets
    sample_type_Wjets = "systCRL/Wjets_nominal" # only one that is available
    backgrounds += [os.path.join(sample_dir, f"{sample_type_Wjets}/{e}/Wjets_0_pseudotop_{c}.root") for c in channels for e in subcampaigns]

    # Z+jets
    sample_type_Zjets = "systCRL/Zjets_nominal" # only one that is available
    backgrounds += [os.path.join(sample_dir, f"{sample_type_Zjets}/{e}/Zjets_0_pseudotop_{c}.root") for c in channels for e in subcampaigns]

    # other samples
    for bkg in ['singleTop', 'ttH', 'ttV', 'VV']:
        if systematics != 'nominal':
            sample_type_bkg = f"detNP/{bkg}_{systematics}"
        elif alternative_sample.startswith(f"{bkg}_"):
            sample_type_bkg = f"mcWAlt/{alternative_sample}"
        else:
            sample_type_bkg = f"systCRL/{bkg}_nominal"

        backgrounds += [os.path.join(sample_dir, f"{sample_type_bkg}/{e}/{bkg}_0_pseudotop_{c}.root") for c in channels for e in subcampaigns]

    ###
    # signal
    signal = []

    if systematics != 'nominal':
        sample_type_sig = f"detNP/ttbar_{systematics}"
    elif alternative_sample.startswith("ttbar_"):
        sample_type_sig = f"mcWAlt/{alternative_sample}"
    else:
        sample_type_sig = "systCRL/ttbar_nominal"

    for e in subcampaigns:
        for c in channels:
            s = glob.glob(os.path.join(sample_dir, f"{sample_type_sig}/{e}/ttbar_*_pseudotop_parton_{c}.root"))
            s.sort()
            signal += s

    assert(data)
    assert(signal)
    assert(backgrounds)

    return data, signal, backgrounds

def createRun2Config(
        sample_local_dir,
        category, # "ejets" or "mjets" or "ljets"
        outname_config = 'runConfig',
        output_top_dir = '.',
        subcampaigns = ["mc16a", "mc16d", "mc16e"],
        do_bootstrap = False,
        do_systematics = False,
        systematics_keywords = [],
        common_cfg = {}
    ):

    # get the real paths of the sample directory and output directory
    sample_local_dir = os.path.expanduser(sample_local_dir)
    sample_local_dir = os.path.realpath(sample_local_dir)

    output_top_dir = os.path.expanduser(output_top_dir)
    output_top_dir = os.path.realpath(output_top_dir)

    # in case outname_config comes with an extension
    outname_config = os.path.splitext(outname_config)[0]

    # create the output directory in case it does not exist
    outputdir = os.path.dirname(outname_config)
    if not os.path.isdir(outputdir):
        print(f"Create directory {outputdir}")
        os.makedirs(outputdir)

    # nominal input files
    write_config_nominal(
        sample_local_dir,
        category = category,
        subcampaigns = subcampaigns,
        output_top_dir = output_top_dir,
        outname_config = outname_config,
        common_cfg = common_cfg
        )

    # bootstrap for statistical uncertainties
    if do_bootstrap:
        write_config_bootstrap(
            sample_local_dir,
            nresamples = 10,
            start_index = 0,
            category = category,
            subcampaigns = subcampaigns,
            output_top_dir = output_top_dir,
            outname_config = outname_config,
            common_cfg = common_cfg
            )

        write_config_bootstrap_mc(
            sample_local_dir,
            nresamples = 10,
            start_index = 0,
            category = category,
            subcampaigns = subcampaigns,
            output_top_dir = output_top_dir,
            outname_config = outname_config,
            common_cfg = common_cfg
            )

        write_config_bootstrap_mc_clos(
            sample_local_dir,
            nresamples = 10,
            start_index = 0,
            category = category,
            subcampaigns = subcampaigns,
            output_top_dir = output_top_dir,
            outname_config = outname_config,
            common_cfg = common_cfg
            )

    # for systematic uncertainties
    if do_systematics:
        # mc closure
        write_config_mc_clos(
            sample_local_dir,
            category = category,
            subcampaigns = subcampaigns,
            output_top_dir = output_top_dir,
            outname_config = outname_config,
            common_cfg = common_cfg
        )

        write_config_systematics(
            sample_local_dir,
            systematics_keywords = systematics_keywords,
            category = category,
            subcampaigns = subcampaigns,
            output_top_dir = output_top_dir,
            outname_config = outname_config,
            common_cfg = common_cfg
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--sample-dir", type=str,
                        default="/mnt/xrootdg/ztao/NtupleTT/latest",
                        help="Sample directory")
    parser.add_argument("-n", "--config-name", type=str,
                        default="configs/run/ttbarDiffXsRun2/runCfg_run2_ljets")
    parser.add_argument("-c", "--category", choices=["ejets", "mjets", "ljets"],
                        default="ljets")
    parser.add_argument("-r", "--result-dir", type=str,
                        default="/mnt/xrootdg/ztao/OmniFoldOutputs/Run2",
                        help="Output directory of unfolding runs")
    parser.add_argument("-e", "--subcampaigns", nargs='+', choices=["mc16a", "mc16d", "mc16e"], default=["mc16a", "mc16d", "mc16e"])
    parser.add_argument("-s", "--do-systematics", action="store_true",
                        help="If True, also generate run configs for evaluating systematics")
    parser.add_argument("-k", "--systematics-keywords", type=str, nargs="*", default=[],
                        help="List of keywords to filter systematic uncertainties to evaluate.If empty, include all available")
    parser.add_argument("-b", "--do-bootstrap", action="store_true",
                        help="If True, also generate run configs to do bootstrap")
    parser.add_argument("--observables", nargs='+',
                        default=['th_pt', 'th_y', 'tl_pt', 'tl_y', 'ptt', 'ytt', 'mtt'],
                        help="List of observables to unfold")

    args = parser.parse_args()

    # hard code common config here for now
    common_cfg = {
        "observable_config" : "configs/observables/vars_ttbardiffXs_pseudotop.json",
        "binning_config" : "configs/binning/bins_ttdiffxs.json",
        "iterations" : 4,
        "batch_size" : 20000,
        "normalize" : False,
        "nruns" : 7,
        "parallel_models" : 3,
        "resample_data" : False,
        "correct_acceptance" : True,
        "run_ibu": True,
    }

    if args.observables:
        common_cfg["observables"] = args.observables

    createRun2Config(
        args.sample_dir,
        category = args.category,
        outname_config = args.config_name,
        output_top_dir = args.result_dir,
        subcampaigns = args.subcampaigns,
        do_bootstrap = args.do_bootstrap,
        do_systematics = args.do_systematics,
        systematics_keywords = args.systematics_keywords,
        common_cfg = common_cfg
        )