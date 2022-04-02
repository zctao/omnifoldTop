#!/usr/bin/env python3
import os
import glob
from unfoldv2 import getArgsParser, unfold

# directories
sample_local_dir = os.path.join(os.getenv("HOME"),"data/ttbarDiffXs13TeV/latest")
output_top_dir = os.path.join(os.getenv("HOME"),"data/OmniFoldOutputs/syst_run2")

# systematics
# based on https://github.com/zctao/ntuplerTT/blob/master/configs/datasets/systematics.yaml
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

def getSamples_detNP(
    sample_dir, # top directory to read sample files
    category, # "ejets" or "mjets"
    systematics = 'nominal', # type of systematics
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

    ###
    # observed data
    data = [os.path.join(sample_dir, f"obs/{y}/data_0_pseudotop_{category}.root") for y in years]

    ###
    # background
    backgrounds = []

    # Fakes
    backgrounds += [os.path.join(sample_dir, f"fakes/{y}/data_0_pseudotop_{category}.root") for y in years]

    # W+jets
    backgrounds += [os.path.join(sample_dir, f"systCRL/Wjets_nominal/{e}/Wjets_0_pseudotop_{category}.root") for e in subcampaigns]

    # Z+jets
    backgrounds += [os.path.join(sample_dir, f"systCRL/Zjets_nominal/{e}/Zjets_0_pseudotop_{category}.root") for e in subcampaigns]

    # other samples
    for bkg in ['singleTop', 'ttH', 'ttV', 'VV']:
        backgrounds += [os.path.join(sample_dir, f"detNP/{bkg}_{systematics}/{e}/{bkg}_0_pseudotop_{category}.root") for e in subcampaigns]

    ###
    # signal
    signal = []
    for e in subcampaigns:
        s = glob.glob(os.path.join(sample_dir, f"detNP/ttbar_{systematics}/{e}/ttbar_*_pseudotop_parton_{category}.root"))
        s.sort()
        signal += s

    assert(data)
    assert(signal)
    assert(backgrounds)

    return data, signal, backgrounds

# get the default run config
default_run_cfg = vars(getArgsParser(['-d','dummy1', '-s', 'dummy2']))

# make some changes
default_run_cfg.update({
    "observable_config" : "configs/observables/vars_ttbardiffXs_pseudotop.json",
    "iterations" : 5,
    "batch_size" : 20000,
    "normalize" : True,
    "error_type" : "bootstrap_full",
    "nresamples" : 10,
    #"plot_verbosity" : 0,
    #"run_ibu" : True,
})

for c in ["ejets", "mjets"]:
    print(c)

    print("nominal")
    # nominal
    obs_nominal, sig_nominal, bkg_nominal = getSamples_detNP(
        sample_local_dir,
        category = c,
        systematics = 'nominal',
        subcampaigns = ["mc16e"] #["mc16a", "mc16d", "mc16e"]
    )

    outdir_nominal = os.path.join(output_top_dir, "nominal", f"output_run2_{c}")

    run_cfg = default_run_cfg.copy()
    run_cfg.update({
        "data": obs_nominal, "signal": sig_nominal, "background": bkg_nominal,
        "outputdir": outdir_nominal,
        "plot_verbosity": 3,
        "run_ibu" : True
        })

    unfold(**run_cfg)

    # systematic uncertainties
    for k in syst_dict:
        prefix = syst_dict[k]["prefix"]
        for s in syst_dict[k]["uncertainties"]:
            for v in syst_dict[k]["variations"]:
                syst = f"{prefix}_{s}_{v}"
                print(syst)

                obs_syst, sig_syst, bkg_syst = getSamples_detNP(
                    sample_local_dir,
                    category = c,
                    systematics = syst,
                    subcampaigns = ["mc16e"] #["mc16a", "mc16d", "mc16e"]
                )

                outdir_syst = os.path.join(output_top_dir, syst, f"output_run2_{c}")

                run_cfg = default_run_cfg.copy()
                run_cfg.update({
                    "data": obs_syst, "signal": sig_syst, "background": bkg_syst,
                    "outputdir": outdir_syst,
                    "load_models": outdir_nominal
                    })

                unfold(**run_cfg)
