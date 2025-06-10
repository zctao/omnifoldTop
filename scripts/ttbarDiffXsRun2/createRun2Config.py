#!/usr/bin/env python3
import os
import glob
import json
import util

from generate_slurm_jobs import generate_slurm_jobs

# systematics
from ttbarDiffXsRun2.systematics import get_systematics, get_gen_weight_index, get_sum_weights_dict

all_backgrounds = ['fakes', 'Wjets', 'Zjets', 'singleTop_sch', 'singleTop_tch', 'singleTop_tW_DR_dyn', 'ttH', 'ttV', 'VV']

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
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    check_exist = True, # If True, check if the files exist
    ):

    years = subCampaigns_to_years(subcampaigns)

    data = [os.path.join(sample_dir, f"obs/{y}/data_0_pseudotop_ljets.h5") for y in years]

    assert data, "Data sample empty"
    if check_exist:
        for d in data:
            assert os.path.isfile(d), "Not all data sample files exist"

    return data

def get_samples_signal(
    sample_dir, # top direcotry to look for sample files
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    sample_suffix = '',
    syst_type = 'nominal',
    check_exist = True, # If True, check if the files exist
    ):

    sample_name = f"ttbar_{sample_suffix}" if sample_suffix else "ttbar"

    samples_sig = []
    for e in subcampaigns:
        s_wildcard = os.path.join(sample_dir, f"{sample_name}/{syst_type}/{e}/ttbar_*_pseudotop*_ljets.h5")
        s = glob.glob(s_wildcard)
        assert s, f"Signal sample empty: {s_wildcard}"
        s.sort()
        samples_sig += s

    if check_exist:
        for f in samples_sig:
            assert os.path.isfile(f), "Not all signal sample files exist"

    return samples_sig

def get_samples_backgrounds(
    sample_dir, # top direcotry to look for sample files
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    backgrounds = all_backgrounds,
    syst_type = 'nominal',
    check_exist = True, # If True, check if the files exist
    ):

    samples_bkg = []

    for bkg in backgrounds:
        if bkg.lower() == "fakes":
            # QCD
            years = subCampaigns_to_years(subcampaigns)
            samples_bkg += [os.path.join(sample_dir, f"fakes/{y}/data_0_pseudotop_ljets.h5") for y in years]
        else:
            sample_name = f"{bkg}/{syst_type}"

            #samples_bkg += [os.path.join(sample_dir, f"{sample_name}/{e}/{bkg}_*_pseudotop_ljets.h5") for e in subcampaigns]
            samples_b = []
            for e in subcampaigns:
                b_wildcard = os.path.join(sample_dir, f"{sample_name}/{e}/{bkg}_*_pseudotop_ljets.h5")
                b = glob.glob(b_wildcard)
                assert b, f"Background sample empty: {b_wildcard}"
                b.sort()
                samples_b += b
            samples_bkg += samples_b

    assert samples_bkg, f"Background sample empty: {backgrounds}"
    if check_exist:
        for f in samples_bkg:
            assert os.path.isfile(f), "Not all background sample files exist"

    return samples_bkg

def write_config_nominal(
    sample_local_dir,
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):
    print("nominal")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, subcampaigns)

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
    print(f"Create run config: {outname_config_nominal}")
    util.write_dict_to_json(nominal_cfg, outname_config_nominal)

    for site in batch_job:
        generate_slurm_jobs(outname_config_nominal, sample_local_dir, site=site)

def write_config_bootstrap(
    sample_local_dir,
    nresamples,
    start_index = 0,
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):
    print("bootstrap data")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, subcampaigns)

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
    print(f"Create run config: {outname_config_bootstrap}")
    util.write_dict_to_json(bootstrap_cfg, outname_config_bootstrap)

    for site in batch_job:
        generate_slurm_jobs(outname_config_bootstrap, sample_local_dir, site=site)

def write_config_bootstrap_mc(
    sample_local_dir,
    nresamples,
    start_index = 0,
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):
    print("bootstrap mc")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, subcampaigns)

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
    print(f"Create run config: {outname_config_bootstrap_mc}")
    util.write_dict_to_json(bootstrap_mc_cfg, outname_config_bootstrap_mc)

    for site in batch_job:
        generate_slurm_jobs(outname_config_bootstrap_mc, sample_local_dir, site=site)

def write_config_bootstrap_mc_clos(
    sample_local_dir,
    nresamples,
    start_index = 0,
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):
    print("bootstrap mc closure")

    # nominal samples:
    sig_nominal = get_samples_signal(sample_local_dir, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, subcampaigns)

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
    print(f"Create run config: {outname_config_bootstrap_mc}")
    util.write_dict_to_json(bootstrap_mc_cfg, outname_config_bootstrap_mc)

    for site in batch_job:
        generate_slurm_jobs(outname_config_bootstrap_mc, sample_local_dir, site=site)

def write_config_closure_resample(
    sample_local_dir,
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):
    print("closure resample")

    # nominal samples:
    sig_nominal = get_samples_signal(sample_local_dir, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, subcampaigns)

    # output directory
    outdir_clos = os.path.join(output_top_dir, "closure_resample")

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
        "run_ibu": True,
        "correct_acceptance" : False,
        "resample_everyrun" : True,
        "plot_verbosity": 2,
        "truth_known": True
        })

    # write run configuration to file
    outname_config_mc_clos = f"{outname_config}_closure_resample.json"
    print(f"Create run config: {outname_config_mc_clos}")
    util.write_dict_to_json(mc_clos_cfg, outname_config_mc_clos)

    for site in batch_job:
        generate_slurm_jobs(outname_config_mc_clos, sample_local_dir, site=site)

def write_config_closure_oddeven(
    sample_local_dir,
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):
    print("closure oddeven")

    # nominal samples:
    sig_nominal = get_samples_signal(sample_local_dir, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, subcampaigns)

    # output directory
    outdir_clos = os.path.join(output_top_dir, "closure_oddeven")

    # config
    mc_clos_cfg = common_cfg.copy()
    mc_clos_cfg.update({
        "data": sig_nominal,
        "bdata": bkg_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_clos,
        "resample_data": False,
        "resample_mc": False,
        "run_ibu": True,
        "correct_acceptance": False,
        "normalize": True,
        "plot_verbosity": 2,
        "truth_known": True
        })

    # write run configuration to file
    outname_config_clos = f"{outname_config}_closure_oddeven.json"
    print(f"Create run config: {outname_config_clos}")
    util.write_dict_to_json(mc_clos_cfg, outname_config_clos)

    for site in batch_job:
        generate_slurm_jobs(outname_config_clos, sample_local_dir, site=site)

def write_config_theory(
    sample_local_dir,
    systematics_keywords = [],
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    write_single_file = False,
    batch_job = []
    ):

    cfg_theo_list = []

    # Read sum of weights for generator weight variations
    sumWgts_d = get_sum_weights_dict()
    if not sumWgts_d:
        print("WARNING: no file for sum of weight variations provided. Will not rescale total sample weights.")

    for syst, wtype in zip(*get_systematics(systematics_keywords, syst_type='GenWeight', get_weight_types=True)):
        print(syst)

        wname, windex = wtype.split(':')
        windex = int(windex)

        samples_data, samples_signal = [], []

        if 'PDF4LHC' in syst: # PDF uncertainty
            weight_data = "mc_generator_weights_PDF4LHC15_0"
            weight_mc = wname

            # loop over subcampaigns to rescale the sum of weights
            for era in subcampaigns:
                # signal sample
                sig_era = get_samples_signal(
                    sample_local_dir,
                    subcampaigns = [era],
                    )

                if sumWgts_d:
                    rescale_sumw_data = sumWgts_d[410470][era][0] / sumWgts_d[410470][era][get_gen_weight_index('PDF4LHC15_0')]
                    samples_data += [f"{sample}*{rescale_sumw_data}" for sample in sig_era]

                    rescale_sumw_mc = sumWgts_d[410470][era][0] / sumWgts_d[410470][era][windex]
                    samples_signal += [f"{sample}*{rescale_sumw_mc}" for sample in sig_era]
                else:
                    samples_data += sig_era
                    samples_signal += sig_era

        elif '_muR_' in syst or '_muF_' in syst or '_alphaS_' in syst: # ISR, FSR
            weight_data = wname
            weight_mc = 'nominal'

            # loop over subcampaigns to rescale the sum of weights
            for era in subcampaigns:
                # signal sample
                sig_era = get_samples_signal(
                    sample_local_dir,
                    subcampaigns = [era],
                    )

                if sumWgts_d:
                    rescale_sumw = sumWgts_d[410470][era][0] / sumWgts_d[410470][era][windex]
                    # Index 0 is the nominal sum of weights
                    samples_data += [f"{sample}*{rescale_sumw}" for sample in sig_era]
                else:
                    samples_data += sig_era

                samples_signal += sig_era
        else:
            print(f"ERROR: unknown systematic uncertainty {syst}")
            continue

        syst_cfg = common_cfg.copy()
        syst_cfg.update({
            "data": samples_data,
            "signal": samples_signal,
            "weight_data": weight_data,
            "weight_mc": weight_mc,
            "outputdir": os.path.join(output_top_dir, syst),
            "correct_acceptance" : False,
            "truth_known": True
        })

        cfg_theo_list.append(syst_cfg)

        if not write_single_file:
            outname_config_syst = f"{outname_config}_{syst}.json"
            print(f"Create run config: {outname_config_syst}")
            util.write_dict_to_json(syst_cfg, outname_config_syst)

            for site in batch_job:
                generate_slurm_jobs(outname_config_syst, sample_local_dir, site=site)

    return cfg_theo_list

def write_config_systematics_modelling(
    sample_local_dir,
    systematics_keywords = [],
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    write_single_file = False,
    batch_job = []
    ):

    cfg_model_list = []

    # nominal signal sample
    signal_nom = get_samples_signal(
        sample_local_dir,
        subcampaigns,
        sample_suffix = 'AFII',
        )

    for syst in get_systematics(systematics_keywords, syst_type='Modelling'):
        print(syst)

        # alternative sample as the pseudo data
        signal_alt = get_samples_signal(
            sample_local_dir, 
            subcampaigns,
            sample_suffix = f"{syst.split('_')[-1]}",
            )

        syst_cfg = common_cfg.copy()
        syst_cfg.update({
            "data": signal_alt,
            "signal": signal_nom,
            "outputdir": os.path.join(output_top_dir, syst),
            "correct_acceptance" : False,
            "truth_known": True
        })

        cfg_model_list.append(syst_cfg)

        if not write_single_file:
            outname_config_syst = f"{outname_config}_{syst}.json"
            print(f"Create run config: {outname_config_syst}")
            util.write_dict_to_json(syst_cfg, outname_config_syst)

            for site in batch_job:
                generate_slurm_jobs(outname_config_syst, sample_local_dir, site=site)

    return cfg_model_list

def write_config_systematics_background(
    sample_local_dir,
    systematics_keywords = [],
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    write_single_file = False,
    batch_job = []
    ):

    cfg_bkg_list = []

    # nominal signal and background samples
    sig_nom = get_samples_signal(sample_local_dir, subcampaigns)
    bkg_nom = get_samples_backgrounds(
        sample_local_dir, subcampaigns,
        backgrounds = all_backgrounds
        )

    # background modelling
    for syst in get_systematics(systematics_keywords, syst_type="BackgroundModelling"):
        print(syst)
        # singleTop
        if syst == 'singleTop_tW_DS_dyn':
            # replace 'singleTop_tW_DR_dyn' in the nominal list
            all_backgrounds_alt = all_backgrounds.copy()
            all_backgrounds_alt.remove('singleTop_tW_DR_dyn')
            all_backgrounds_alt.append('singleTop_tW_DS_dyn')

            # alternative background samples
            bkg_alt = get_samples_backgrounds(
                sample_local_dir, subcampaigns,
                backgrounds = all_backgrounds_alt
                )

        else:
            raise RuntimeError(f"Something went wrong. Unknown background modelling systematic {syst}.")

        syst_cfg = common_cfg.copy()
        syst_cfg.update({
            "data": sig_nom,
            "bdata": bkg_alt,
            "signal": sig_nom,
            "background": bkg_nom,
            "outputdir": os.path.join(output_top_dir, syst),
            "correct_acceptance": False
        })

        cfg_bkg_list.append(syst_cfg)

        if not write_single_file:
            outname_config_syst = f"{outname_config}_{syst}.json"
            print(f"Create run config: {outname_config_syst}")
            util.write_dict_to_json(syst_cfg, outname_config_syst)

            for site in batch_job:
                generate_slurm_jobs(outname_config_syst, sample_local_dir, site=site)

    # background normalization
    for syst in get_systematics(systematics_keywords, syst_type="BackgroundNorm"):
        print(syst)
        bkg_prefix = syst.split('_norm_')[0]
        f_rescale = float(syst.split('_norm_')[-1])

        # separate background samples
        bkg_names_rescale = []
        bkg_names_others = []
        for bkg in all_backgrounds:
            if bkg.startswith(bkg_prefix):
                bkg_names_rescale.append(bkg)
            else:
                bkg_names_others.append(bkg)

        bkg_rescale = get_samples_backgrounds(
            sample_local_dir, subcampaigns,
            backgrounds = bkg_names_rescale
        )

        bkg_others = get_samples_backgrounds(
            sample_local_dir, subcampaigns,
            backgrounds = bkg_names_others
        )

        bkg_alt = [f"{sample}*{f_rescale}" for sample in bkg_rescale]

        # add the rest of the background samples
        bkg_alt += bkg_others

        syst_cfg = common_cfg.copy()
        syst_cfg.update({
            "data": sig_nom,
            "bdata": bkg_alt,
            "signal": sig_nom,
            "background": bkg_nom,
            "outputdir": os.path.join(output_top_dir, syst),
            "correct_acceptance": False
        })

        cfg_bkg_list.append(syst_cfg)

        if not write_single_file:
            outname_config_syst = f"{outname_config}_{syst}.json"
            print(f"Create run config: {outname_config_syst}")
            util.write_dict_to_json(syst_cfg, outname_config_syst)

            for site in batch_job:
                generate_slurm_jobs(outname_config_syst, sample_local_dir, site=site)

    return cfg_bkg_list

def write_config_systematics(
    sample_local_dir,
    systematics_keywords = [],
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    write_single_file = False,
    batch_job = []
    ):

    cfg_dict_list = []

    # nominal samples:
    sig_nom = get_samples_signal(sample_local_dir, subcampaigns)
    bkg_nom = get_samples_backgrounds(sample_local_dir, subcampaigns)

    print("central")
    central_cfg = common_cfg.copy()
    central_cfg.update({
        "data": sig_nom,
        "bdata": bkg_nom,
        "signal": sig_nom,
        "background": bkg_nom,
        "outputdir": os.path.join(output_top_dir, "central"),
        "correct_acceptance" : False,
        })

    cfg_dict_list.append(central_cfg)

    if not write_single_file:
        # write to file
        outname_config_central = f"{outname_config}_central.json"
        print(f"Create run config: {outname_config_central}")
        util.write_dict_to_json(central_cfg, outname_config_central)

        for site in batch_job:
            generate_slurm_jobs(outname_config_central, sample_local_dir, site=site)

    # systematics as alternative sets of events in TTrees
    for syst in get_systematics(systematics_keywords, syst_type="Branch"):
        print(syst)

        # samples
        # varied samples as pseudo data
        sig_syst = get_samples_signal(
            sample_local_dir,
            subcampaigns,
            syst_type = syst,
            )

        # background samples to be mixed with the above signal samples to make pseudo data
        bkg_syst = get_samples_backgrounds(
            sample_local_dir,
            subcampaigns,
            syst_type = syst
            )

        # unfold using the nominal samples

        syst_cfg = common_cfg.copy()
        syst_cfg.update({
            "data": sig_syst,
            "bdata": bkg_syst,
            "signal": sig_nom,
            "background": bkg_nom,
            "outputdir": os.path.join(output_top_dir, syst),
            "correct_acceptance" : False,
            #"load_models": ?
            #"nruns": ?
            })

        cfg_dict_list.append(syst_cfg)

        if not write_single_file:
            outname_config_syst = f"{outname_config}_{syst}.json"
            print(f"Create run config: {outname_config_syst}")
            util.write_dict_to_json(syst_cfg, outname_config_syst)

            for site in batch_job:
                generate_slurm_jobs(outname_config_syst, sample_local_dir, site=site)

    # systematics as scale factor variations
    for syst, wtype in zip(*get_systematics(systematics_keywords, syst_type="ScaleFactor", get_weight_types=True)):
        print(syst)

        syst_cfg = common_cfg.copy()

        # use nominal samples but different weights as pseudo data
        syst_cfg.update({
            "data": sig_nom,
            "bdata": bkg_nom,
            "signal": sig_nom,
            "background": bkg_nom,
            "weight_data": wtype,
            "weight_mc": "nominal",
            "outputdir": os.path.join(output_top_dir, syst),
            "correct_acceptance" : False,
            #"load_models": ?
            #"nruns": ?
            #"unfolded_weights": ?
            })

        cfg_dict_list.append(syst_cfg)

        if not write_single_file:
            outname_config_syst = f"{outname_config}_{syst}.json"
            print(f"Create run config: {outname_config_syst}")
            util.write_dict_to_json(syst_cfg, outname_config_syst)

            for site in batch_job:
                generate_slurm_jobs(outname_config_syst, sample_local_dir, site=site)

    # Theory uncertainties
    # ISR,FSR, and PDF
    cfg_dict_list += write_config_theory(
        sample_local_dir,
        systematics_keywords = systematics_keywords,
        subcampaigns = subcampaigns,
        output_top_dir = output_top_dir,
        outname_config = outname_config,
        common_cfg = common_cfg,
        write_single_file = write_single_file,
        batch_job = batch_job
        )

    # Modelling uncertainties
    cfg_dict_list += write_config_systematics_modelling(
        sample_local_dir,
        systematics_keywords = systematics_keywords,
        subcampaigns = subcampaigns,
        output_top_dir = output_top_dir,
        outname_config = outname_config,
        common_cfg = common_cfg,
        write_single_file = write_single_file,
        batch_job = batch_job
        )

    # background uncertainties
    cfg_dict_list += write_config_systematics_background(
        sample_local_dir,
        systematics_keywords = systematics_keywords,
        subcampaigns = subcampaigns,
        output_top_dir = output_top_dir,
        outname_config = outname_config,
        common_cfg = common_cfg,
        write_single_file = write_single_file,
        batch_job = batch_job
        )

    print(f"Number of run configs for systematics: {len(cfg_dict_list)}")

    if write_single_file:
        # write run configs to file
        outname_config_syst = f"{outname_config}_syst.json"
        print(f"Create run config: {outname_config_syst}")
        util.write_dict_to_json(cfg_dict_list, outname_config_syst)

        if batch_job:
            print("ERROR!! Can only generate Slurm job files for each individual systematic variation.")

def write_config_model(
    sample_local_dir,
    ttbar_alt, # 'hw', 'amc'
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):

    print(f"Model tests: {ttbar_alt}")

    # alternative ttbar vs nominal ttbar
    # samples
    signal_nominal = get_samples_signal(
        sample_local_dir,
        subcampaigns,
        sample_suffix = 'AFII',
    )

    signal_alt = get_samples_signal(
        sample_local_dir,
        subcampaigns,
        sample_suffix = ttbar_alt,
    )

    outdir_alt = os.path.join(output_top_dir, f"ttbar_{ttbar_alt}_vs_nominal")

    # config
    ttbar_alt_cfg = common_cfg.copy()
    ttbar_alt_cfg.update({
        "data": signal_alt,
        "signal": signal_nominal,
        "outputdir": outdir_alt,
        "plot_verbosity": 2,
        "normalize": True,
        "correct_acceptance": False,
        "truth_known": True
    })

    # write run config to file
    outname_config_alt = f"{outname_config}_model_{ttbar_alt}.json"
    print(f"Create run config: {outname_config_alt}")
    util.write_dict_to_json(ttbar_alt_cfg, outname_config_alt)

    for site in batch_job:
        generate_slurm_jobs(outname_config_alt, sample_local_dir, site=site)

def write_config_stress(
    sample_local_dir,
    fpath_reweights = [],
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):

    print("Stress tests")

    sig_nominal = get_samples_signal(sample_local_dir, subcampaigns)

    stress_common_cfg = common_cfg.copy()
    stress_common_cfg.update({
        "data": sig_nominal,
        "signal": sig_nominal,
        "plot_verbosity": 2,
        "normalize": True,
        "correct_acceptance": False,
        "truth_known": True
    })

    ######
    # linear th_pt
    stress_th_pt_cfg = stress_common_cfg.copy()
    stress_th_pt_cfg.update({
        "outputdir": os.path.join(output_top_dir, f"stress_th_pt"),
        "reweight_data": "linear_th_pt"
    })

    # write run config to file
    outname_stress_th_pt = f"{outname_config}_stress_th_pt.json"
    print(f"Create run config: {outname_stress_th_pt}")
    util.write_dict_to_json(stress_th_pt_cfg, outname_stress_th_pt)

    for site in batch_job:
        generate_slurm_jobs(outname_stress_th_pt, sample_local_dir, site=site)

    ######
    # mtt bump
    stress_bump_cfg = stress_common_cfg.copy()
    stress_bump_cfg.update({
        "outputdir": os.path.join(output_top_dir, f"stress_bump"),
        "reweight_data": "gaussian_bump"
    })

    # write run config to file
    outname_stress_bump = f"{outname_config}_stress_bump.json"
    print(f"Create run config: {outname_stress_bump}")
    util.write_dict_to_json(stress_bump_cfg, outname_stress_bump)

    for site in batch_job:
        generate_slurm_jobs(outname_stress_bump, sample_local_dir, site=site)

    # data
    if not fpath_reweights:
        print("WARNING cannot generate run config for data induced stress test: no external weight files are provided.")
    else:
        # reweighted signal MC as pseudo-data
        stress_data_cfg = stress_common_cfg.copy()
        stress_data_cfg.update({
            "outputdir": os.path.join(output_top_dir, f"stress_data"),
            "weight_data": f"external:{','.join(fpath_reweights)}",
            "weight_mc": "nominal"
        })

        # write run config to file
        outname_stress_data = f"{outname_config}_stress_data.json"
        print(f"Create run config: {outname_stress_data}")
        util.write_dict_to_json(stress_data_cfg, outname_stress_data)

        for site in batch_job:
            generate_slurm_jobs(outname_stress_data, sample_local_dir, site=site)

        # use the reweighted signal MC to unfold data
        data_nominal = get_samples_data(sample_local_dir, subcampaigns)

        stress_data_alt_cfg = stress_common_cfg.copy()
        stress_data_alt_cfg.update({
            "outputdir": os.path.join(output_top_dir, f"stress_data_alt"),
            "data": data_nominal,
            "signal": sig_nominal,
            "plot_verbosity": 2,
            "normalize": False,
            "correct_acceptance": True,
            "truth_known": False,
            "weight_data": "nominal",
            "weight_mc": f"external:{','.join(fpath_reweights)}"
        })

        # write run config to file
        outname_stress_data_alt = f"{outname_config}_stress_data_alt.json"
        print(f"Create run config: {outname_stress_data_alt}")
        util.write_dict_to_json(stress_data_alt_cfg, outname_stress_data_alt)

        for site in batch_job:
            generate_slurm_jobs(outname_stress_data_alt, sample_local_dir, site=site)

def write_config_stress_binned(
    sample_local_dir,
    fpath_reweights,
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {},
    batch_job = []
    ):

    print("Stress tests: binned reweighting")
    if not fpath_reweights:
        print("ERROR cannot generate run config for data induced stress test: no external weight files are provided.")
    else:
        sig_nominal = get_samples_signal(sample_local_dir, subcampaigns)

        observables = common_cfg["observables"]

        # output directory
        if "stress_data_binned" in output_top_dir:
            output_dir = output_top_dir
        else:
            output_dir = os.path.join(output_top_dir, f"stress_data_binned")

        stress_binned_cfg = common_cfg.copy()
        stress_binned_cfg.update({
            "data": sig_nominal,
            "signal": sig_nominal,
            "plot_verbosity": 2,
            "normalize": True,
            "correct_acceptance": False,
            "truth_known": True,
            "observables": {f"{obs}" : [obs] for obs in observables},
            "outputdir": output_dir,
            "weight_data": f"external:{','.join(fpath_reweights)}",
            "weight_mc": "nominal"
        })

        # write run configs to file
        outname_stress_data_binned = f"{outname_config}_stress_data_binned.json"
        print(f"Create run config: {outname_stress_data_binned}")
        util.write_dict_to_json(stress_binned_cfg, outname_stress_data_binned)

        for site in batch_job:
            generate_slurm_jobs(outname_stress_data_binned, sample_local_dir, site=site)

def createRun2Config(
        sample_local_dir,
        outname_config = 'runConfig',
        output_top_dir = '.',
        subcampaigns = ["mc16a", "mc16d", "mc16e"],
        run_list = None,
        systematics_keywords = [],
        external_reweights = [],
        common_cfg = {},
        batch_job = []
    ):

    # get the real paths of the sample directory and output directory
    sample_local_dir = os.path.expanduser(sample_local_dir)

    output_top_dir = os.path.expanduser(output_top_dir)

    # in case outname_config comes with an extension
    outname_config = os.path.splitext(outname_config)[0]

    # create the output directory in case it does not exist
    outputdir = os.path.dirname(outname_config)
    if not os.path.isdir(outputdir):
        print(f"Create directory {outputdir}")
        os.makedirs(outputdir)

    # common arguments for write_config_*
    write_common_args = {
        'sample_local_dir': sample_local_dir,
        'subcampaigns': subcampaigns,
        'output_top_dir': output_top_dir,
        'outname_config': outname_config,
        'common_cfg': common_cfg,
        'batch_job': batch_job,
        }

    # nominal
    if 'nominal' in run_list:
        write_config_nominal(**write_common_args)

    # bootstrap for statistical uncertainties
    if 'bootstrap' in run_list:
        write_config_bootstrap(
            nresamples = 10,
            start_index = 0,
            **write_common_args
            )

        write_config_bootstrap_mc(
            nresamples = 10,
            start_index = 0,
            **write_common_args
            )

    # Systematic uncertainties
    if 'systematics' in run_list:
        write_config_systematics(
            systematics_keywords = systematics_keywords,
            **write_common_args
        )

    # MC closure
    if 'closure' in run_list:
        write_config_closure_resample(**write_common_args)
        write_config_closure_oddeven(**write_common_args)

        #if 'bootstrap' in run_list:
        #    write_config_bootstrap_mc_clos(
        #        nresamples = 10,
        #        start_index = 0,
        #        **write_common_args
        #        )

    if 'model' in run_list:
        for ttbar_alt in ['hw', 'amc']:
            write_config_model(ttbar_alt=ttbar_alt, **write_common_args)

    if 'stress' in run_list:
        write_config_stress(
            fpath_reweights = external_reweights,
            **write_common_args
        )

    if 'stress_binned' in run_list:
        write_config_stress_binned(
            fpath_reweights = external_reweights,
            **write_common_args
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--sample-dir", type=str, action=util.ParseEnvVar,
                        default="${DATA_DIR}/ntuplerTT/latest",
                        help="Sample directory")
    parser.add_argument("-n", "--config-name", type=str,
                        default="configs/run/ttbarDiffXsRun2/runCfg_run2_ljets")
    parser.add_argument("-r", "--result-dir", type=str, action=util.ParseEnvVar,
                        default="${DATA_DIR}/OmniFoldOutputs/Run2",
                        help="Output directory of unfolding runs")
    parser.add_argument("-e", "--subcampaigns", nargs='+', choices=["mc16a", "mc16d", "mc16e"], default=["mc16a", "mc16d", "mc16e"])
    parser.add_argument("--observables", nargs='+',
                        default=['th_pt', 'th_y', 'tl_pt', 'tl_y', 'ptt', 'ytt', 'mtt'],
                        help="List of observables to unfold")

    run_options = ['nominal', 'bootstrap', 'systematics', 'model', 'closure', 'stress', 'stress_binned']
    parser.add_argument("-l", "--run-list", nargs="+",
                        choices=run_options, default=run_options,
                        help="List of run types to generate config files. If None, generate run configs for all types")

    parser.add_argument("-k", "--systematics-keywords", type=str, nargs="*", default=[],
                        help="List of keywords to filter systematic uncertainties to evaluate. If empty, include all available.")

    parser.add_argument("--external-reweights",
                        type=str, nargs='+', default=[], action=util.ParseEnvVar,
                        help="List of path to external weight files from reweighting")

    parser.add_argument("--config-string", type=str,
                        help="String in JSON format to be parsed for updating run configs")

    parser.add_argument("-b", "--batch-job", choices=['cc', 'ubc'], nargs='+',
                        default=[],
                        help="If provided, generate both run configs and batch job files for the specified sites")

    args = parser.parse_args()

    # hard code common config here for now
    common_cfg = {
        "observable_config" : "${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json",
        "binning_config" : "${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json",
        "iterations" : 3,
        "batch_size" : 500000,
        "normalize" : False,
        "nruns" : 12,
        "parallel_models" : 2,
        "resample_data" : False,
        "correct_acceptance" : True,
        "run_ibu": True,
    }

    if args.observables:
        common_cfg["observables"] = args.observables

    if args.config_string:
        try:
            jcfg = json.loads(args.config_string)
            common_cfg.update(jcfg)
        except json.decoder.JSONDecodeError:
            print("ERROR Cannot parse the extra config string: {args.config_string}")

    createRun2Config(
        args.sample_dir,
        outname_config = args.config_name,
        output_top_dir = args.result_dir,
        subcampaigns = args.subcampaigns,
        run_list = args.run_list,
        systematics_keywords = args.systematics_keywords,
        external_reweights = args.external_reweights,
        common_cfg = common_cfg,
        batch_job = args.batch_job
        )