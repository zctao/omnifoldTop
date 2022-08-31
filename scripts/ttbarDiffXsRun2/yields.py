import os
import numpy as np

import util
from datahandler_root import DataHandlerROOT

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--run-config', type=str,
                    default='configs/run/runCfg_run2_ljets_nominal.json',
                    help="Run config to read input samples")
parser.add_argument('--observables', nargs='+', default=['mtt'],
                    help="Observables to load from the files")

args = parser.parse_args()

# get input file lists from run config file
runCfg_d = util.read_dict_from_json(args.run_config)

# observable config
obsConfig_d = util.read_dict_from_json(runCfg_d['observable_config'])

# reco level variable names
varnames_reco = [obsConfig_d[ob]['branch_det'] for ob in args.observables]

# truth level variable names
varnames_truth = [obsConfig_d[ob]['branch_mc'] for ob in args.observables]

# Data handlers
print("Load data")
filepaths_obs = runCfg_d['data']

handle_obs = DataHandlerROOT(
    filepaths_obs,
    varnames_reco,
    treename_reco = 'reco',
    treename_truth = None
    )

print(f"Number of data events: {len(handle_obs)}")
print(f"Total weights of data events: {handle_obs.sum_weights():.2f}")

print("Load signal samples")
filepaths_sig = runCfg_d['signal']

handle_sig = DataHandlerROOT(
    filepaths_sig,
    varnames_reco,
    varnames_truth,
    treename_reco = 'reco',
    treename_truth = 'parton',
    dummy_value = -99,
    )

print(f"Number of signal events: {len(handle_sig)}")
print(f"Total weights of signal events: {handle_sig.sum_weights():.2f}")

print(f"Number of signal events passed truth cuts: {np.count_nonzero(handle_sig.pass_truth)}")
print(f"Total truth level weights of signal events: {handle_sig.sum_weights(reco_level=False):.2f}")

print("Load background samples")
filepaths_bkg = runCfg_d['background']

handle_bkg = DataHandlerROOT(
    filepaths_bkg,
    varnames_reco,
    treename_reco = 'reco',
    treename_truth = None
    )

print(f"Number of background events: {len(handle_bkg)}")
print(f"Total weights of background events: {handle_bkg.sum_weights():.2f}")

# loop over every component
backgrounds = ['fakes', 'Wjets', 'Zjets', 'singleTop', 'ttH', 'ttV', 'VV']

nfiles_bkg = 0
nevents_bkg = 0
yields_bkg = 0
for bkg in backgrounds:
    filelist_bkg = [fname for fname in filepaths_bkg if bkg in fname]

    if not filelist_bkg:
        continue

    handle_bkg_i = DataHandlerROOT(
        filelist_bkg,
        varnames_reco,
        treename_reco = 'reco',
        treename_truth = None
        )

    print(f" {bkg}: N_events = {len(handle_bkg_i)} yields = {handle_bkg_i.sum_weights():.2f}")
    nfiles_bkg += len(filelist_bkg)
    nevents_bkg += len(handle_bkg_i)
    yields_bkg += handle_bkg_i.sum_weights()

assert(nfiles_bkg == len(filepaths_bkg))
assert(nevents_bkg == len(handle_bkg))
assert(np.isclose(yields_bkg, handle_bkg.sum_weights()))
