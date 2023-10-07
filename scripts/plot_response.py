import os
import logging
import argparse

import util
from datahandler import getDataHandler
import histogramming as myhu
import plotter

parser = argparse.ArgumentParser()

parser.add_argument('samples', nargs='+', type=str,
                    help="List of sample files")
parser.add_argument('--observables', nargs='+', type=str,
                    default=['th_pt','th_y','tl_pt','tl_y','mtt','ptt','ytt'],
                    help="List of observables used to train the classifier")
parser.add_argument('--observable-config',
                    default='configs/observables/vars_ttbardiffXs_pseudotop.json',
                    help="JSON configurations for observables")
parser.add_argument('-o', '--outputdir', type=str, default='.',
                    help="Output directory")
parser.add_argument("--binning-config", type=str,
                    default='configs/binning/bins_ttdiffxs.json',
                    help="Path to the binning config file for variables.")
parser.add_argument('--match-dR', type=float,
                    help="Require dR between the reco and truth tops less than the provided value")
parser.add_argument('--raw-entries', action='store_true',
                    help="If True, do not normalize truth bins")

args = parser.parse_args()

# logger
util.configRootLogger()
logger = logging.getLogger("plot_response")

# variables
obsCfg_d = util.read_dict_from_json(args.observable_config)
varnames_reco = [obsCfg_d[obs]['branch_det'] for obs in args.observables]
varnames_truth = [obsCfg_d[obs]['branch_mc'] for obs in args.observables]

# binning
binCfg_d = util.get_bins_dict(args.binning_config)

# Load samples
logger.info(f"Load samples from {args.samples}")
dh = getDataHandler(
    args.samples, varnames_reco, varnames_truth,
    match_dR = args.match_dR,
    plot_dir = args.outputdir
    )

response_obs_d = {}

for obs, vname_reco, vname_truth in zip(args.observables, varnames_reco, varnames_truth):
    bins_reco = binCfg_d[obs]
    bins_truth = binCfg_d[obs]

    response_obs_d[obs] = dh.get_response(
        vname_reco, vname_truth,
        bins_reco, bins_truth,
        absoluteValue = '_abs' in obs,
        normalize_truthbins = not args.raw_entries
    )

# output directory
if not os.path.isdir(args.outputdir):
    os.makedirs(args.outputdir)

fpath_root = os.path.join(args.outputdir, 'responses.root')
logger.info(f"Write responses to file {fpath_root}")
myhu.write_histograms_dict_to_file(response_obs_d, fpath_root)

# plot
for obs in response_obs_d:
    figname = os.path.join(args.outputdir, f"response_{obs}")
    logger.info(f"Plot response for {obs}: {figname}")
    plotter.plot_response(figname, response_obs_d[obs], obs, percentage=not args.raw_entries, title='')