#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import numpy as np
import h5py

import util
from datahandler import getDataHandler
import histogramming as myhu
import plotter

util.configRootLogger()
logger = logging.getLogger("reweight_sample_binned")

def reweight_samples_binned(**parsed_args):

    # Prepare logging
    logger.setLevel(logging.DEBUG if parsed_args['verbose']>0 else logging.INFO)

    logger.info(f"Hostname: {os.uname().nodename}")

    if not os.path.isdir(parsed_args['outputdir']):
        os.makedirs(parsed_args['outputdir'])

    ######
    # variables
    obsCfg_d = util.read_dict_from_json(parsed_args['observable_config'])
    # FIXME hardcode 'branch_det' for reco-level variables only for now
    varnames = [ obsCfg_d[key]['branch_det'] for key in parsed_args['observables'] ]

    ######
    # Load input data files
    logger.info(f"Load target samples: {' '.join(parsed_args['target'])}")
    dh_target = getDataHandler(parsed_args['target'], varnames)
    logger.info(f"Total number of events in target: {len(dh_target)}")

    logger.info(f"Load source samples: {' '.join(parsed_args['source'])}")
    dh_source = getDataHandler(parsed_args['source'], varnames)
    logger.info(f"Total number of events in source: {len(dh_source)}")

    if parsed_args['background']:
        logger.info(f"Load background samples: {' '.join(parsed_args['background'])}")
        dh_bkg = getDataHandler(parsed_args['background'], varnames)
        logger.info(f"Total number of events in background: {len(dh_bkg)}")
    else:
        dh_bkg = None

    #######
    # TODO multiple observables
    # For now only deal with one
    if len(varnames) > 1:
        logger.critical("Only one variable for now")
        return
    else:
        # binning config
        binCfg_d = util.get_bins_dict(parsed_args['binning_config'])

        obs = parsed_args['observables'][0]
        vname = varnames[0]

        binedges = binCfg_d[obs]
        absV = '_abs' in obs

        hist_target = dh_target.get_histogram(vname, binedges, absoluteValue=absV)
        if dh_bkg:
            hist_bkg = dh_bkg.get_histogram(vname, binedges, absoluteValue=absV)
            hist_target += (-1 * hist_bkg)

        hist_source = dh_source.get_histogram(vname, binedges, absoluteValue=absV)

        # ratios
        hist_ratio = myhu.divide(hist_target, hist_source)

        values_ratio = hist_ratio.values()
        errors_ratio = np.sqrt(hist_ratio.variances())
        bincenters = hist_ratio.axes.centers[0]

        # fit
        poly_fitted = np.polynomial.Polynomial.fit(
            x = bincenters,
            y = values_ratio,
            deg = parsed_args['polynomial_degree'],
            w = 1 / errors_ratio)

        file_rw = h5py.File(os.path.join(parsed_args['outputdir'], "reweights.h5"), "w")

        # create datasets
        # new event weights
        weights_rw = file_rw.create_dataset(
            parsed_args["weight_name"], data = dh_source.get_weights(valid_only=False)
            )

        weights_rw[:] *= poly_fitted(dh_source.get_arrays(vname, valid_only=False))

        if parsed_args['plot_verbosity'] > 0:
            # plot ratio and fit function
            plotter.plot_histogram_and_function(
                figname = os.path.join(parsed_args['outputdir'], f"fit_{obs}"),
                histogram = hist_ratio,
                draw_option_histogram = {'color':'black', 'histtype':'errorbar'},
                function = poly_fitted,
                draw_option_function = {'color':'tab:red', 'linestyle':'-'},
                xlabel = obsCfg_d[obs]['xlabel'],
                ylabel = "Target / Source",
            )

            # reweighted source vs target
            target_style = {'color':'black', 'label':'Target', 'histtype':'step'}
            source_style = {'color':'blue', 'label':'Source', 'histtype':'step'}
            reweighted_style = {'color':'tab:red', 'label':'Reweighted', 'histtype':'step'}

            hist_source_rw = dh_source.get_histogram(vname, binedges, weights=weights_rw[dh_source.pass_reco], absoluteValue=absV)

            plotter.plot_histograms_and_ratios(
                figname = os.path.join(parsed_args['outputdir'], obs),
                hists_numerator = [hist_source_rw, hist_source],
                hist_denominator = hist_target,
                draw_options_numerator = [reweighted_style, source_style],
                draw_option_denominator = target_style,
                xlabel = obsCfg_d[obs]['xlabel'],
                ylabel = obsCfg_d[obs]['ylabel'],
                ylabel_ratio = "Ratio to Target"
            )

        file_rw.close()

def getArgsParser(arguments_list=None, print_help=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--target',
                        required=True, nargs='+', type=str, action=util.ParseEnvVar,
                        help="Target sample to reweight to as a list of root files")
    parser.add_argument('-s', '--source',
                        required=True, nargs='+', type=str, action=util.ParseEnvVar,
                        help="Source sample to be reweighted")
    parser.add_argument('-b', '--background',
                        nargs='+', type=str, action=util.ParseEnvVar,
                        help="Background contributions to be subtracted from target")
    parser.add_argument('--observables', nargs='+', type=str,
                        default=['th_pt','th_y','tl_pt','tl_y','mtt','ptt','ytt'],
                        help="List of observables used to train the classifier")
    parser.add_argument('--observable-config', action=util.ParseEnvVar,
                        default='${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json',
                        help="JSON configurations for observables")
    parser.add_argument('-o', '--outputdir',
                        type=str, default='.', action=util.ParseEnvVar,
                        help="Output directory")
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help="If True, set logging level to DEBUG else INFO")
    parser.add_argument('-p', '--plot-verbosity', action='count', default=0,
                        help="Plot verbosity level")
    #parser.add_argument('-w', '--weights-file', type=str,
    #                    help="File path to the weights file. If provided, load weights from this file and skip training")
    parser.add_argument("--binning-config", type=str, action=util.ParseEnvVar,
                        default='${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json',
                        help="Path to the binning config file for variables.")
    parser.add_argument("--weight-name", type=str, default="normalized_weight",
                        help="Weight array name stored in the output file")
    parser.add_argument("-d", "--polynomial-degree", type=int, default=4,
                        help="Degree of the fitting polynomial")

    if print_help:
        parser.print_help()

    args = parser.parse_args(arguments_list)

    return args

if __name__ == "__main__":

    try:
        args = getArgsParser()
    except Exception as e:
        sys.exit(f"Config Failure: {e}")

    # reweight
    reweight_samples_binned(**vars(args))