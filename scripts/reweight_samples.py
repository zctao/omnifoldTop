#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
import glob
import h5py

import util
import modelUtils
from OmniFoldTTbar import getDataHandler
from nnreweighter import train_and_reweight
import plotter

def reweight_samples(**parsed_args):

    # Prepare logging
    util.configRootLogger()
    logger = logging.getLogger("nnreweight")
    logger.setLevel(logging.DEBUG if parsed_args['verbose']>0 else logging.INFO)

    logger.info(f"Hostname: {os.uname().nodename}")

    if not os.path.isdir(parsed_args['outputdir']):
        os.makedirs(parsed_args['outputdir'])

    # Print arguments to logger
    for argkey, argvalue in sorted(parsed_args.items()):
        if argvalue is None:
            continue
        logger.info(f"Argument {argkey}: {argvalue}")

    # store arguments to json file
    argname = 'arguments.json'
    fname_args = os.path.join(parsed_args['outputdir'], argname)
    logger.info(f"Write arguments to file {fname_args}")
    util.write_dict_to_json(parsed_args, fname_args)

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

    ######
    # feature arrays
    logger.info("Cconstruct feature and label arrays")
    target_arr = dh_target[varnames]
    source_arr = dh_source[varnames]

    # preprocessing
    X = np.concatenate([target_arr, source_arr])
    xmean = np.mean(X, axis=0)
    xstd = np.std(X, axis=0)

    target_arr -= xmean
    target_arr /= xstd

    source_arr -= xmean
    source_arr /= xstd

    # event weights
    logger.info("Prepare weights")
    w_target = dh_target.get_weights()

    if parsed_args['normalize_weight']:
        logger.info(f"Rescale source weights to be consistant with the target weights")
        sumw_target = dh_target.sum_weights()
        sumw_source = dh_source.sum_weights()
        dh_source.rescale_weights(sumw_target/sumw_source)

    w_source = dh_source.get_weights()

    logger.debug(f"total weights (target): {dh_target.sum_weights()}")
    logger.debug(f"total weights (source): {dh_source.sum_weights()}")

    ##
    # Plot input distributions
    if parsed_args['plot_verbosity'] > 1:
        logger.info("Plot input distributions")
        for obs, varr_1, varr_0 in zip(parsed_args['observables'], target_arr.T, source_arr.T):
            logger.debug(f" Plot {obs}")
            plotter.plot_hist(
                os.path.join(parsed_args['outputdir'], f"input_{obs}"),
                [varr_1, varr_0],
                weight_arrs = [w_target, w_source],
                labels = ['Target', 'Source'],
                xlabel = obsCfg_d[obs]['xlabel'],
                title = "Training inputs"
            )

        logger.info("Plot input weights")
        plotter.plot_hist(
            os.path.join(parsed_args['outputdir'], f"input_w"),
            [w_target, w_source],
            labels = ['Target', 'Source'],
            xlabel = 'w',
            title = "Training inputs"
        )

    if parsed_args['weights_file']:
        file_rw = h5py.File(parsed_args['weights_file'], 'r')
        rw = file_rw['rw']
    else:
        modelUtils.n_models_in_parallel = 1

        model_dir = os.path.join(parsed_args['outputdir'],"Models")
        if not os.path.isdir(model_dir):
            logger.debug(f"Create directory {model_dir}")
            os.makedirs(model_dir)

        file_rw = h5py.File(os.path.join(parsed_args['outputdir'], "reweights.h5"), "w")

        rw = file_rw.create_dataset("rw", shape=(len(source_arr),))
        rw[:] = train_and_reweight(
            X_target = target_arr,
            X_source = source_arr,
            w_target = w_target,
            w_source = w_source,
            model_type = parsed_args["model_type"],
            model_filepath_save = os.path.join(model_dir, "model"),
            nsplit_cv = 2,
            batch_size = parsed_args['batch_size'],
            epochs = 100,
            calibrate = parsed_args['reweight_method']=='histogram',
            verbose= parsed_args['verbose'],
            plot = parsed_args['plot_verbosity'] > 0
        )[0]
        # [0] because modelUtils.n_models_in_parallel = 1

    if parsed_args['plot_verbosity'] > 0:
        # plot train history
        logger.info("Plot training history")
        for csvfile in glob.glob(os.path.join(model_dir, '*.csv')):
            plotter.plot_train_log(csvfile)

        # plot distributions of the weights
        logger.info("Plot the weight distribution")
        plotter.plot_LR_distr(
            os.path.join(parsed_args['outputdir'], "rw_distr"),
            [rw[:]],
            nbins = 10 + int(len(w_source) ** (1. / 3.))
            )

        # binning config
        binCfg_d = util.get_bins_dict(parsed_args['binning_config'])

        target_style = {'color':'black', 'label':'Target', 'histtype':'step'}
        source_style = {'color':'blue', 'label':'Source', 'histtype':'step'}
        reweighted_style = {'color':'tab:red', 'label':'Reweighted', 'histtype':'step'}

        # compare reweighted source with target
        for obs, vname in zip(parsed_args['observables'], varnames):
            logger.info(f"Plot observable {obs}")

            binedges = binCfg_d[obs]
            absV = '_abs' in obs

            hist_target = dh_target.get_histogram(vname, binedges, absoluteValue=absV)
            hist_source = dh_source.get_histogram(vname, binedges, absoluteValue=absV)
            hist_source_rw = dh_source.get_histogram(vname, binedges, weights=w_source * rw[:], absoluteValue=absV)

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

    parser.add_argument('-t', '--target', required=True, nargs='+', type=str,
                        help="Target sample to reweight to as a list of root files")
    parser.add_argument('-s', '--source', required=True, nargs='+', type=str,
                        help="Source sample to be reweighted")
    parser.add_argument('--observables', nargs='+', type=str,
                        default=['th_pt','th_y','tl_pt','tl_y','mtt','ptt','ytt'],
                        help="List of observables used to train the classifier")
    parser.add_argument('--observable-config',
                        default='configs/observables/vars_ttbardiffXs_pseudotop.json',
                        help="JSON configurations for observables")
    parser.add_argument('-o', '--outputdir', type=str, default='.',
                        help="Output directory")
    parser.add_argument('-n', '--normalize-weight', action='store_true',
                        help="If True, renormalize source weights to target weights")
    parser.add_argument('-m', '--model-type', type=str, default='dense_100x3',
                        help="Type of NN")
    parser.add_argument('--batch-size', type=int, default=20000,
                        help="Training batch size")
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help="If True, set logging level to DEBUG else INFO")
    parser.add_argument('-r', '--reweight-method', choices=['direct','histogram'],
                        default='histogram', help="Method to reweight source to target")
    parser.add_argument('-p', '--plot-verbosity', action='count', default=0,
                        help="Plot verbosity level")
    parser.add_argument('-w', '--weights-file', type=str,
                        help="File path to the weights file. If provided, load weights from this file and skip training")
    parser.add_argument("--binning-config", type=str,
                        default='configs/binning/bins_ttdiffxs.json',
                        help="Path to the binning config file for variables.")

    if print_help:
        parser.print_help()

    args = parser.parse_args(arguments_list)

    return args

if __name__ == "__main__":

    try:
        args = getArgsParser()
    except Exception as e:
        sys.exit(f"Config Failure: {e}")

    # unfold
    reweight_samples(**vars(args))