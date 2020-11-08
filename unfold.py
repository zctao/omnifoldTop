#!/usr/bin/env python3
import os
import glob
import logging
from packaging import version

import tensorflow as tf

from util import load_dataset, getLogger
from plotting import plot_train_log
logger = getLogger('Unfold')

from omnifold_wbkg import OmniFoldwBkg
from omnifold_wbkg import OmniFold_subHist, OmniFold_negW, OmniFold_multi
from observables import observable_dict

import time
import tracemalloc

def unfold(**parsed_args):

    tracemalloc.start()

    #################
    # Load and prepare datasets
    #################

    logger.info("Observables used for training: {}".format(', '.join(parsed_args['observables_train'])))
    parsed_args['observables'] = list(set().union(parsed_args['observables'], parsed_args['observables_train']))
    logger.info("Observables to unfold: {}".format(', '.join(parsed_args['observables'])))

    # weight name
    wname = parsed_args['weight']
    wname_mc = parsed_args['weight_mc']

    # collision data
    logger.info("Loading datasets")
    t_data_start = time.time()
    fnames_obs = parsed_args['data']
    data_obs = load_dataset(fnames_obs, weight_columns=[wname,wname_mc])

    # signal MC
    fnames_mc_sig = parsed_args['signal']
    data_mc_sig = load_dataset(fnames_mc_sig, weight_columns=[wname,wname_mc])

    # background MC
    fnames_mc_bkg = parsed_args['background']
    data_mc_bkg = load_dataset(fnames_mc_bkg, weight_columns=[wname,wname_mc]) if fnames_mc_bkg is not None else None

    t_data_finish = time.time()
    logger.info("Loading dataset took {:.2f} seconds".format(t_data_finish-t_data_start))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    # detector level variable names for training
    vars_det = [ observable_dict[key]['branch_det'] for key in parsed_args['observables_train'] ]
    # truth level variable names for training
    vars_mc = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables_train'] ]

    #####################
    # Start unfolding
    if parsed_args['background_mode'] == 'subHist':
        unfoler = OmniFold_subHist(vars_det, vars_mc, wname, wname_mc, it=parsed_args['iterations'], outdir=parsed_args['outputdir'], binned_rw=parsed_args['alt_rw'])
    elif parsed_args['background_mode'] == 'negW':
        unfolder = OmniFold_negW(vars_det, vars_mc, wname, wname_mc, it=parsed_args['iterations'], outdir=parsed_args['outputdir'], binned_rw=parsed_args['alt_rw'])
    elif parsed_args['background_mode'] == 'multiClass':
        unfolder = OmniFold_multi(vars_det, vars_mc, wname, wname_mc, it=parsed_args['iterations'], outdir=parsed_args['outputdir'], binned_rw=parsed_args['alt_rw'])
    else:
        unfolder = OmniFoldwBkg(vars_det, vars_mc, wname, wname_mc, it=parsed_args['iterations'], outdir=parsed_args['outputdir'], binned_rw=parsed_args['alt_rw'])

    ##################
    # prepare input data
    logger.info("Preprocessing data")
    t_prep_start = time.time()
    unfolder.prepare_inputs(data_obs, data_mc_sig, data_mc_bkg, standardize=True,
                            plot_corr=parsed_args['plot_correlations'],
                            truth_known=parsed_args['truth_known'],
                            reweight_type=parsed_args['reweight_data'])
    t_prep_finish = time.time()

    logger.info("Preprocessnig data took {:.2f} seconds".format(t_prep_finish-t_prep_start))
    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    ##################
    if parsed_args['unfolded_weights']:
        # use the provided weights directly
        logger.info("Skipping training")
        logger.info("Reading weights from file {}".format(parsed_args['unfolded_weights']))
        unfolder.set_weights_from_file(parsed_args['unfolded_weights'])
    else:
        # training parameters
        fitargs = {'batch_size': 500, 'epochs': 100, 'verbose': 1}

        ##################
        # Unfold
        logger.info("Start unfolding")

        t_unfold_start = time.time()
        unfolder.unfold(fitargs, val=0.2)
        t_unfold_finish = time.time()

        logger.info("Done!")
        logger.info("Unfolding took {:.2f} seconds".format(t_unfold_finish-t_unfold_start))
        mcurrent, mpeak = tracemalloc.get_traced_memory()
        logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    ##################
    # Show results
    resObs_dict = { var:observable_dict[var] for var in parsed_args['observables']}
    t_result_start = time.time()
    unfolder.results(resObs_dict, data_obs, data_mc_sig, data_mc_bkg, binning_config=parsed_args['binning_config'], truth_known=parsed_args['truth_known'], normalize=parsed_args['normalize'])
    t_result_finish = time.time()

    logger.info("Getting results took {:.2f} seconds (average {:.2f} seconds per variable)".format(t_result_finish-t_result_start, (t_result_finish-t_result_start)/len(resObs_dict)))
    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    # Plot training log
    logger.info("Plot training history")
    for csvfile in glob.glob(os.path.join(parsed_args['outputdir'], 'Models', '*.csv')):
        logger.info("  Plot training log {}".format(csvfile))
        plot_train_log(csvfile)

    tracemalloc.stop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--observables-train', dest='observables_train',
                        nargs='+', choices=observable_dict.keys(),
                        default=['th_pt', 'th_y', 'th_phi', 'th_m', 'tl_pt', 'tl_y', 'tl_phi', 'tl_m'],
                        help="List of observables to use in training.")
    parser.add_argument('--observables',
                        nargs='+', choices=observable_dict.keys(),
                        default=['mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht', 'th_pt', 'th_y', 'th_eta', 'th_phi', 'th_m', 'th_e', 'th_pout', 'tl_pt', 'tl_y', 'tl_eta', 'tl_phi', 'tl_m', 'tl_e', 'tl_pout'],
                        help="List of observables to unfold")
    parser.add_argument('-d', '--data', required=True, nargs='+',
                        type=str,
                        help="Observed data npz file names")
    parser.add_argument('-s', '--signal', required=True, nargs='+',
                        type=str,
                        help="Signal MC npz file names")
    parser.add_argument('-b', '--background', nargs='+',
                        type=str,
                        help="Background MC npz file names")
    parser.add_argument('-o', '--outputdir',
                        default='./output',
                        help="Directory for storing outputs")
    parser.add_argument('-t', '--truth-known', dest='truth_known',
                        action='store_true',
                        help="MC truth is known for 'data' sample")
    parser.add_argument('-n', '--normalize',
                        action='store_true',
                        help="Normalize the distributions when plotting the result")
    parser.add_argument('-c', '--plot-correlations', dest='plot_correlations',
                        action='store_true',
                        help="Plot pairwise correlations of training variables")
    parser.add_argument('-i', '--iterations', type=int, default=5,
                        help="Numbers of iterations for unfolding")
    parser.add_argument('--weight', default='w',
                        help="name of event weight")
    parser.add_argument('--weight-mc', dest='weight_mc', default='wTruth',
                        help="name of MC weight")
    parser.add_argument('--alt-rw', dest='alt_rw',
                        action='store_true',
                        help="Use alternative reweighting if true")
    parser.add_argument('-m', '--background-mode', dest='background_mode',
                        choices=['default', 'subHist', 'negW', 'multiClass'],
                        default='default', help="Background mode")
    parser.add_argument('-r', '--reweight-data', dest='reweight_data',
                        choices=['linear_th_pt', 'gaussian_bump', 'gaussian_tail'], default=None,
                        help="Reweight strategy of the input spectrum for stress tests")
    parser.add_argument('-v', '--verbose',
                        action='count', default=0,
                        help="Verbosity level")
    parser.add_argument('-g', '--gpu',
                        type=int, choices=[0, 1], default=None,
                        help="Manually select one of the GPUs to run")
    parser.add_argument('--unfolded-weights', dest='unfolded_weights',
                        default='', type=str,
                        help="File name of the stored weights after unfolding. If provided, skip training/unfolding and use the weights directly to show results.")
    parser.add_argument('--binning-config', dest='binning_config',
                        default='', type=str,
                        help="Binning config file for variables")

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)

    #################
    assert(version.parse(tf.__version__) >= version.parse('2.0.0'))
    # tensorflow configuration
    # device placement
    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(args.verbose > 0)

    # limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        logger.error("No GPU found!")
        raise RuntimeError("No GPU found!")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    if not os.path.isdir(args.outputdir):
        logger.info("Create output directory {}".format(args.outputdir))
        os.makedirs(args.outputdir)

    if args.gpu is not None:
        tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    #with tf.device('/GPU:{}'.format(args.gpu)):
    unfold(**vars(args))
