#!/usr/bin/env python3
import logging
from packaging import version

import numpy as np
import tensorflow as tf

import energyflow as ef
import energyflow.archs

from util import prepare_data_multifold, getLogger
logger = getLogger('Unfold')

from omnifold_wbkg import OmniFoldwBkg
from observables import observable_dict

def load_dataset(file_name, array_name='arr_0'):
    """
    Load and return a structured numpy array from npz file
    """
    npzfile = np.load(file_name, allow_pickle=True, encoding='bytes')
    data = npzfile[array_name]
    npzfile.close()
    return data

def unfold(**parsed_args):

    #################
    # Load and prepare datasets
    #################

    #observables_all = list(set().union(parsed_args['observables'], parsed_args['observables_train']))
    logger.info("Observables used for training: {}".format(', '.join(parsed_args['observables_train'])))
    logger.info("Observables to unfold: {}".format(', '.join(parsed_args['observables'])))

    # collision data
    fname_obs = parsed_args['data']
    data_obs = load_dataset(fname_obs)

    # signal MC
    fname_mc_sig = parsed_args['signal']
    data_mc_sig = load_dataset(fname_mc_sig)

    # background MC
    fname_mc_bkg = parsed_args['background']
    data_mc_bkg = load_dataset(fname_mc_bkg) if fname_mc_bkg is not None else None

    # detector level variable names for training
    vars_det = [ observable_dict[key]['branch_det'] for key in parsed_args['observables_train'] ]
    # truth level variable names for training
    vars_mc = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables_train'] ]
    # weight name
    wname = parsed_args['weight']
    wname_mc = parsed_args['weight_mc']

    #####################
    # Start unfolding
    unfolder = OmniFoldwBkg(vars_det, vars_mc, wname, wname_mc, it=3, outdir=parsed_args['outputdir'])

    ##################
    # preprocess_data
    # detector level (step 1 reweighting)
    unfolder.preprocess_det(data_obs, data_mc_sig, data_mc_bkg)
    # mc truth (step 2 reweighting)
    # only signal simulation is of interest here
    unfolder.preprocess_gen(data_mc_sig)

    ##################
    # Models
    # FIXME

    # step 1 model and arguments
    model_det = ef.archs.DNN
    args_det = {'input_dim': len(vars_det), 'dense_sizes': [100, 100],
                'patience': 10, 'filepath': 'Step1_{}',
                'save_weights_only': False,
                'modelcheck_opts': {'save_best_only': True, 'verbose':1}}

    # step 2 model and arguments
    model_sim = ef.archs.DNN
    args_sim = {'input_dim': len(vars_mc), 'dense_sizes': [100, 100],
                'patience': 10, 'filepath': 'Step2_{}',
                'save_weights_only': False,
                'modelcheck_opts': {'save_best_only': True, 'verbose':1}}

    # training parameters
    fitargs = {'batch_size': 500, 'epochs': 3, 'verbose': 1}

    ##################
    # Unfold
    unfolder.omnifold((model_det, args_det), (model_sim, args_sim), fitargs, val=0.2)

    ##################
    # Show results
    subObs_dict = { var:observable_dict[var] for var in parsed_args['observables']}
    unfolder.results(subObs_dict, data_obs, data_mc_sig, data_mc_bkg, truth_known=parsed_args['closure_test'], normalize=parsed_args['normalize'])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--observables_train', nargs='+',
                        choices=observable_dict.keys(),
                        default=['mtt', 'ptt', 'ytt', 'ystar', 'yboost', 'dphi', 'Ht'],
                        help="List of observables to use in training.")
    parser.add_argument('--observables', nargs='+',
                        choices=observable_dict.keys(),
                        default=observable_dict.keys(),
                        help="List of observables to unfold")
    parser.add_argument('-d', '--data', required=True,
                        type=str,
                        help="Observed data npz file name")
    parser.add_argument('-s', '--signal', required=True,
                        type=str,
                        help="Signal MC npz file name")
    parser.add_argument('-b', '--background',
                        type=str,
                        help="Background MC npz file name")
    parser.add_argument('-o', '--outputdir',
                        default='./output',
                        help="Directory for storing outputs")
    parser.add_argument('--ibu',
                        action='store_true',
                        help="Do iterative bayesian unfolding")
    parser.add_argument('-t', '--closure_test',
                        action='store_true',
                        help="Is a closure test")
    parser.add_argument('-n', '--normalize',
                        action='store_true',
                        help="Normalize the distributions when plotting the result")
    parser.add_argument('--weight', type=str, default='w',
                        help="name of event weight")
    parser.add_argument('--weight_mc', type=str, default='wTruth',
                        help="name of MC weight")
    parser.add_argument('-m', '--multiclass',
                        action='store_true',
                        help="If set, background MC is treated as a separate class")
    parser.add_argument('-v', '--verbose',
                        action='count', default=0,
                        help="Verbosity level")
    parser.add_argument('-g', '--gpu',
                        type=int, choices=[0, 1], default=1,
                        help="Manually select one of the GPUs to run")

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

    with tf.device('/GPU:{}'.format(args.gpu)):
        unfold(**vars(args))
