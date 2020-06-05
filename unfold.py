#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import energyflow as ef
import energyflow.archs

from util import prepare_data_multifold
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--observables', nargs='+', choices=observable_dict.keys(), default=['mtt', 'ptt', 'ytt', 'ystar', 'yboost', 'dphi', 'Ht'], help="List of observables to unfold")
    parser.add_argument('--observables_train', nargs='+', choices=observable_dict.keys(), help="List of observables to use in training.")
    parser.add_argument('-i', '--inputdir', default = './input', help="Directory of input data")
    parser.add_argument('-o', '--outputdir', default='./output', help="Directory for storing outputs")
    parser.add_argument('-d', '--data', required=True, help="Observed data npz file name")
    parser.add_argument('-s', '--signal', required=True, help="Signal MC npz file name")
    parser.add_argument('-b', '--background', help="Background MC npz file name")
    parser.add_argument('--ibu', action='store_true', help="Do iterative bayesian unfolding")
    parser.add_argument('-t', '--closure_test', action='store_true', help="Is a closure test")
    parser.add_argument('-m', '--multiclass', action='store_true', help="If set, background MC is treated as a separate class")
    parser.add_argument('-v', '--verbose', action='count', help="Verbosity")
    
    args = parser.parse_args()

    #################
    # GPU configuration
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
    
    #################
    # Load and prepare datasets
    #################

    if args.observables_train is None:
        args.observables_train = args.observables
        
    observables_all = list(set().union(args.observables, args.observables_train))

    print("Observables: ", args.observables)

    # collision data
    fname_obs = args.inputdir.strip('/')+'/'+args.data
    data_obs = load_dataset(fname_obs)
    
    # signal MC
    fname_mc_sig = args.inputdir.strip('/')+'/'+args.signal
    data_mc_sig = load_dataset(fname_mc_sig)

    # background MC
    fname_mc_bkg = args.inputdir.strip('/')+'/'+args.background if args.background is not None else None
    data_mc_bkg = load_dataset(fname_mc_bkg) if fname_mc_bkg is not None else None
    
    # detector level variable names
    vars_det = [ observable_dict[key]['branch_det'] for key in observables_all ]
    # truth level variable names
    vars_mc = [ observable_dict[key]['branch_mc'] for key in observables_all ]
    # weight name
    wname = 'w'

    #####################
    # Start unfolding
    unfolder = OmniFoldwBkg(vars_det, vars_mc, wname, it=3, outdir=args.outputdir)

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
    unfold_ws = unfolder.omnifold((model_det, args_det), (model_sim, args_sim), fitargs, val=0.2)

    ##################
    # Show results
    unfolder.results(observable_dict, data_obs, data_mc_sig, data_mc_bkg, truth_known=args.closure_test)


