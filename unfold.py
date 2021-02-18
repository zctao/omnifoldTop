#!/usr/bin/env python3
import os
import sys
import logging
from packaging import version
import tensorflow as tf
import time
import tracemalloc

from datahandler import DataHandler
from omnifoldwbkg import OmniFoldwBkg
from ibu import IBU
from util import read_dict_from_json, get_bins
import logging

def configRootLogger(filename=None, level=logging.INFO):
    msgfmt = '%(asctime)s %(levelname)-7s %(name)-15s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=level, format=msgfmt, datefmt=datefmt)
    if filename:
        # check if the directory exists
        dirname = os.path.dirname(filename)
        nodir = not os.path.isdir(dirname)
        if nodir:
            os.makedirs(dirname)

        fhdr = logging.FileHandler(filename, mode='w')
        fhdr.setFormatter(logging.Formatter(msgfmt, datefmt))
        logging.getLogger().addHandler(fhdr)

        if nodir:
            logging.info("Create directory {}".format(dirname))

def configGPUs(gpu=None, limit_gpu_mem=False, verbose=0):
    assert(version.parse(tf.__version__) >= version.parse('2.0.0'))
    # tensorflow configuration
    # device placement
    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(verbose > 0)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        logger.error("No GPU found!")
        raise RuntimeError("No GPU found!")

    if gpu is not None:
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    # limit GPU memory growth
    if limit_gpu_mem:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g,True)

def expandFilePath(filepath):
    filepath = filepath.strip()
    if not os.path.isfile(filepath):
        # try expanding the path in the directory of this file
        src_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(src_dir, filepath)

    if os.path.isfile(filepath):
        return os.path.abspath(filepath)
    else:
        return None

def unfold(**parsed_args):
    tracemalloc.start()

    logger = logging.getLogger('Unfold')

    # log arguments
    for argkey, argvalue in sorted(parsed_args.items()):
        if argvalue is None:
            continue
        logger.info('Argument {}: {}'.format(argkey, argvalue))

    #################
    # Variables
    #################
    # Dictionary for observable configurations
    observable_dict = read_dict_from_json(parsed_args['observable_config'])

    logger.info("Observables used in training: {}".format(', '.join(parsed_args['observables_train'])))
    parsed_args['observables'] = list(set().union(parsed_args['observables'], parsed_args['observables_train']))
    logger.info("Observables to unfold: {}".format(', '.join(parsed_args['observables'])))

    # all variable names at detector level
    vars_det_all = [ observable_dict[key]['branch_det'] for key in parsed_args['observables'] ]
    # all variable names at truth level
    vars_mc_all = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables'] ]

    # detector-level variable names for training
    vars_det_train = [ observable_dict[key]['branch_det'] for key in parsed_args['observables_train'] ]
    # truth-level variable names for training
    vars_mc_train = [ observable_dict[key]['branch_mc'] for key in parsed_args['observables_train'] ] 

    # weight name
    wname = parsed_args['weight']

    #################
    # Load data
    #################
    logger.info("Loading datasets")
    t_data_start = time.time()

    # collision data
    fnames_obs = parsed_args['data']
    logger.info("Data files: {}".format(' '.join(fnames_obs)))
    data_obs = DataHandler(fnames_obs, wname,
                            truth_known=parsed_args['truth_known'],
                            variable_names = vars_det_all+vars_mc_all)
                            #vars_dict = observable_dict

    # signal simulation
    fnames_sig = parsed_args['signal']
    logger.info("Simulation files: {}".format(' '.join(fnames_sig)))
    data_sig = DataHandler(fnames_sig, wname, variable_names = vars_det_all+vars_mc_all) #vars_dict = observable_dict

    # background simulation
    fnames_bkg = parsed_args['background']
    if fnames_bkg is not None:
        logger.info("Background simulation files: {}".format(' '.join(fnames_bkg)))
        data_bkg =  DataHandler(fnames_bkg, wname, variable_names = vars_det_all+vars_mc_all)
    else:
        data_bkg = None

    t_data_done = time.time()
    logger.info("Loading dataset took {:.2f} seconds".format(t_data_done-t_data_start))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    #################
    # Unfold
    #################
    unfolder = OmniFoldwBkg(vars_det_train, vars_mc_train,
                            iterations = parsed_args['iterations'],
                            outdir = parsed_args['outputdir'])
                            #binned_rw = parsed_args['alt_rw'])
    # TODO: parsed_args['background_mode']

    # prepare input data
    logger.info("Prepare data")
    t_prep_start = time.time()

    unfolder.prepare_inputs(data_obs, data_sig, data_bkg,
                            parsed_args['plot_correlations'],
                            standardize=True,
                            reweight_type=parsed_args['reweight_data'],
                            vars_dict=observable_dict)

    t_prep_done = time.time()
    logger.info("Preparing data took {:.2f} seconds".format(t_prep_done - t_prep_start))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    # start unfolding
    logger.info("Start unfolding")
    t_unfold_start = time.time()

    if parsed_args['unfolded_weights']:
        # load unfolded event weights from the saved files
        unfolder.load(parsed_args['unfolded_weights'])
    else:
        # set up hardware
        configGPUs(parsed_args['gpu'], verbose=parsed_args['verbose'])

        # run training
        unfolder.run(parsed_args['error_type'], parsed_args['nresamples'],
                     load_previous_iteration=False,
                     batch_size=parsed_args['batch_size'])

    t_unfold_done = time.time()
    logger.info("Done!")
    logger.info("Unfolding took {:.2f} seconds".format(t_unfold_done - t_unfold_start))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    #################
    # Show results
    #################
    t_result_start = time.time()

    for varname in parsed_args['observables']:
        logger.info("Unfold variable: {}".format(varname))
        varConfig = observable_dict[varname]

        # detector-level distributions
        bins_det = get_bins(varname, parsed_args['binning_config'])
        if bins_det is None:
            bins_det = np.linspace(varConfig['xlim'][0], varConfig['xlim'][1], varConfig['nbins_det']+1)

        unfolder.plot_distributions_reco(varname, varConfig, bins_det)

        # truth-level distributions
        bins_mc = get_bins(varname, parsed_args['binning_config'])
        if bins_mc is None:
            bins_mc = np.linspace(varConfig['xlim'][0], varConfig['xlim'][1], varConfig['nbins_mc']+1)

        # iterative Bayesian unfolding
        if True: # doIBU
            array_obs = data_obs.get_variable_arr(varConfig['branch_det'])
            array_sim = data_sig.get_variable_arr(varConfig['branch_det'])
            array_gen = data_sig.get_variable_arr(varConfig['branch_mc'])
            array_simbkg = data_bkg.get_variable_arr(varConfig['branch_det']) if data_bkg else None
            ibu = IBU(varname, bins_det, bins_mc,
                      array_obs, array_sim, array_gen, array_simbkg,
                      # use the same weights from OmniFold
                      unfolder.weights_obs, unfolder.weights_sim, unfolder.weights_bkg,
                      iterations=parsed_args['iterations'], # same as OmniFold
                      nresample=25, #parsed_args['nresamples']
                      outdir = unfolder.outdir)
            ibu.run()
        else:
            ibu = None

        unfolder.plot_distributions_unfold(varname, varConfig, bins_mc, ibu=ibu, iteration_history=parsed_args['plot_history'])

    t_result_done = time.time()
    logger.info("Plotting results took {:.2f} seconds ({:.2f} seconds per variable)".format(t_result_done - t_result_start, (t_result_done - t_result_start)/len(parsed_args['observables']) ))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.info("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    tracemalloc.stop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--observables-train', dest='observables_train',
                        nargs='+',
                        default=['th_pt', 'th_y', 'th_phi', 'th_e', 'tl_pt', 'tl_y', 'tl_phi', 'tl_e'],
                        help="List of observables to use in training.")
    parser.add_argument('--observables',
                        nargs='+',
                        default=['mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht', 'th_pt', 'th_y', 'th_eta', 'th_phi', 'th_m', 'th_e', 'th_pout', 'tl_pt', 'tl_y', 'tl_eta', 'tl_phi', 'tl_m', 'tl_e', 'tl_pout'],
                        help="List of observables to unfold")
    parser.add_argument('--observable-config', dest='observable_config',
                        default='configs/observables/vars_klfitter.json',
                        help="JSON configurations for observables")
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
    parser.add_argument('-c', '--plot-correlations', dest='plot_correlations',
                        action='store_true',
                        help="Plot pairwise correlations of training variables")
    parser.add_argument('-i', '--iterations', type=int, default=4,
                        help="Numbers of iterations for unfolding")
    parser.add_argument('--weight', default='w',
                        help="name of event weight")
    #parser.add_argument('-m', '--background-mode', dest='background_mode',
    #                    choices=['default', 'subHist', 'negW', 'multiClass'],
    #                    default='default', help="Background mode")
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
                        nargs='*', type=str,
                        help="Unfolded weights file names. If provided, load event weights directly from the files and skip training.")
    parser.add_argument('--binning-config', dest='binning_config',
                        default='configs/binning/bins_10equal.json', type=str,
                        help="Binning config file for variables")
    parser.add_argument('--plot-history', dest='plot_history',
                        action='store_true',
                        help="If true, plot intermediate steps of unfolding")
    parser.add_argument('--nresamples', type=int, default=25,
                        help="number of times for resampling to estimate the unfolding uncertainty using the bootstrap method")
    parser.add_argument('-e', '--error-type', dest='error_type',
                        choices=['sumw2','bootstrap_full','bootstrap_model'],
                        default='sumw2', help="Method to evaluate uncertainties")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=512,
                        help="Batch size for training")

    #parser.add_argument('-n', '--normalize',
    #                    action='store_true',
    #                    help="Normalize the distributions when plotting the result")
    #parser.add_argument('--alt-rw', dest='alt_rw',
    #                    action='store_true',
    #                    help="Use alternative reweighting if true")

    args = parser.parse_args()

    logfile = os.path.join(args.outputdir, 'log.txt')
    configRootLogger(filename=logfile)
    logger = logging.getLogger('Unfold')
    logger.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)

    #################
    if not os.path.isdir(args.outputdir):
        logger.info("Create output directory {}".format(args.outputdir))
        os.makedirs(args.outputdir)

    # check if configuraiton files exist and expand the file path
    fullpath_obsconfig = expandFilePath(args.observable_config)
    if fullpath_obsconfig is not None:
        args.observable_config = fullpath_obsconfig
    else:
        logger.error("Cannot find file: {}".format(args.observable_config))
        sys.exit("Config Failure")

    fullpath_binconfig = expandFilePath(args.binning_config)
    if fullpath_binconfig is not None:
        args.binning_config = fullpath_binconfig
    else:
        logger.error("Cannot find file: {}".format(args.binning_config))
        sys.exit("Config Failure")

    #with tf.device('/GPU:{}'.format(args.gpu)):
    unfold(**vars(args))
