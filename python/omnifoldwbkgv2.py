import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging

import util
import plotting
from datahandler import DataHandler
from datahandler_root import DataHandlerROOT
from model import get_model, get_callbacks
from histogramming import get_mean_from_hists, get_sigma_from_hists, get_bin_correlations_from_hists, set_hist_contents, set_hist_errors

logger = logging.getLogger('OmniFoldTTbar')
logger.setLevel(logging.DEBUG)

# maybe put the following in datahandler.py
def getDataHandler(
    filepaths, # list of str
    variables_reco, # list of str
    variables_truth = [], # list of str
    dummy_value = None, # float
    reweighter = None
):    
    if util.getFilesExtension(filepaths) == ".root":
        # ROOT files
        # hard code tree names here for now
        tree_reco = 'reco'
        tree_truth = 'parton' if variables_truth else None

        dh = DataHandlerROOT(
            filepaths, variables_reco, variables_truth,
            treename_reco=tree_reco, treename_truth=tree_truth,
            weight_type='nominal',
            dummy_value=dummy_value)
    else:
        # for limited backward compatibility to deal with npz file
        wname = 'totalWeight_nominal'
        dh = DataHandler(filepaths, varnames_reco, varnames_truth, wname)

    if reweighter is not None:
        # TODO: check if variables required by reweighter are included
        dh.rescale_weights(reweighter=reweighter)

    return dh
##

def read_weights_from_file(filepath_weights, array_name):
    wfile = np.load(filepath_weights)
    weights = wfile[array_name]
    wfile.close()
    return weights

# move this to model.py?
def train_model(model, X, Y, w, callbacks=[], figname='', **fitargs):
    """
    Train model
    """

    if callbacks:
        fitargs.setdefault('callbacks', []).extend(callbacks)

    X_train, X_val, Y_train, Y_val, w_train, w_val = train_test_split(X, Y, w)

    # zip event weights with labels
    Yw_train = np.column_stack((Y_train, w_train))
    Yw_val = np.column_stack((Y_val, w_val))

    model.fit(X_train, Yw_train, validation_data=(X_val, Yw_val), **fitargs)

    if figname:
        logger.info(f"Plot model output distributions: {figname}")
        preds_train = model.predict(X_train, batch_size=int(0.1*len(X_train)))[:,1]
        preds_val = model.predict(X_val, batch_size=int(0.1*len(X_val)))[:,1]
        plotting.plot_training_vs_validation(figname, preds_train, Y_train, w_train, preds_val, Y_val, w_val)

def set_up_model(
    model_type, # str, type of the network
    input_shape, # tuple, shape of the input layer
    iteration = 0, # int, iteration index
    name_prefix = 'model', # str, prefix of the model name
    save_models_to = '', # str, directory to save the trained model to
    load_models_from = '', # str, directory to load trained model weights
    start_from_previous_iter = False, # bool, if True, initialize model from previous iteration
    ):

    # get network
    model = get_model(input_shape, nclass=2, model_name=model_type)

    # name of the model checkpoint
    mname = name_prefix + "_{}".format(iteration)

    # callbacks
    filepath_save = None
    if save_models_to:
        filepath_save = os.path.join(save_models_to, mname)

    callbacks = get_callbacks(filepath_save)
    
    # load trained model if needed
    if load_models_from:
        filepath_load = os.path.join(load_models_from, mname)
        model.load_weights(filepath_load).expect_partial()
        logger.info(f"Load model from {filepath_load}")
    else:
        if start_from_previous_iter and save_models_to and iteration > 0:
            # initialize model weights from the previous iteration
            mname_prev = name_prefix+"_{}".format(iteration-1)
            filepath_load = os.path.join(save_models_to, mname_prev)
            model.load_weights(filepath_load)
            logger.debug(f"Initialize model from {filepath_load}")

    return model, callbacks

def reweight(model, events, figname=None):
    preds = model.predict(events, batch_size=int(0.1*len(events)))[:,1]
    r = np.nan_to_num( preds / (1. - preds) )

    if figname: # plot the distribution
        logger.info(f"Plot likelihood ratio distribution {figname}")
        plotting.plot_LR_distr(figname, [r])

    return r

def unfold(
    # Data
    X_data, # feature array of observed data
    X_sim, # feature array of signal simulation at reco level
    X_gen, # feature array of signal simulation at truth level
    w_data, # event weights of observed data
    w_sim, # reco weights of signal simulation events
    w_gen, # MC weights of signal simulation events
    # Event selection flags
    passcut_data, # flags to indicate if data events pass reco level selections
    passcut_sim, # flags to indicate if signal events pass reco level selections
    passcut_gen, # flags to indicate if signal events pass truth level selections
    # Parameters
    niterations, # number of iterations
    model_type='dense_100x3', # name of the model type 
    save_models_to='', # directory to save models to if provided
    load_models_from='', # directory to load trained models if provided
    start_from_previous_iter=False, # If True, initialize model with the previous iteration
    **fitargs
    ):
    """
    OmniFold
    arXiv:1911.09107, arXiv:2105.04448

    """
    ################
    # Prepare data arrays for training
    assert(len(X_data)==len(w_data))
    assert(len(X_sim)==len(w_sim))
    assert(len(X_gen)==len(w_gen))
    assert(len(X_sim)==len(passcut_sim))
    assert(len(X_gen)==len(passcut_gen))

    # Step 1
    # Use events that pass reco level selections
    # features
    X_step1 = np.concatenate([ X_data[passcut_data], X_sim[passcut_sim] ])
    # labels: data=1, sim=0
    Y_step1 = np.concatenate([ np.ones(len(X_data[passcut_data])), np.zeros(len(X_sim[passcut_sim])) ])
    # make Y categorical
    Y_step1 = tf.keras.utils.to_categorical(Y_step1)

    logger.debug(f"Size of the feature array for step 1: {X_step1.nbytes*2**-20:.3f} MB")
    logger.debug(f"Size of the label array for step 1: {Y_step1.nbytes*2**-20:.3f} MB")

    # Step 1b
    if np.any(~passcut_sim):
        X_step1b = np.concatenate([ X_gen[passcut_sim & passcut_gen], X_gen[passcut_sim & passcut_gen] ])
        Y_step1b = np.concatenate([ np.ones(len(X_gen[passcut_sim & passcut_gen])), np.zeros(len(X_gen[passcut_sim & passcut_gen])) ])
        Y_step1b = tf.keras.utils.to_categorical(Y_step1b)

        logger.debug(f"Size of the feature array for step 1b: {X_step1b.nbytes*2**-20:.3f} MB")
        logger.debug(f"Size of the label array for step 1b: {Y_step1b.nbytes*2**-20:.3f} MB")

    # Step 2
    # features
    X_step2 = np.concatenate([ X_gen[passcut_gen], X_gen[passcut_gen] ])
    # labels
    Y_step2 = np.concatenate([ np.ones(len(X_gen[passcut_gen])), np.zeros(len(X_gen[passcut_gen])) ])
    # make Y categorical
    Y_step2 = tf.keras.utils.to_categorical(Y_step2)

    logger.debug(f"Size of the feature array for step 2: {X_step2.nbytes*2**-20:.3f} MB")
    logger.debug(f"Size of the label array for step 2: {Y_step2.nbytes*2**-20:.3f} MB")

    # Step 2b
    if np.any(~passcut_gen):
        X_step2b = np.concatenate([ X_sim[passcut_sim & passcut_gen], X_sim[passcut_sim & passcut_gen] ])
        Y_step2b = np.concatenate([ np.ones(len(X_sim[passcut_sim & passcut_gen])), np.zeros(len(X_sim[passcut_sim & passcut_gen])) ])
        Y_step2b = tf.keras.utils.to_categorical(Y_step2b)

        logger.debug(f"Size of the feature array for step 2b: {X_step2b.nbytes*2**-20:.3f} MB")
        logger.debug(f"Size of the label array for step 2b: {Y_step2b.nbytes*2**-20:.3f} MB")

    ################
    # Prepare models
    if save_models_to and not os.path.isdir(save_models_to):
        logger.info("Make directory for saving models")
        os.makedirs(save_models_to)

    if load_models_from and not os.path.isdir(load_models_from):
        raise RuntimeError(f"Cannot load models from {load_models_from}: directory does not exist!")

    ################
    # Start iterations
    # Weights
    weights_push = np.ones(len(X_sim))
    weights_pull = np.ones(len(X_gen))

    weights_unfold = np.empty(shape=(niterations, len(X_gen[passcut_gen])))
    # shape: (n_iterations, n_events[passcut_gen])

    for i in range(niterations):
        logger.info(f"Iteration {i}")
        #####
        # step 1: reweight to sim to data
        logger.info("Step 1")
        # set up the model
        model_step1, cb_step1 = set_up_model(
            model_type, X_step1.shape[1:], i, "model_step1",
            save_models_to, load_models_from, start_from_previous_iter)

        if load_models_from:
            logger.info("Use trained model for reweighting")
        else: # train model
            w_step1 = np.concatenate([
                w_data[passcut_data], (weights_push*w_sim)[passcut_sim]
                ])

            logger.info("Start training")
            fname_preds = save_models_to + f"/preds_step1_{i}" if save_models_to else ''
            train_model(model_step1, X_step1, Y_step1, w_step1,
                        callbacks = cb_step1,
                        #figname = fname_preds,
                        **fitargs)
            logger.info("Training done")

        # reweight
        logger.info("Reweight")
        fname_rdistr = save_models_to + f"/rdistr_step1_{i}" if save_models_to else ''
        weights_pull = weights_push * reweight(model_step1, X_sim, fname_rdistr)

        #####
        # step 1b: deal with events that do not pass reco cuts
        if np.any(~passcut_sim):
            logger.info("Step 1b")
            # weights_pull[~passcut_sim] = 1.
            # Or alternatively, estimate the average weights: <w|x_true>
            model_step1b, cb_step1b = set_up_model(
                model_type, X_step1b.shape[1:], i, "model_step1b",
                save_models_to, load_models_from, start_from_previous_iter)

            if load_models_from:
                logger.info("Use trained model for reweighting")
            else: # train model
                w_step1b = np.concatenate([
                    (weights_pull*w_gen)[passcut_sim & passcut_gen],
                    w_gen[passcut_sim & passcut_gen]
                    ])

                logger.info("Start training")
                train_model(model_step1b, X_step1b, Y_step1b, w_step1b,
                            callbacks = cb_step1b, **fitargs)
                logger.info("Training done")

            # reweight
            logger.info("Reweight")
            fname_rdistr = save_models_to + f"/rdistr_step1b_{i}" if save_models_to else ''
            weights_pull[~passcut_sim] = reweight(model_step1b, X_gen[~passcut_sim], fname_rdistr)

        #####
        # step 2
        logger.info("Step 2")
        model_step2, cb_step2 = set_up_model(
            model_type, X_step2.shape[1:], i, "model_step2",
            save_models_to, load_models_from, start_from_previous_iter)

        if load_models_from:
            logger.info("Use trained model for reweighting")
        else: # train model
#            rw_step2 = 1. # always reweight against the prior
            rw_step2 = 1. if i==0 else weights_unfold[i-1] # previous iteration

            
            w_step2 = np.concatenate([
                (weights_pull*w_gen)[passcut_gen], w_gen[passcut_gen]*rw_step2
                ])

            logger.info("Start training")
            fname_preds = save_models_to + f"/preds_step2_{i}" if save_models_to else ''
            train_model(model_step2, X_step2, Y_step2, w_step2,
                        callbacks = cb_step2,
                        #figname = fname_preds,
                        **fitargs)
            logger.info("Training done")

        # reweight
        logger.info("Reweight")
        fname_rdistr = save_models_to + f"/rdistr_step2_{i}" if save_models_to else ''
        weights_push[passcut_gen] = rw_step2 * reweight(model_step2, X_gen[passcut_gen], fname_rdistr)

        #####
        # step 2b: deal with events that do not pass truth cuts
        if np.any(~passcut_gen):
            logger.info("Step 2b")
            # weights_push[~passcut_gen] = 1.
            # Or alternatively, estimate the average weights: <w|x_reco>
            model_step2b, cb_step2b = set_up_model(
                model_type, X_step2b.shape[1:], i, "model_step2b",
                save_models_to, load_models_from, start_from_previous_iter)

            if load_models_from:
                logger.info("Use trained model for reweighting")
            else: # train model
                w_step2b = np.concatenate([
                    (weights_push*w_sim)[passcut_sim & passcut_gen],
                    w_sim[passcut_sim & passcut_gen]
                    ])

                logger.info("Start training")
                train_model(model_step2b, X_step2b, Y_step2b, w_step2b,
                            callbacks = cb_step2b, **fitargs)
                logger.info("Training done")

            # reweight
            logger.info("Reweight")
            fname_rdistr = save_models_to + f"/rdistr_step2b_{i}" if save_models_to else ''
            weights_push[~passcut_gen] = reweight(model_step2b, X_sim[~passcut_gen], fname_rdistr)

        # save truth level weights of this iteration
        weights_unfold[i,:] = weights_push[passcut_gen]
    # end of iteration loop

    return weights_unfold

def plot_training_history(model_dir):
    logger.info("Plot model training history")
    for csvfile in glob.glob(os.path.join(model_dir, '*.csv')):
        logger.info(f"  Plot training log {csvfile}")
        plotting.plot_train_log(csvfile)

def plot_training_inputs_step1(
        variable_names,
        Xdata, Xsim,
        wdata, wsim,
        outdir):
    # features
    logger.info("Plot distributions of input variables for step 1")
    for vname, vdata, vsim in zip(variable_names, Xdata.T, Xsim.T):
        logger.debug(f"  Plot variable {vname}")
        plotting.plot_data_arrays(
            os.path.join(outdir, f"Train_step1_{vname}"),
            [ vdata, vsim ],
            weight_arrs = [wdata, wsim],
            labels = ['Data', 'Sim.'],
            xlabel = vname,
            title = "Step-1 training inputs")

    # weights
    logger.info("Plot distributions of the prior weights for step 1")
    plotting.plot_data_arrays(
        os.path.join(outdir, "Train_step1_weights"),
        [wdata, wsim],
        labels = ['Data', 'Sim.'],
        xlabel = 'w (training)',
        title = "Step-1 prior weights at reco level")

def plot_training_inputs_step2(variable_names, Xgen, wgen, outdir):
    # features
    logger.info("Plot distributions of input variables for step 2")
    for vname, vgen in zip(variable_names, Xgen.T):
        logger.debug(f"  Plot variable {vname}")
        plotting.plot_data_arrays(
            os.path.join(outdir, f"Train_step2_{vname}"),
            [vgen], weight_arrs = [wgen],
            labels = ['Gen.'],
            xlabel = vname,
            title = "Step-2 training inputs")

    # weights
    logger.info("Plot distributions of prior weights for step 2")
    plotting.plot_data_arrays(
        os.path.join(outdir, "Train_step2_weights"),
        [wgen],
        labels = ['Gen.'],
        xlabel = 'w (training)',
        title = "Step-2 prior weights at truth level")

class OmniFoldTTbar():
    def __init__(
        self,
        ## Observable names and configuration dictionary
        # Name of the variables used to train networks and to unfold, list of str
        # reco level (for step 1)
        variables_reco,
        # truth level (for step 2)
        variables_truth,
        ## Input files
        # observed data or pseudo data, list of str
        filepaths_obs,
        # signal MC simulation, list of str
        filepaths_sig,
        # background MC simulation, list of str
        filepaths_bkg = [],
        # background MC events to be mixed with pseudo data, list of str
        filepaths_obsbkg = [], 
        ## Additional options
        # flag to indicate if the input is pseudo data in which MC truth is known
        truth_known = False,
        # If or not to normalize simulation weights to match data
        normalize_to_data = False,
        # extra variables to unfold but not used in training, list of str
        variables_reco_extra = [],
        variables_truth_extra = [],
        # dummy value to assign for events fail selections
        dummy_value = -99.,
        # output directory
        outputdir = None,
        # data reweighting for stress test
        data_reweighter=None
    ):
        # unfolded weights
        self.unfolded_weights = None
        self.unfolded_weights_resample = None

        # output directory
        self.outdir = outputdir

        # variables used in training
        self.varnames_reco = variables_reco
        logger.debug(f"Variables in step 1 training: {', '.join(self.varnames_reco)}")
        self.varnames_truth = variables_truth
        logger.debug(f"Variables in step 2 training: {', '.join(self.varnames_truth)}")

        # all variables to load from input files
        assert(len(variables_reco)==len(variables_truth))
        assert(len(variables_reco_extra)==len(variables_truth_extra))

        vars_reco_all = list(set().union(variables_reco, variables_reco_extra)) if variables_reco_extra else variables_reco
        vars_truth_all = list(set().union(variables_truth, variables_truth_extra)) if variables_truth_extra else variables_truth

        # data handlers
        self.handle_obs = None
        self.handle_sig = None
        self.handle_bkg = None
        self.handle_obsbkg = None

        # load input files to the data handlers
        self._prepare_inputs(
            vars_reco_all, vars_truth_all,
            filepaths_obs, filepaths_sig, filepaths_bkg, filepaths_obsbkg,
            normalize = normalize_to_data,
            truth_known = truth_known,
            dummy_value = dummy_value,
            data_reweighter = data_reweighter
        )

    def _prepare_inputs(
        self,
        vars_reco, # list of str
        vars_truth, # list of str
        filepaths_obs, # list of str
        filepaths_sig, # list of str
        filepaths_bkg, # list of str, optional
        filepaths_obsbkg, # list of str, optional
        normalize = False, # bool
        truth_known = False, # bool
        dummy_value = -99., # float
        data_reweighter = None # reweight.Reweighter
        ):
        """
        Load input files into data handlers: self.handle_obs, self.handle_sig, 
        self.handle_bkg (optional), self.handle_obsbkg (optional)
        Also rescale simulation weights in needed
        """

        # Observed data
        logger.info(f"Load data files: {' '.join(filepaths_obs)}")
        self.handle_obs = getDataHandler(
            filepaths_obs,
            vars_reco,
            vars_truth if truth_known else [],
            dummy_value,
            data_reweighter)
        logger.info(f"Total number of observed events: {len(self.handle_obs)}")

        # Signal MC simulation
        logger.info(f"Load signal simulation files: {' '.join(filepaths_sig)}")
        self.handle_sig = getDataHandler(
            filepaths_sig, vars_reco, vars_truth, dummy_value)
        logger.info(f"Total number of signal events: {len(self.handle_sig)}")

        # Background MC simulation if needed
        if filepaths_bkg:
            logger.info(f"Load background simulation files: {' '.join(filepaths_bkg)}")
            # only reco level events are needed
            self.handle_bkg = getDataHandler(
                filepaths_bkg, vars_reco, [], dummy_value)
            logger.info(f"Total number of background events: {len(self.handle_bkg)}")

        # Simulated background events to be mixed with pseudo data for testing
        if filepaths_obsbkg:
            logger.info(f"Load background simulation files to be mixed with data: {' '.join(filepaths_obsbkg)}")
            # only reco level events are needed
            self.handle_obsbkg = getDataHandler(
                filepaths_obsbkg, vars_reco_all, [], dummy_value)
            logger.info(f"Total number of background events mixed with data: {len(self.handle_obsbkg)}")

        ####
        # Event weights
        # total weights of data
        sumw_obs = self.handle_obs.sum_weights()
        if self.handle_obsbkg is not None:
            sumw_obs += self.handle_obsbkg.sum_weights()
        logger.debug(f"Total weights of data events: {sumw_obs}")

        # total weights of simulated events
        sumw_sig = self.handle_sig.sum_weights()
        sumw_bkg = 0. if self.handle_bkg is None else self.handle_bkg.sum_weights()
        sumw_sim = sumw_sig + sumw_bkg

        # normalize total weights of simulation to that of data if needed
        if normalize:
            logger.info("Rescale simulation weights to data")

            self.handle_sig.rescale_weights(sumw_obs/sumw_sim)

            if self.handle_bkg is not None:
                self.handle_bkg.rescale_weights(sumw_obs/sumw_sim)

        logger.info(f"Total weights of signal events: {self.handle_sig.sum_weights()}")
        if self.handle_bkg is not None:
            logger.info(f"Total weights of background events: {self.handle_bkg.sum_weights()}")

    def _get_input_arrays(self, preprocess=True):
        logger.debug("Prepare input arrays")

        # observed data (or pseudo data)
        arr_data = self.handle_obs.get_arrays(self.varnames_reco, valid_only=False)

        # only for testing with pseudo data:
        # mix background simulation with signal simulation as pseudo data
        if self.handle_obsbkg is not None:
            arr_dataobs = self.handle_obsbkg.get_arrays(self.varnames_reco, valid_only=False)
            arr_data = np.concatenate([arr_data, arr_dataobs])

        # add backgrouund simulation to the data array (with negative weights)
        if self.handle_bkg is not None:
            arr_bkg = np.concatenate([arr_data, arr_bkg])

        # signal simulation
        # reco level
        arr_sim = self.handle_sig.get_arrays(self.varnames_reco, valid_only=False)
        # truth level
        arr_gen = self.handle_sig.get_arrays(self.varnames_truth, valid_only=False)

        if preprocess:
            logger.info("Preprocess feature arrays")

            # estimate the order of magnitude
            logger.debug("Divide each variable by its order of magnitude")
            # use only the valid events
            xmean_reco = np.mean(np.abs(self.handle_obs[self.varnames_reco]), axis=0)
            xoom_reco = 10**(np.log10(xmean_reco).astype(int))
            arr_data /= xoom_reco
            arr_sim /= xoom_reco

            xmean_truth = np.mean(np.abs(self.handle_sig[self.varnames_truth]), axis=0)
            xoom_truth = 10**(np.log10(xmean_truth).astype(int))
            arr_gen /= xoom_truth

            # TODO: check alternative
            # standardize feature arrays to mean of zero and variance of one
        
        return arr_data, arr_sim, arr_gen

    def _get_event_weights(self, resample=False, standardize=True):
        logger.debug("Prepare event weights")

        wdata = self.handle_obs.get_weights(bootstrap=resample, valid_only=False)

        if standardize:
            logger.debug("Standardize data weights to mean of one for training")
            # exclude dummy value when calculating mean
            wmean_obs = np.mean(wdata[self.handle_obs.pass_reco])
            wdata /= wmean_obs

        if self.handle_obsbkg is not None:
            wobsbkg = self.handle_obsbkg.get_weights(bootstrap=resample, valid_only=False)
            if standardize: # rescale by the same factor as data
                wobsbkg /= wmean_obs

            wdata = np.concatenate([wdata, wobsbkg])

        # add background simulation as observed data but with negative weights
        if self.handle_bkg is not None:
            wbkg = self.handle_bkg.get_weights(valid_only=False)
            if standardize:
                wbkg /= wmean_obs

            wdata = np.concatenate([wdata, -1*wbkg])

        # signal simulation
        # reco level
        wsim = self.handle_sig.get_weights(valid_only=False)
        if standardize:
            wsim /= wmean_obs
            #TODO check alternative: divide by its own mean

        # truth level
        wgen = self.handle_sig.get_weights(valid_only=False, reco_level=False)

        if standardize:
            # CHECK HERE!!
            #wgen /= wmean_obs
            # this is what's been done previously
            wmean_gen = np.mean(wgen[self.handle_sig.pass_truth])
            wgen /= wmean_gen

        return wdata, wsim, wgen

    def _get_event_flags(self):
        logger.debug("Get event selection flags")

        data_pass_reco = self.handle_obs.pass_reco
        if self.handle_obsbkg is not None:
            data_pass_reco = np.concatenate([data_pass_reco, self.handle_obsbkg.pass_reco])
        if self.handle_bkg is not None:
            data_pass_reco = np.concatenate([data_pass_reco, self.handle_bkg.pass_reco])

        mc_pass_reco = self.handle_sig.pass_reco
        mc_pass_truth = self.handle_sig.pass_truth

        return data_pass_reco, mc_pass_reco, mc_pass_truth

    def run(
        self,
        niterations, # number of iterations
        error_type='sumw2',
        nresamples=10,
        model_type='dense_100x3',
        save_models=True,
        load_previous_iteration=True,
        load_models_from='',
        batch_size=256,
    ):
        """
        Run unfolding
        """
        fitargs = {"batch_size": batch_size, "epochs": 100, "verbose": 1}

        # model directory
        if load_models_from:
            load_model_dir = os.path.join(load_models_from, "Models")
            save_model_dir = '' # no need to save the model again
        else:
            load_model_dir = ''
            save_model_dir = ''
            if save_models and self.outdir:
                save_model_dir = os.path.join(self.outdir, "Models")

        # preprocess data and weights
        X_data, X_sim, X_gen = self._get_input_arrays()
        w_data, w_sim, w_gen = self._get_event_weights()
        passcut_data, passcut_sim, passcut_gen = self._get_event_flags()

        # plot variable and event weight distributions for training
        # TODO: add a flag to enable/disable plotting
        if True:
            plot_training_inputs_step1(
                self.varnames_reco,
                X_data[passcut_data], X_sim[passcut_sim],
                w_data[passcut_data], w_sim[passcut_sim],
                self.outdir)

            plot_training_inputs_step2(
                self.varnames_truth,
                X_gen[passcut_gen],
                w_gen[passcut_gen],
                self.outdir)

        # unfold
        self.unfolded_weights = unfold(
            X_data, X_sim, X_gen,
            w_data, w_sim, w_gen,
            passcut_data, passcut_sim, passcut_gen,
            niterations = niterations,
            model_type = model_type,
            save_models_to = save_model_dir,
            load_models_from = load_model_dir,
            start_from_previous_iter=load_previous_iteration,
            **fitargs)

        plot_training_history(save_model_dir)

        # save weights
        wfile = os.path.join(self.outdir, 'weights.npz')
        np.savez(wfile, weights = self.unfolded_weights)

        # resamples
        if error_type in ['bootstrap_full', 'bootstrap_model']:

            self.unfolded_weights_resample = np.empty(
                shape=(nresamples,)+self.unfolded_weights.shape
            )

            for ir in range(nresamples):
                logger.info(f"Resample #{ir}")

                # model directory
                load_model_dir_rs = os.path.join(load_models_from, f"Models_rs{ir}") if load_model_dir else ''
                save_model_dir_rs = os.path.join(self.outdir, f"Models_rs{ir}") if save_model_dir else ''

                # bootstrap data weights
                w_data, w_sim, w_gen = self._get_event_weights(resample=True)

                # unfold
                self.unfolded_weights_resample[ir,:,:] = unfold(
                    X_data, X_sim, X_gen,
                    w_data, w_sim, w_gen,
                    passcut_data, passcut_sim, passcut_gen,
                    niterations = niterations,
                    model_type = model_type,
                    save_models_to = save_model_dir_rs,
                    load_models_from = load_model_dir_rs,
                    start_from_previous_iter=load_previous_iteration,
                    **fitargs)

                plot_training_history(save_model_dir_rs)

            # save weights
            wfile = os.path.join(self.outdir, f"weights_resample{nresamples}.npz")
            np.savez(wfile, weights_resample = self.unfolded_weights_resample)

    def load(self, filepaths_unfolded_weights):
        """
        Load unfolded weights from files on disk
        """
        logger.info("Load unfolded weights directly and skip training")

        if isinstance(filepaths_unfolded_weights, str):
            wfilelist = [filepaths_unfolded_weights]
        else:
            wfilelist = list(filepaths_unfolded_weights)
        wfilelist.sort()
        assert(len(wfilelist)>0)

        logger.info(f"Load weights from {wfilelist[0]}")
        self.unfolded_weights = read_weights_from_file(wfilelist[0], array_name='weights')
        logger.debug(f"unfolded_weights.shape: {self.unfolded_weights.shape}")

        if len(wfilelist) > 1:
            logger.info(f"Load weights from resampling: {wfilelist[1]}")
            self.unfolded_weights_resample = read_weights_from_file(wfilelist[1], array_name='weights_resample')
            # FIXME: load weights from multiple files

            logger.debug(f"unfolded_weights_resample.shape: {self.unfolded_weights_resample.shape}")

    def get_unfolded_hists_resamples(
        self,
        varname, # str, name of the variable
        bins,
        normalize,
        all_iterations=False
        ):

        hists_resample = []

        if self.unfolded_weights_resample is None:
            logger.warn("No resample weights! Return an empty list.")
            return hists_resample

        # shape of self.unfolded_weights_resample:
        # (n_resamples, n_iterations, n_events)
        for iresample in range(len(self.unfolded_weights_resample)):
            if all_iterations:
                rw = self.unfolded_weights_resample[iresample]
            else: # last iteration
                rw = self.unfolded_weights_resample[iresample][-1]

            # truth-level prior weights
            wprior = self.handle_sig.get_weights(valid_only=True, reco_level=False)
            h = self.handle_sig.get_histogram(varname, bins, wprior*rw)

            if normalize:
                # normalize each resample to the prior
                if all_iterations:
                    rs = wprior.sum() / (wprior*rw).sum(axis=1)
                    for hh, r in zip(h, rs): # for each iteration
                        hh *= r
                else:
                    h *= (wprior.sum() / (wprior*rw).sum())

            hists_resample.append(h)

        return hists_resample
            
    def get_unfolded_distribution(
        self,
        varname,
        bins,
        normalize,
        all_iterations=False,
        bootstrap_uncertainty=True
        ):
        rw = self.unfolded_weights if all_iterations else self.unfolded_weights[-1]
        wprior = self.handle_sig.get_weights(valid_only=True, reco_level=False)
        h_uf = self.handle_sig.get_histogram(varname, bins, wprior*rw)
        # h_uf is a hist object or a list of hist objects

        bin_corr = None # bin correlation
        if bootstrap_uncertainty and self.unfolded_weights_resample is not None:
            h_uf_rs = self.get_unfolded_hists_resamples(varname, bins, all_iterations)

            # add the "nominal" histogram to the resampled ones
            h_uf_rs.append(h_uf)

            # take the mean of each bin
            hmean = get_mean_from_hists(h_uf_rs)

            # take the standard deviation of each bin as bin uncertainties
            hsigma = get_sigma_from_hists(h_uf_rs)

            # compute bin correlations
            bin_corr = get_bin_correlations_from_hists(h_uf_rs)

            # update the nominal histogam
            set_hist_contents(h_uf, hmean)
            set_hist_errors(h_uf, hsigma)

        if normalize:
            # normalize the unfolded histograms to the prior distribution
            if all_iterations:
                rs = wprior.sum() / (wprior*rw).sum(axis=1)
                for h, r in zip(h_uf, rs):
                    h *= r
            else:
                h_uf *= (wprior.sum() / (wprior*rw).sum())

        return h_uf, bin_corr

#    def get_unfolded_weights():
#        pass

    def plot_more(self):
        # Event weights
        logger.info("Plot event weights")
        plotting.plot_data_arrays(
            os.path.join(self.outdir, 'Event_weights'),
            [self.handle_sig.get_weights(), self.handle_obs.get_weights()],
            label = ['Sim.', 'Data'],
            title = "Event weights",
            xlabel = 'w')
