import os
import glob
import numpy as np

import util
import plotter
import histogramming as myhu
from datahandler import DataHandler
from datahandler_root import DataHandlerROOT
from omnifold import omnifold

import logging
logger = logging.getLogger('OmniFoldTTbar')
logger.setLevel(logging.DEBUG)

# maybe put the following in datahandler.py
def getDataHandler(
    filepaths, # list of str
    variables_reco, # list of str
    variables_truth = [], # list of str
    dummy_value = None, # float
    reweighter = None,
    weight_type = 'nominal'
):    
    if util.getFilesExtension(filepaths) == ".root":
        # ROOT files
        # hard code tree names here for now
        tree_reco = 'reco'
        tree_truth = 'parton' if variables_truth else None

        dh = DataHandlerROOT(
            filepaths, variables_reco, variables_truth,
            treename_reco=tree_reco, treename_truth=tree_truth,
            weight_type=weight_type,
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
        data_reweighter=None,
        # type of event weights
        weight_type='nominal'
    ):
        # unfolded weights
        self.unfolded_weights = None
        self.unfolded_weights_resample = None

        # output directory
        self.outdir = outputdir
        if outputdir and not os.path.isdir(outputdir):
            logger.info(f"Create output directory: {outputdir}")
            os.makedirs(outputdir)

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
            data_reweighter = data_reweighter,
            weight_type = weight_type
        )

        # set learning rate

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
        data_reweighter = None, # reweight.Reweighter
        weight_type = 'nominal' # str, optional
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
            reweighter = data_reweighter
            )
        logger.info(f"Total number of observed events: {len(self.handle_obs)}")

        # Signal MC simulation
        logger.info(f"Load signal simulation files: {' '.join(filepaths_sig)}")
        self.handle_sig = getDataHandler(
            filepaths_sig, vars_reco, vars_truth, dummy_value,
            weight_type = weight_type
            )
        logger.info(f"Total number of signal events: {len(self.handle_sig)}")

        # Background MC simulation if needed
        if filepaths_bkg:
            logger.info(f"Load background simulation files: {' '.join(filepaths_bkg)}")
            # only reco level events are needed
            self.handle_bkg = getDataHandler(
                filepaths_bkg, vars_reco, [], dummy_value,
                weight_type = weight_type
                )
            logger.info(f"Total number of background events: {len(self.handle_bkg)}")

        # Simulated background events to be mixed with pseudo data for testing
        if filepaths_obsbkg:
            logger.info(f"Load background simulation files to be mixed with data: {' '.join(filepaths_obsbkg)}")
            # only reco level events are needed
            self.handle_obsbkg = getDataHandler(
                filepaths_obsbkg, vars_reco, [], dummy_value,
                weight_type = weight_type
                )
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
            arr_bkg = self.handle_bkg.get_arrays(self.varnames_reco, valid_only=False)
            arr_data = np.concatenate([arr_data, arr_bkg])

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
            #wmean_sim = np.mean(wsim[self.handle_sig.pass_reco])
            #wsim /= wmean_sim

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
        plot_status=False, # if True, make extra plots for monitoring/debugging
        # learning rate, set to tf.keras.optimizers.Adam's default if not supplied
        learning_rate=0.001
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
        if plot_status:
            plotter.plot_training_inputs_step1(
                os.path.join(self.outdir, "Train_step1"),
                self.varnames_reco,
                X_data[passcut_data], X_sim[passcut_sim],
                w_data[passcut_data], w_sim[passcut_sim])

            plotter.plot_training_inputs_step2(
                os.path.join(self.outdir, "Train_step2"),
                self.varnames_truth,
                X_gen[passcut_gen],
                w_gen[passcut_gen])

        # unfold
        self.unfolded_weights = omnifold(
            X_data, X_sim, X_gen,
            w_data, w_sim, w_gen,
            passcut_data, passcut_sim, passcut_gen,
            learning_rate,
            niterations = niterations,
            model_type = model_type,
            save_models_to = save_model_dir,
            load_models_from = load_model_dir,
            start_from_previous_iter=load_previous_iteration,
            plot = plot_status,
            **fitargs)

        if plot_status:
            logger.info("Plot model training history")
            for csvfile in glob.glob(os.path.join(save_model_dir, '*.csv')):
                logger.info(f"  Plot training log {csvfile}")
                plotter.plot_train_log(csvfile)

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
                self.unfolded_weights_resample[ir,:,:] = omnifold(
                    X_data, X_sim, X_gen,
                    w_data, w_sim, w_gen,
                    passcut_data, passcut_sim, passcut_gen,
                    learning_rate,
                    niterations = niterations,
                    model_type = model_type,
                    save_models_to = save_model_dir_rs,
                    load_models_from = load_model_dir_rs,
                    start_from_previous_iter=load_previous_iteration,
                    **fitargs)

                logger.info("Plot model training history")
                for csvfile in glob.glob(os.path.join(save_model_dir_rs, '*.csv')):
                    logger.info(f"  Plot training log {csvfile}")
                    plotter.plot_train_log(csvfile)

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
        norm=None,
        all_iterations=False,
        iteration=-1, # default: the last iteration
        absoluteValue=False
        ):

        hists_resample = []

        if self.unfolded_weights_resample is None:
            logger.debug("No resample weights! Return an empty list.")
            return hists_resample

        # shape of self.unfolded_weights_resample:
        # (n_resamples, n_iterations, n_events)
        # check if weights are available for iteration
        if iteration >= self.unfolded_weights_resample.shape[1]:
            raise RuntimeError(f"Weights for iteration {iteration} unavailable")

        for iresample in range(len(self.unfolded_weights_resample)):
            if all_iterations:
                rw = self.unfolded_weights_resample[iresample]
            else:
                rw = self.unfolded_weights_resample[iresample][iteration]

            # truth-level prior weights
            wprior = self.handle_sig.get_weights(valid_only=True, reco_level=False)
            h = self.handle_sig.get_histogram(varname, bins, wprior*rw, absoluteValue=absoluteValue)

            if norm is not None:
                # normalize distributions from each resampling
                if all_iterations:
                    for hh in h: # for each iteration
                        hh *= (norm / hh.sum(flow=True)['value'])
                else:
                    h *= (norm / h.sum(flow=True)['value'])

            hists_resample.append(h)

        return hists_resample
            
    def get_unfolded_distribution(
        self,
        varname,
        bins,
        norm=None,
        all_iterations=False,
        iteration=-1, # default: the last iteration
        bootstrap_uncertainty=True,
        absoluteValue=False
        ):
        # check if weights for iteration is available
        if iteration >= self.unfolded_weights.shape[0]:
            raise RuntimeError(f"Weights for iteration {iteration} unavailable")

        rw = self.unfolded_weights if all_iterations else self.unfolded_weights[iteration]
        wprior = self.handle_sig.get_weights(valid_only=True, reco_level=False)
        h_uf = self.handle_sig.get_histogram(varname, bins, wprior*rw, absoluteValue=absoluteValue)
        # h_uf is a hist object or a list of hist objects

        bin_corr = None # bin correlation
        if bootstrap_uncertainty and self.unfolded_weights_resample is not None:
            h_uf_rs = self.get_unfolded_hists_resamples(varname, bins, norm=None, all_iterations=all_iterations, iteration=iteration, absoluteValue=absoluteValue)

            # add the "nominal" histogram to the resampled ones
            h_uf_rs.append(h_uf)

            # take the mean of each bin
            hmean = myhu.get_mean_from_hists(h_uf_rs)

            # take the standard deviation of each bin as bin uncertainties
            hsigma = myhu.get_sigma_from_hists(h_uf_rs)

            # the bin uncertainties are the standard error of the mean
            hstderr = hsigma / np.sqrt(len(h_uf_rs))

            # compute bin correlations
            bin_corr = myhu.get_bin_correlations_from_hists(h_uf_rs)

            # update the nominal histogam
            myhu.set_hist_contents(h_uf, hmean)
            myhu.set_hist_errors(h_uf, hstderr)

        if norm is not None:
            # rescale the unfolded distribution to the norm
            if all_iterations:
                for hh in h_uf:
                    hh *= (norm / hh.sum(flow=True)['value'])
            else:
                h_uf *= (norm / h_uf.sum(flow=True)['value'])

        return h_uf, bin_corr

    def plot_more(self):
        # Event weights
        logger.info("Plot event weights")
        plotter.plot_hist(
            os.path.join(self.outdir, 'Event_weights'),
            [self.handle_sig.get_weights(), self.handle_obs.get_weights()],
            label = ['Sim.', 'Data'],
            title = "Event weights",
            xlabel = 'w')
