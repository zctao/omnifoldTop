import os
import glob
import numpy as np
from copy import copy

import util
import plotter
import reweight
import histogramming as myhu
from datahandler import DataHandler
from datahandler_root import DataHandlerROOT
from omnifold import omnifold
from modelUtils import n_models_in_parallel
import preprocessor

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

def read_weights_from_file(filepaths_weights):

    weights = None

    for wfpath in filepaths_weights:
        logger.info(f"Load weights from {wfpath}")
        wfile = np.load(wfpath)

        for wname, warr in wfile.items():
            logger.info(f"Read weight array: {wname}")

            if warr.ndim == 3: # shape: (nruns, niterations, nevents)
                if weights is None:
                    weights = warr
                else:
                    weights = np.concatenate((weights, warr))

            # for backward compatibility
            elif warr.ndim == 2:
                # read old weights of shape (niterations, nevents)
                if weights is None:
                    weights = np.expand_dims(warr, 0)
                else:
                    weights = np.concatenate((weights, np.expand_dims(warr, 0)))

            else:
                logger.error(f"Fail to read weight array {wname}. Abort...")
                return None

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

        # preprocess input arrays

        p = preprocessor.get()

        arr_data = p.preprocess(self.varnames_reco, arr_data)
        # TODO: a truth known option?
        
        arr_sim = p.preprocess(self.varnames_reco, arr_sim)
        arr_gen = p.preprocess(self.varnames_truth, arr_gen)

        # if preprocess:
        #     logger.info("Preprocess feature arrays")

        #     # estimate the order of magnitude
        #     logger.debug("Divide each variable by its order of magnitude")
        #     # use only the valid events
        #     xmean_reco = np.mean(np.abs(self.handle_obs[self.varnames_reco]), axis=0)
        #     xoom_reco = 10**(np.log10(xmean_reco).astype(int))
        #     arr_data /= xoom_reco
        #     arr_sim /= xoom_reco

        #     xmean_truth = np.mean(np.abs(self.handle_sig[self.varnames_truth]), axis=0)
        #     xoom_truth = 10**(np.log10(xmean_truth).astype(int))
        #     arr_gen /= xoom_truth

        #     # TODO: check alternative
        #     # standardize feature arrays to mean of zero and variance of one

        arr_data, arr_sim = p.group_standardize((arr_data, arr_sim))

        # TODO: a truth known option?
        arr_gen = p.standardize(arr_gen)

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
        resample_data=False,
        nruns=1,
        resample_everyrun=False,
        model_type='dense_100x3',
        save_models=True,
        load_previous_iteration=True,
        load_models_from='',
        batch_size=256,
        plot_status=False # if True, make extra plots for monitoring/debugging
    ):
        """
        Run unfolding
        """

        # preprocess data and weights
        X_data, X_sim, X_gen = self._get_input_arrays()
        w_data, w_sim, w_gen = self._get_event_weights(resample=resample_data)
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
        assert(nruns>0)
        self.unfolded_weights = np.empty(
            shape=(nruns * n_models_in_parallel, niterations, np.count_nonzero(passcut_gen))
            )

        for ir in range(nruns):
            logger.info(f"Run #{ir}")

            # model directory
            if load_models_from:
                load_model_dir = os.path.join(load_models_from, "Models", f"run{ir}")
                save_model_dir = '' # no need too save the model again
            else:
                load_model_dir = ''
                if save_models and self.outdir:
                    save_model_dir = os.path.join(self.outdir, "Models", f"run{ir}")
                else:
                    save_model_dir = ''

            if resample_data and resample_everyrun:
                # fluctuate data weights
                w_data, w_sim, w_gen = self._get_event_weights(resample=True)

            # omnifold
            self.unfolded_weights[ir*n_models_in_parallel:(ir+1)*n_models_in_parallel,:,:] = omnifold(
                X_data, X_sim, X_gen,
                w_data, w_sim, w_gen,
                passcut_data, passcut_sim, passcut_gen,
                niterations = niterations,
                model_type = model_type,
                save_models_to = save_model_dir,
                load_models_from = load_model_dir,
                start_from_previous_iter=load_previous_iteration,
                plot = plot_status and ir==0, # only make plots for the first run
                batch_size = batch_size
            )

            if plot_status:
                logger.info("Plot model training history")
                for csvfile in glob.glob(os.path.join(save_model_dir, '*.csv')):
                    logger.info(f"  Plot training log {csvfile}")
                    plotter.plot_train_log(csvfile)

        # save weights to disk
        wfile = os.path.join(self.outdir, "weights_unfolded.npz")
        np.savez(wfile, unfolded_weights = self.unfolded_weights)

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

        self.unfolded_weights = read_weights_from_file(wfilelist)
        logger.debug(f"unfolded_weights.shape: {self.unfolded_weights.shape}")

    def get_unfolded_hists_resamples(
        self,
        varname, # str, name of the variable
        bins,
        norm=None,
        all_iterations=False,
        iteration=-1, # default: the last iteration
        nresamples=None, # default: take all that are available
        density=False,
        absoluteValue=False
        ):

        hists_resample = []

        if self.unfolded_weights is None:
            logger.error("No unfolded weights! Return an empty list.")
            return hists_resample

        # shape of self.unfolded_weights: (nruns, niterations, nevents)
        # check if weights are available for the required iteration
        if iteration >= self.unfolded_weights.shape[1]:
            raise RuntimeError(f"Weights for iteration {iteration} unavailable")

        # number of ensambles
        if nresamples is None:
            # default: use all available weights
            nresamples = self.unfolded_weights.shape[0]
        else:
            if nresamples > self.unfolded_weights.shape[0]:
                logger.warn(f"Requested number of runs {nresamples} is larger than what is available in the unfolded weights.")
            nresamples = min(nresamples, self.unfolded_weights.shape[0])

        for iresample in range(nresamples):
            if all_iterations:
                rw = self.unfolded_weights[iresample]
            else:
                rw = self.unfolded_weights[iresample][iteration]

            # truth-level prior weights
            wprior = self.handle_sig.get_weights(valid_only=True, reco_level=False)
            h = self.handle_sig.get_histogram(varname, bins, wprior*rw, density=density, norm=norm, absoluteValue=absoluteValue)

            hists_resample.append(h)

        return hists_resample
            
    def get_unfolded_distribution(
        self,
        varname,
        bins,
        norm=None,
        all_iterations=False,
        iteration=-1, # default: the last iteration
        nresamples=None, # default, take all that are avaiable
        density=False,
        absoluteValue=False
        ):

        hists_uf = self.get_unfolded_hists_resamples(
            varname,
            bins,
            norm=None,
            all_iterations=all_iterations,
            iteration=iteration,
            nresamples=nresamples,
            density=density,
            absoluteValue=absoluteValue
            )
        # hists_uf is a list of hist objects or a list of a list of hist objects

        # compute the average of each bin
        h_uf = myhu.average_histograms(hists_uf)

        if norm is not None:
            # rescale the unfolded histograms to the norm
            if all_iterations:
                for hh in h_uf:
                    hh = myhu.renormalize_hist(hh, norm=norm, density=density, flow=True)
            else:
                h_uf = myhu.renormalize_hist(h_uf, norm=norm, density=density, flow=True)

        # bin correlations
        bin_corr = None
        if len(hists_uf) > 1:
            bin_corr = myhu.get_bin_correlations_from_hists(hists_uf)

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

#####
# helper function to instantiate an unfolder from a previous result directory
def load_unfolder(
    # Path to the arguments json config
    fpath_arguments, # str
    # List of observables. If empty, use the observables from arguments config
    observables=[], # list of str
    # Observable configuration. If empty, use the config from arguments
    obsConfig={}, # dict
    # Binning configuration. If empty, use the config from arguments
    ):

    logger.info(f"Read arguments from {fpath_arguments}")
    args_d = util.read_dict_from_json(fpath_arguments)

    # observables
    if not observables:
        # if not specified, take the list of observabes from arguments config
        observables[:] = args_d['observables'] + args_d['observables_extra']

    # configuration for observables
    if not obsConfig:
        # if not provided, use the one from the arguments config
        observable_config = args_d['observable_config']
        logger.info(f"Get observable config from {observable_config}")
        obsConfig.update(util.read_dict_from_json(observable_config))

        if not obsConfig:
            logger.error("Failed to load observable config. Abort...")
            return None

    # reco level variable names
    varnames_reco = [obsConfig[k]['branch_det'] for k in observables]

    # truth level variable names
    varnames_truth = [obsConfig[k]["branch_mc"] for k in observables]

    # reweighter
    rw = None
    if args_d['reweight_data']:
        var_lookup = np.vectorize(lambda v: obsConfig[v]["branch_mc"])
        rw = copy(reweight.rw[args_d["reweight_data"]])
        rw.variables = var_lookup(rw.variables)

    logger.info("Construct unfolder")
    unfolder = OmniFoldTTbar(
        varnames_reco,
        varnames_truth,
        filepaths_obs = args_d['data'],
        filepaths_sig = args_d['signal'],
        filepaths_bkg = args_d['background'],
        normalize_to_data = args_d['normalize'],
        dummy_value = args_d['dummy_value'],
        weight_type = args_d['weight_type'],
        truth_known = args_d['truth_known'],
        outputdir = args_d['outputdir'],
        data_reweighter = rw
    )

    # read unfolded event weights
    fnames_uw = os.path.join(args_d['outputdir'], "weights_unfolded.npz")
    unfolder.load(fnames_uw)

    return unfolder
