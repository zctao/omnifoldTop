import os
import glob
import numpy as np
import h5py
from copy import copy

import util
import plotter
import reweight
import histogramming as myhu
import FlattenedHistogram as fh
from datahandler import getDataHandler
from omnifold import omnifold
import modelUtils
import preprocessor

import logging
logger = logging.getLogger('OmniFoldTTbar')
logger.setLevel(logging.DEBUG)

from numpy.random import default_rng
rng = default_rng()

def read_weights_from_npz(filepaths_weights):

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
        # output directory
        outputdir = None,
        # data reweighting for stress test
        data_reweighter=None,
        # type of event weights
        weight_type_data='nominal',
        weight_type_mc='nominal',
        # If or not use toy data handler
        use_toydata=False,
        # Flags for efficiency and acceptance corrections
        correct_efficiency=False,
        correct_acceptance=False,
        # max dR for matching reco and truth tops
        match_dR=None
    ):

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

        # unfolded weights
        self.unfolded_weights = None
        # h5py file
        self.file_uw = None

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
            data_reweighter = data_reweighter,
            weight_type_data = weight_type_data,
            weight_type_mc = weight_type_mc,
            use_toydata = use_toydata,
            correct_efficiency = correct_efficiency,
            correct_acceptance = correct_acceptance,
            match_dR = match_dR
        )

    def __del__(self):
        if self.file_uw is not None:
            self.file_uw.close()

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
        data_reweighter = None, # reweight.Reweighter
        weight_type_data = 'nominal', # str, optional
        weight_type_mc = 'nominal', # str, optional
        use_toydata = False, # bool, optional
        correct_efficiency = False, # bool
        correct_acceptance = False, # bool
        match_dR = None, # float
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
            reweighter = data_reweighter,
            weight_type = weight_type_data,
            use_toydata = use_toydata,
            match_dR = match_dR # For pseudo data
            )
        logger.info(f"Total number of observed events: {len(self.handle_obs)}")

        # Signal MC simulation
        logger.info(f"Load signal simulation files: {' '.join(filepaths_sig)}")
        self.handle_sig = getDataHandler(
            filepaths_sig, vars_reco, vars_truth,
            weight_type = weight_type_mc,
            use_toydata = use_toydata,
            match_dR = match_dR
            )
        logger.info(f"Total number of signal events: {len(self.handle_sig)}")

        # Background MC simulation if needed
        if filepaths_bkg:
            logger.info(f"Load background simulation files: {' '.join(filepaths_bkg)}")
            # only reco level events are needed
            self.handle_bkg = getDataHandler(
                filepaths_bkg, vars_reco, [],
                weight_type = weight_type_mc,
                use_toydata = use_toydata
                )
            logger.info(f"Total number of background events: {len(self.handle_bkg)}")

        # Simulated background events to be mixed with pseudo data for testing
        if filepaths_obsbkg:
            logger.info(f"Load background simulation files to be mixed with data: {' '.join(filepaths_obsbkg)}")
            # only reco level events are needed
            self.handle_obsbkg = getDataHandler(
                filepaths_obsbkg, vars_reco, [],
                weight_type = weight_type_data,
                use_toydata = use_toydata
                )
            logger.info(f"Total number of background events mixed with data: {len(self.handle_obsbkg)}")

        ###
        self.with_efficiency_correction = correct_efficiency
        if not correct_efficiency:
            self.remove_events_failing_reco()

        self.with_acceptance_correction = correct_acceptance
        if not correct_acceptance:
            self.remove_events_failing_truth()

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

    def _get_input_arrays(self):
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

        return arr_data, arr_sim, arr_gen

    def _get_event_weights(self, resample_data=False, resample_mc=False, standardize=True):
        logger.debug("Prepare event weights")

        wdata = self.handle_obs.get_weights(bootstrap=resample_data, valid_only=False)

        if standardize:
            logger.debug("Standardize data weights to mean of one for training")
            # exclude dummy value when calculating mean
            wmean_obs = np.mean(wdata[self.handle_obs.pass_reco])
            wdata /= wmean_obs

        if self.handle_obsbkg is not None:
            wobsbkg = self.handle_obsbkg.get_weights(bootstrap=resample_data, valid_only=False)
            if standardize: # rescale by the same factor as data
                wobsbkg /= wmean_obs

            wdata = np.concatenate([wdata, wobsbkg])

        # add background simulation as observed data but with negative weights
        if self.handle_bkg is not None:
            wbkg = self.handle_bkg.get_weights(bootstrap=resample_mc, valid_only=False)
            if standardize:
                # rescale by the same factor as data
                wbkg /= wmean_obs

            wdata = np.concatenate([wdata, -1*wbkg])

        # signal simulation
        # reco level
        wsim = self.handle_sig.get_weights(valid_only=False)

        if resample_mc:
            w_bsmc = rng.poisson(1, size=len(wsim))
            wsim *= w_bsmc

        if standardize:
            # rescale by the same factor as data
            wsim /= wmean_obs

            #TODO check alternative: divide by its own mean
            #wmean_sim = np.mean(wsim[self.handle_sig.pass_reco])
            #wsim /= wmean_sim

        # truth level
        wgen = self.handle_sig.get_weights(valid_only=False, reco_level=False)

        if resample_mc:
            # use the same resampling weights
            wgen *= w_bsmc

        if standardize:
            # rescale by the same factor as data
            wgen /= wmean_obs
            # this is what's been done previously
            #wmean_gen = np.mean(wgen[self.handle_sig.pass_truth])
            #wgen /= wmean_gen

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
        resample_mc=False,
        nruns=1,
        resample_everyrun=False,
        model_type='dense_100x3',
        save_models=True,
        load_previous_iteration=True,
        load_models_from='',
        fast_correction=False,
        batch_size=256,
        plot_status=False, # If True, make extra plots for monitoring/debugging
        resume_training=False, # If True, load trained models if available, continue to train for more runs/steps if needed
        dummy_value=-99.,
    ):
        """
        Run unfolding
        """

        # preprocess data and weights
        X_data, X_sim, X_gen = self._get_input_arrays()
        w_data, w_sim, w_gen = self._get_event_weights(resample_data=resample_data, resample_mc=resample_mc)
        passcut_data, passcut_sim, passcut_gen = self._get_event_flags()

        # total weights for rescaling the unfolded weights
        sumw_data = w_data[passcut_data].sum()
        sumw_sim_valid = w_sim[passcut_sim].sum()
        sumw_sim_matched = w_sim[passcut_sim & passcut_gen].sum()
        sumw_gen_matched = w_gen[passcut_sim & passcut_gen].sum()
        fscale_unfolded = sumw_data * sumw_sim_matched / sumw_sim_valid / sumw_gen_matched

        # preprocessing
        p = preprocessor.get()

        # step 1: mapping
        X_data, X_data_order = p.feature_preprocess(X_data)
        X_sim, X_sim_order = p.feature_preprocess(X_sim)
        X_gen, X_gen_order = p.feature_preprocess(X_gen)

        # step 2: dummy values

        X_data[~passcut_data] = dummy_value
        X_sim[~passcut_sim] = dummy_value
        X_gen[~passcut_gen] = dummy_value

        # step 3: weight preprocessing

        w_data = p.preprocess_weight(X_data, w_data, X_data_order)
        w_sim = p.preprocess_weight(X_sim, w_sim, X_sim_order)
        w_gen = p.preprocess_weight(X_gen, w_gen, X_gen_order)

        # step 4 normalization

        assert(np.array_equal(X_data_order, X_sim_order) and np.array_equal(X_sim_order, X_gen_order))

        X_data[passcut_data], X_sim[passcut_sim], X_gen[passcut_gen] = p.apply_normalizer(X_data[passcut_data], X_sim[passcut_sim], X_gen[passcut_gen], X_data_order)

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

        # open an h5py file
        self.file_uw = h5py.File(
            os.path.join(self.outdir, "weights_unfolded.h5"), 'w'
            )

        uw_shape_per_run = (niterations, np.count_nonzero(passcut_gen))
        uw_shape = (nruns * modelUtils.n_models_in_parallel,) + uw_shape_per_run

        self.unfolded_weights = self.file_uw.create_dataset(
            "unfolded_weights", shape=uw_shape, chunks = (1,)+uw_shape_per_run,
            dtype=np.float)

        for ir in range(nruns):
            logger.info(f"Run #{ir}")

            # model directory
            model_dir = os.path.join(self.outdir, "Models", f"run{ir}")

            if resume_training:
                if os.path.isdir(model_dir):
                    # load from trained models
                    load_model_dir = model_dir
                else:
                    # continue training
                    load_model_dir = ''
            elif load_models_from:
                load_model_dir = os.path.join(load_models_from, "Models", f"run{ir}")
            else:
                load_model_dir = ''

            if save_models and self.outdir:
                save_model_dir = model_dir
                # save_model_dir is igored in omnifold() if models are loaded from load_model_dir

            if (resample_data or resample_mc) and resample_everyrun:
                # fluctuate data weights
                w_data, w_sim, w_gen = self._get_event_weights(resample_data=resample_data, resample_mc=resample_mc)

            # omnifold
            self.unfolded_weights[ir*modelUtils.n_models_in_parallel:(ir+1)*modelUtils.n_models_in_parallel,:,:] = omnifold(
                X_data, X_sim, X_gen,
                w_data, w_sim, w_gen,
                passcut_data, passcut_sim, passcut_gen,
                niterations = niterations,
                model_type = model_type,
                save_models_to = save_model_dir,
                load_models_from = load_model_dir,
                continue_training = resume_training,
                start_from_previous_iter=load_previous_iteration,
                fast_correction = fast_correction,
                plot = plot_status and ir==0, # only make plots for the first run
                batch_size = batch_size,
                feature_names_sim = X_sim_order,
                feature_names_gen = X_gen_order,
            )

            if plot_status:
                logger.info("Plot model training history")
                for csvfile in glob.glob(os.path.join(save_model_dir, '*.csv')):
                    logger.info(f"  Plot training log {csvfile}")
                    plotter.plot_train_log(csvfile)

        # scale the unfolded weights so they are consistent with what is measured in data
        self.unfolded_weights[:] *= fscale_unfolded
        # TODO: scale all weights to a fixed norm as the data?

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

        try:
            self.load_h5(wfilelist)
        except:
            try:
                self.unfolded_weights = read_weights_from_npz(wfilelist)
            except:
                logger.critical(f"Cannot load weights from file: {wfilelist}")

        logger.debug(f"unfolded_weights.shape: {self.unfolded_weights.shape}")

    def load_h5(self, filepaths_unfolded_weights):
        if len(filepaths_unfolded_weights) == 1:
            self.file_uw = h5py.File(filepaths_unfolded_weights[0], 'r')
            self.unfolded_weights = self.file_uw["unfolded_weights"]
        else:
            raise NotImplementedError("TODO Virtual Datasets")

    def get_unfolded_hists_resamples(
        self,
        varnames, # str or list of str, name of the variable(s)
        bins, # ndarray or dict
        norm=None,
        all_iterations=False,
        iteration=-1, # default: the last iteration
        nresamples=None, # default: take all that are available
        density=False,
        absoluteValue=False, # bool or list of bool
        extra_cuts=None
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

        wprior = self.handle_sig.get_weights(valid_only=True, reco_level=False)

        for iresample in range(nresamples):
            if all_iterations:
                rw = self.unfolded_weights[iresample]
            else:
                rw = self.unfolded_weights[iresample][iteration]

            # truth-level prior weights
            if isinstance(varnames, list):
                assert(isinstance(bins, dict))
                h = self.handle_sig.get_histograms_flattened(varnames, bins, wprior*rw, density=density, norm=norm, absoluteValues=absoluteValue, extra_cuts=extra_cuts)
            else:
                h = self.handle_sig.get_histogram(varnames, bins, wprior*rw, density=density, norm=norm, absoluteValue=absoluteValue, extra_cuts=extra_cuts)

            hists_resample.append(h)

        return hists_resample

    def get_unfolded_distribution(
        self,
        varname,
        bins,
        norm=None,
        all_iterations=False,
        iteration=-1, # default: the last iteration
        nresamples=None, # default, take all that are available
        density=False,
        absoluteValue=False,
        extra_cuts=None
        ):

        hists_uf = self.get_unfolded_hists_resamples(
            varname,
            bins,
            norm=None,
            all_iterations=all_iterations,
            iteration=iteration,
            nresamples=nresamples,
            density=density,
            absoluteValue=absoluteValue,
            extra_cuts=extra_cuts
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

    def get_unfolded_distribution_multidim(
        self,
        varnames,
        bins_d,
        norm=None,
        iteration=-1, # default: the last iteration
        nresamples=None, # default, take all that are available
        density=False,
        absoluteValues=False,
        extra_cuts=None
        ):

        hists_md_uf = self.get_unfolded_hists_resamples(
            varnames,
            bins_d,
            norm=norm,
            iteration = iteration,
            nresamples=nresamples,
            density=density,
            absoluteValue=absoluteValues,
            extra_cuts=extra_cuts
            )

        # compute the average of each bin
        h_md_uf = fh.average_histograms(hists_md_uf)

        if norm:
            h_md_uf.renormalize(norm, density=density, flow=True)

        return h_md_uf

    def get_correlations_unfolded(self, varnames, irun=0, iteration=-1):
        # truth-level prior weights
        w_prior = self.handle_sig.get_weights(valid_only=True, reco_level=False)

        # weights after unfolding
        w_unf = w_prior * self.unfolded_weights[irun][iteration]

        return self.handle_sig.get_correlations(varnames, weights=w_unf)

    def plot_more(self):
        # Event weights
        logger.info("Plot event weights")
        plotter.plot_hist(
            os.path.join(self.outdir, 'Event_weights'),
            [self.handle_sig.get_weights(), self.handle_obs.get_weights()],
            label = ['Sim.', 'Data'],
            title = "Event weights",
            xlabel = 'w')

    def remove_events_failing_reco(self):
        logger.info("Remove events that fail reco-level selections")
        self.handle_obs.remove_events_failing_reco()
        self.handle_sig.remove_events_failing_reco()

        if self.handle_bkg is not None:
            self.handle_bkg.remove_events_failing_reco()

        if self.handle_obsbkg is not None:
            self.handle_obsbkg.remove_events_failing_reco()

    def remove_events_failing_truth(self):
        logger.info("Remove events that fail truth-level selections")
        self.handle_obs.remove_events_failing_truth()
        self.handle_sig.remove_events_failing_truth()

        if self.handle_bkg is not None:
            self.handle_bkg.remove_events_failing_truth()

        if self.handle_obsbkg is not None:
            self.handle_obsbkg.remove_events_failing_truth()

    # methods for setting underflow overflow flags
    def reset_underflow_overflow_flags(self):
        self.handle_obs.reset_underflow_overflow_flags()
        self.handle_sig.reset_underflow_overflow_flags()

        if self.handle_bkg is not None:
            self.handle_bkg.reset_underflow_overflow_flags()

        if self.handle_obsbkg is not None:
            self.handle_obsbkg.reset_underflow_overflow_flags()

    def update_underflow_overflow_flags(self, vname, bins):
        self.handle_obs.update_underflow_overflow_flags(vname, bins)
        self.handle_sig.update_underflow_overflow_flags(vname, bins)

        if self.handle_bkg is not None:
            self.handle_bkg.update_underflow_overflow_flags(vname, bins)

        if self.handle_obsbkg is not None:
            self.handle_obsbkg.update_underflow_overflow_flags(vname, bins)

    def clear_underflow_overflow_events(self):
        self.handle_obs.clear_underflow_overflow_events()
        self.handle_sig.clear_underflow_overflow_events()

        if self.handle_bkg is not None:
            self.handle_bkg.clear_underflow_overflow_events()

        if self.handle_obsbkg is not None:
            self.handle_obsbkg.clear_underflow_overflow_events()

    # helper function to clear all events in underflow and overflow bins
    def clearAllUnderflowOverflow(self, observables, fpath_binning, obs_config):
        logger.info("Remove events in underflow or overflow bins")

        if not os.path.isfile(fpath_binning):
            logger.error(f"Cannot access binning config file: {fpath_binning}")
            raise RuntimeError("Fail to clear events in underflow/oveflow bins")

        self.reset_underflow_overflow_flags()

        bins_reco_d = util.get_bins_dict(fpath_binning)
        bins_truth_d = util.get_bins_dict(fpath_binning)

        for ob in observables:
            logger.debug(f" {ob}")
            # Same binning at reco and truth level for now
            bins_reco = bins_reco_d[ob]
            bins_truth = bins_truth_d[ob]

            vname_reco = obs_config[ob]['branch_det']
            vname_truth = obs_config[ob]['branch_mc']

            self.update_underflow_overflow_flags(vname_reco, bins_reco)
            self.update_underflow_overflow_flags(vname_truth, bins_truth)

        self.clear_underflow_overflow_events()

# helper function to instantiate an unfolder from a previous result directory
def load_unfolder(
    # Path to the arguments json config
    fpath_arguments, # str
    # List of observables. If empty, use the observables from arguments config
    observables=[], # list of str
    # Observable configuration. If empty, use the config from arguments
    obsConfig={}, # dict
    # File path to binning configuration. If None, use the config from arguments if available
    fpath_binning=None,
    # Flag for renormalizing sample. If True, normalize simulation weights to data.
    # If None, set the flag according to the run arguments
    normalize_to_data = None, # bool
    args_update = {}, # dict, new arguments to overwrite the ones read from fpath_arguments if any
    ):

    logger.info(f"Read arguments from {fpath_arguments}")
    args_d = util.read_dict_from_json(fpath_arguments)

    if args_update:
        args_d.update(args_update)

    # observables
    if not observables:
        # if not specified, take the list of observabes from arguments config
        observables[:] = set(args_d['observables'] + args_d['observables_extra'])

    observable_all = set(args_d['observables']+args_d['observables_extra']+observables)
    observable_extra = observable_all.difference(args_d['observables'])

    # configuration for observables
    if not obsConfig:
        # if not provided, use the one from the arguments config
        observable_config = args_d['observable_config']
        logger.info(f"Get observable config from {observable_config}")
        obsConfig.update(util.read_dict_from_json(observable_config))

        if not obsConfig:
            logger.error("Failed to load observable config. Abort...")
            return None

    # binning configuration
    if fpath_binning is None:
        fpath_binning = args_d['binning_config']

    # reco level variable names
    varnames_reco = [obsConfig[k]['branch_det'] for k in args_d['observables']]
    varnames_reco_extra = [obsConfig[k]['branch_det'] for k in observable_extra]

    # truth level variable names
    varnames_truth = [obsConfig[k]["branch_mc"] for k in args_d['observables']]
    varnames_truth_extra = [obsConfig[k]["branch_mc"] for k in observable_extra]

    # reweighter
    rw = None
    if args_d['reweight_data']:
        var_lookup = np.vectorize(lambda v: obsConfig[v]["branch_mc"])
        rw = copy(reweight.rw[args_d["reweight_data"]])
        rw.variables = var_lookup(rw.variables)

    # normalize simulation weights
    if normalize_to_data is None:
        normalize_to_data = args_d['normalize']

    # for backward compatibility
    if args_d.get('dummy_value') is not None:
        args_d['correct_acceptance'] = True

    if args_d.get('weight_type') is not None:
        args_d['weight_mc'] = args_d['weight_type']
        if args_d.get('weight_data') is None:
            args_d['weight_data'] = 'nominal'

    logger.info("Construct unfolder")
    unfolder = OmniFoldTTbar(
        varnames_reco,
        varnames_truth,
        variables_reco_extra = varnames_reco_extra,
        variables_truth_extra = varnames_truth_extra,
        filepaths_obs = args_d['data'],
        filepaths_sig = args_d['signal'],
        filepaths_bkg = args_d['background'],
        filepaths_obsbkg = args_d['bdata'],
        normalize_to_data = normalize_to_data,
        weight_type_data = args_d['weight_data'],
        weight_type_mc = args_d['weight_mc'],
        truth_known = args_d['truth_known'],
        outputdir = args_d['outputdir'],
        data_reweighter = rw,
        correct_efficiency = args_d.get('correct_efficiency', False),
        correct_acceptance = args_d['correct_acceptance'],
        match_dR = args_d.get('match_dR')
    )

    if args_d['exclude_flow']:
        try:
            unfolder.clearAllUnderflowOverflow(observables, fpath_binning, obsConfig)
        except:
            return None

    # read unfolded event weights
    fnames_uw = args_d.get("unfolded_weights")
    if fnames_uw is None:
        fnames_uw = os.path.join(args_d['outputdir'], "weights_unfolded")

        if os.path.isfile(fnames_uw+'.h5'):
            fnames_uw += '.h5'
        elif os.path.isfile(fnames_uw+'.npz'):
            fnames_uw += '.npz'

    unfolder.load(fnames_uw)

    return unfolder
