import numpy as np
import tensorflow as tf

# The original omnifold
import OmniFold.omnifold as omnifold
import OmniFold.modplot as modplot

from util import read_dataset, prepare_data_multifold
from util import DataShufflerDet, DataShufflerGen
from util import set_up_bins, ibu

from plotting import plot_results

# Base class of OmniFold for non-negligible background
# Adapted from the original omnifold:
# https://github.dcom/ericmetodiev/OmniFold/blob/master/omnifold.py
class OmniFoldwBkg(object):
    def __init__(self, variables_det, variables_gen, wname, it, outdir='./'):
        assert(len(variables_det)==len(variables_gen))
        self.vars_det = variables_det # list of detector-level variables
        self.vars_gen = variables_gen # list of truth-level variables
        self.weight_name = wname # name of sample weight
        self.label_obs = 1
        self.label_sig = 0
        self.label_bkg = 0
        # number of iterationns
        self.iterations = it
        # feature, label, and weight arrays
        self.X_det = None # detector-level feature array
        self.Y_det = None # detector-level label array
        self.X_gen = None # simulation-level feature array
        self.Y_gen = None # simulation-level label array
        self.wdata = None # ndarray for data sample weights
        self.winit = None # ndarray for initial signal simulation weights
        self.wbkg = None  # ndarray for background simulation sample weights
        # unfolded event weights
        self.ws_unfolded = None
        # output directory
        self.outdir = outdir.strip('/')+'/'

    def _rescale_sample_weights(self):
        ndata = self.wdata.sum()
        nsim = self.winit.sum()

        if self.wbkg is not None:
            nsim += self.wbkg.sum()

        self.winit *= ndata/nsim
        if self.wbkg is not None:
            self.wbkg *= ndata/nsim
        
    def _set_up_model_i(self, i, model_config, model_filepath=None):
        """ Set up the model for the ith iteration of reweighting
        Args:
            i: int for iteration
            model_config: a ntuple with two elements. model_config[0] is the architecture of model, model_config[1] is a dictionary for the model's arguments
            model_filepath: str, file path to save the trained model
        Return:
            model: classifier object
        """
        # set file path
        if model_filepath is not None:
            model_config[1]['filepath'] = model_filepath.format(i) + '_Epoch-{epoch}'

        # define model
        model = model_config[0](**model_config[1])

        # load model from previous iteration if not the first one
        if i > 0 and model_filepath is not None:
            model.load_weights(model_filepath.format(i-1))
    
        return model

    def _set_up_model_det_i(self, i, model_config, model_filepath=None):
        return self._set_up_model_i(i, model_config, model_filepath)

    def _set_up_model_sim_i(self, i, model_config, model_filepath=None):
        return self._set_up_model_i(i, model_config, model_filepath)

    def _reweight_step1(self, X, Y, w, model, filepath, fitargs, val_data=None):
        # the original one
        return omnifold.reweight(X, Y, w, model, filepath, fitargs, val_data=val_data)

    def _reweight_step2(self, X, Y, w, model, filepath, fitargs, val_data=None):
        # the original one
        return omnifold.reweight(X, Y, w, model, filepath, fitargs, val_data=val_data)
    
    def preprocess_det(self, dataset_obs, dataset_sig, dataset_bkg=None, standardize=True):
        """
        Args:
            dataset_obs, dataset_sig, dataset_bkg: structured numpy array whose field names are variables 
            standardize: bool, if true standardize feature array X
        """
        X_obs, Y_obs, self.wdata = read_dataset(dataset_obs, self.vars_det, label=self.label_obs)
        X_sig, Y_sig, self.winit = read_dataset(dataset_sig, self.vars_det, label=self.label_sig, weight_name=self.weight_name)

        self.X_det = np.concatenate((X_obs, X_sig))
        self.Y_det = np.concatenate((Y_obs, Y_sig))

        if dataset_bkg is not None:
            X_bkg, Y_bkg, self.wbkg = read_dataset(dataset_bkg, self.vars_det, label=self.label_bkg, weight_name=self.weight_name)
            self.X_det = np.concatenate((self.X_det, X_bkg))
            self.Y_det = np.concatenate((self.Y_det, Y_bkg))

        if standardize: # standardize X
            self.X_det = (self.X_det - np.mean(self.X_det, axis=0)) / np.std(self.X_det, axis=0)

        # make Y categorical
        self.Y_det = tf.keras.utils.to_categorical(self.Y_det)

        # reweight the sim and data to have the same total weight
        self._rescale_sample_weights()

        # TODO: plot?
        
    def preprocess_gen(self, dataset_sig, standardize=True):
        """
        Args:
            dataset_sig: structured numpy array whose field names are variables 
            standardize: bool, if true standardize feature array X
        """
        X_gen_sig = prepare_data_multifold(dataset_sig, self.vars_gen)
        nsim = len(X_gen_sig)

        self.X_gen = np.concatenate((X_gen_sig, X_gen_sig))

        if standardize: # standardize X
            self.X_gen = (self.X_gen - np.mean(self.X_gen, axis=0)) / np.std(self.X_gen, axis=0)

        # make Y categorical
        self.Y_gen = tf.keras.utils.to_categorical(np.concatenate([np.ones(nsim), np.zeros(nsim)]))
    
    def omnifold(self, det_model, sim_model, fitargs, val=0.2, weights_filename=None):
        # initialize the truth weights to the prior
        ws = [self.winit]

        # split dataset for training and validation
        # detector level
        splitter_det = DataShufflerDet(len(self.X_det), val)
        X_det_train, X_det_val = splitter_det.shuffle_and_split(self.X_det)
        Y_det_train, Y_det_val = splitter_det.shuffle_and_split(self.Y_det)

        # simulation
        splitter_gen = DataShufflerGen(len(self.X_gen), val)
        X_gen_train, X_gen_val = splitter_gen.shuffle_and_split(self.X_gen)
        Y_gen_train, Y_gen_val = splitter_gen.shuffle_and_split(self.Y_gen)

        # model filepath
        model_det_fp = det_model[1].get('filepath', None)
        if model_det_fp is not None:
            model_det_fp = self.outdir + model_det_fp
        model_sim_fp = sim_model[1].get('filepath', None)
        if model_sim_fp is not None:
            model_sim_fp = self.outdir + model_sim_fp

        # start iterations
        for i in range(self.iterations):

            # set up models for this iteration
            model_det = self._set_up_model_det_i(i, det_model, model_det_fp)
            model_sim = self._set_up_model_sim_i(i, sim_model, model_sim_fp)

            # step 1: reweight sim to look like data
            w = np.concatenate((self.wdata, ws[-1]))
            if self.wbkg is not None:
                w = np.concatenate((w, self.wbkg))
            assert(len(w)==len(self.X_det))

            w_train, w_val = splitter_det.shuffle_and_split(w)

            rw = self._reweight_step1(X_det_train, Y_det_train, w_train, model_det, model_det_fp.format(i), fitargs, val_data=(X_det_val, Y_det_val, w_val))
            wnew = splitter_det.unshuffle(rw)
            if self.wbkg is not None:
                wnew = wnew[len(self.wdata):-len(self.wbkg)]
            else:
                wnew = wnew[len(self.wdata):]
            ws.append(wnew)

            # step 2: reweight the simulation prior to the learned weights
            w = np.concatenate((ws[-1], ws[-2]))
            w_train, w_val = splitter_gen.shuffle_and_split(w)

            rw = self._reweight_step2(X_gen_train, Y_gen_train, w_train, model_sim, model_sim_fp.format(i), fitargs, val_data=(X_gen_val, Y_gen_val, w_val))
            wnew = splitter_gen.unshuffle(rw)[len(ws[-1]):]
            ws.append(wnew)

        # save the weights
        if weights_filename is not None:
            np.save(weights_file, ws)

        self.ws_unfolded = ws[-1]

    def results(self, vars_dict, dataset_obs, dataset_sig, dataset_bkg=None, truth_known=False, normalize=False):
        """
        Args:
            vars_dict:
            dataset_obs/sig/bkg: structured numpy array labeled by variable names
        Return:
            
        """
        for varname, config in vars_dict.items():
            dataobs = np.hstack(dataset_obs[config['branch_det']])
            truth = np.hstack(dataset_obs[config['branch_mc']]) if truth_known else None
            
            sim_sig = np.hstack(dataset_sig[config['branch_det']])
            gen_sig = np.hstack(dataset_sig[config['branch_mc']])
            sim_bkg = np.hstack(dataset_bkg[config['branch_det']]) if dataset_bkg is not None else None
            #gen_bkg = np.hstack(dataset_sig[config['branch_mc']]) if dataset_bkg is not None else None

            # histograms
            # set up bins
            bins_det = np.linspace(config['xlim'][0], config['xlim'][1], config['nbins_det']+1)
            bins_mc = np.linspace(config['xlim'][0], config['xlim'][1], config['nbins_mc']+1)

            # observed distributions
            hist_obs, hist_obs_unc = modplot.calc_hist(dataobs, weights=self.wdata, bins=bins_det, density=False)[:2]

            # signal simulation
            hist_sim, hist_sim_unc = modplot.calc_hist(sim_sig, weights=self.winit, bins=bins_det, density=False)[:2]
            # FIXME: weights=winit?

            # background simulation
            if sim_bkg is not None:
                # negate background weights if it has been negated earlier
                wbkg = self.wbkg if self.wbkg.sum() > 0 else -self.wbkg
                hist_simbkg, hist_simbkg_unc = modplot.calc_hist(sim_bkg, weights=self.wbkg, bins=bins_det, density=False)[:2]
                # subtract background contribution from the observed data
                hist_obs -= hist_simbkg
                # TODO: uncertainties?

            # generated distribution (prior)
            hist_gen, hist_gen_unc = modplot.calc_hist(gen_sig, weights=self.winit, bins=bins_mc, density=False)[:2]
            # FIXME: weights=winit?

            # truth distribution if known
            hist_truth, hist_truth_unc = None, None
            if truth is not None:
                hist_truth, hist_truth_unc = modplot.calc_hist(truth, bins=bins_mc, density=False)[:2]

            # unfolded distributions
            # iterative Bayesian unfolding
            hist_ibu, hist_ibu_unc = ibu(hist_obs, sim_sig, gen_sig, bins_det, bins_mc, self.winit, it=self.iterations, density=False)

            # omnifold
            hist_of, hist_of_unc = modplot.calc_hist(gen_sig, weights=self.ws_unfolded, bins=bins_mc, density=False)[:2]

            # plot results
            plot_results(varname, bins_det, bins_mc,
                         (hist_obs,hist_obs_unc), (hist_sim,hist_sim_unc),
                         (hist_gen,hist_gen_unc), (hist_of,hist_of_unc),
                         (hist_ibu,hist_ibu_unc), (hist_truth, hist_truth_unc),
                         outdir = self.outdir, normalize = normalize,
                         **config)

##############################################################################
#############
# Approach 1
#############
# unfold as is first, then subtract the background histogram from the unfolded distribution if any observable of interest.

# preprocess_data: data vs mc signal only w/o background (?)
# classifier: data vs signal mc
# reweight: standard
# show result: subtract background histograms
        
class OmniFold_subHist(OmniFoldwBkg):
    def __init__(self, variables_det, variables_gen, wname, outdir='./'):
        OmniFoldwBkg.__init__(self, variables_det, variables_gen, wname, outdir)

    def preprocess_det(self, dataset_obs, dataset_sig, dataset_bkg=None, standardize=True):
        OmniFoldwBkg.preprocess_det(dataset_obs, dataset_sig, None, standardize)
    
    #def result(self): # fix me
    
#############
# Approach 2
#############
# unfold as is, but set the background event weights to be negative

# preprocess_data: set background weights to be negative
# background label is the same as data
# detector level (step 1 reweighting)
# show result: standard

class OmniFold_negW(OmniFoldwBkg):
    def __init__(self, variables_det, variables_gen, wname, outdir='./'):
        OmniFoldwBkg.__init__(self, variables_det, variables_gen, wname, outdir)
        # make background label as data
        self.label_bkg = self.label_obs

    def preprocess_det(self, dataset_obs, dataset_sig, dataset_bkg, standardize=True):
        OmniFoldwBkg.preprocess_det(dataset_obs, dataset_sig, dataset_bkg, standardize)
        # make mc background weight negative
        self.wbkg = -self.wbkg
    
#############
# Approach 3
#############
# add a correction term to the ratio for updating weights in step 1

# preprocess_data: standard w/ signal + background mc
# classifier: data vs mc
# reweight: correct the original weight from classifier
# show result: standard

class OmniFold_corR(OmniFoldwBkg):
    def __init__(self, variables_det, variables_gen, wname, outdir='./'):
        OmniFoldwBkg.__init__(self, variables_det, variables_gen, wname, outdir)

    # redefine step 1 reweighting
    def _reweight_step1(self, X, Y, w, model, filepath, fitargs, val_data=None):
        # validation data
        val_dict = {'validation_data': val_data} if val_data is not None else {}
        model.fit(X, Y, sample_weight=w, **fitargs, **val_dict)
        model.save_weights(filepath)
        preds = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,1]

        # concatenate validation predictions into training predictions
        if val_data is not None:
            preds_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,1]
            preds = np.concatenate((preds, preds_val))
            w = np.concatenate((w, val_data[2]))

        r = preds/(1 - preds + 10**-50)
        # correction term
        # FIXME
        nbkg = self.wbkg.sum() 
        nsig = self.wdata.sum() - nbkg
        cor = (r-1)*nbkg/nsim
        r += cor
        w *= np.clip(r, fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
        return w
    
#############
# Approach 4
#############
# approximate likelihood ratio of data vs signal and background vs signal separately with separate classifiers

# preprocess_data: (data, signal_mc) and (background_mc, signal_mc) separately
# classifier: data vs signal, background vs signal
# reweight: weight from data_vs_signal minus weight from background_vs_signal
# show result: standard
    
#############
# Approach 5
#############
# use a multi-class classifier to approximate the likelihood ratio of signal events in data and signal mc

# preprocess_data: standard w/ background mc
# classifier: mutli-class
# reweight: new_weight = old_weight * (y_data - y_bkg) / y_sig
# show result: standard

class OmniFold_multi(OmniFoldwBkg):
    def __init__(self, variables_det, variables_gen, wname, outdir='./'):
        OmniFoldwBkg.__init__(self, variables_det, variables_gen, wname, outdir)
        # new label for background
        self.label_bkg = 2

    # multi-class classifier for step 1 reweighting
    #def _set_up_model_det_i(self, i, model_config, model_filepath=None):
        # TODO

    # reweighting with multi-class classifer
    def _reweight_step1(self, X, Y, w, model, filepath, fitargs, val_data=None):
        # validation data
        val_dict = {'validation_data': val_data} if val_data is not None else {}
        model.fit(X, Y, sample_weight=w, **fitargs, **val_dict)
        model.save_weights(filepath)
        preds_obs = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,self.label_obs]
        preds_bkg = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,self.label_bkg]

        # concatenate validation predictions into training predictions
        if val_data is not None:
            preds_obs_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,self.label_obs]
            preds_obs = np.concatenate((preds_obs, preds_obs_val))
            preds_bkg_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,self.label_bkg]
            preds_bkg = np.concatenate((preds_bkg, preds_bkg_val))
            w = np.concatenate((w, val_data[2]))

        r = (preds_obs - preds_bkg) / (1 - preds_obs - preds_bkg + 10**-50)
        w *= np.clip(r, fitargs.get('weight_clip_min', 0.), fitargs.get('weight_clip_max', np.inf))
        return w
