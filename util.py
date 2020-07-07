#!/usr/bin/env python3

import numpy as np
import logging

def get_fourvector_array(pt_arr, eta_arr, phi_arr, e_arr, padding=True):
    """
    Combine and format pt, eta, phi, and e arrays into array of four vectors

    Each of the input array can be either an array of float or array of arrays
    with a shape of (nevents,) 

    If padding is set to True, pad the output array into homogeneous ndarray 
    with zeros, i.e. each event will have the same number of four vectors 
    Otherwise the output array will be an array of objects
    """

    # Check if the input arrays are of the same dimension
    if (not (pt_arr.shape==eta_arr.shape and pt_arr.shape==phi_arr.shape and pt_arr.shape==e_arr.shape)):
        raise Exception("Input arrays have to be of the same dimension.")

    # Stack arrays
    v4_arr = np.stack([pt_arr, eta_arr, phi_arr, e_arr], axis=1)
    
    # Determine if input arrays are arrays of floats or objects
    if pt_arr.dtype != np.dtype('O'):
        # array of float i.e. one four vector per event
        return v4_arr
    else:
        # array of objects (arrays)
        # i.e. each event can contain various number of four vectors

        stackarr = lambda v : np.stack([v[0],v[1],v[2],v[3]], axis=1)
        
        if not padding:
            v4_arr_new = np.asarray([stackarr(v4) for v4 in v4_arr])
            return v4_arr_new
        else:
            nv4max = max(len(v4[0]) for v4 in v4_arr)
            v4_arr_new = np.asarray([np.concatenate((stackarr(v4), np.zeros([nv4max-len(v4[0]),4]))) for v4 in v4_arr])
            return v4_arr_new


def prepare_data_omnifold(ntuple, padding=True):
    """
    ntuple: structure array from root tree

    return an numpy array of the shape (n_events, n_particles, 4)
    """

    # Total number of events
    nevents = len(ntuple)

    # TODO: check if ntuple['xxx'] exsits
    # jets
    jetv4 = get_fourvector_array(ntuple['jet_pt'], ntuple['jet_eta'], ntuple['jet_phi'], ntuple['jet_e'], padding=True)
    # shape: (nevents, NJetMax, 4)

    # lepton
    lepv4 = get_fourvector_array(ntuple['lep_pt'], ntuple['lep_eta'], ntuple['lep_phi'], ntuple['lep_e'], padding=True)
    # shape: (nevents, 4)

    # MET
    metv4 = get_fourvector_array(ntuple['met_met'], np.zeros(nevents), ntuple['met_phi'], ntuple['met_met'], padding=True)
    # shape: (nevent, 4)

    data = np.concatenate([np.stack([lepv4],axis=1), np.stack([metv4],axis=1), jetv4], axis=1)

    return data

def prepare_data_multifold(ntuple, variables, standardize=False, reshape1D=False):
    """
    ntuple: structure array from root tree

    return an numpy array of the shape (n_events, n_variables)
    """
    # TODO: check if ntuple[var] exsits
    data = np.hstack([np.vstack(ntuple[var]) for var in variables])

    if standardize:
        data = (data - np.mean(data, axis=0))/np.std(data, axis=0)

    if reshape1D and len(variables)==1:
        # In case only one variable, reshape a column array into a 1D array of the shape (n_events,)
        data = data.reshape(len(data))

    return data

def read_dataset(dataset, variables, label, weight_name=None):
    """
    Args:
        dataset: a structured numpy array labeled by variable names
        variables: a list of feature names
        label: int for class label
        weight_name: name of the variable for sample weights
    Return:
        X: 2-d numpy array of the shape (n_events, n_features)
        Y: 1-d numpy array for label
        W: 1-d numpy array for sample weights
    """
    X = prepare_data_multifold(dataset, variables)
    Y = np.full(len(X), label)
    W = np.ones(len(X)) if weight_name is None else np.hstack(dataset[weight_name])

    return X, Y, W

def set_up_bins(xmin, xmax, nbins):
    bins = np.linspace(xmin, xmax, nbins+1)
    midbins = (bins[:-1]+bins[1:])/2
    binwidth = bins[1]-bins[0]

    return bins, midbins, binwidth

# Data shuffle and split for step 1 (detector-level) reweighting
# Adapted based on https://github.com/ericmetodiev/OmniFold/blob/master/omnifold.py#L54-L59
class DataShufflerDet(object):
    def __init__(self, ndata, val):
        """
        ndata: number of events
        val: percentage of data for validation
        """
        self.nval = int(ndata * val)
        self.perm = np.random.permutation(ndata)
        self.invperm = np.argsort(self.perm)

    def shuffle(self, arr):
        assert(len(arr)==len(self.perm))
        return arr[self.perm]

    def shuffle_and_split(self, arr):
        assert(len(arr)==len(self.perm))
        arr_train = arr[self.perm[:-self.nval]]
        arr_val = arr[self.perm[-self.nval:]]
        return arr_train, arr_val

    def unshuffle(self, arr):
        assert(len(arr)==len(self.invperm))
        return arr[self.invperm]

# Data shuffle and split for step 2 (simulation-level) reweighting
# Adapted based on https://github.com/ericmetodiev/OmniFold/blob/master/omnifold.py#L69-L86
class DataShufflerGen(DataShufflerDet):
    # The dataset consists of two identical sets of events
    # A event is labelled as "1" if its event weight is updated from step 1
    # Otherwise it is labelled as "0"
    def __init__(self, ndata, val):
        """
        ndata: number of total events
        val: percentage of data for validation
        """     
        nevt = int(ndata/2) # number of events labeled either as "1" or as "0"
        nval = int(val*nevt)
        # initial permuation for events with new weights and events with old weights
        baseperm0 = np.random.permutation(nevt)
        baseperm1 = baseperm0 + nevt # same permutation but with an offset

        # wish to have the same event end up in the training (or validation) dataset
        trainperm = np.concatenate((baseperm0[:-nval], baseperm1[:-nval]))
        valperm = np.concatenate((baseperm0[-nval:], baseperm1[-nval:]))
        np.random.shuffle(trainperm)
        np.random.shuffle(valperm)

        self.nval = len(valperm) # number of validation events in the entire dataset
        self.perm = np.concatenate((trainperm, valperm))
        self.invperm = np.argsort(self.perm)


#########################################################
# iterative Bayesian unfolding
# based on the implementation in https://github.com/ericmetodiev/OmniFold/blob/master/ibu.py
def ibu(hist_obs, datasim, datagen, bins_det, bins_gen, winit, it, density=False, nresamples=50):
    binwidth_det = bins_det[1]-bins_det[0]
    binwidth_gen = bins_gen[1]-bins_gen[0]

    # response matrix
    r = np.histogram2d(datasim, datagen, bins=(bins_det, bins_gen))[0]
    r /= (r.sum(axis=0) + 10**-50)

    # initialize the truth distribution to the prior
    phis = [np.histogram(datagen, bins=bins_gen, weights=winit, density=density)[0]]

    # iterate
    for i in range(it):

        # update the estimate for the matrix m
        m = r * phis[-1]
        m /= (m.sum(axis=1)[:,np.newaxis] + 10**-50)

        # update the estimate for the truth distribution
        phis.append(np.dot(m.T, hist_obs)*binwidth_det/binwidth_gen)

    hist_ibu = phis[-1]

    # statistical uncertainty on the IBU distribution only from uncertainty on the prior
    # TODO
    hist_ibu_unc = np.zeros(len(hist_ibu))
    
    return hist_ibu, hist_ibu_unc

def getLogger(name, level=logging.DEBUG):
    msgfmt = '%(asctime)s %(levelname)-7s %(name)-35s %(message)s'
    datefmt = '%H:%M:%S'
    logger = logging.getLogger(name)
    logging.basicConfig(format = msgfmt, datefmt = datefmt)
    logger.setLevel(level)
    return logger
