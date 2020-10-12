import numpy as np
import logging
from observables import observable_dict

def parse_input_name(fname):
    fname_list = fname.split('*')
    if len(fname_list) == 1:
        return fname, 1.
    else:
        try:
            name = fname_list[0]
            rwfactor = float(fname_list[1])
        except ValueError:
            name = fname_list[1]
            rwfactor = float(fname_list[0])
        except:
            print('Unknown data file name {}'.format(fname))

        return name, rwfactor

def load_dataset(file_names, array_name='arr_0', allow_pickle=True, encoding='bytes', weight_columns=[]):
    """
    Load and return a structured numpy array from a list of npz files
    """
    data = None
    for fname in file_names:
        fn, rwfactor = parse_input_name(fname)

        npzfile = np.load(fn, allow_pickle=allow_pickle, encoding=encoding)
        di = npzfile[array_name]
        if len(di)==0:
            raise RuntimeError('There is no events in input file {}'.format(fname))

        # rescale total event weights for this input file
        if rwfactor != 1.:
            for wname in weight_columns:
                try:
                    di[wname] = di[wname]*rwfactor
                except ValueError:
                    print('Unknown field name {}'.format(wname))
                    continue

        if data is None:
            data = di
        else:
            data  = np.concatenate([data, di])
        npzfile.close()

    return data

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

def get_variable_arr(ntuple, variable):
    # Reture a 1-d numpy array of the 'variable' from a structured array 'ntuple'
    # np.hstack() ensures the returned array is always of the shape (length,)
    # in case the original array in ntuple is a column array

    # check if variable is in the structured array
    if variable in ntuple.dtype.names:
        return np.hstack( ntuple[variable] )
    # special cases
    elif '_px' in variable:
        var_pt = variable.replace('_px', '_pt')
        var_phi = variable.replace('_px', '_phi')
        return np.hstack(ntuple[var_pt])*np.cos(np.hstack(ntuple[var_phi]))
    elif '_py' in variable:
        var_pt = variable.replace('_py', '_pt')
        var_phi = variable.replace('_py', '_phi')
        return np.hstack(ntuple[var_pt])*np.sin(np.hstack(ntuple[var_phi]))
    elif variable == 'pz':
        var_pt = variable.replace('_pz', '_pt')
        var_eta = variable.replace('_pz', '_eta')
        return np.hstack(ntuple[var_pt])*np.sinh(np.hstack(ntuple[var_eta]))
    else:
        raise RuntimeError("Unknown variable {}".format(variable))

def prepare_data_multifold(ntuple, variables, standardize=False, reshape1D=False):
    """
    ntuple: structure array from root tree

    return an numpy array of the shape (n_events, n_variables)
    """
    data = np.hstack([np.vstack( get_variable_arr(ntuple,var) ) for var in variables])

    if standardize:
        data = (data - np.mean(data, axis=0))/np.std(data, axis=0)

    if reshape1D and len(variables)==1:
        # In case only one variable, reshape a column array into a 1D array of the shape (n_events,)
        data = data.reshape(len(data))

    return data

def read_dataset(dataset, variables, label, weight_name=None, standardize=False):
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
    # features
    X = np.hstack([np.vstack( get_variable_arr(dataset,var) ) for var in variables])
    if standardize:
        X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    # label
    Y = np.full(len(X), label)

    # weight
    W = np.ones(len(X)) if weight_name is None else np.hstack(dataset[weight_name])

    return X, Y, W

def reweight_sample(weights_orig, dataset, reweight_type=None):
    if reweight_type is None:
        return weights_orig

    elif reweight_type == 'linear_th_pt':
        # truth-level hadronic top pt
        varname_thpt = observable_dict['th_pt']['branch_mc']
        th_pt = get_variable_arr(dataset, varname_thpt)
        # reweight function
        rw = 1 + 1/800.*th_pt
        return weights_orig * rw

    elif reweight_type == 'gaussian_bump':
        # truth-level variable name of the ttbar mass
        varname_mtt = observable_dict['mtt']['branch_mc']
        mtt = get_variable_arr(dataset, varname_mtt)
        # reweight function
        k = 0.5
        m0 = 800
        sigma = 100
        rw = 1 + k*np.exp( -( (mtt-m0)/sigma )**2 )
        return weights_orig * rw

    else:
        raise RuntimeError("Unknown sample reweighting type: {}".format(reweight_type))

def set_up_bins(xmin, xmax, nbins):
    bins = np.linspace(xmin, xmax, nbins+1)
    midbins = (bins[:-1]+bins[1:])/2
    binwidth = bins[1]-bins[0]

    return bins, midbins, binwidth

def normalize_histogram(bin_edges, hist, hist_unc=None):
    binwidths = bin_edges[1:] - bin_edges[:-1]
    norm = np.dot(hist, binwidths)
    hist /= norm
    if hist_unc is not None:
        hist_unc /= norm

def normailize_stacked_histogrms(bin_edges, hists, hists_unc=None):
    binwidths = bin_edges[1:] - bin_edges[:-1]
    hstacked = np.asarray(hists).sum(axis=0)
    norm = np.dot(hstacked, binwidths)

    if hists_unc is None:
        hists_unc = [None]*len(hists)
    else:
        assert(len(hists_unc)==len(hists))

    for h, herr in zip(hists, hists_unc):
        h /= norm
        if herr is not None:
            herr /= norm

def add_histograms(h1, h2=None, h1_err=None, h2_err=None, c1=1., c2=1.):
    hsum = c1*h1
    if h2 is not None:
        assert(len(h1)==len(h2))
        hsum += c2*h2

    sumw2 = np.zeros_like(h1)

    if h1_err is not None:
        assert(len(h1_err)==len(h1))
        sumw2 += (c1*h1_err)**2

    if h2_err is not None:
        assert(len(h2_err)==len(h2))
        sumw2 += (c2*h2_err)**2

    hsum_err = np.sqrt(sumw2)

    return hsum, hsum_err

def divide_histograms(h_numer, h_denom, h_numer_err=None, h_denom_err=None):
    ratio = np.divide(h_numer, h_denom, out=np.zeros_like(h_denom), where=(h_denom!=0))

    # bin errors
    if h_numer_err is None:
        h_numer_err = np.zeros_like(h_numer_err)
    if h_denom_err is None:
        h_denom_err = np.zeros_like(h_denom_err)

    #rerrsq = (h_numer_err**2 + h_denom_err**2 * ratio**2) / h_denom**2
    rerrsq = np.divide(h_numer_err**2 + h_denom_err**2 * ratio**2, h_denom**2, out=np.zeros_like(h_denom), where=(h_denom!=0))

    ratio_err = np.sqrt(rerrsq)

    return ratio, ratio_err

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

def getLogger(name, level=logging.DEBUG):
    msgfmt = '%(asctime)s %(levelname)-7s %(name)-15s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logger = logging.getLogger(name)
    logging.basicConfig(format = msgfmt, datefmt = datefmt)
    logger.setLevel(level)
    return logger

def triangular_discr(histogram_1, histogram_2):
    if len(histogram_1) != len(histogram_2):
        raise RuntimeError("Input histograms are not of the same size")

    delta = 0.
    for p, q in zip(histogram_1, histogram_2):
        if q==0 and q==0:
            continue
        delta += ((p-q)**2)/(p+q)*0.5

    return delta * 1000

def compute_triangular_discriminators(hist_ref, hists, labels):
    assert(len(hists)==len(labels))
    stamps = ["Triangular discriminator ($\\times 10^{-3}$):"]

    for h, l in zip(hists, labels):
        d = triangular_discr(h, hist_ref)
        stamps.append("{} = {:.3f}".format(l, d))

    return stamps
