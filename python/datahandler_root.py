import uproot
import numpy as np
import numpy.lib.recfunctions as rfn

import datahandler as dh
from datahandler import DataHandler

def MeVtoGeV(array):
    """
    Convert unit from MeV to GeV

    Parameters
    ----------
    array : numpy structured array

    """
    for fname in list(array.dtype.names):
        # jet_pt, jet_e, met_met, mwt, lep_pt, lep_m
        isObjectVar = fname in ['jet_pt', 'jet_e', 'met_met', 'mwt', 'lep_pt', 'lep_m']
        # MC_*_afterFSR_[pt,m,E, Ht, pout]
        isPartonVar = fname.startswith('MC_') and (
            fname.endswith('_pt') or fname.endswith('_m') or
            fname.endswith('_E') or fname.endswith('_Ht') or
            fname.endswith('_pout')
            )

        if isObjectVar or isPartonVar:
            array[fname] /= 1000.

def setDummyValue(array, masks, dummy_value):
    """
    Set dummy value of entries in array that are masked by masks

    Parameters
    ----------
    array : numpy structured array
    masks : numpy ndarray
    dummy_value : float
    """

    if array.dtype.names is None:
        array[masks] = dummy_value
    else:
        for vname in list(array.dtype.names):
            # avoid modifying event flags
            if vname in ['isDummy', 'isMatched']:
                continue

            array[vname][masks] = dummy_value

def load_dataset_root(
        file_names,
        tree_name,
        variable_names = [],
        weight_type = 'nominal',
        dummy_value = None
    ):
    """
    Load data from a list of ROOT files
    Return a structured numpy array of data and a numpy ndarray as masks

    Parameters
    ----------
    file_names : str or file-like object; or sequence of str or file-like objects
        List of root files to load
    tree_name : str
        Name of the tree in root files
    variable_names : list of str, optional
        List of variables to read. If not provided, read all available variables
    weight_name : str, default None
        Name of the event weights. If not None, add it to the list of variables
    dummy_value : float, default None
        Dummy value for setting events that are flagged as dummy. If None, only
        include events that are not dummy in the array
    """
    if isinstance(file_names, str):
        file_names = [file_names]

    intrees = [fname + ':' + tree_name for fname in file_names]

    if variable_names:
        branches = list(variable_names)

        # event weights
        wvarlist = branches_for_weights(weight_type)
        branches.append(wvarlist)

        # flags for identifying events
        branches += ['isMatched', 'isDummy']

        # in case of KLFitter
        branches.append('klfitter_logLikelihood')

        # for checking invalid value in truth trees
        branches.append('MC_thad_afterFSR_y')
    else:
        # load everything
        branches = None

    data_array = uproot.lazy(intrees, filter_name=branches)
    # variables in branches but not in intrees are ignored

    # convert awkward array to numpy array for now
    # can probably use awkward array directly once it is more stable
    data_array = data_array.to_numpy()

    # convert units
    MeVtoGeV(data_array)

    #####
    # event selection flag
    pass_sel = (data_array['isDummy'] == 0)

    # In case of reco variables with KLFitter, cut on KLFitter logLikelihood
    if 'klfitter_logLikelihood' in data_array.dtype.names:
        pass_sel &= data_array['klfitter_logLikelihood'] > -52.

    # A special case where some matched events in truth trees contain invalid
    # (nan or inf) values
    if 'MC_thad_afterFSR_y' in data_array.dtype.names:
        invalid = np.isnan(data_array['MC_thad_afterFSR_y'])
        pass_sel &= (~invalid)
        #print("number of events with invalid value", np.sum(invalid))

    # TODO: return a masked array?
    return data_array, pass_sel

def branches_for_weights(weight_type='nominal'):
    """
    Return a list of branches/variables needed to retrieve event weights
    """
    branches = []

    if weight_type == 'nominal':
        branches = ['totalWeight_nominal','normedWeight', 'weight_mc']
    #elif: TODO for systematics from varying weights

    return branches

def retrieve_weights(array_reco, array_truth=None, weight_type='nominal'):
    """
    Retrieve event weights from data arrays
    """
    if not 'totalWeight_nominal' in array_reco.dtype.names:
        # not MC sample
        w_reco = np.ones(len(array_reco))
        w_truth = None
    else:
        # MC sample
        # normalization: cross section * luminosity / sumWeights
        # Currently sumWeights is not saved, instead there is a 'normedWeight'
        # 'normedWeight' = 'totalWeight_nominal' * 'xs_times_lumi' / sumWeights
        # the normalization factor
        f_norm = np.zeros(len(array_reco))
        np.divide(
            array_reco['normedWeight'], array_reco['totalWeight_nominal'],
            out=f_norm, where = array_reco['totalWeight_nominal']!=0
            )
        # TODO: store sumWeights in the ntuple
        # TODO: some events have zero weight due to 'weight_pileup' = 0

        w_reco = array_reco['normedWeight']

        if array_truth is not None:
            assert(len(array_truth) == len(array_reco))
            w_truth = array_truth['weight_mc'] * f_norm
            # note: truth weight now becomes 0 too when reco weight is 0
        else:
            w_truth = None

        # TODO: weight_type != 'nominal'

    return w_reco, w_truth

class DataHandlerROOT(DataHandler):
    """
    Load data from root files

    Parameters
    ----------
    filepaths :  str or sequence of str
        List of root file names to load
    variable_names : list of str, optional
        List of reco level variable names to read. If not provided, read all
    variable_names_mc : list of str, optional
        List of truth level variable names to read. If not provided, read all
    weights_name : str, optional 
        Name of event weights
    weights_name_mc : str, optional
        Name of mc weights
    treename_reco : str, default 'reco'
        Name of the reconstruction-level tree
    treename_truth : str, default 'parton'
        Name of the truth-level tree. If empty or None, skip loading the 
        truth-level tree
    dummy_value : float, default None
        Dummy value for setting events that are flagged as dummy. If None, only
        include events that are not dummy in the array
    """
    def __init__(
        self,
        filepaths,
        variable_names=[],
        variable_names_mc=[],
        weight_type='nominal',
        treename_reco='reco',
        treename_truth='parton',
        dummy_value=None
    ):
        # load data from root files
        variable_names = dh._filter_variable_names(variable_names)
        self.data_reco, self.pass_reco = load_dataset_root(
            filepaths, treename_reco, variable_names, weight_type, dummy_value
            )

        # truth variables if available
        if treename_truth:
            variable_names_mc = dh._filter_variable_names(variable_names_mc)
            self.data_truth, self.pass_truth = load_dataset_root(
                filepaths, treename_truth, variable_names_mc, weight_type,
                dummy_value
            )
        else:
            self.data_truth = None

        # event weightss
        self.weights, self.weights_mc = retrieve_weights(
            self.data_reco, self.data_truth, weight_type
            )

        # rename truth array variable name if it already exists in reco array
        if self.data_truth is not None:
            prefix = 'truth_'
            newnames = {}
            for fname in self.data_truth.dtype.names:
                if fname in self.data_reco.dtype.names:
                    newnames[fname] = prefix+fname
            self.data_truth = rfn.rename_fields(self.data_truth, newnames)

        # deal with events that fail selections
        if dummy_value is None:
            # include only events that pass all selections
            if self.data_truth is not None:
                pass_all = self.pass_reco & self.pass_truth
                self.data_reco = self.data_reco[pass_all]
                self.data_truth = self.data_truth[pass_all]
                self.weights = self.weights[pass_all]
                self.weights_mc = self.weights_mc[pass_all]

                self.pass_reco = self.pass_reco[pass_all]
                self.pass_truth = self.pass_truth[pass_all]
            else:
                self.data_reco = self.data_reco[self.pass_reco]
                self.weights = self.weights[self.pass_reco]
                self.pass_reco = self.pass_reco[self.pass_reco]
        else:
            # set dummy value
            dummy_value = float(dummy_value)
            setDummyValue(self.data_reco, ~self.pass_reco, dummy_value)
            setDummyValue(self.weights, ~self.pass_reco, dummy_value)
            if self.data_truth is not None:
                setDummyValue(self.data_truth, ~self.pass_truth, dummy_value)
                setDummyValue(self.weights_mc, ~self.pass_truth, dummy_value)

        # sanity check
        assert(len(self.data_reco)==len(self.weights))
        assert(len(self.data_reco)==len(self.pass_reco))
        if self.data_truth is not None:
            assert(len(self.data_reco)==len(self.data_truth))
            assert(len(self.data_truth)==len(self.weights_mc))
            assert(len(self.data_truth)==len(self.pass_truth))
