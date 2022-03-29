import uproot
import numpy as np

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
        variable_names
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
    variable_names : list of str
        List of variables to read. If not provided, read all available variables

    Returns
    -------
    A numpy ndarray for event features and a numpy ndarray of event selection flags
    """
    if isinstance(file_names, str):
        file_names = [file_names]

    intrees = [fname + ':' + tree_name for fname in file_names]

    #####
    # feature variables
    if variable_names:
        branches = [variable_names] if isinstance(variable_names, str) else variable_names

        # event flags
        branches += ['isMatched', 'isDummy']

        # in case of KLFitter
        branches.append('klfitter_logLikelihood')

        # for filtering a small fraction of reco events with zero weights
        # due to weight_pileup
        branches += ['totalWeight_nominal']

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

    if 'totalWeight_nominal' in data_array.dtype.names:
        # remove reco events with zero event weights
        pass_sel &= data_array['totalWeight_nominal'] > 0.

    # A special case where some matched events in truth trees contain invalid
    # (nan or inf) values
    if 'MC_thad_afterFSR_y' in data_array.dtype.names:
        invalid = np.isnan(data_array['MC_thad_afterFSR_y'])
        pass_sel &= (~invalid)
        #print("number of events with invalid value", np.sum(invalid))

    # TODO: return a masked array?
    return data_array, pass_sel

def load_weights_root(
        file_names,
        tree_name,
        weight_name,
        normalize_to_weight=''
    ):
    """
    Load event weights from ROOT files

    Parameters
    ----------
    file_names : str or file-like object; or sequence of str or file-like objects
        List of root files to load
    tree_name : str
        Name of the tree in root files
    weight_name : str
        Name of the event weight
    normalize_to_weight : str, default ''
        Name of the nominal weights to normalize to.
        Only needed when converting weights for systematic uncertainty variations

    Returns
    -------
    A numpy ndarray of event weights
    """
    if isinstance(file_names, str):
        file_names = [file_names]

    intrees = [fname + ':' + tree_name for fname in file_names]

    branches_w = [weight_name]

    # hard code the names here for now
    # components of the nominal weights
    weight_names_part = ["weight_bTagSF_DL1r_70", "weight_jvt", "weight_leptonSF", "weight_pileup", "weight_mc"]

    if normalize_to_weight:
        branches_w.append(normalize_to_weight)

        for wname in weight_names_part:
            if weight_name.startswith(wname):
                branches_w.append(wname)
                break

        # a special case
        if weight_name == "mc_generator_weights":
            branches_w.append("weight_mc")

    weights_array = uproot.lazy(intrees, filter_name = branches_w)

    warr = weights_array[weight_name].to_numpy().T

    if normalize_to_weight:
        warr_norm = weights_array[normalize_to_weight].to_numpy()

        wname_part = branches_w[-1]
        warr_part = weights_array[wname_part].to_numpy()

        sf = np.ones_like(warr_norm, float)
        np.divide(warr_norm, warr_part, out=sf, where = warr_part!=0)
        # e.g. w_pileup_DOWN * (w_normalized / w_pileup)
        warr = warr * sf

    return warr

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
    weight_type : str, default 'nominal'
        Type of event weights to load for systematic uncertainty variations
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
            filepaths, treename_reco, variable_names)

        # event weights
        # TODO?: weight_type
        # all nominal for now
        self.weights = load_weights_root(
            filepaths, treename_reco, weight_name = 'normalized_weight'
            )

        # truth variables if available
        if treename_truth:
            variable_names_mc = dh._filter_variable_names(variable_names_mc)
            self.data_truth, self.pass_truth = load_dataset_root(
                filepaths, treename_truth, variable_names_mc)

            # event weights
            self.weights_mc = load_weights_root(
                filepaths, treename_truth, weight_name = 'normalized_weight_mc'
                )
        else:
            self.data_truth = None
            self.pass_truth = None
            self.weights_mc = None

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

            # acceptance effect only for now
            if self.data_truth is not None:
                setDummyValue(self.data_truth, ~self.pass_truth, dummy_value)
                setDummyValue(self.weights_mc, ~self.pass_truth, dummy_value)
                #
                self.data_truth = self.data_truth[self.pass_reco]
                self.weights_mc = self.weights_mc[self.pass_reco]
                self.pass_truth = self.pass_truth[self.pass_reco]

            self.data_reco = self.data_reco[self.pass_reco]
            self.weights = self.weights[self.pass_reco]
            self.pass_reco = self.pass_reco[self.pass_reco]
            #
            # if account for both acceptance and efficiency corrections
            #setDummyValue(self.data_reco, ~self.pass_reco, dummy_value)
            #setDummyValue(self.weights, ~self.pass_reco, dummy_value)
            #if self.data_truth is not None:
            #    setDummyValue(self.data_truth, ~self.pass_truth, dummy_value)
            #    setDummyValue(self.weights_mc, ~self.pass_truth, dummy_value)

        # sanity check
        assert(len(self.data_reco)==len(self.weights))
        assert(len(self.data_reco)==len(self.pass_reco))
        if self.data_truth is not None:
            assert(len(self.data_reco)==len(self.data_truth))
            assert(len(self.data_truth)==len(self.weights_mc))
            assert(len(self.data_truth)==len(self.pass_truth))
