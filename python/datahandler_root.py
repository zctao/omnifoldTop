import uproot
import numpy as np

import datahandler as dh
from datahandler import DataHandler

import logging
logger = logging.getLogger('datahandler_root')

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
        branches = [variable_names] if isinstance(variable_names, str) else variable_names.copy()

        # event flags
        branches += ['isMatched', 'isDummy']

        # in case of KLFitter
        # check if any variable is KLFitter variable
        for vname in variable_names:
            if 'klfitter' in vname:
                branches.append('klfitter_logLikelihood')
                break

        # for filtering a small fraction of reco events with zero weights
        # due to weight_pileup
        branches += ['normalized_weight']

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

    if 'normalized_weight' in data_array.dtype.names:
        # remove reco events with zero event weights
        pass_sel &= data_array['normalized_weight'] != 0.

    # A special case where some matched events in truth trees contain invalid
    # (nan or inf) values
    if 'MC_thad_afterFSR_y' in data_array.dtype.names:
        invalid = np.isnan(data_array['MC_thad_afterFSR_y'])
        pass_sel &= (~invalid)
        #print("number of events with invalid value", np.sum(invalid))

    # TODO: return a masked array?
    return data_array, pass_sel

def read_weight_array(
        filename, # str, name of the root file to read
        tree_name, # str, name of TTree to read
        weight_nominal, # str, name of the nominal weight
        weight_component = None, # str, name of the weight component for normalization
        weight_variation = None, # str, name of the weight from systematic uncertainty variation
        weight_index = None, # int, index of the weight variation in case it is a vector  
    ):

    with uproot.open(filename) as f:
        events = f[tree_name]

        # nominal event weights
        if not weight_nominal in events:
            logger.error(f"Unkown branch {weight_nominal} in {filename}")
            return None

        warr = events[weight_nominal].array().to_numpy()

        warr_comp = None
        if weight_component is not None:
            if not weight_component in events:
                logger.warn(f"No branch {weight_component} in {filename}. Will use nominal event weights.")
            else:
                warr_comp = events[weight_component].array().to_numpy()
                assert(warr.shape == warr_comp.shape)

        warr_syst = None
        if weight_variation is not None:
            if not weight_variation in events:
                logger.warn(f"No branch {weight_variation} in {filename}. Will use nominal event weights.")
            else:
                if weight_index is None:
                    warr_syst = events[weight_variation].array().to_numpy()
                else:
                    warr_syst = events[weight_variation].array().to_numpy()[:,weight_index]
                assert(warr.shape == warr_syst.shape)

    #warr *= warr_syst / warr_comp
    if warr_comp is not None and warr_syst is not None:
        sf = np.zeros_like(warr, float)
        np.divide(warr_syst, warr_comp, out = sf, where = warr_comp!=0)
        warr *= sf

    return warr

def load_weights_root(
        file_names,
        tree_name,
        weight_name,
        weight_type = 'nominal'
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
        Name of the TTree branch that stores the nominal event weights
    weight_type : str
        Type of the event weights variations. Default: nominal

    Returns
    -------
    A numpy ndarray of event weights
    """

    # parse weight_type
    if weight_type != 'nominal':
        # Examples of expected 'weight_type':
        # 'weight_pileup_UP' or 'weight_bTagSF_DL1r_70_eigenvars_B_up:5'
        if len(weight_type.split(':')) > 1:
            # The weight variation branch is a vector of float
            weight_syst, index_w = weight_type.split(':')
            weight_syst = weight_syst.strip()
            index_w = int(index_w.strip())
        else:
            weight_syst = weight_type
            index_w = None

        # component of the nominal weights corresponding to the weight variation
        # All components of the nominal weights (hard code here for now)
        weight_comp = None
        all_weight_components = ["weight_bTagSF_DL1r_70", "weight_jvt", "weight_leptonSF", "weight_pileup", "weight_mc"]
        for wname in all_weight_components:
            if weight_syst.startswith(wname):
                weight_comp = wname
                break

        # A special case for the MC generator weight variations
        if weight_syst == 'mc_generator_weights':
            weight_comp = 'weight_mc'

        if weight_comp is None: # something's wrong
            raise RuntimeError(f"Unknown base component for event weight {weight_type}")

    else:
        weight_syst = None
        weight_comp = None
        index_w = None

    # loop over input files
    if isinstance(file_names, str):
        file_names = [file_names]

    weights_arr = np.empty(shape=(0,))

    for fname in file_names:

        weights_arr = np.concatenate([
            weights_arr,
            read_weight_array(
                fname,
                tree_name,
                weight_nominal = weight_name,
                weight_component = weight_comp,
                weight_variation = weight_syst,
                weight_index = index_w
            )
            ])

    return weights_arr

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
        self.weights = load_weights_root(
            filepaths, treename_reco, weight_name = 'normalized_weight',
            weight_type = weight_type
            )

        # truth variables if available
        if treename_truth:
            variable_names_mc = dh._filter_variable_names(variable_names_mc)
            self.data_truth, self.pass_truth = load_dataset_root(
                filepaths, treename_truth, variable_names_mc)

            # event weights
            #self.weights_mc = load_weights_root(
            #    filepaths, treename_truth, weight_name = 'normalized_weight_mc',
            #    weight_type = weight_type
            #    )
            self.weights_mc = self.weights.copy()
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

        # overflow/underflow flags to be set later
        self.underflow_overflow_reco = False
        self.underflow_overflow_truth = False