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

def setDummyValue(array, dummy_value):
    """
    Set dummy value of events that are flagged as dummy

    Parameters
    ----------
    array : numpy structured array
    dummy_value
    """
    isdummy = array['isDummy'] == 1
    for vname in list(array.dtype.names):
        if vname in ['isDummy', 'isMatched']:
            continue

        array[vname][isdummy] = dummy_value

def load_dataset_root(
        file_names,
        tree_name,
        variable_names = [],
        weight_name = None,
        dummy_value = None
    ):
    """
    Load and return a structured numpy array from a list of root files

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

        # flags for identifying events
        branches += ['isMatched', 'isDummy']

        if weight_name:
            branches.append(weight_name)
    else:
        # load everything
        branches = None

    data_array = uproot.lazy(intrees, filter_name=branches)

    # convert awkward array to numpy array for now
    # can probably use awkward array directly once it is more stable
    data_array = data_array.to_numpy()

    # convert units
    MeVtoGeV(data_array)

    if 'isDummy' in data_array.dtype.names:
        if dummy_value is None:
            # select only events that are matched and are not dummy
            good_event = np.logical_and(
                data_array['isMatched']==1, data_array['isDummy']==0
            )

            data_array = data_array[good_event]
        else:
            # include dummy events and set the value
            setDummyValue(data_array, dummy_value)

    return data_array

class DataHandlerROOT(DataHandler):
    """
    Load data from root files

    Parameters
    ----------
    filepaths :  str or sequence of str
        List of root file names to load
    treename_reco : str, default 'reco'
        Name of the reconstruction-level tree
    treename_truth : str, default 'parton'
        Name of the truth-level tree. If empty or None, skip loading the 
        truth-level tree
    variable_names : list of str, optional
        List of reco level variable names to read. If not provided, read all
    variable_names_mc : list of str, optional
        List of truth level variable names to read. If not provided, read all
    dummy_value : float, default None
        Dummy value for setting events that are flagged as dummy. If None, only
        include events that are not dummy in the array
    """
    def __init__(
        self,
        filepaths,
        treename_reco='reco',
        treename_truth='parton',
        variable_names=[],
        variable_names_mc=[],
        weights_name=None, #"normedWeight",
        weights_name_mc=None, #"weight_mc",
        dummy_value=None
    ):
        # load data from root files
        variable_names = dh._filter_variable_names(variable_names)
        self.data_reco = load_dataset_root(
            filepaths, treename_reco, variable_names, weights_name, dummy_value
            )

        # event weights
        if weights_name:
            self.weights = self.data_reco[weights_name]
        else:
            self.weights = np.ones(len(self.data_reco))

        # truth variables if available
        if treename_truth:
            variable_names_mc = dh._filter_variable_names(variable_names_mc)
            self.data_truth = load_dataset_root(
                filepaths, treename_truth, variable_names_mc, weights_name_mc,
                dummy_value
            )

            # rename fields of the truth array if the name is already in the reco
            # array
            prefix = 'truth_'
            newnames = {}
            for fname in self.data_truth.dtype.names:
                if fname in self.data_reco.dtype.names:
                    newnames[fname] = prefix+fname
            self.data_truth = rfn.rename_fields(self.data_truth, newnames)

            assert(len(self.data_reco)==len(self.data_truth))

            # mc weights
            if weights_name_mc:
                self.weights_mc = self.data_truth[weights_name_mc]
            else:
                self.weights_mc = np.ones(len(self.data_truth))
        else:
            self.data_truth = None
            self.weights_mc = None

    def __len__(self):
        """
        Get the number of events in the dataset.

        Returns
        -------
        non-negative int
        """
        return len(self.data_reco)

    def __contains__(self, variable):
        """
        Check if a variable is in the dataset.

        Parameters
        ----------
        variable : str

        Returns
        -------
        bool
        """
        inReco = variable in self.data_reco.dtype.names

        if self.data_truth is None:
            inTruth = False
        else:
            inTruth = variable in self.data_truth.dtype.names

        return inReco or inTruth

    def __iter__(self):
        """
        Create an iterator over the variable names in the dataset.

        Returns
        -------
        iterator of strings
        """
        if self.data_truth is None:
            return iter(self.data_reco.dtype.names)
        else:
            return iter(
                list(self.data_reco.dtype.names) +
                list(self.data_truth.dtype.names)
                )

    def _get_array(self, variable):
        """
        Return a 1D numpy array of the variable

        Parameters
        ----------
        variable : str
            Name of the variable

        Returns
        -------
        np.ndarray of shape (n_events,)
        """
        if variable in self.data_reco.dtype.names:
            return self.data_reco[str(variable)]

        if self.data_truth is not None:
            if variable in self.data_truth.dtype.names:
                return self.data_truth[str(variable)]

        # no 'variable' in the data arrays
        return None
